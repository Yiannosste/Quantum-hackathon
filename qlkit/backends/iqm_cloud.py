"""IQM Resonance cloud backend (Final Verification phase).

Compiles each CircuitSpec's QUBO into a QAOA circuit via qiskit-iqm and
submits it to an IQM quantum computer. Requires the optional dependency:

    pip install "qlkit[iqm]"     # or: pip install qiskit-iqm

Auth: set the IQM_TOKEN environment variable (never hardcode tokens in
notebooks — they end up in git).

NOTE for organizers: this module is exercised against real hardware, not in
CI. Keep the mapping logic thin; everything testable (batching, retries,
normalization) lives in the orchestration layer, which only sees the
Backend interface.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

from qlkit.backends.base import (
    Backend,
    BackendCapabilities,
    BackendError,
    CircuitSpec,
    JobHandle,
    JobStatus,
    RawResult,
    TransientBackendError,
)

_STATUS_MAP = {
    "pending compilation": JobStatus.QUEUED,
    "pending execution": JobStatus.QUEUED,
    "queued": JobStatus.QUEUED,
    "running": JobStatus.RUNNING,
    "ready": JobStatus.DONE,
    "done": JobStatus.DONE,
    "failed": JobStatus.FAILED,
    "aborted": JobStatus.CANCELLED,
}


class IQMCloudBackend(Backend):
    name = "iqm_cloud"
    capabilities = BackendCapabilities(
        max_circuits_per_batch=20,
        max_shots=20_000,
        supports_async=True,
        is_hardware=True,
    )

    def __init__(self, server_url: str, qaoa_layers: int = 1, token: Optional[str] = None):
        try:
            from iqm.qiskit_iqm import IQMProvider  # noqa: F401
            from qiskit import QuantumCircuit, transpile  # noqa: F401
        except ImportError as exc:
            raise BackendError(
                "IQMCloudBackend requires the optional IQM dependencies. "
                'Install them with: pip install "qlkit[iqm]"'
            ) from exc

        token = token or os.environ.get("IQM_TOKEN")
        if not token:
            raise BackendError(
                "No IQM token found. Set the IQM_TOKEN environment variable."
            )
        os.environ.setdefault("IQM_TOKEN", token)

        from iqm.qiskit_iqm import IQMProvider

        self._provider = IQMProvider(server_url)
        self._backend = self._provider.get_backend()
        self.qaoa_layers = qaoa_layers
        self._jobs: Dict[str, object] = {}

    # -- Backend interface -------------------------------------------------

    def submit(self, circuits: Sequence[CircuitSpec], shots: int) -> JobHandle:
        from qiskit import transpile

        compiled = [
            transpile(self._to_qaoa_circuit(spec), self._backend)
            for spec in circuits
        ]
        try:
            job = self._backend.run(compiled, shots=shots)
        except Exception as exc:  # network / throttling — let dispatcher retry
            raise TransientBackendError(str(exc)) from exc
        handle = self._new_handle(len(circuits), iqm_job_id=str(job.job_id()))
        self._jobs[handle.job_id] = job
        return handle

    def status(self, handle: JobHandle) -> JobStatus:
        job = self._jobs.get(handle.job_id)
        if job is None:
            return JobStatus.FAILED
        raw = str(job.status()).split(".")[-1].lower()
        return _STATUS_MAP.get(raw, JobStatus.RUNNING)

    def result(self, handle: JobHandle) -> RawResult:
        job = self._jobs[handle.job_id]
        result = job.result()
        counts_list = []
        for i in range(handle.num_circuits):
            counts = result.get_counts(i) if handle.num_circuits > 1 else result.get_counts()
            # Qiskit bitstrings are little-endian (qubit 0 rightmost);
            # qlkit convention is variable 0 leftmost — reverse here so the
            # rest of the stack never has to know.
            counts_list.append({key[::-1]: v for key, v in counts.items()})
        return RawResult(counts_list, {"iqm_job_id": handle.meta.get("iqm_job_id")})

    # -- QUBO -> QAOA ------------------------------------------------------

    def _to_qaoa_circuit(self, spec: CircuitSpec):
        """Standard QAOA ansatz for the QUBO's Ising form. Angles come from
        spec.params (defaults are deliberately mediocre — tuning them is
        part of the hackathon's advanced track)."""
        from qiskit import QuantumCircuit

        qubo = spec.qubo
        n = qubo.num_vars
        gammas = spec.params.get("gammas", [0.8] * self.qaoa_layers)
        betas = spec.params.get("betas", [0.4] * self.qaoa_layers)

        # QUBO -> Ising: x_i = (1 - z_i) / 2
        h = [0.0] * n
        j: Dict[tuple, float] = {}
        for (a, b), coeff in qubo.terms.items():
            if a == b:
                h[a] += -coeff / 2.0
            else:
                j[(a, b)] = j.get((a, b), 0.0) + coeff / 4.0
                h[a] += -coeff / 4.0
                h[b] += -coeff / 4.0

        qc = QuantumCircuit(n, n)
        qc.h(range(n))
        for gamma, beta in zip(gammas, betas):
            for (a, b), coupling in j.items():
                qc.cx(a, b)
                qc.rz(2.0 * gamma * coupling, b)
                qc.cx(a, b)
            for i, field in enumerate(h):
                if field:
                    qc.rz(2.0 * gamma * field, i)
            qc.rx(2.0 * beta, range(n))
        qc.measure(range(n), range(n))
        return qc
