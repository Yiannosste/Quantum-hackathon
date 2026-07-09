"""Local simulator backend for the Simulation Phase.

For small problems (<= ``exact_threshold`` variables) it enumerates the full
state space and samples the Gibbs/Boltzmann distribution at a configurable
temperature — a stand-in for the noisy, low-energy-biased distribution a
tuned QAOA run produces. For larger problems it falls back to simulated
annealing restarts.

Deterministic under a fixed ``seed``, so participants can debug reproducibly.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence

from qlkit.backends.base import (
    Backend,
    BackendCapabilities,
    BackendError,
    CircuitSpec,
    JobHandle,
    JobStatus,
    RawResult,
)
from qlkit.core.qubo import QUBO


class LocalSimulatorBackend(Backend):
    name = "local_simulator"
    capabilities = BackendCapabilities(
        max_circuits_per_batch=64,
        max_shots=100_000,
        supports_async=False,
        is_hardware=False,
    )

    def __init__(
        self,
        seed: Optional[int] = None,
        temperature: float = 0.1,
        exact_threshold: int = 16,
        sa_restarts: int = 128,
        sa_sweeps: int = 200,
    ):
        self._rng = random.Random(seed)
        self.temperature = temperature
        self.exact_threshold = exact_threshold
        self.sa_restarts = sa_restarts
        self.sa_sweeps = sa_sweeps
        self._results: Dict[str, RawResult] = {}

    def submit(self, circuits: Sequence[CircuitSpec], shots: int) -> JobHandle:
        counts = [self._sample(spec.qubo, shots) for spec in circuits]
        handle = self._new_handle(len(circuits))
        self._results[handle.job_id] = RawResult(counts, {"simulator": "gibbs/sa"})
        return handle

    def status(self, handle: JobHandle) -> JobStatus:
        return JobStatus.DONE if handle.job_id in self._results else JobStatus.FAILED

    def result(self, handle: JobHandle) -> RawResult:
        try:
            return self._results[handle.job_id]
        except KeyError:
            raise BackendError(f"Unknown job {handle.job_id}") from None

    # -- sampling ---------------------------------------------------------

    def _sample(self, qubo: QUBO, shots: int) -> Dict[str, int]:
        n = qubo.num_vars
        if n == 0:
            return {}
        if n <= self.exact_threshold:
            return self._sample_exact(qubo, shots)
        return self._sample_annealing(qubo, shots)

    def _sample_exact(self, qubo: QUBO, shots: int) -> Dict[str, int]:
        n = qubo.num_vars
        states: List[str] = []
        energies: List[float] = []
        for idx in range(1 << n):
            bits = [(idx >> i) & 1 for i in range(n)]
            states.append("".join(map(str, bits)))
            energies.append(qubo.energy(bits))
        e_min = min(energies)
        spread = max(energies) - e_min or 1.0
        beta = 1.0 / (self.temperature * spread)
        weights = [math.exp(-beta * (e - e_min)) for e in energies]
        picks = self._rng.choices(range(len(states)), weights=weights, k=shots)
        counts: Dict[str, int] = {}
        for p in picks:
            counts[states[p]] = counts.get(states[p], 0) + 1
        return counts

    def _sample_annealing(self, qubo: QUBO, shots: int) -> Dict[str, int]:
        n = qubo.num_vars
        restarts = min(self.sa_restarts, shots)
        per_restart, remainder = divmod(shots, restarts)
        counts: Dict[str, int] = {}
        for r in range(restarts):
            bits = [self._rng.randint(0, 1) for _ in range(n)]
            energy = qubo.energy(bits)
            t_hot, t_cold = 2.0, 0.01
            for sweep in range(self.sa_sweeps):
                t = t_hot * (t_cold / t_hot) ** (sweep / max(1, self.sa_sweeps - 1))
                i = self._rng.randrange(n)
                bits[i] ^= 1
                new_energy = qubo.energy(bits)
                delta = new_energy - energy
                if delta <= 0 or self._rng.random() < math.exp(-delta / t):
                    energy = new_energy
                else:
                    bits[i] ^= 1
            key = "".join(map(str, bits))
            counts[key] = counts.get(key, 0) + per_restart + (1 if r < remainder else 0)
        return counts
