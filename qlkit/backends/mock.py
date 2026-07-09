"""Deterministic mock backend: zero dependencies, instant results.

Used for CI, day-one onboarding, and for exercising the middleware's error
paths (``fail_first_submits`` raises TransientBackendError to test retry
logic; ``latency_s`` and ``queue_polls`` simulate a slow cloud queue so the
polling/progress UX can be demoed without hardware).
"""

from __future__ import annotations

import random
import time
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


class MockBackend(Backend):
    name = "mock"
    capabilities = BackendCapabilities(
        max_circuits_per_batch=20,
        max_shots=20_000,
        supports_async=True,
        is_hardware=False,
    )

    def __init__(
        self,
        seed: int = 0,
        latency_s: float = 0.0,
        fail_first_submits: int = 0,
        queue_polls: int = 0,
    ):
        self._rng = random.Random(seed)
        self.latency_s = latency_s
        self.fail_first_submits = fail_first_submits
        self.queue_polls = queue_polls
        self.submit_calls = 0
        self.submitted_batch_sizes: list[int] = []
        self._results: Dict[str, RawResult] = {}
        self._polls_left: Dict[str, int] = {}

    def submit(self, circuits: Sequence[CircuitSpec], shots: int) -> JobHandle:
        self.submit_calls += 1
        if self.submit_calls <= self.fail_first_submits:
            raise TransientBackendError("mock: simulated 429 Too Many Requests")
        if self.latency_s:
            time.sleep(self.latency_s)
        self.submitted_batch_sizes.append(len(circuits))
        counts = []
        for spec in circuits:
            n = spec.qubo.num_vars
            per_circuit: Dict[str, int] = {}
            for _ in range(min(shots, 256)):
                key = "".join(str(self._rng.randint(0, 1)) for _ in range(n))
                per_circuit[key] = per_circuit.get(key, 0) + max(1, shots // 256)
            counts.append(per_circuit)
        handle = self._new_handle(len(circuits))
        self._results[handle.job_id] = RawResult(counts, {"mock": True})
        self._polls_left[handle.job_id] = self.queue_polls
        return handle

    def status(self, handle: JobHandle) -> JobStatus:
        if handle.job_id not in self._results:
            return JobStatus.FAILED
        if self._polls_left.get(handle.job_id, 0) > 0:
            self._polls_left[handle.job_id] -= 1
            return JobStatus.QUEUED
        return JobStatus.DONE

    def result(self, handle: JobHandle) -> RawResult:
        try:
            return self._results[handle.job_id]
        except KeyError:
            raise BackendError(f"Unknown job {handle.job_id}") from None
