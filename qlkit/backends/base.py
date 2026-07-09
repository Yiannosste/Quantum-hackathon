"""Backend abstraction: the only contract the orchestration layer relies on.

Key rule: ``submit()`` is ALWAYS non-blocking, even on the local simulator
(which simply resolves instantly). Participant code written against the
simulator is therefore identical to QPU code — no async rewrite on day two.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Sequence

from qlkit.core.qubo import QUBO


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class BackendError(Exception):
    """Permanent failure; the dispatcher will not retry."""


class TransientBackendError(BackendError):
    """Retryable failure (throttling, queue hiccup, network blip)."""


@dataclass
class CircuitSpec:
    """Backend-agnostic description of one sampling task.

    Backends interpret it natively: the local simulator samples the QUBO's
    Boltzmann distribution, the IQM backend compiles it to a QAOA circuit.
    Participants never construct circuits by hand.
    """

    qubo: QUBO
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobHandle:
    job_id: str
    backend_name: str
    num_circuits: int
    submitted_at: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RawResult:
    """One counts dict per submitted circuit: bitstring -> shot count.
    Bitstring convention: leftmost character is variable 0."""

    counts: List[Dict[str, int]]
    backend_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendCapabilities:
    max_circuits_per_batch: int
    max_shots: int
    supports_async: bool
    is_hardware: bool


class Backend(ABC):
    name: str = "base"
    capabilities: BackendCapabilities

    @abstractmethod
    def submit(self, circuits: Sequence[CircuitSpec], shots: int) -> JobHandle:
        """Non-blocking; returns a trackable handle immediately."""

    @abstractmethod
    def status(self, handle: JobHandle) -> JobStatus: ...

    @abstractmethod
    def result(self, handle: JobHandle) -> RawResult:
        """Only valid once status() is DONE."""

    def close(self) -> None:
        pass

    def _new_handle(self, num_circuits: int, **meta: Any) -> JobHandle:
        return JobHandle(
            job_id=uuid.uuid4().hex[:12],
            backend_name=self.name,
            num_circuits=num_circuits,
            submitted_at=time.time(),
            meta=dict(meta),
        )
