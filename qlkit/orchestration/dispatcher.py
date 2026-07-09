"""BatchDispatcher: aggregates concurrent circuit requests into
backend-sized batches, with retry/backoff on transient failures.

Participants never see this class. It exists so that when a team's outer
loop fires 20 evaluate() calls, they coalesce into 1-2 QPU jobs instead of
20 — which matters enormously against a shared hardware queue.

Mechanics:
- ``submit()`` enqueues a request and returns a Future immediately.
- A single background worker drains the queue: it takes the first pending
  request, waits up to ``max_wait_s`` for more to arrive, packs them up to
  the backend's max batch size (grouped by shot count, since shots are a
  batch-level parameter), submits ONE backend job, polls it to completion
  via the tracker, then fans the per-circuit counts back out to the
  originating futures.
- TransientBackendError (throttling etc.) triggers exponential backoff with
  jitter and up to ``max_retries`` resubmissions; permanent errors fail all
  futures in the batch.
"""

from __future__ import annotations

import queue
import random
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from qlkit.backends.base import (
    Backend,
    BackendError,
    CircuitSpec,
    JobStatus,
    RawResult,
    TransientBackendError,
)
from qlkit.orchestration.tracker import JobLedger, poll_until_done


class BudgetExceededError(BackendError):
    """Client-side circuit breaker tripped: max hardware jobs reached."""


@dataclass
class _Request:
    circuits: List[CircuitSpec]
    shots: int
    future: Future = field(default_factory=Future)


_STOP = object()


class BatchDispatcher:
    def __init__(
        self,
        backend: Backend,
        ledger: Optional[JobLedger] = None,
        max_wait_s: float = 0.2,
        max_retries: int = 3,
        retry_base_s: float = 0.5,
        poll_base_s: float = 0.25,
        poll_cap_s: float = 30.0,
        max_jobs: Optional[int] = None,
        on_status: Optional[Callable[[JobStatus], None]] = None,
    ):
        self.backend = backend
        self.ledger = ledger
        self.max_wait_s = max_wait_s
        self.max_retries = max_retries
        self.retry_base_s = retry_base_s
        self.poll_base_s = poll_base_s
        self.poll_cap_s = poll_cap_s
        self.max_jobs = max_jobs
        self.on_status = on_status
        self.jobs_submitted = 0
        self._queue: "queue.Queue" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._closed = False

    # -- public API --------------------------------------------------------

    def submit(self, circuits: Sequence[CircuitSpec], shots: int) -> "Future[RawResult]":
        """Returns a Future resolving to a RawResult with one counts dict per
        circuit, in the order given."""
        if self._closed:
            raise RuntimeError("dispatcher is shut down")
        request = _Request(list(circuits), shots)
        self._ensure_worker()
        self._queue.put(request)
        return request.future

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(_STOP)
        if wait and self._worker is not None:
            self._worker.join(timeout=60)

    # -- worker ------------------------------------------------------------

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._run, name="qlkit-dispatcher", daemon=True
                )
                self._worker.start()

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is _STOP:
                return
            batch = self._collect_window(first)
            if batch is None:  # _STOP seen mid-window; batch already failed
                return
            for shots, group in self._group_by_shots(batch):
                self._execute(group, shots)

    def _collect_window(self, first: _Request) -> Optional[List[_Request]]:
        batch = [first]
        total = len(first.circuits)
        cap = self.backend.capabilities.max_circuits_per_batch
        deadline = time.monotonic() + self.max_wait_s
        while total < cap:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty:
                break
            if item is _STOP:
                self._queue.put(_STOP)  # re-post so _run exits after this batch
                break
            batch.append(item)
            total += len(item.circuits)
        return batch

    @staticmethod
    def _group_by_shots(batch: List[_Request]):
        by_shots: dict = {}
        for req in batch:
            by_shots.setdefault(req.shots, []).append(req)
        return sorted(by_shots.items())

    def _execute(self, group: List[_Request], shots: int) -> None:
        circuits: List[CircuitSpec] = []
        slices: List[tuple] = []
        for req in group:
            start = len(circuits)
            circuits.extend(req.circuits)
            slices.append((start, len(circuits)))

        cap = self.backend.capabilities.max_circuits_per_batch
        try:
            if self.max_jobs is not None and self.jobs_submitted >= self.max_jobs:
                raise BudgetExceededError(
                    f"Hardware job budget of {self.max_jobs} reached. "
                    "Raise SolverConfig.max_qpu_jobs deliberately if you really need more."
                )
            raw_parts: List[RawResult] = []
            for chunk_start in range(0, len(circuits), cap):
                chunk = circuits[chunk_start : chunk_start + cap]
                raw_parts.append(self._submit_with_retry(chunk, shots))
            counts = [c for part in raw_parts for c in part.counts]
            meta = raw_parts[0].backend_meta if raw_parts else {}
        except BaseException as exc:
            for req in group:
                if not req.future.done():
                    req.future.set_exception(exc)
            return

        for req, (start, end) in zip(group, slices):
            req.future.set_result(RawResult(counts[start:end], dict(meta)))

    def _submit_with_retry(self, circuits: List[CircuitSpec], shots: int) -> RawResult:
        attempt = 0
        while True:
            try:
                handle = self.backend.submit(circuits, shots)
                self.jobs_submitted += 1
                if self.ledger is not None:
                    self.ledger.record(handle, JobStatus.QUEUED)
                poll_until_done(
                    self.backend,
                    handle,
                    ledger=self.ledger,
                    base_delay_s=self.poll_base_s,
                    cap_s=self.poll_cap_s,
                    on_update=self.on_status,
                )
                return self.backend.result(handle)
            except TransientBackendError:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                delay = self.retry_base_s * (2 ** (attempt - 1))
                time.sleep(delay * (0.8 + 0.4 * random.random()))
