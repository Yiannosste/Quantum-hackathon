"""Job tracking: persistent ledger + adaptive polling.

Ledger: every submitted job is written to disk immediately, so a restarted
notebook kernel doesn't orphan QPU jobs — the job ID and status survive.

Polling: exponential backoff with jitter (base -> cap), one status request
at a time per job. The dispatcher multiplexes all in-flight jobs through a
single worker, so N notebooks don't turn into N*jobs poll loops hammering
the API.
"""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from qlkit.backends.base import Backend, BackendError, JobHandle, JobStatus


class JobLedger:
    def __init__(self, path: str = ".qlkit/jobs.json"):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._jobs: Dict[str, dict] = {}
        if self.path.exists():
            try:
                self._jobs = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._jobs = {}

    def record(self, handle: JobHandle, status: JobStatus) -> None:
        with self._lock:
            entry = self._jobs.setdefault(
                handle.job_id,
                {
                    "backend": handle.backend_name,
                    "num_circuits": handle.num_circuits,
                    "submitted_at": handle.submitted_at,
                    "meta": handle.meta,
                },
            )
            entry["status"] = status.value
            entry["updated_at"] = time.time()
            self._flush()

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self._jobs)

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._jobs, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def poll_until_done(
    backend: Backend,
    handle: JobHandle,
    ledger: Optional[JobLedger] = None,
    base_delay_s: float = 0.25,
    backoff_factor: float = 1.5,
    cap_s: float = 30.0,
    timeout_s: Optional[float] = None,
    on_update: Optional[Callable[[JobStatus], None]] = None,
) -> None:
    """Block until DONE; raise BackendError on FAILED/CANCELLED,
    TimeoutError on timeout. No sleep before the first poll, so instant
    backends (simulator, mock) return immediately."""
    started = time.monotonic()
    delay = base_delay_s
    while True:
        status = backend.status(handle)
        if ledger is not None:
            ledger.record(handle, status)
        if on_update is not None:
            on_update(status)
        if status is JobStatus.DONE:
            return
        if status in (JobStatus.FAILED, JobStatus.CANCELLED):
            raise BackendError(f"Job {handle.job_id} ended in state {status.value}")
        if timeout_s is not None and time.monotonic() - started > timeout_s:
            raise TimeoutError(f"Job {handle.job_id} still {status.value} after {timeout_s}s")
        time.sleep(delay * (0.8 + 0.4 * random.random()))
        delay = min(delay * backoff_factor, cap_s)
