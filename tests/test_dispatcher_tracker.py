import threading

import pytest

from qlkit.backends.base import BackendError, CircuitSpec, JobStatus
from qlkit.backends.mock import MockBackend
from qlkit.core.qubo import QUBOBuilder
from qlkit.orchestration.dispatcher import BatchDispatcher, BudgetExceededError
from qlkit.orchestration.tracker import JobLedger, poll_until_done


def _spec():
    return CircuitSpec(QUBOBuilder().add_linear(0, -1.0).add_linear(1, 1.0).build())


def test_concurrent_submits_coalesce_into_one_backend_job():
    backend = MockBackend(seed=0)
    dispatcher = BatchDispatcher(backend, max_wait_s=0.5)
    try:
        barrier = threading.Barrier(5)
        futures = []

        def submit():
            barrier.wait()
            futures.append(dispatcher.submit([_spec()], shots=100))

        threads = [threading.Thread(target=submit) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        results = [f.result(timeout=10) for f in futures]

        assert all(len(r.counts) == 1 for r in results)
        # 5 requests, same shots, within one batch window -> 1 backend job.
        assert backend.submit_calls == 1
        assert backend.submitted_batch_sizes == [5]
    finally:
        dispatcher.shutdown()


def test_dispatcher_retries_transient_failures():
    backend = MockBackend(seed=0, fail_first_submits=2)
    dispatcher = BatchDispatcher(backend, max_wait_s=0.05, max_retries=3, retry_base_s=0.01)
    try:
        result = dispatcher.submit([_spec()], shots=50).result(timeout=10)
        assert len(result.counts) == 1
        assert backend.submit_calls == 3  # 2 failures + 1 success
    finally:
        dispatcher.shutdown()


def test_dispatcher_gives_up_after_max_retries():
    backend = MockBackend(seed=0, fail_first_submits=10)
    dispatcher = BatchDispatcher(backend, max_wait_s=0.05, max_retries=2, retry_base_s=0.01)
    try:
        future = dispatcher.submit([_spec()], shots=50)
        with pytest.raises(BackendError):
            future.result(timeout=10)
    finally:
        dispatcher.shutdown()


def test_budget_circuit_breaker():
    backend = MockBackend(seed=0)
    dispatcher = BatchDispatcher(backend, max_wait_s=0.01, max_jobs=1)
    try:
        dispatcher.submit([_spec()], shots=10).result(timeout=10)
        with pytest.raises(BudgetExceededError):
            dispatcher.submit([_spec()], shots=10).result(timeout=10)
    finally:
        dispatcher.shutdown()


def test_ledger_persists_across_instances(tmp_path):
    path = tmp_path / "jobs.json"
    backend = MockBackend(seed=0, queue_polls=1)
    ledger = JobLedger(str(path))
    handle = backend.submit([_spec()], 10)
    poll_until_done(backend, handle, ledger=ledger, base_delay_s=0.01)

    reloaded = JobLedger(str(path))
    entry = reloaded.get(handle.job_id)
    assert entry is not None
    assert entry["status"] == "DONE"
    assert entry["backend"] == "mock"


def test_poll_until_done_sees_queue_then_done():
    backend = MockBackend(seed=0, queue_polls=3)
    handle = backend.submit([_spec()], 10)
    seen = []
    poll_until_done(backend, handle, base_delay_s=0.01, on_update=seen.append)
    assert seen[-1] is JobStatus.DONE
    assert JobStatus.QUEUED in seen
