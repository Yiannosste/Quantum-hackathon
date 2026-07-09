import pytest

from qlkit.backends.base import CircuitSpec, JobStatus, TransientBackendError
from qlkit.backends.local_simulator import LocalSimulatorBackend
from qlkit.backends.mock import MockBackend
from qlkit.core.qubo import QUBOBuilder, bits_from_string


def _simple_qubo():
    # Minimum at x0=1, x1=0: energies 0/-2/1/0
    return QUBOBuilder().add_linear(0, -2.0).add_linear(1, 1.0).add(0, 1, 1.0).build()


def test_local_simulator_is_instantly_done_and_biased_to_ground_state():
    backend = LocalSimulatorBackend(seed=1)
    handle = backend.submit([CircuitSpec(_simple_qubo())], shots=1000)
    assert backend.status(handle) is JobStatus.DONE
    counts = backend.result(handle).counts[0]
    assert sum(counts.values()) == 1000
    # Ground state '10' must dominate a low-temperature Gibbs sample.
    assert counts.get("10", 0) > 600


def test_local_simulator_deterministic_under_seed():
    backend1 = LocalSimulatorBackend(seed=42)
    backend2 = LocalSimulatorBackend(seed=42)
    h1 = backend1.submit([CircuitSpec(_simple_qubo())], 500)
    h2 = backend2.submit([CircuitSpec(_simple_qubo())], 500)
    assert backend1.result(h1).counts == backend2.result(h2).counts


def test_local_simulator_annealing_path_for_large_problems():
    n = 20  # above exact_threshold=16 -> simulated annealing
    builder = QUBOBuilder(num_vars=n)
    for i in range(n):
        builder.add_linear(i, -1.0)  # ground state: all ones
    backend = LocalSimulatorBackend(seed=3, sa_restarts=16, sa_sweeps=400)
    handle = backend.submit([CircuitSpec(builder.build())], shots=64)
    counts = backend.result(handle).counts[0]
    assert sum(counts.values()) == 64
    best = min(counts, key=lambda k: builder.build().energy(bits_from_string(k)))
    assert best == "1" * n


def test_mock_backend_counts_shape_and_determinism():
    spec = CircuitSpec(_simple_qubo())
    m1, m2 = MockBackend(seed=5), MockBackend(seed=5)
    r1 = m1.result(m1.submit([spec, spec], 100))
    r2 = m2.result(m2.submit([spec, spec], 100))
    assert len(r1.counts) == 2
    assert r1.counts == r2.counts
    assert all(len(key) == 2 for key in r1.counts[0])


def test_mock_backend_transient_failures_then_success():
    backend = MockBackend(seed=0, fail_first_submits=2)
    spec = CircuitSpec(_simple_qubo())
    with pytest.raises(TransientBackendError):
        backend.submit([spec], 10)
    with pytest.raises(TransientBackendError):
        backend.submit([spec], 10)
    handle = backend.submit([spec], 10)  # third attempt succeeds
    assert backend.status(handle) is JobStatus.DONE


def test_mock_backend_simulated_queue():
    backend = MockBackend(seed=0, queue_polls=2)
    handle = backend.submit([CircuitSpec(_simple_qubo())], 10)
    assert backend.status(handle) is JobStatus.QUEUED
    assert backend.status(handle) is JobStatus.QUEUED
    assert backend.status(handle) is JobStatus.DONE
