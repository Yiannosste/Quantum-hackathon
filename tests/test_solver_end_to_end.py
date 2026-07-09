import json

import pytest

from problems.warehouse import WarehouseAllocationProblem
from qlkit import (
    GreedyAssignmentRepair,
    LocalSimulatorBackend,
    LogisticsSolver,
    MockBackend,
    SolverConfig,
)
from tests.conftest import brute_force_best_feasible

LAMBDAS = {"one_hot": 12.0, "capacity": 2.0}


def _solver(problem, backend, ledger_path, repair=True, **cfg):
    return LogisticsSolver(
        problem,
        backend,
        repair=GreedyAssignmentRepair() if repair else None,
        config=SolverConfig(ledger_path=ledger_path, batch_window_s=0.05, **cfg),
    )


def test_full_pipeline_on_mock_backend(vrp, ledger_path):
    with _solver(vrp, MockBackend(seed=0), ledger_path) as solver:
        result = solver.solve(LAMBDAS, shots=512)

    assert result.samples.total_shots > 0
    assert result.metrics.lambdas_used == LAMBDAS
    # Mock samples are uniform noise; repair should still deliver feasibility.
    assert result.best_solution is not None
    assert solver.validate(result.best_solution).passed


def test_local_simulator_finds_the_true_optimum(vrp, ledger_path):
    _, exact_cost = brute_force_best_feasible(vrp)
    with _solver(vrp, LocalSimulatorBackend(seed=7), ledger_path) as solver:
        result = solver.solve(LAMBDAS, shots=2048)

    assert result.best_solution is not None
    best = result.samples.best(feasible_only=True)
    assert best.true_objective == pytest.approx(exact_cost)
    # The optimal split of the small instance is the two distance clusters.
    routes = result.best_solution.payload["routes"]
    assert sorted(map(sorted, routes.values())) == [[0, 1, 2], [3, 4, 5]]


def test_same_code_runs_warehouse_problem(ledger_path):
    """Backend-agnostic AND problem-agnostic: identical call sites."""
    problem = WarehouseAllocationProblem.small_instance()
    _, exact_cost = brute_force_best_feasible(problem)
    with _solver(problem, LocalSimulatorBackend(seed=7), ledger_path) as solver:
        result = solver.solve(LAMBDAS, shots=2048)

    best = result.samples.best(feasible_only=True)
    assert best is not None
    assert best.true_objective == pytest.approx(exact_cost)


def test_evaluate_returns_tuner_ready_metrics(vrp, ledger_path):
    with _solver(vrp, LocalSimulatorBackend(seed=7), ledger_path) as solver:
        metrics = solver.evaluate(LAMBDAS, shots=1024)

    assert set(metrics.violations) == {"one_hot", "capacity"}
    assert 0.0 <= metrics.feasible_fraction <= 1.0
    assert metrics.shots == 1024
    assert metrics.scalarize() >= metrics.true_objective


def test_repair_hook_increases_feasible_fraction(vrp, ledger_path):
    """On pure-noise samples (mock), repair is doing all the work."""
    with _solver(vrp, MockBackend(seed=0), ledger_path, repair=False) as solver:
        without = solver.solve(LAMBDAS, shots=512)
    with _solver(vrp, MockBackend(seed=0), ledger_path) as solver:
        with_repair = solver.solve(LAMBDAS, shots=512)

    assert with_repair.samples.feasible_fraction > without.samples.feasible_fraction
    assert with_repair.samples.repaired_fraction > 0
    # Pre-repair solutions are preserved for analysis.
    repaired_records = [r for r in with_repair.samples.records if r.pre_repair]
    assert repaired_records


def test_submit_gather_non_blocking_flow(vrp, ledger_path):
    with _solver(vrp, LocalSimulatorBackend(seed=7), ledger_path) as solver:
        handles = [
            solver.submit({"one_hot": lam, "capacity": 2.0}, shots=256)
            for lam in (5.0, 10.0, 15.0)
        ]
        results = solver.gather(*handles)

    assert len(results) == 3
    for handle, result in zip(handles, results):
        assert result.lambdas == handle.lambdas


def test_from_config_switches_backend(vrp, tmp_path):
    config = {
        "backend": {"type": "mock", "options": {"seed": 1}},
        "solver": {"shots": 128, "ledger_path": str(tmp_path / "jobs.json")},
    }
    path = tmp_path / "backend.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with LogisticsSolver.from_config(vrp, path, repair=GreedyAssignmentRepair()) as solver:
        assert solver.backend.name == "mock"
        assert solver.config.shots == 128
        result = solver.solve(LAMBDAS)
        assert result.metrics.shots > 0


def test_ledger_records_solver_jobs(vrp, ledger_path):
    from qlkit.orchestration.tracker import JobLedger

    with _solver(vrp, MockBackend(seed=0), ledger_path) as solver:
        solver.solve(LAMBDAS, shots=64)

    jobs = JobLedger(ledger_path).all()
    assert len(jobs) == 1
    assert next(iter(jobs.values()))["status"] == "DONE"
