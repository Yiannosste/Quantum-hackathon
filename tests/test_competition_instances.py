"""Competition-instance calibration guarantees.

Day-2 tests are skipped when private/competition_day2_disruption.json is
absent (it is gitignored and only exists on organizer machines), so CI on
the public repo stays green without leaking the disruption.
"""

from pathlib import Path

import pytest

from problems.vrp.definition import VehicleRoutingProblem
from qlkit import GreedyAssignmentRepair, LocalSimulatorBackend, LogisticsSolver, SolverConfig
from tools.oracle import instance_report

DAY2_PATH = Path(__file__).resolve().parent.parent / "private" / "competition_day2_disruption.json"

requires_day2 = pytest.mark.skipif(
    not DAY2_PATH.exists(), reason="private Day-2 disruption file not present"
)


@pytest.fixture(scope="module")
def day1():
    return VehicleRoutingProblem.competition_instance()


@pytest.fixture(scope="module")
def day1_report(day1):
    return instance_report(day1)


def _partition(problem, bits):
    routes = problem.decode(bits).payload["routes"]
    return sorted(sorted(members) for members in routes.values())


def test_day1_shape(day1):
    assert day1.num_vars() == 16
    assert day1.num_customers == 8
    assert day1.num_vehicles == 2
    # Capacity must genuinely bind: little slack over total demand.
    assert sum(day1.capacities) - sum(day1.demands) <= 2


def test_day1_oracle_and_feasibility_band(day1_report):
    assert day1_report.optimum_bits is not None
    assert day1_report.optimum_cost == pytest.approx(38.74, abs=0.01)
    assert 0.10 <= day1_report.assignment_feasible_fraction <= 0.70


def test_day1_repair_reaches_feasibility_from_zeros(day1):
    broken = day1.decode((0,) * day1.num_vars())
    fixed = GreedyAssignmentRepair().repair(broken, day1)
    hard = [c for c in day1.constraints() if c.hard]
    assert sum(c.violation(fixed) for c in hard) == 0


def test_day1_simulator_recovers_oracle_optimum(day1, day1_report, tmp_path):
    with LogisticsSolver(
        day1,
        LocalSimulatorBackend(seed=7),
        repair=GreedyAssignmentRepair(),
        config=SolverConfig(ledger_path=str(tmp_path / "jobs.json")),
    ) as solver:
        result = solver.solve({"one_hot": 12.0, "capacity": 2.0}, shots=4096)
    best = result.samples.best(feasible_only=True)
    assert best is not None
    assert best.true_objective == pytest.approx(day1_report.optimum_cost)


@requires_day2
def test_day2_day1_optimum_becomes_infeasible(day1, day1_report):
    day2 = VehicleRoutingProblem.from_json(DAY2_PATH)
    stale = day2.decode(day1_report.optimum_bits)
    hard_violation = sum(
        c.violation(stale) for c in day2.constraints() if c.hard
    )
    assert hard_violation > 0


@requires_day2
def test_day2_optimum_is_a_different_partition(day1, day1_report):
    day2 = VehicleRoutingProblem.from_json(DAY2_PATH)
    day2_report = instance_report(day2)
    assert day2_report.optimum_bits is not None
    assert _partition(day2, day2_report.optimum_bits) != _partition(
        day1, day1_report.optimum_bits
    )


@requires_day2
def test_day2_still_solvable_by_the_pipeline(tmp_path):
    """The live-event guarantee: an unchanged tuner+repair stack re-converges
    on the disrupted instance."""
    day2 = VehicleRoutingProblem.from_json(DAY2_PATH)
    day2_report = instance_report(day2)
    with LogisticsSolver(
        day2,
        LocalSimulatorBackend(seed=7),
        repair=GreedyAssignmentRepair(),
        config=SolverConfig(ledger_path=str(tmp_path / "jobs.json")),
    ) as solver:
        result = solver.solve({"one_hot": 12.0, "capacity": 2.0}, shots=4096)
    best = result.samples.best(feasible_only=True)
    assert best is not None
    # Repair-assisted pipeline must land on (or very near) the new optimum.
    assert best.true_objective <= day2_report.optimum_cost * 1.10
