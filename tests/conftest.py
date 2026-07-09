import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from problems.vrp import VehicleRoutingProblem
from problems.warehouse import WarehouseAllocationProblem


@pytest.fixture
def vrp():
    return VehicleRoutingProblem.small_instance()


@pytest.fixture
def warehouse():
    return WarehouseAllocationProblem.small_instance()


@pytest.fixture
def ledger_path(tmp_path):
    return str(tmp_path / "jobs.json")


def brute_force_best_feasible(problem):
    """Exhaustive reference optimum: best true objective over all hard-
    feasible bitstrings. Ground truth for solver assertions."""
    n = problem.num_vars()
    objective = problem.objective_qubo()
    hard = [c for c in problem.constraints() if c.hard]
    best_bits, best_cost = None, float("inf")
    for idx in range(1 << n):
        bits = tuple((idx >> i) & 1 for i in range(n))
        solution = problem.decode(bits)
        if any(c.violation(solution) > 1e-9 for c in hard):
            continue
        cost = objective.energy(bits)
        if cost < best_cost:
            best_bits, best_cost = bits, cost
    return best_bits, best_cost
