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


# Exhaustive reference optimum, shared with the organizer calibration tool.
from tools.oracle import brute_force_best_feasible  # noqa: E402, F401
