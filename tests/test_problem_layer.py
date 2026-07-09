import pytest

from problems.common import CapacityConstraint, OneHotConstraint
from qlkit.core.solution import Solution


def _feasible_bits(vrp):
    """Customers 0-2 on vehicle 0, customers 3-5 on vehicle 1 (the natural
    cluster split of the small instance)."""
    bits = [0] * vrp.num_vars()
    for c in range(3):
        bits[vrp.var(c, 0)] = 1
    for c in range(3, 6):
        bits[vrp.var(c, 1)] = 1
    return tuple(bits)


def test_one_hot_constraint_violation_is_continuous():
    constraint = OneHotConstraint("oh", groups=[[0, 1], [2, 3]])
    assert constraint.violation(Solution(bits=(1, 0, 0, 1))) == 0.0
    assert constraint.violation(Solution(bits=(0, 0, 0, 1))) == 1.0
    assert constraint.violation(Solution(bits=(1, 1, 0, 0))) == 2.0


def test_one_hot_penalty_matches_squared_form():
    constraint = OneHotConstraint("oh", groups=[[0, 1, 2]])
    q = constraint.penalty_terms(3.0)
    for bits in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]:
        expected = 3.0 * (1 - sum(bits)) ** 2
        assert q.energy(bits) == pytest.approx(expected)


def test_capacity_constraint_only_counts_overload():
    constraint = CapacityConstraint(
        "cap", loads={"bin": {0: 5.0, 1: 4.0}}, capacities={"bin": 6.0}
    )
    assert constraint.violation(Solution(bits=(1, 0))) == 0.0
    assert constraint.violation(Solution(bits=(1, 1))) == pytest.approx(3.0)


def test_capacity_penalty_matches_squared_form():
    constraint = CapacityConstraint(
        "cap", loads={"bin": {0: 5.0, 1: 4.0}}, capacities={"bin": 6.0}
    )
    q = constraint.penalty_terms(2.0)
    for bits in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        load = 5.0 * bits[0] + 4.0 * bits[1]
        assert q.energy(bits) == pytest.approx(2.0 * (load - 6.0) ** 2)


def test_build_qubo_injects_lambdas(vrp):
    bits = _feasible_bits(vrp)
    objective_only = vrp.objective_qubo().energy(bits)

    # Feasible bits: one-hot penalty vanishes; the equality-style capacity
    # penalty still charges the underloaded vehicle (loads 11 and 10 vs
    # capacity 11) exactly lambda * (10 - 11)^2 = lambda. This is the
    # documented caveat of the slack-free penalty (see problems/common.py).
    low = vrp.build_qubo({"one_hot": 1.0, "capacity": 1.0}).energy(bits)
    high = vrp.build_qubo({"one_hot": 50.0, "capacity": 50.0}).energy(bits)
    assert low == pytest.approx(objective_only + 1.0)
    assert high == pytest.approx(objective_only + 50.0)

    # Infeasible bits must be penalized proportionally to lambda.
    broken = list(bits)
    broken[vrp.var(0, 0)] = 0  # customer 0 unassigned
    low_e = vrp.build_qubo({"one_hot": 1.0}).energy(broken)
    high_e = vrp.build_qubo({"one_hot": 50.0}).energy(broken)
    assert high_e > low_e


def test_build_qubo_rejects_unknown_lambda_names(vrp):
    with pytest.raises(ValueError, match="Unknown constraint name"):
        vrp.build_qubo({"onehot_typo": 3.0})


def test_vrp_decode(vrp):
    solution = vrp.decode(_feasible_bits(vrp))
    assert solution.payload["routes"] == {0: [0, 1, 2], 1: [3, 4, 5]}


def test_warehouse_uses_same_generic_constraints(warehouse, vrp):
    """The agnosticism proof: both problems expose the same constraint
    names/types, so the same tuner code works on either."""
    assert {c.name for c in warehouse.constraints()} == {
        c.name for c in vrp.constraints()
    }
    assert warehouse.num_vars() == 12
    assert len(warehouse.one_hot_groups()) == 4
