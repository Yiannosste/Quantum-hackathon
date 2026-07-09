from qlkit.repair.base import GreedyAssignmentRepair
from qlkit.validation.validator import validate


def _total_hard_violation(problem, solution):
    return sum(
        c.violation(solution) for c in problem.constraints() if c.hard
    )


def test_repair_fixes_all_zero_bitstring(vrp):
    broken = vrp.decode((0,) * vrp.num_vars())
    assert _total_hard_violation(vrp, broken) > 0

    fixed = GreedyAssignmentRepair().repair(broken, vrp)
    assert fixed.repaired
    assert _total_hard_violation(vrp, fixed) == 0
    assert validate(vrp, fixed).passed


def test_repair_fixes_everything_on_one_vehicle(vrp):
    bits = [0] * vrp.num_vars()
    for c in range(vrp.num_customers):
        bits[vrp.var(c, 0)] = 1  # all 6 customers on vehicle 0: overload
    broken = vrp.decode(tuple(bits))
    assert _total_hard_violation(vrp, broken) > 0

    fixed = GreedyAssignmentRepair().repair(broken, vrp)
    assert _total_hard_violation(vrp, fixed) == 0


def test_repair_never_worsens_feasible_input(vrp):
    bits = [0] * vrp.num_vars()
    for c in range(3):
        bits[vrp.var(c, 0)] = 1
    for c in range(3, 6):
        bits[vrp.var(c, 1)] = 1
    feasible = vrp.decode(tuple(bits))
    fixed = GreedyAssignmentRepair().repair(feasible, vrp)
    assert _total_hard_violation(vrp, fixed) == 0


def test_repair_does_not_mutate_input(vrp):
    broken = vrp.decode((1,) * vrp.num_vars())
    original_bits = broken.bits
    GreedyAssignmentRepair().repair(broken, vrp)
    assert broken.bits == original_bits
    assert not broken.repaired


def test_repair_works_on_warehouse_too(warehouse):
    """Same heuristic, different problem — the generic contract at work."""
    broken = warehouse.decode((1,) * warehouse.num_vars())
    fixed = GreedyAssignmentRepair().repair(broken, warehouse)
    assert _total_hard_violation(warehouse, fixed) == 0
