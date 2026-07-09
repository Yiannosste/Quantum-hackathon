"""YOUR repair heuristic goes here.

The toolkit calls repair() on every infeasible sample, automatically,
inside solve()/evaluate(). Your score depends heavily on this class:
on real hardware most raw samples violate constraints, and a smart repair
turns them into feasible solutions instead of wasted shots.

Ideas to beat the shipped GreedyAssignmentRepair:
  - repair toward the objective, not just toward feasibility
    (choose the reassignment that is both legal AND cheap)
  - local search: after reaching feasibility, keep making
    objective-improving swaps that stay feasible
  - problem-specific moves (e.g. swap two customers between vehicles
    instead of moving one)
"""

from qlkit import GreedyAssignmentRepair, RepairHeuristic, Solution
from qlkit.core.problem import ProblemDefinition


class MyRepair(GreedyAssignmentRepair):
    """Extends the shipped greedy repair. Replace/override as you like —
    you only must honor the contract: return a solution at least as
    feasible as the input, never mutate the input."""

    def repair(self, solution: Solution, problem: ProblemDefinition) -> Solution:
        repaired = super().repair(solution, problem)
        # TODO: add a feasibility-preserving local search pass here.
        return repaired
