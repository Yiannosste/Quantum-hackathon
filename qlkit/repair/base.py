"""Classical repair hooks: fix "illegal" quantum results.

Participants extend RepairHeuristic. The solver applies it automatically to
every infeasible sample inside solve()/evaluate(), and SampleSet keeps both
pre- and post-repair solutions so teams can measure how much their
heuristic is doing versus the QPU.

GreedyAssignmentRepair is the shipped reference implementation. It is fully
generic over any problem that exposes one_hot_groups() — it knows nothing
about vehicles or warehouses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from qlkit.core.problem import ProblemDefinition
from qlkit.core.solution import Solution

_TOL = 1e-9


class RepairHeuristic(ABC):
    @abstractmethod
    def repair(self, solution: Solution, problem: ProblemDefinition) -> Solution:
        """Return a solution at least as feasible as the input. Must not
        mutate the input solution."""


class GreedyAssignmentRepair(RepairHeuristic):
    """Two-phase greedy repair for one-hot-encoded assignment problems.

    Phase 1: enforce every one-hot group (exactly one variable set),
    keeping/choosing the option with the lowest objective contribution.
    Phase 2: while hard constraints are still violated, try single-group
    reassignments and take the move that most reduces total hard violation
    (objective as tie-break), up to ``max_moves``.
    """

    def __init__(self, max_moves: int = 50):
        self.max_moves = max_moves

    def repair(self, solution: Solution, problem: ProblemDefinition) -> Solution:
        groups = problem.one_hot_groups()
        if not groups:
            return solution
        objective = problem.objective_qubo()
        bits = list(solution.bits)

        # Phase 1: exactly one bit per group.
        for group in groups:
            ones = [i for i in group if bits[i]]
            if len(ones) == 1:
                continue
            candidates = ones if len(ones) > 1 else group
            best = min(candidates, key=lambda i: self._solo_energy(objective, bits, group, i))
            for i in group:
                bits[i] = 1 if i == best else 0

        # Phase 2: greedy reassignment to reduce hard violations.
        hard = [c for c in problem.constraints() if c.hard]

        def total_violation(b: List[int]) -> float:
            sol = problem.decode(tuple(b))
            return sum(c.violation(sol) for c in hard)

        for _ in range(self.max_moves):
            current = total_violation(bits)
            if current <= _TOL:
                break
            best_bits = None
            best_score = (current, objective.energy(bits))
            for group in groups:
                chosen = next(i for i in group if bits[i])
                for alternative in group:
                    if alternative == chosen:
                        continue
                    trial = list(bits)
                    trial[chosen] = 0
                    trial[alternative] = 1
                    score = (total_violation(trial), objective.energy(trial))
                    if score < best_score:
                        best_score = score
                        best_bits = trial
            if best_bits is None:
                break  # local optimum; hand back best effort
            bits = best_bits

        repaired = problem.decode(tuple(bits))
        repaired.repaired = True
        return repaired

    @staticmethod
    def _solo_energy(objective, bits: List[int], group: List[int], keep: int) -> float:
        trial = list(bits)
        for i in group:
            trial[i] = 1 if i == keep else 0
        return objective.energy(trial)
