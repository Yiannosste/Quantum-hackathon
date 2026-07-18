"""Exhaustive reference oracle for small instances. ORGANIZER-SIDE ONLY.

Ground truth for calibration and tests: enumerates the full state space,
so it is only usable up to ~20 variables. Never hand this to participants —
it answers the exact question the hackathon asks them to answer via the
quantum pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import List, Optional, Tuple

from qlkit.core.problem import ProblemDefinition

_TOL = 1e-9


def brute_force_best_feasible(
    problem: ProblemDefinition,
) -> Tuple[Optional[Tuple[int, ...]], float]:
    """Best true objective over all hard-feasible bitstrings.
    Returns (None, inf) if no feasible state exists."""
    n = problem.num_vars()
    objective = problem.objective_qubo()
    hard = [c for c in problem.constraints() if c.hard]
    best_bits, best_cost = None, float("inf")
    for idx in range(1 << n):
        bits = tuple((idx >> i) & 1 for i in range(n))
        solution = problem.decode(bits)
        if any(c.violation(solution) > _TOL for c in hard):
            continue
        cost = objective.energy(bits)
        if cost < best_cost:
            best_bits, best_cost = bits, cost
    return best_bits, best_cost


@dataclass
class InstanceReport:
    num_vars: int
    optimum_bits: Optional[Tuple[int, ...]]
    optimum_cost: float
    top_feasible: List[Tuple[Tuple[int, ...], float]]  # (bits, cost), ascending
    num_feasible: int
    raw_feasible_fraction: float          # feasible / 2^n
    num_assignments: int                  # product of one-hot group sizes
    assignment_feasible_fraction: float = field(default=0.0)
    # feasible / num_assignments — the meaningful difficulty dial for
    # one-hot encodings, since only num_assignments states satisfy one-hot
    # at all.


def instance_report(problem: ProblemDefinition, top_k: int = 5) -> InstanceReport:
    n = problem.num_vars()
    objective = problem.objective_qubo()
    hard = [c for c in problem.constraints() if c.hard]
    feasible: List[Tuple[Tuple[int, ...], float]] = []
    for idx in range(1 << n):
        bits = tuple((idx >> i) & 1 for i in range(n))
        solution = problem.decode(bits)
        if any(c.violation(solution) > _TOL for c in hard):
            continue
        feasible.append((bits, objective.energy(bits)))
    feasible.sort(key=lambda pair: pair[1])

    groups = problem.one_hot_groups()
    num_assignments = prod(len(g) for g in groups) if groups else (1 << n)
    optimum_bits, optimum_cost = (feasible[0] if feasible else (None, float("inf")))
    return InstanceReport(
        num_vars=n,
        optimum_bits=optimum_bits,
        optimum_cost=optimum_cost,
        top_feasible=feasible[:top_k],
        num_feasible=len(feasible),
        raw_feasible_fraction=len(feasible) / (1 << n),
        num_assignments=num_assignments,
        assignment_feasible_fraction=(
            len(feasible) / num_assignments if num_assignments else 0.0
        ),
    )
