"""Result normalization: raw backend counts -> clean SampleSet.

For every distinct measured bitstring:
  1. decode to a domain solution,
  2. score constraint violations,
  3. if infeasible and a repair hook is present, repair and re-score,
  4. record raw energy (what the QPU minimized) alongside the true
     objective of the FINAL solution (what the business cares about).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from qlkit.core.problem import ProblemDefinition
from qlkit.core.qubo import QUBO, bits_from_string
from qlkit.core.solution import SampleRecord, SampleSet, Solution
from qlkit.repair.base import RepairHeuristic

_TOL = 1e-9


def _score(problem: ProblemDefinition, solution: Solution) -> tuple:
    violations = {c.name: c.violation(solution) for c in problem.constraints()}
    hard_ok = all(
        v <= _TOL for c, v in zip(problem.constraints(), violations.values()) if c.hard
    )
    return violations, hard_ok


def counts_to_sampleset(
    counts_list: List[Dict[str, int]],
    problem: ProblemDefinition,
    qubo: QUBO,
    repair: Optional[RepairHeuristic] = None,
    metadata: Optional[dict] = None,
) -> SampleSet:
    merged: Dict[str, int] = {}
    for counts in counts_list:
        for key, count in counts.items():
            merged[key] = merged.get(key, 0) + count

    objective = problem.objective_qubo()
    records: List[SampleRecord] = []
    for bitstring in sorted(merged):
        count = merged[bitstring]
        bits = bits_from_string(bitstring)
        raw_energy = qubo.energy(bits)
        solution = problem.decode(bits)
        violations, hard_ok = _score(problem, solution)
        pre_repair = None
        if repair is not None and not hard_ok:
            pre_repair = solution
            solution = repair.repair(solution, problem)
            solution.repaired = True
            violations, hard_ok = _score(problem, solution)
        records.append(
            SampleRecord(
                solution=solution,
                count=count,
                energy=raw_energy,
                true_objective=objective.energy(solution.bits),
                violations=violations,
                hard_feasible=hard_ok,
                pre_repair=pre_repair,
            )
        )
    return SampleSet(records, dict(metadata or {}))
