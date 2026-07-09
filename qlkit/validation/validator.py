"""Validation Phase: judge a solution against the problem's hard constraints.

This is the exact code the judges run. It ships inside the toolkit (not in
participant-editable files) so there can be no "but it passed on my
machine" disputes at judging time.
"""

from __future__ import annotations

from qlkit.core.metrics import ValidationReport
from qlkit.core.problem import ProblemDefinition
from qlkit.core.solution import Solution

_TOL = 1e-9


def validate(problem: ProblemDefinition, solution: Solution) -> ValidationReport:
    violations = {}
    messages = []
    passed = True
    for constraint in problem.constraints():
        v = constraint.violation(solution)
        violations[constraint.name] = v
        if v > _TOL:
            if constraint.hard:
                passed = False
                messages.append(
                    f"Hard constraint {constraint.name!r} violated by {v:g}."
                )
            else:
                messages.append(
                    f"Soft constraint {constraint.name!r} violated by {v:g} (allowed, costs score)."
                )
    return ValidationReport(passed=passed, violations=violations, messages=messages)
