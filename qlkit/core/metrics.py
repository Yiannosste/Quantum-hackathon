"""Metrics consumed by outer-loop tuners and by the validation phase.

``EvalMetrics`` deliberately separates the three numbers that raw QPU
"energy" conflates: true cost, constraint violation, and feasibility rate.
A lambda tuner that only saw energy could not tell "cheap but illegal"
from "legal but expensive".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class EvalMetrics:
    true_objective: float
    violations: Dict[str, float]
    feasible_fraction: float
    best_energy: float
    lambdas_used: Dict[str, float]
    shots: int = 0

    @property
    def total_violation(self) -> float:
        return sum(self.violations.values())

    def scalarize(self, violation_weight: float = 100.0, infeasibility_weight: float = 0.0) -> float:
        """Single-float summary for tuners that need one (e.g. scikit-optimize).

        Lower is better. ``infeasibility_weight`` optionally penalizes a low
        feasible-shot rate, rewarding lambdas that make the QPU land in the
        feasible region more often.
        """
        return (
            self.true_objective
            + violation_weight * self.total_violation
            + infeasibility_weight * (1.0 - self.feasible_fraction)
        )


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    violations: Dict[str, float]
    messages: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Validation {status}"]
        for name, v in self.violations.items():
            marker = "ok" if v <= 1e-9 else f"VIOLATED by {v:g}"
            lines.append(f"  - {name}: {marker}")
        lines.extend(f"  ! {m}" for m in self.messages)
        return "\n".join(lines)
