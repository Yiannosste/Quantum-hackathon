"""Normalized result containers: Solution, SampleRecord, SampleSet."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Solution:
    """A decoded candidate solution.

    ``bits`` is the raw binary assignment; ``payload`` is whatever domain
    object the problem's ``decode()`` produced (routes, allocations, ...).
    """

    bits: Tuple[int, ...]
    payload: Any = None
    repaired: bool = False


@dataclass
class SampleRecord:
    """One distinct measured bitstring, fully post-processed.

    ``energy`` is the raw QUBO energy (objective + penalties) of the bits as
    measured — i.e. what the QPU was actually minimizing. ``true_objective``
    and ``violations`` refer to the *final* solution (post-repair if repair
    ran), so teams can see exactly what repair bought them.
    """

    solution: Solution
    count: int
    energy: float
    true_objective: float
    violations: Dict[str, float]
    hard_feasible: bool
    pre_repair: Optional[Solution] = None


@dataclass
class SampleSet:
    records: List[SampleRecord]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def best(self, feasible_only: bool = False) -> Optional[SampleRecord]:
        """Best record: by true objective among feasible ones if
        ``feasible_only``, otherwise by raw energy."""
        pool = self.records
        if feasible_only:
            pool = [r for r in pool if r.hard_feasible]
            if not pool:
                return None
            return min(pool, key=lambda r: r.true_objective)
        if not pool:
            return None
        return min(pool, key=lambda r: r.energy)

    @property
    def total_shots(self) -> int:
        return sum(r.count for r in self.records)

    @property
    def feasible_fraction(self) -> float:
        total = self.total_shots
        if total == 0:
            return 0.0
        return sum(r.count for r in self.records if r.hard_feasible) / total

    @property
    def repaired_fraction(self) -> float:
        """Fraction of shots whose final solution came from the repair hook."""
        total = self.total_shots
        if total == 0:
            return 0.0
        return sum(r.count for r in self.records if r.solution.repaired) / total
