"""The problem-agnostic contract: ProblemDefinition and Constraint.

This is the ONLY interface participants must implement to plug a new
logistics problem (VRP, warehouse allocation, ...) into the toolkit. The
orchestration layer never imports anything problem-specific — it sees a
QUBO, a decode function, and a list of constraints.

Design rules encoded here:
- ``objective_qubo()`` is the pure cost function with NO penalties baked in.
- Penalty weights (lambdas) are injected at ``build_qubo()`` time, keyed by
  constraint name. This is what makes the outer-loop lambda tuner possible.
- ``Constraint.violation()`` returns a continuous magnitude, not a bool, so
  tuners get gradient signal ("3 units over" vs "12 units over").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

from qlkit.core.qubo import QUBO
from qlkit.core.solution import Solution


class Constraint(ABC):
    def __init__(self, name: str, hard: bool = True):
        self.name = name
        self.hard = hard

    @abstractmethod
    def violation(self, solution: Solution) -> float:
        """0.0 = satisfied; positive = magnitude of violation."""

    @abstractmethod
    def penalty_terms(self, lam: float) -> QUBO:
        """This constraint's contribution to the QUBO, scaled by its lambda."""


class ProblemDefinition(ABC):
    @abstractmethod
    def num_vars(self) -> int: ...

    @abstractmethod
    def objective_qubo(self) -> QUBO:
        """Pure cost function — no penalties."""

    @abstractmethod
    def constraints(self) -> List[Constraint]: ...

    @abstractmethod
    def decode(self, bits: Sequence[int]) -> Solution:
        """Binary assignment -> domain solution object."""

    def one_hot_groups(self) -> List[List[int]]:
        """Optional structural hint: groups of variables of which exactly one
        should be 1. Generic repair heuristics exploit this; return [] if the
        encoding has no such structure."""
        return []

    def default_lambdas(self) -> Dict[str, float]:
        return {c.name: 1.0 for c in self.constraints()}

    def build_qubo(self, lambdas: Optional[Dict[str, float]] = None) -> QUBO:
        """objective + sum(lambda_c * penalty_c). Participants never assemble
        this by hand; the outer loop just varies ``lambdas``."""
        constraints = self.constraints()
        known = {c.name for c in constraints}
        resolved = self.default_lambdas()
        for name, lam in (lambdas or {}).items():
            if name not in known:
                raise ValueError(
                    f"Unknown constraint name {name!r}; known: {sorted(known)}"
                )
            resolved[name] = lam
        qubo = self.objective_qubo()
        for c in constraints:
            qubo = qubo + c.penalty_terms(resolved[c.name])
        if qubo.num_vars < self.num_vars():
            qubo = QUBO(qubo.terms, qubo.offset, self.num_vars())
        return qubo
