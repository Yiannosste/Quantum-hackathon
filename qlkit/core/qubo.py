"""Canonical QUBO intermediate representation.

Everything in the toolkit compiles down to this: problem objectives,
constraint penalties, and the circuit specs sent to backends all speak QUBO.

Conventions:
- ``terms`` is sparse and upper-triangular: key ``(i, j)`` with ``i <= j``.
  ``(i, i)`` entries are linear coefficients (since ``x_i^2 == x_i``).
- Bit order: index ``i`` of a bits tuple / leftmost-first bitstring is
  variable ``i``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

Key = Tuple[int, int]


def _key(i: int, j: int) -> Key:
    return (i, j) if i <= j else (j, i)


@dataclass(frozen=True)
class QUBO:
    terms: Dict[Key, float] = field(default_factory=dict)
    offset: float = 0.0
    num_vars: int = 0

    def energy(self, bits: Sequence[int]) -> float:
        e = self.offset
        for (i, j), coeff in self.terms.items():
            if i == j:
                if bits[i]:
                    e += coeff
            elif bits[i] and bits[j]:
                e += coeff
        return e

    def __add__(self, other: "QUBO") -> "QUBO":
        merged = dict(self.terms)
        for k, coeff in other.terms.items():
            merged[k] = merged.get(k, 0.0) + coeff
        return QUBO(merged, self.offset + other.offset, max(self.num_vars, other.num_vars))

    def scaled(self, factor: float) -> "QUBO":
        return QUBO(
            {k: coeff * factor for k, coeff in self.terms.items()},
            self.offset * factor,
            self.num_vars,
        )


class QUBOBuilder:
    """Mutable accumulator for QUBO terms; ``build()`` produces a frozen QUBO."""

    def __init__(self, num_vars: int = 0):
        self._terms: Dict[Key, float] = {}
        self._offset = 0.0
        self.num_vars = num_vars

    def add(self, i: int, j: int, coeff: float) -> "QUBOBuilder":
        if coeff == 0.0:
            return self
        k = _key(i, j)
        self._terms[k] = self._terms.get(k, 0.0) + coeff
        self.num_vars = max(self.num_vars, k[1] + 1)
        return self

    def add_linear(self, i: int, coeff: float) -> "QUBOBuilder":
        return self.add(i, i, coeff)

    def add_offset(self, value: float) -> "QUBOBuilder":
        self._offset += value
        return self

    def build(self) -> QUBO:
        return QUBO(dict(self._terms), self._offset, self.num_vars)


def bits_from_string(bitstring: str) -> Tuple[int, ...]:
    """'0110' -> (0, 1, 1, 0); leftmost character is variable 0."""
    return tuple(1 if ch == "1" else 0 for ch in bitstring)


def string_from_bits(bits: Sequence[int]) -> str:
    return "".join("1" if b else "0" for b in bits)
