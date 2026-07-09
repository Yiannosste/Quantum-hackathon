"""Generic, reusable constraints for one-hot assignment encodings.

Both the VRP and the Warehouse problem are built from these two classes —
that reuse is the proof that the toolkit's problem contract is agnostic.
"""

from __future__ import annotations

from typing import Dict, List

from qlkit.core.problem import Constraint
from qlkit.core.qubo import QUBO, QUBOBuilder
from qlkit.core.solution import Solution


class OneHotConstraint(Constraint):
    """Each group of variables must have exactly one bit set
    (e.g. every customer is assigned to exactly one vehicle).

    Penalty per group: lam * (1 - sum(x))^2, which expands over binaries to
    lam * (1 - sum_i x_i + 2 * sum_{i<j} x_i x_j).
    Violation: sum over groups of |1 - sum(x)| — continuous, so tuners see
    "2 groups broken" vs "5 groups broken".
    """

    def __init__(self, name: str, groups: List[List[int]], hard: bool = True):
        super().__init__(name, hard)
        self.groups = groups

    def violation(self, solution: Solution) -> float:
        bits = solution.bits
        return float(sum(abs(1 - sum(bits[i] for i in group)) for group in self.groups))

    def penalty_terms(self, lam: float) -> QUBO:
        builder = QUBOBuilder()
        for group in self.groups:
            builder.add_offset(lam)
            for a_idx, a in enumerate(group):
                builder.add_linear(a, -lam)
                for b in group[a_idx + 1 :]:
                    builder.add(a, b, 2.0 * lam)
        return builder.build()


class CapacityConstraint(Constraint):
    """Per-bin load limit (vehicle capacity, warehouse zone capacity, ...).

    ``loads`` maps each bin name to {variable_index: weight}; ``capacities``
    maps bin name to its limit.

    Violation: sum over bins of max(0, load - capacity) — only OVERLOAD
    counts, in real units (e.g. kg over).

    Penalty: lam * (load - capacity)^2 per bin. NOTE this is an equality-
    style penalty: it also (wrongly) punishes underloaded bins. It is QUBO-
    expressible without extra variables, which is why the starter kit ships
    it — encoding the true inequality with slack variables is a documented
    stretch goal for participants (see notebook 01).
    """

    def __init__(
        self,
        name: str,
        loads: Dict[str, Dict[int, float]],
        capacities: Dict[str, float],
        hard: bool = True,
    ):
        super().__init__(name, hard)
        self.loads = loads
        self.capacities = capacities

    def violation(self, solution: Solution) -> float:
        bits = solution.bits
        total = 0.0
        for bin_name, weights in self.loads.items():
            load = sum(w for i, w in weights.items() if bits[i])
            total += max(0.0, load - self.capacities[bin_name])
        return total

    def penalty_terms(self, lam: float) -> QUBO:
        builder = QUBOBuilder()
        for bin_name, weights in self.loads.items():
            cap = self.capacities[bin_name]
            items = sorted(weights.items())
            # (sum w_i x_i - C)^2 = C^2 - 2C sum w_i x_i + sum w_i^2 x_i
            #                       + 2 sum_{i<j} w_i w_j x_i x_j
            builder.add_offset(lam * cap * cap)
            for idx, (i, w_i) in enumerate(items):
                builder.add_linear(i, lam * (w_i * w_i - 2.0 * cap * w_i))
                for j, w_j in items[idx + 1 :]:
                    builder.add(i, j, 2.0 * lam * w_i * w_j)
        return builder.build()
