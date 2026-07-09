"""Warehouse Resource Allocation as a QUBO problem.

Encoding: binary x[t, z] = 1 if task t is allocated to zone z.
Variable index: var(t, z) = t * num_zones + z.

Objective: linear handling cost per (task, zone) pair, plus a pairwise
congestion cost for tasks sharing a zone.

Deliberately reuses the SAME generic constraints as the VRP
(problems/common.py) — the orchestration layer cannot tell the two
problems apart, which is the point of the agnostic contract.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from problems.common import CapacityConstraint, OneHotConstraint
from qlkit.core.problem import Constraint, ProblemDefinition
from qlkit.core.qubo import QUBO, QUBOBuilder
from qlkit.core.solution import Solution


class WarehouseAllocationProblem(ProblemDefinition):
    def __init__(
        self,
        handling_costs: List[List[float]],   # [task][zone]
        loads: List[float],                  # per task
        zone_capacities: List[float],
        congestion_cost: float = 0.5,
    ):
        self.handling_costs = handling_costs
        self.loads = loads
        self.zone_capacities = zone_capacities
        self.congestion_cost = congestion_cost
        self.num_tasks = len(loads)
        self.num_zones = len(zone_capacities)
        self._constraints = self._build_constraints()

    def var(self, task: int, zone: int) -> int:
        return task * self.num_zones + zone

    def num_vars(self) -> int:
        return self.num_tasks * self.num_zones

    def one_hot_groups(self) -> List[List[int]]:
        return [
            [self.var(t, z) for z in range(self.num_zones)]
            for t in range(self.num_tasks)
        ]

    def objective_qubo(self) -> QUBO:
        builder = QUBOBuilder(num_vars=self.num_vars())
        for t in range(self.num_tasks):
            for z in range(self.num_zones):
                builder.add_linear(self.var(t, z), self.handling_costs[t][z])
        if self.congestion_cost:
            for z in range(self.num_zones):
                for t1 in range(self.num_tasks):
                    for t2 in range(t1 + 1, self.num_tasks):
                        builder.add(
                            self.var(t1, z), self.var(t2, z), self.congestion_cost
                        )
        return builder.build()

    def constraints(self) -> List[Constraint]:
        return self._constraints

    def decode(self, bits: Sequence[int]) -> Solution:
        allocation: Dict[int, List[int]] = {
            t: [z for z in range(self.num_zones) if bits[self.var(t, z)]]
            for t in range(self.num_tasks)
        }
        return Solution(bits=tuple(bits), payload={"allocation": allocation})

    def _build_constraints(self) -> List[Constraint]:
        zone_loads = {
            f"zone_{z}": {self.var(t, z): self.loads[t] for t in range(self.num_tasks)}
            for z in range(self.num_zones)
        }
        capacities = {f"zone_{z}": self.zone_capacities[z] for z in range(self.num_zones)}
        return [
            OneHotConstraint("one_hot", self.one_hot_groups(), hard=True),
            CapacityConstraint("capacity", zone_loads, capacities, hard=True),
        ]

    @classmethod
    def small_instance(cls) -> "WarehouseAllocationProblem":
        """4 tasks x 3 zones = 12 binary variables."""
        return cls(
            handling_costs=[
                [1.0, 4.0, 6.0],
                [5.0, 1.5, 4.0],
                [6.0, 5.0, 1.0],
                [2.0, 2.5, 3.0],
            ],
            loads=[3.0, 4.0, 3.0, 2.0],
            zone_capacities=[5.0, 5.0, 5.0],
            congestion_cost=0.5,
        )
