"""Vehicle Routing (assignment formulation) as a QUBO problem.

Encoding: binary x[c, v] = 1 if customer c is served by vehicle v.
Variable index: var(c, v) = c * num_vehicles + v.

Objective: cluster-compactness proxy for route length — for each vehicle,
sum of pairwise distances between the customers assigned to it. (A full
tour-ordering QUBO needs O(n^2) more qubits; upgrading this objective is an
advanced-track task.)

Constraints (both built from the generic classes in problems/common.py):
  - "one_hot":  every customer on exactly one vehicle (hard)
  - "capacity": per-vehicle demand <= capacity (hard)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from problems.common import CapacityConstraint, OneHotConstraint
from qlkit.core.problem import Constraint, ProblemDefinition
from qlkit.core.qubo import QUBO, QUBOBuilder
from qlkit.core.solution import Solution


class VehicleRoutingProblem(ProblemDefinition):
    def __init__(
        self,
        distances: List[List[float]],
        demands: List[float],
        capacities: List[float],
    ):
        self.distances = distances
        self.demands = demands
        self.capacities = capacities
        self.num_customers = len(demands)
        self.num_vehicles = len(capacities)
        self._constraints = self._build_constraints()

    # -- encoding ------------------------------------------------------------

    def var(self, customer: int, vehicle: int) -> int:
        return customer * self.num_vehicles + vehicle

    def num_vars(self) -> int:
        return self.num_customers * self.num_vehicles

    def one_hot_groups(self) -> List[List[int]]:
        return [
            [self.var(c, v) for v in range(self.num_vehicles)]
            for c in range(self.num_customers)
        ]

    # -- ProblemDefinition contract -------------------------------------------

    def objective_qubo(self) -> QUBO:
        builder = QUBOBuilder(num_vars=self.num_vars())
        for v in range(self.num_vehicles):
            for c1 in range(self.num_customers):
                for c2 in range(c1 + 1, self.num_customers):
                    d = self.distances[c1][c2]
                    if d:
                        builder.add(self.var(c1, v), self.var(c2, v), d)
        return builder.build()

    def constraints(self) -> List[Constraint]:
        return self._constraints

    def decode(self, bits: Sequence[int]) -> Solution:
        assignment: Dict[int, List[int]] = {}
        for c in range(self.num_customers):
            assignment[c] = [
                v for v in range(self.num_vehicles) if bits[self.var(c, v)]
            ]
        routes: Dict[int, List[int]] = {v: [] for v in range(self.num_vehicles)}
        for c, vehicles in assignment.items():
            if len(vehicles) == 1:
                routes[vehicles[0]].append(c)
        return Solution(
            bits=tuple(bits),
            payload={"assignment": assignment, "routes": routes},
        )

    # -- helpers ---------------------------------------------------------------

    def _build_constraints(self) -> List[Constraint]:
        loads = {
            f"vehicle_{v}": {
                self.var(c, v): self.demands[c] for c in range(self.num_customers)
            }
            for v in range(self.num_vehicles)
        }
        capacities = {
            f"vehicle_{v}": self.capacities[v] for v in range(self.num_vehicles)
        }
        return [
            OneHotConstraint("one_hot", self.one_hot_groups(), hard=True),
            CapacityConstraint("capacity", loads, capacities, hard=True),
        ]

    @classmethod
    def from_json(cls, path: str | Path) -> "VehicleRoutingProblem":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["distances"], data["demands"], data["capacities"])

    @classmethod
    def small_instance(cls) -> "VehicleRoutingProblem":
        """6 customers x 2 vehicles = 12 binary variables. Sized so a decent
        repair heuristic reliably yields feasible solutions — nothing
        demoralizes a room like best(feasible_only=True) == None."""
        here = Path(__file__).parent / "data" / "small_instance.json"
        return cls.from_json(here)

    @classmethod
    def competition_instance(cls) -> "VehicleRoutingProblem":
        """The scored Day-1 instance: 8 customers x 2 vehicles = 16 binary
        variables. Two clusters plus two 'bridge' customers, tight capacity.
        Heads up: the instance WILL change during the event — build your
        tuner for generality, not for these exact numbers."""
        here = Path(__file__).parent / "data" / "competition_day1.json"
        return cls.from_json(here)
