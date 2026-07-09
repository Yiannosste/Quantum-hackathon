"""Starter kit: the whole hybrid pipeline in ~30 lines.

Run me first:  python starter_kit/base_solver.py

Swap 'configs/local.json' for 'configs/iqm.json' when you're ready for real
hardware — nothing else in this file changes. That is the whole point.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from problems.vrp import VehicleRoutingProblem
from qlkit import GreedyAssignmentRepair, LogisticsSolver

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "local.json"


def main() -> None:
    problem = VehicleRoutingProblem.small_instance()
    with LogisticsSolver.from_config(
        problem, CONFIG, repair=GreedyAssignmentRepair()
    ) as solver:
        # Lambdas are YOUR knob. Notebook 04 shows how to tune them
        # automatically with solver.evaluate().
        result = solver.solve(lambdas={"one_hot": 12.0, "capacity": 2.0})

        print(f"backend:            {result.samples.metadata['backend']}")
        print(f"shots:              {result.metrics.shots}")
        print(f"feasible fraction:  {result.samples.feasible_fraction:.1%}")
        print(f"repaired fraction:  {result.samples.repaired_fraction:.1%}")
        print(f"best true cost:     {result.metrics.true_objective:.2f}")

        best = result.best_solution
        if best is None:
            print("No feasible solution found — raise lambdas or improve repair.")
            return
        print(f"routes:             {best.payload['routes']}")
        print()
        print(solver.validate(best))


if __name__ == "__main__":
    main()
