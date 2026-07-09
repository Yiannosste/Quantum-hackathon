"""YOUR outer-loop lambda tuner goes here.

The toolkit gives you exactly one primitive for this:

    metrics = solver.evaluate(lambdas)   # EvalMetrics

and EvalMetrics separates the three signals a tuner needs:
    metrics.true_objective     real cost, no penalties
    metrics.violations         per-constraint magnitudes
    metrics.feasible_fraction  how often the QPU lands in the legal region

The random-search baseline below works. Beating it is the challenge:
plug in scikit-optimize (gp_minimize), Optuna, or your own RL agent —
anything that maps lambdas -> scalarized score.

Run me:  python starter_kit/my_tuner.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from problems.vrp import VehicleRoutingProblem
from qlkit import GreedyAssignmentRepair, LogisticsSolver

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "local.json"


def random_search(solver: LogisticsSolver, iterations: int = 15, seed: int = 42):
    rng = random.Random(seed)
    best_lambdas, best_score = None, float("inf")
    for i in range(iterations):
        lambdas = {
            "one_hot": rng.uniform(1.0, 25.0),
            "capacity": rng.uniform(0.1, 10.0),
        }
        metrics = solver.evaluate(lambdas)
        score = metrics.scalarize(violation_weight=100.0, infeasibility_weight=5.0)
        marker = ""
        if score < best_score:
            best_lambdas, best_score = lambdas, score
            marker = "  <-- new best"
        print(
            f"[{i:02d}] one_hot={lambdas['one_hot']:5.1f} "
            f"capacity={lambdas['capacity']:5.1f} | "
            f"cost={metrics.true_objective:7.2f} "
            f"viol={metrics.total_violation:5.2f} "
            f"feas={metrics.feasible_fraction:5.1%} "
            f"score={score:8.2f}{marker}"
        )
    return best_lambdas, best_score


def main() -> None:
    problem = VehicleRoutingProblem.small_instance()
    with LogisticsSolver.from_config(
        problem, CONFIG, repair=GreedyAssignmentRepair()
    ) as solver:
        best_lambdas, best_score = random_search(solver)
        print(f"\nbest lambdas: {best_lambdas}  (score {best_score:.2f})")
        # Final confirmation run with the winner:
        result = solver.solve(best_lambdas)
        print(f"final routes: {result.best_solution.payload['routes']}")


if __name__ == "__main__":
    main()
