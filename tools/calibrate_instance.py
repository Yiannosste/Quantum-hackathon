"""Organizer-side instance calibration report. NEVER ship to participants.

Usage:
    python tools/calibrate_instance.py problems/vrp/data/competition_day1.json
    python tools/calibrate_instance.py problems/vrp/data/competition_day1.json \
        --disruption private/competition_day2_disruption.json

Checks, per instance:
  - brute-force optimum (the answer the QPU pipeline should converge to)
  - feasibility band: 10%-70% of assignments feasible (harder = repair
    matters; too hard = the room stalls)
  - lambda sensitivity: a small sweep on the local simulator must show that
    solution quality actually depends on lambda (otherwise the tuning
    challenge is dead)

With --disruption, additionally asserts the live-event properties:
  - the Day-1 optimum is INFEASIBLE under the disruption
  - the disrupted optimum is a different customer partition

Exit code 0 = all checks pass; 1 = at least one hard check failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from problems.vrp.definition import VehicleRoutingProblem
from qlkit import GreedyAssignmentRepair, LocalSimulatorBackend, LogisticsSolver, SolverConfig
from tools.oracle import instance_report

FEASIBILITY_BAND = (0.10, 0.70)
LAMBDA_GRID = [
    {"one_hot": 0.5, "capacity": 0.1},
    {"one_hot": 8.0, "capacity": 2.0},
    {"one_hot": 40.0, "capacity": 40.0},
]


def _partition(problem, bits):
    routes = problem.decode(bits).payload["routes"]
    return sorted(sorted(members) for members in routes.values())


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def report_instance(path: str, enforce_band: bool = True) -> tuple:
    problem = VehicleRoutingProblem.from_json(path)
    rep = instance_report(problem)
    print(f"\n=== {path} ===")
    print(f"  variables: {rep.num_vars}   customers: {problem.num_customers}   "
          f"vehicles: {problem.num_vehicles}")
    print(f"  total demand: {sum(problem.demands)}   capacities: {problem.capacities}")
    if rep.optimum_bits is None:
        print("  [FAIL] no feasible solution exists")
        return problem, rep, False
    print(f"  ORACLE optimum: cost={rep.optimum_cost:.2f}  "
          f"routes={problem.decode(rep.optimum_bits).payload['routes']}")
    print("  top feasible solutions:")
    for bits, cost in rep.top_feasible:
        print(f"    cost={cost:7.2f}  partition={_partition(problem, bits)}")

    band_detail = (
        f"{rep.num_feasible}/{rep.num_assignments} assignments "
        f"({rep.assignment_feasible_fraction:.1%}), band "
        f"{FEASIBILITY_BAND[0]:.0%}-{FEASIBILITY_BAND[1]:.0%}"
    )
    if enforce_band:
        ok = check(
            "feasibility band",
            FEASIBILITY_BAND[0] <= rep.assignment_feasible_fraction <= FEASIBILITY_BAND[1],
            band_detail,
        )
    else:
        # Disruption instances are ALLOWED to be tighter than the band —
        # that is the crisis. Solvability is asserted separately.
        print(f"  [info] feasibility (band not enforced for disruptions): {band_detail}")
        ok = True
    return problem, rep, ok


def lambda_sweep(problem, oracle_cost: float) -> bool:
    # Deliberately NO repair here: repair masks lambda entirely (it fixes
    # whatever the QPU emits). The tuning challenge lives in the RAW
    # distribution, so that is what we measure.
    print("  lambda sweep (local simulator, seed 7, 2048 shots, NO repair):")
    costs, feasible_fractions = [], []
    with LogisticsSolver(
        problem,
        LocalSimulatorBackend(seed=7),
        repair=None,
        config=SolverConfig(ledger_path=".qlkit/calibration_jobs.json"),
    ) as solver:
        for lambdas in LAMBDA_GRID:
            metrics = solver.evaluate(lambdas, shots=2048)
            costs.append(metrics.true_objective)
            feasible_fractions.append(metrics.feasible_fraction)
            gap = metrics.true_objective - oracle_cost
            print(f"    {lambdas}  ->  cost={metrics.true_objective:7.2f} "
                  f"(gap {gap:+.2f})  feasible={metrics.feasible_fraction:5.1%}")
    # Sensitivity: quality must depend on lambda. Repair can mask cost
    # differences, so accept EITHER a cost spread or a feasibility spread.
    cost_spread = max(costs) - min(costs)
    feas_spread = max(feasible_fractions) - min(feasible_fractions)
    return check(
        "lambda sensitivity",
        cost_spread > 1e-9 or feas_spread > 0.05,
        f"cost spread {cost_spread:.2f}, feasible-fraction spread {feas_spread:.1%} "
        "across the grid",
    )


def disruption_checks(day1_problem, day1_rep, disruption_path: str) -> bool:
    day2_problem, day2_rep, ok = report_instance(disruption_path, enforce_band=False)
    if day2_rep.optimum_bits is None:
        return False
    d1_solution = day2_problem.decode(day1_rep.optimum_bits)
    d1_violation = sum(
        c.violation(d1_solution) for c in day2_problem.constraints() if c.hard
    )
    ok &= check(
        "day-1 optimum breaks",
        d1_violation > 1e-9,
        f"day-1 optimum violates disrupted constraints by {d1_violation:g}",
    )
    ok &= check(
        "optimum moves",
        _partition(day1_problem, day1_rep.optimum_bits)
        != _partition(day2_problem, day2_rep.optimum_bits),
        f"day-1 partition {_partition(day1_problem, day1_rep.optimum_bits)} vs "
        f"day-2 partition {_partition(day2_problem, day2_rep.optimum_bits)}",
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", help="path to the Day-1 instance JSON")
    parser.add_argument("--disruption", help="path to the Day-2 disruption JSON")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="skip the simulator lambda sweep (faster)")
    args = parser.parse_args()

    problem, rep, ok = report_instance(args.instance)
    if rep.optimum_bits is not None and not args.skip_sweep:
        ok &= lambda_sweep(problem, rep.optimum_cost)
    if args.disruption:
        ok &= disruption_checks(problem, rep, args.disruption)

    print(f"\n{'ALL CHECKS PASSED' if ok else 'CALIBRATION FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
