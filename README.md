# qlkit — Quantum-Accelerated Systems Engineering Toolkit

Backend-agnostic Python infrastructure for the logistics hackathon. You build
the *Hybrid Optimizer* (repair heuristics + lambda tuning); the toolkit owns
the plumbing (circuits, batching, retries, polling, normalization, scoring).

## Event flow

Practice on `VehicleRoutingProblem.small_instance()` (12 qubits, used by all
tutorials). The **scored** problem is `VehicleRoutingProblem.competition_instance()`
(8 customers x 2 vehicles, 16 qubits). On Day 2 a **live disruption instance**
is released mid-event and re-optimized under a time-boxed sprint with a capped
QPU budget — build your tuner for generality, not for the Day-1 numbers. The
Warehouse problem is unscored practice material for exactly that generality.

## Quickstart

```bash
python starter_kit/base_solver.py     # whole pipeline on the local simulator
python starter_kit/my_tuner.py        # outer-loop lambda tuning baseline
python -m pytest                      # 37 tests, all green
```

Then work through `starter_kit/notebooks/01...05` in order.

## The one-line backend switch

```python
solver = LogisticsSolver.from_config(problem, "configs/local.json")   # develop
solver = LogisticsSolver.from_config(problem, "configs/iqm.json")     # verify
```

Nothing else in your code changes. `submit()` is non-blocking on every
backend — including the local simulator — so there is no async rewrite when
you move to hardware.

## API contract

| You call | You get | Blocking? | Guarantees |
|---|---|---|---|
| `solver.solve(lambdas, shots)` | `SolveResult` | yes | normalized, decoded, repaired, scored |
| `solver.submit(lambdas)` | `SolveHandle` | no | job ID persisted to `.qlkit/jobs.json` on submit |
| `solver.gather(*handles)` | `list[SolveResult]` | yes | order matches handles |
| `solver.evaluate(lambdas)` | `EvalMetrics` | yes | the outer-loop primitive; separates cost / violations / feasibility |
| `solver.validate(solution)` | `ValidationReport` | no I/O | **the exact code the judges run** |
| `result.samples.best(feasible_only=True)` | `SampleRecord` | — | post-repair, lowest true objective |

You implement, at most, three things:

1. **A `ProblemDefinition`** (only if you bring a new problem): `objective_qubo()`,
   `constraints()`, `decode()`. Penalties are *never* baked into the objective —
   lambdas are injected at `build_qubo(lambdas)` time.
2. **A `RepairHeuristic`** (`starter_kit/my_repair.py`): fixes illegal samples.
   Applied automatically inside `solve()`/`evaluate()`; pre/post-repair pairs
   are kept in the `SampleSet` so you can measure it.
3. **An outer-loop tuner** (`starter_kit/my_tuner.py`): anything that maps
   `lambdas -> solver.evaluate(lambdas)` to a search strategy.

## Layout

```
qlkit/            ORCHESTRATION — sealed middleware, do not edit
  core/           QUBO, ProblemDefinition/Constraint contract, SampleSet, metrics
  backends/       local_simulator | mock | iqm_cloud  (all satisfy Backend ABC)
  orchestration/  LogisticsSolver, batch dispatcher, job tracker, normalization
  repair/         RepairHeuristic ABC + GreedyAssignmentRepair reference
  validation/     the judge
problems/         PROBLEM DEFINITIONS — organizer-controlled ground truth
  common.py       generic OneHot/Capacity constraints (shared by both problems)
  vrp/            vehicle routing (6 customers x 2 vehicles starter instance)
  warehouse/      warehouse allocation (4 tasks x 3 zones)
configs/          local.json | mock.json | iqm.json
starter_kit/      YOUR code lives here (+ notebooks 01-05)
tests/            end-to-end coverage incl. batching, retries, ledger persistence
```

## Middleware behavior you get for free

- **Batching**: concurrent `submit()`/`evaluate()` calls coalesce into shared
  backend jobs (window `SolverConfig.batch_window_s`, capped by the backend's
  `max_circuits_per_batch`, grouped by shot count).
- **Retries**: `TransientBackendError` (throttling, queue hiccups) triggers
  exponential backoff with jitter, up to `max_retries`.
- **Polling**: adaptive backoff (`poll_base_s` → `poll_cap_s`), no sleep before
  the first poll, single worker multiplexing all in-flight jobs.
- **Crash safety**: every job ID is written to `.qlkit/jobs.json` at submit
  time — a restarted kernel does not orphan QPU jobs.
- **Budget breaker**: `SolverConfig.max_qpu_jobs` (default 50) hard-stops
  hardware spend client-side. Simulator/mock are unlimited.

## Hardware (Submission Phase)

```bash
pip install "qlkit[iqm]"
set IQM_TOKEN=...        # never in notebooks
# edit configs/iqm.json with your assigned QC URL
```

## Known wart, on purpose

`CapacityConstraint` ships with an equality-style penalty
`lambda * (load - cap)^2`, which also punishes underloaded bins. The true
inequality needs slack variables — a documented stretch goal (notebook 01).
