# Quantum-Accelerated Systems Engineering Hackathon — Official Participant Guide

You are not here to run a quantum circuit. You are here to build a **Hybrid Optimizer**: a classical intelligence layer that squeezes business-grade logistics solutions out of noisy quantum hardware. The quantum plumbing is done. The optimization brain is yours to build.

---

## 1. Architecture & Repository Rules

The repository enforces a strict three-layer separation. Learn this map before writing a single line:

```
quantum-hackathon/
├── qlkit/            🔒 LOCKED MIDDLEWARE — import it, never edit it
├── problems/         🔒 GROUND TRUTH — read it, never edit it
├── configs/          ⚙️  Backend switch (local / mock / IQM)
├── starter_kit/      ✏️  YOUR PLAYGROUND — everything you write lives here
└── tests/            🔒 Organizer CI
```

### `qlkit/` — the locked middleware

This package is the infrastructure your code rides on. It handles, silently and automatically:

- **IQM cloud communication** — auth, job submission, status polling with adaptive backoff, crash-safe job ledgers (a restarted notebook kernel never orphans a QPU job).
- **Batching & rate-limiting** — concurrent requests from your code are coalesced into shared hardware jobs; throttling (HTTP 429) is absorbed with exponential backoff and retries.
- **QUBO compilation** — your problem's QUBO is compiled to QAOA circuits in IQM's native gate set. You never touch a transpiler.
- **Result normalization** — raw bitstring counts come back decoded, repaired, scored, and validated.

> **Rule #1: You MUST NOT edit anything inside `qlkit/`.** Submissions that modify the middleware are disqualified. If you think you found a bug, report it to the organizers — do not patch it locally.

### `problems/` — the ground truth

Contains the logistics problems you are scored on: **Vehicle Routing** (`problems/vrp/`) and **Warehouse Allocation** (`problems/warehouse/`), built on shared constraint primitives in `problems/common.py`. These files define the judging criteria — the constraint code here is *exactly* what runs at final validation. Read them carefully; do not modify them.

### `starter_kit/` — your playground

**Your final score depends entirely on what you write in two files:**

| File | What you build there |
|---|---|
| `starter_kit/my_tuner.py` | The **outer optimization loop** that tunes penalty weights (λ) |
| `starter_kit/my_repair.py` | The **classical repair heuristics** that fix illegal quantum samples |

Everything else in `starter_kit/` (the `base_solver.py` wiring script and notebooks `01`–`05`) is teaching material — run it, copy from it, learn from it.

---

## 2. The Core Mathematical Concept: The Lambda Dilemma

Quantum hardware does exactly one thing here: it **minimizes energy**. The energy function it sees is a QUBO of the form:

```
E(x)  =  Cost(x)  +  Σ  λ_c · Penalty_c(x)
```

`Cost(x)` is the true business objective (route distance, handling cost). Each `Penalty_c(x)` is a quadratic term that grows when a constraint `c` is broken (a customer assigned to no vehicle, an overloaded truck). The `λ_c` weights decide how loudly each constraint screams.

Here is the dilemma — and your main job in `my_tuner.py`:

- **λ too low** → the penalty is cheaper than the detour. The QPU happily returns *illegal* routes: it drops customers and overloads trucks to save distance, because the math told it that was a bargain.
- **λ too high** → the penalty terms dwarf the cost terms. The energy landscape **flattens**: every legal solution looks nearly identical to the QPU, and it can no longer distinguish a brilliant route from a terrible-but-legal one. Solution quality collapses to random.

The sweet spot is narrow, instance-dependent, and shifts with hardware noise. Finding it by hand is folklore; finding it **automatically** is engineering. That is the hackathon.

---

## 3. The Mandatory API Contract

All interaction with the quantum system goes through **one class: `LogisticsSolver`**. You cannot bypass it — there is no supported path to construct circuits, talk to backends, or touch job handles directly, and attempting to is both against the rules and a waste of your weekend.

```python
from qlkit import LogisticsSolver, GreedyAssignmentRepair
from problems.vrp import VehicleRoutingProblem

problem = VehicleRoutingProblem.small_instance()
solver = LogisticsSolver.from_config(problem, "configs/local.json",
                                     repair=GreedyAssignmentRepair())
```

Switching from the local simulator to real IQM hardware is **that one string**: `configs/local.json` → `configs/iqm.json`. Nothing else in your code changes.

### The four core methods

| Method | Blocking | Returns | Use it for |
|---|---|---|---|
| `solver.solve(lambdas, shots)` | **Yes** | `SolveResult` | Getting an actual solution |
| `solver.evaluate(lambdas)` | **Yes** | `EvalMetrics` | Feeding your outer-loop tuner |
| `solver.submit(lambdas, shots)` | **No** | `SolveHandle` | Advanced: pipelined dispatch |
| `solver.gather(*handles)` | **Yes** | `list[SolveResult]` | Advanced: collecting pipelined jobs |

#### `solver.solve(lambdas, shots)` — blocking

Runs the **full pipeline** in one call: compiles your QUBO with the given λs, dispatches the batch, polls the backend, normalizes raw counts, decodes every bitstring into a domain solution, applies your repair heuristic to every infeasible sample, and scores everything. Returns a `SolveResult`. This is the workhorse for "give me the best routes you can, right now."

#### `solver.evaluate(lambdas)` — blocking

**The primary tool for your outer-loop tuner.** One call = one λ candidate assessed. It runs the same pipeline as `solve()` but hands back only the tuner-relevant signals as `EvalMetrics` — with cost, constraint violations, and feasibility rate **separated** (see §4 for why that separation is the whole point). This is the function you wrap with Optuna, an RL policy, or CMA-ES.

#### `solver.submit(lambdas, shots)` — non-blocking (Advanced Track)

Dispatches a job to the background and returns a `SolveHandle` **immediately**. Here is why this matters on hardware: the middleware's dispatcher collects every request that arrives within a short batching window and packs them into a **single HTTP request / QPU job**. Three λ candidates submitted back-to-back cost one trip through the shared hardware queue instead of three. On a Saturday with thirty teams hammering one QPU, this is the difference between iterating and waiting.

#### `solver.gather(*handles)` — blocking (Advanced Track)

Resolves multiple `SolveHandle` objects and returns their `SolveResult`s **in the exact order the handles were provided** — result `i` always corresponds to handle `i`, regardless of completion order. If one job failed, the exception is raised when its turn comes; the others are unaffected.

Two more calls you will use constantly (not part of the core four, but fully supported):

- `solver.validate(solution)` → `ValidationReport`. **This is the exact code the judges run.** If it prints `Validation PASSED`, it passes at judging. No disputes possible.
- `solver.close()` (or use `with LogisticsSolver.from_config(...) as solver:`) — flushes the dispatcher cleanly.

---

## 4. API Return Types Explained

### `SolveResult` — what `solve()` and `gather()` give you

| Attribute | Type | Meaning |
|---|---|---|
| `.best_solution` | `Solution \| None` | Best **feasible** solution found (post-repair, lowest true cost). `None` if no sample was feasible — raise your λs or improve your repair. |
| `.metrics` | `EvalMetrics` | The same tuner-ready metrics `evaluate()` returns. |
| `.samples` | `SampleSet` | The full statistical distribution of all quantum shots. |

The `Solution` object inside `.best_solution` carries the decoded domain answer in `.payload`:

```python
result.best_solution.payload["routes"]   # VRP: {0: [3, 4, 5], 1: [0, 1, 2]}
```

`SampleSet` is your microscope: `.records` lists every distinct measured bitstring with its shot count, raw energy, true cost, per-constraint violations, and — crucially — its **pre-repair** version when your heuristic fired. `.feasible_fraction` and `.repaired_fraction` tell you, respectively, how often you landed in the legal region and how much of that was your repair code's doing.

### `EvalMetrics` — what your tuner consumes

This object deliberately **un-mixes** the three signals that raw QPU energy conflates. A tuner that only saw energy could not tell *cheap-but-illegal* from *legal-but-expensive*. Yours can:

| Attribute / Method | Type | Meaning |
|---|---|---|
| `.true_objective` | `float` | Pure business cost of the best solution — **no penalties included**. |
| `.violations` | `dict[str, float]` | Per-constraint violation magnitudes (e.g. `{"one_hot": 0.0, "capacity": 3.0}` = 3 units overloaded). Continuous, not boolean — your tuner gets gradient signal. |
| `.feasible_fraction` | `float` | Percentage of shots that decoded to valid solutions. A great λ makes the QPU *land* in the legal region, not just permit it. |
| `.total_violation` | `float` | Sum of all violation magnitudes — a convenient aggregate. |
| `.scalarize(violation_weight, infeasibility_weight)` | `float` | Boils everything down to **one float (lower = better)** for standard optimizers that need a single objective (Optuna, skopt, CMA-ES). |

### `SolveHandle` — what `submit()` gives you

| Attribute | Type | Meaning |
|---|---|---|
| `.done()` | `bool` | Whether the background job has finished. Poll it if you want; `gather()` blocks for you either way. |
| `.lambdas` | `dict` | The λs this job was built with — handy for bookkeeping in batched sweeps. |

---

## 5. Code Examples

### Example 1 — A basic tuner loop (`my_tuner.py` pattern)

```python
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from problems.vrp import VehicleRoutingProblem
from qlkit import GreedyAssignmentRepair, LogisticsSolver

problem = VehicleRoutingProblem.small_instance()

with LogisticsSolver.from_config(
    problem, "configs/local.json", repair=GreedyAssignmentRepair()
) as solver:

    rng = random.Random(42)
    best_lambdas, best_score = None, float("inf")

    for i in range(15):
        lambdas = {
            "one_hot": rng.uniform(1.0, 25.0),
            "capacity": rng.uniform(0.1, 10.0),
        }
        metrics = solver.evaluate(lambdas)
        score = metrics.scalarize(violation_weight=100.0, infeasibility_weight=5.0)
        if score < best_score:
            best_lambdas, best_score = lambdas, score
        print(f"[{i:02d}] cost={metrics.true_objective:7.2f} "
              f"viol={metrics.total_violation:5.2f} "
              f"feas={metrics.feasible_fraction:5.1%} score={score:8.2f}")

    # Final confirmation run with the winning lambdas:
    result = solver.solve(best_lambdas)
    print("best lambdas:", best_lambdas)
    print("routes:      ", result.best_solution.payload["routes"])
    print(solver.validate(result.best_solution))
```

This is random search — **the baseline you must beat**. Replace the `rng.uniform` sampling with a real acquisition strategy and everything else stays identical.

### Example 2 — Pipelined evaluation with `submit()` / `gather()` (Advanced Track)

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from problems.vrp import VehicleRoutingProblem
from qlkit import GreedyAssignmentRepair, LogisticsSolver

problem = VehicleRoutingProblem.small_instance()

with LogisticsSolver.from_config(
    problem, "configs/local.json", repair=GreedyAssignmentRepair()
) as solver:

    # Fire 3 lambda candidates without waiting. Because they arrive within
    # the dispatcher's batching window, the middleware packs them into ONE
    # backend job — one trip through the QPU queue instead of three.
    candidates = [
        {"one_hot": 5.0,  "capacity": 2.0},
        {"one_hot": 10.0, "capacity": 2.0},
        {"one_hot": 15.0, "capacity": 2.0},
    ]
    handles = [solver.submit(lambdas) for lambdas in candidates]

    # ... free CPU time here: update your surrogate model, log, plot ...

    # Results come back in handle order, guaranteed:
    for handle, result in zip(handles, solver.gather(*handles)):
        print(f"lambdas={handle.lambdas}  "
              f"cost={result.metrics.true_objective:.2f}  "
              f"feasible={result.metrics.feasible_fraction:.1%}")
```

Pair this with a batch-acquisition Bayesian optimizer (e.g. `q-EI`) and your tuner evaluates λ candidates several at a time for the queue price of one.

---

## 6. The Three Hackathon Challenges

### Challenge 1 — Meta-Optimization (`my_tuner.py`)

Ditch the baseline random search. Implement a real outer loop: **Bayesian Optimization** (Optuna's TPE or `skopt.gp_minimize` are the fastest paths), **reinforcement learning**, or **CMA-ES**. Your tuner's interface to the quantum system is exactly one function — `solver.evaluate(lambdas)` — so any optimizer that maps a parameter dict to a float plugs straight in via `.scalarize()`. Sophisticated teams will exploit the *separated* metrics instead: treat `violations` and `feasible_fraction` as constraints rather than folding them into one number. Judged on: final solution quality per QPU call spent.

### Challenge 2 — Classical Repair Hooks (`my_repair.py`)

On real hardware, most raw quantum samples are illegal. Implement a custom `RepairHeuristic` whose `repair(solution, problem)` method takes an invalid bitstring and **optimally "snaps" it onto a valid route** — not just *any* valid route. The shipped `GreedyAssignmentRepair` restores legality; a winning repair restores legality *cheaply* (repair toward the objective, add feasibility-preserving local search, invent problem-specific moves like customer swaps). The contract: return a solution at least as feasible as the input, never mutate the input. Judged on: `feasible_fraction` uplift and the true cost of your repaired solutions — both visible in every `SampleSet`.

### Challenge 3 — The Slack Variable Boss Fight 👹

Open `problems/common.py` and read `CapacityConstraint.penalty_terms()`. The shipped penalty is:

```
λ · (load − capacity)²
```

That is an **equality** penalty — it punishes a vehicle for being *under*loaded just as hard as for being overloaded, even though an underloaded truck is perfectly legal. This wart is documented, deliberate, and worth serious points: encoding the true **inequality** constraint (`load ≤ capacity`) requires introducing **slack variables** — auxiliary binary variables that absorb the unused capacity so the penalty vanishes for every legal load level.

You cannot edit `problems/common.py`. You *can* bring your own `ProblemDefinition` subclass (in your own file under `starter_kit/`) that reuses the official objective and one-hot constraint but swaps in your slack-based capacity constraint — the solver treats it like any other problem. Get the binary expansion of the slack register right, keep the qubit overhead minimal, and demonstrate a better cost/feasibility trade-off than the equality penalty. This is the ultimate stretch goal: it proves you understand penalty-method QUBO at the level the industry actually needs.

---

**Final checklist before you submit:** your solution passes `solver.validate()` (that is literally the judging code), your tuner and repair live in `starter_kit/`, and `qlkit/` and `problems/` are byte-for-byte untouched. Now go make the QPU look smarter than it is. 🚀
