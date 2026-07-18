# Quantum-Accelerated Systems Engineering Hackathon

**A two-day engineering competition: build a hybrid quantum-classical optimizer
that routes real delivery vehicles — on real quantum hardware — and keeps
working when the world changes under it.**

New here? This page explains everything: what the hackathon is, the exact
problem you will solve, how a quantum computer attacks it, and what you will
actually build. When you're ready to code, go to the
**[PARTICIPANT_GUIDE.md](PARTICIPANT_GUIDE.md)** — it is the complete technical
manual (API contract, editing rules, code examples, scoring).

---

## 1. What is this hackathon?

Enterprise logistics runs on optimization: which truck serves which customer,
which task goes to which warehouse zone. These problems are NP-hard — exact
solvers choke as they scale — and they are exactly the shape of problem
quantum computers are being built for.

But today's quantum processors (we use **IQM's superconducting QPUs** via the
IQM Resonance cloud) are noisy. Ask one for an optimal delivery plan and most
of its answers will be *illegal*: customers dropped, trucks overloaded. Raw
quantum output is not a product.

The engineering discipline that turns it into one is the **hybrid
quantum-classical loop**, and that is what you build here:

```
        ┌──────────────────────────────────────────────────┐
        │              YOUR HYBRID OPTIMIZER               │
        │                                                  │
        │   ① tune penalty weights (λ)  ──►  quantum QPU   │
        │        ▲                             │           │
        │        │                             ▼           │
        │   ③ measure quality  ◄──  ② repair illegal       │
        │      & re-tune              answers classically  │
        └──────────────────────────────────────────────────┘
```

You are **not** expected to know quantum physics. The toolkit in this repo
(`qlkit`) handles every quantum detail — circuits, cloud submission, batching,
retries. You write the two classical brains: the **λ tuner** (①/③) and the
**repair heuristic** (②). That division is deliberate: it mirrors how real
quantum-adjacent engineering teams work today.

## 2. The problem we are solving

### The story

A logistics company serves **8 customers** with **2 delivery vehicles**
dispatched from a shared depot region. Each customer needs a known quantity of
goods delivered; each vehicle has a maximum carrying capacity. The company
wants customer-to-vehicle assignments that keep each vehicle's customers
**close together** (compact service zones = short routes = cheap operations)
while **never overloading a vehicle** and **never skipping a customer**.

### The exact data (Day-1 scored instance)

Customers live on a 2D map with two natural neighborhoods and two awkward
"bridge" customers between them — the reason the best answer is not obvious:

```
 y ↑      c1                 c3                    c6
   │   c0                                       c5
   │      c2                    c4                 c7
   └────────────────────────────────────────────────→ x
     west district         "bridge"          east district
```

| Customer | Location (x, y) | Demand |
|:---:|:---:|:---:|
| 0 | (0.0, 0.0) | 4 |
| 1 | (1.2, 1.0) | 3 |
| 2 | (0.8, −1.3) | 4 |
| 3 | (4.6, 1.6) — bridge | 3 |
| 4 | (5.4, −1.5) — bridge | 3 |
| 5 | (9.6, 0.2) | 4 |
| 6 | (10.4, 1.3) | 3 |
| 7 | (10.0, −1.2) | 4 |

| Vehicle | Capacity |
|:---:|:---:|
| 0 | 15 |
| 1 | 14 |

Total demand is **28** against total capacity **29** — one unit of slack.
Capacity is not a formality here; it decides which assignments are even legal.

The machine-readable instance (full pairwise distance matrix, demands,
capacities) is [problems/vrp/data/competition_day1.json](problems/vrp/data/competition_day1.json),
loaded in one line:

```python
from problems.vrp import VehicleRoutingProblem
problem = VehicleRoutingProblem.competition_instance()
```

**Formally:** minimize the sum of pairwise distances between customers assigned
to the same vehicle, subject to (a) every customer on exactly one vehicle,
(b) no vehicle's total demand exceeding its capacity.

*(Yes — a laptop can brute-force 8 customers. That is intentional: it lets us
verify every answer against the exact optimum. You are judged on building the
pipeline that scales, not on this instance's answer — see §5.)*

### How a quantum computer attacks it

The problem is encoded into **16 binary variables**: `x[c,v] = 1` means
"customer *c* rides on vehicle *v*". One variable per customer-vehicle pair =
8 × 2 = 16 — one **qubit** each on the QPU.

The rules can't be given to a QPU as rules. Instead they become *penalty
energies* added to the cost, producing a **QUBO** (Quadratic Unconstrained
Binary Optimization) — the native language of quantum optimizers:

```
E(x)  =  Cost(x)  +  λ₁ · OneHotPenalty(x)  +  λ₂ · CapacityPenalty(x)
```

The QPU physically settles toward low-energy states, so low `E(x)` ≈ good
plans. But everything hinges on the **penalty weights λ**:

- **λ too low** → breaking a rule is cheaper than driving across town. The QPU
  drops customers to save distance. Answers are cheap and illegal.
- **λ too high** → penalties dwarf the cost. Every legal plan looks the same to
  the QPU. Answers are legal and terrible.

We measured this on the actual instance (identical pipeline, only λ changed):

| λ setting | Best cost found | Verdict |
|---|---|---|
| Too low | 10.77 | Illegal — customers dropped |
| **Well-tuned** | **38.74** | **The true optimum** |
| Too high | 72.01 | Legal but ~2× worse than optimal |

Finding the sweet spot **automatically** — for *any* instance, not just this
one — is Challenge 1. Fixing the illegal answers the QPU emits anyway is
Challenge 2. The full challenge definitions (including the slack-variable
stretch goal) are in the [guide, §6](PARTICIPANT_GUIDE.md).

## 3. The event: two days, one twist

| Phase | What happens |
|---|---|
| **Day 1 — Build** | Develop your tuner + repair against the instance above, locally on the bundled quantum emulator. Verified runs on real IQM hardware via a one-line config switch. |
| **Day 2 — ⚡ Live disruption** | At an announced moment, **the problem changes**. A logistics incident strikes — we will not say what kind — and a new instance file is released. You re-optimize in a time-boxed sprint with a capped QPU budget. |
| **Validation** | Your solution is checked by `solver.validate()` — the *exact* code the judges run, shipped in the toolkit. No scoring disputes possible. |

The Day-2 twist is the heart of the event. It makes overfitting fatal: a
"tuner" that is a hardcoded λ pair for the Day-1 data will collapse exactly
when the points are richest. Real logistics re-optimizes under disruption
daily; your optimizer must too. (A second problem — warehouse task allocation,
in [problems/warehouse/](problems/warehouse/) — is included unscored, as the
ideal test that your code generalizes.)

## 4. Quickstart

```bash
git clone https://github.com/Yiannosste/Quantum-hackathon.git
cd Quantum-hackathon
python starter_kit/base_solver.py     # full quantum pipeline, ~5 seconds, no setup
python starter_kit/my_tuner.py        # the λ-tuning baseline you must beat
python -m pytest                      # the toolkit's own test suite (all green)
```

Then work through the guided notebooks in order —
[starter_kit/notebooks/](starter_kit/notebooks/) `01` → `05`:
QUBO basics → local simulation → repair → λ tuning → real hardware.

No dependencies beyond Python 3.10+ for local work (`pip install jupyterlab`
for the notebooks; `pip install "qlkit[iqm]"` only when you go to hardware).

## 5. Repository map & rules of engagement

```
qlkit/          🔒 The quantum middleware. Circuits, IQM cloud, batching,
                   retries, result normalization. IMPORT IT, NEVER EDIT IT.
problems/       🔒 The scored problem definitions & data. The constraint code
                   here IS the judging criteria. Read, never edit.
configs/        ⚙️  local.json (emulator) ⇄ iqm.json (real QPU) — the one-line switch
starter_kit/    ✏️  YOUR code: my_tuner.py, my_repair.py, notebooks
tests/          🔒 The toolkit's test suite (CI) — read, never edit
```

Your entire submission lives in `starter_kit/`. The boundary is the point:
you build optimization intelligence on top of a production-grade quantum
stack, exactly as you would in industry.

## 6. Where to next

- **[PARTICIPANT_GUIDE.md](PARTICIPANT_GUIDE.md)** — the technical manual:
  architecture, the mandatory `LogisticsSolver` API, return types,
  copy-pasteable code, the three challenges, event scoring.
- **[SCIENCE.md](SCIENCE.md)** — the deep dive: QUBO construction, the Ising
  mapping, how QAOA actually works, the two nested optimization loops, and
  the full slack-variable theory behind the boss fight.
- [starter_kit/notebooks/01_graph_to_qubo.ipynb](starter_kit/notebooks/01_graph_to_qubo.ipynb) —
  start here to *understand*; run `starter_kit/base_solver.py` first to *see it work*.
- [problems/vrp/data/competition_day1.json](problems/vrp/data/competition_day1.json) —
  the actual scored data.

Welcome aboard. The QPU is noisy, the deadline is real, and the bridge might
be closed tomorrow. Build accordingly. 🚚⚛️
