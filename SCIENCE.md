# The Science Behind the Hackathon: QUBO, QAOA, and the Two Nested Minima

This document is the deep dive. The [README](README.md) tells you *what* we are
doing and the [PARTICIPANT_GUIDE](PARTICIPANT_GUIDE.md) tells you *how to build*;
this page explains **why the mathematics works**, all the way down. Nothing here
is required to start coding — but the teams that win tend to be the ones that
understand this page.

Notation used throughout: our Day-1 instance has $n = 16$ binary variables
($8$ customers $\times$ $2$ vehicles), $x_{c,v} \in \lbrace 0,1 \rbrace$ means
"customer $c$ rides on vehicle $v$", $d_c$ is customer $c$'s demand, $C_v$ is
vehicle $v$'s capacity, and $D_{ij}$ is the distance between customers $i$
and $j$.

---

## 1. From a business problem to a QUBO

### 1.1 What the business wants

$$\min_{x} \quad \underbrace{\sum_{v} \sum_{i < j} D_{ij} x_{i,v} x_{j,v}}_{\text{compact service zones}} \quad \text{subject to} \quad \underbrace{\sum_v x_{c,v} = 1 \enspace \forall c}_{\text{every customer served once}} \quad \text{and} \quad \underbrace{\sum_c d_c x_{c,v} \le C_v \enspace \forall v}_{\text{no truck overloaded}}$$

This is a constrained quadratic binary program. Quantum optimizers cannot
handle the "subject to" part — they minimize *one unconstrained function*.

### 1.2 The penalty method

A **QUBO** (Quadratic Unconstrained Binary Optimization) has the form

$$E(x) = x^\top Q x + \text{offset}, \qquad x \in \lbrace 0,1 \rbrace^n$$

— nothing but linear and pairwise terms over binary variables. To get there we
convert each constraint into an *energy penalty* that is zero when the
constraint holds and positive when it doesn't, then add it to the objective:

$$E_\lambda(x) = \mathrm{Cost}(x) + \lambda_{\text{oh}} \sum_c \Big(1 - \sum_v x_{c,v}\Big)^2 + \lambda_{\text{cap}} \sum_v \Big(\sum_c d_c x_{c,v} - C_v\Big)^2$$

Two algebraic facts make this legal QUBO material:

**Binary idempotency, $x^2 = x$.** Squaring a linear expression of binaries
yields only linear and pairwise terms. For a one-hot group over two vehicles:

$$\Big(1 - x_{c,0} - x_{c,1}\Big)^2 = 1 - x_{c,0} - x_{c,1} + 2 x_{c,0} x_{c,1}$$

(check it: the cross term $2x_0x_1$ punishes double-assignment, the negative
linear terms reward assignment, the constant $1$ punishes no assignment). This
is exactly what `OneHotConstraint.penalty_terms()` in
[problems/common.py](problems/common.py) emits.

**The capacity expansion** works the same way:

$$\Big(\sum_c d_c x_{c,v} - C_v\Big)^2 = C_v^2 + \sum_c \big(d_c^2 - 2C_v d_c\big) x_{c,v} + 2\sum_{i < j} d_i d_j x_{i,v} x_{j,v}$$

⚠️ **Note the deliberate flaw** you will meet again in §4: this is an
*equality* penalty. It is zero only when the truck is loaded to *exactly*
$C_v$ — a legally underloaded truck still pays energy. Keep reading.

### 1.3 One knob to rule the landscape: λ

The λ multipliers are **free parameters**. The problem definition does not fix
them; *you* do, at solve time. This is not an implementation accident — it is
the entire hackathon. Different λ values produce genuinely different energy
landscapes over the same $2^{16} = 65{,}536$ states, with different minima.

## 2. From QUBO to physics: the Ising Hamiltonian

A QPU does not natively speak in 0/1 variables; it speaks in **spins**
$z_i \in \lbrace -1, +1 \rbrace$ (qubit measured up or down). The change of
variables $x_i = (1 - z_i)/2$ turns any QUBO into an **Ising Hamiltonian**:

$$H_C = \sum_i h_i Z_i + \sum_{i < j} J_{ij} Z_i Z_j + \text{const}$$

where $Z_i$ is the Pauli-Z operator on qubit $i$, the *fields* $h_i$ come from
your linear QUBO terms and the *couplings* $J_{ij}$ from your quadratic ones.
Every classical bitstring $x$ is an eigenstate of $H_C$ with eigenvalue
$E_\lambda(x)$ — the Hamiltonian literally *is* your energy landscape,
implemented as physics. (This conversion is done for you in
`IQMCloudBackend._to_qaoa_circuit()`; it's ~15 lines of bookkeeping.)

Finding the ground state of an Ising Hamiltonian is NP-hard in general — which
is precisely why it's an interesting thing to point a quantum computer at.

## 3. QAOA: how the quantum computer searches the landscape

The **Quantum Approximate Optimization Algorithm** (Farhi, Goldstone & Gutmann,
2014) prepares a quantum state whose measurement outcomes are biased toward
low-energy bitstrings. The circuit, with $p$ layers:

$$|\psi(\vec\gamma, \vec\beta)\rangle = \prod_{k=1}^{p} e^{-i\beta_k H_M} e^{-i\gamma_k H_C} |+\rangle^{\otimes n}$$

Read right-to-left, that is:

1. **Uniform superposition** $|+\rangle^{\otimes n}$: all $65{,}536$ candidate
   assignments at once, equal amplitude. (One Hadamard per qubit.)
2. **Cost layer** $e^{-i\gamma H_C}$: each basis state picks up a *phase*
   proportional to its energy — the landscape is "stamped" into the quantum
   state. On hardware this is RZ rotations for the $h_i$ fields and
   CX–RZ–CX sandwiches for each $J_{ij}$ coupling.
3. **Mixer layer** $e^{-i\beta H_M}$ with $H_M = \sum_i X_i$: RX rotations that
   let amplitude *flow between* bitstrings. Phases alone are unobservable —
   the mixer converts phase differences into amplitude differences, and
   interference then concentrates amplitude on low-energy states.
4. **Measure** all qubits: you get one bitstring, sampled with probability
   $|\langle x|\psi\rangle|^2$. Repeat for `shots` samples.

Three honest facts about QAOA in the NISQ era:

- **It is a sampler, not a solver.** You get a *distribution* tilted toward
  low energy, not the minimum. That's why `qlkit` returns a `SampleSet`, not
  one answer — and why your repair heuristic has a job.
- **The angles $(\vec\gamma, \vec\beta)$ matter** and are themselves usually
  tuned variationally (technically a *third* optimization loop). The toolkit
  ships serviceable defaults and can sweep variations in one batch
  (`circuits_per_solve` / `CircuitSpec.params`); tuning angles properly is
  advanced-track territory.
- **Noise flattens everything.** Real hardware decoheres; the beautiful
  interference pattern blurs, and many samples land in high-energy (illegal)
  states. The local emulator mimics the *shape* of this behavior by sampling
  a Gibbs distribution $P(x) \propto e^{-E_\lambda(x)/T}$ over the exact
  landscape — the same "low-energy-biased but imperfect" statistics, minus
  the queue time.

## 4. The two nested minima (read this section twice)

Here is the conceptual core of the whole event. There are **two optimization
loops, minimizing two different things**, and confusing them is the most
common mistake in hybrid quantum computing.

### The Inner Minimum — quantum, blind, literal

**The QAOA algorithm is blind to trucks and routes.** It has never heard of
customer 3 or capacity 14. It sees exactly one mathematical object: the energy
landscape $E_\lambda(\cdot)$ that *you* created the moment you chose a
specific λ. Its entire job is:

$$\text{(inner)} \qquad \min_{x \in \lbrace 0,1 \rbrace^{16}} E_\lambda(x)$$

It finds (samples near) the lowest valley **of that specific landscape** —
faithfully, literally, with zero judgment. If your λ made an illegal
assignment the deepest valley, the QPU will proudly return that illegal
assignment. It did its job perfectly; *you* asked the wrong question.

### The Outer Minimum — classical, judging, yours

Your tuner in `my_tuner.py` runs the outer loop. It searches λ-space, and its
goal is **landscape design**: shape $E_\lambda$ so that the quantum computer's
lowest valley *coincides with* the cheapest 100%-legal logistics route:

$$\text{(outer)} \qquad \min_{\lambda} \mathrm{Cost}\big(\text{decode}(x^\star_\lambda)\big) \quad \text{where} \quad x^\star_\lambda = \arg\min_x E_\lambda(x) \quad \text{is feasible}$$

This is a **bilevel optimization**: the outer variable (λ) does not appear in
the final answer at all — it only *steers where the inner minimum lands*.

Watch the landscape deform on our actual Day-1 instance (identical pipeline,
only λ changed — measured during organizer calibration):

| λ regime | Where the lowest valley sits | Measured best cost |
|---|---|---|
| Too low | An **illegal** state: dropping customers costs less energy than driving | 10.77 (fake — infeasible) |
| Well-tuned | The cheapest **legal** route — inner and outer minima coincide | **38.74 (true optimum)** |
| Too high | Somewhere in a **flat plateau** of legal states; QPU can't tell them apart | 72.01 (legal, ~2× optimum) |

Three practical consequences for building the outer loop:

1. **Never tune on raw energy.** $E_\lambda$ mixes cost and penalty, and its
   scale changes every time you move λ — comparing energies across λ values is
   comparing different rulers. This is why `solver.evaluate()` returns the
   *separated* signals: `true_objective` (penalty-free cost),
   `violations` (continuous magnitudes — "3 units overloaded" is better
   feedback than "infeasible"), and `feasible_fraction` (how often the QPU's
   distribution lands in the legal region — a direct readout of landscape
   quality).
2. **The outer oracle is stochastic.** Each `evaluate()` is a finite-shot
   sample from a noisy device: the same λ twice gives similar-not-identical
   metrics. Use optimizers that tolerate noisy objectives — Bayesian
   optimization with a Gaussian process (noise-aware by construction), TPE
   (Optuna's default), CMA-ES — rather than methods that trust every
   evaluation exactly.
3. **The outer loop pays queue time.** Every λ candidate costs a QPU round
   trip. Sample efficiency is the metric that matters, and batch acquisition
   (several λ candidates per hardware job via `submit()`/`gather()`) is the
   cheat code the middleware gives you.

*(And repair? It sits between the loops: the inner minimum hands up illegal
samples, repair "snaps" them to the nearest legal states, and the outer loop
judges the repaired result. A strong repair widens the range of λ that works —
the two deliverables reinforce each other.)*

## 5. Slack variables: fixing the capacity wart properly

### 5.1 The problem, precisely

The shipped capacity penalty $\lambda (\text{load}_v - C_v)^2$ is zero **only
at exact full load**. The energy of a *legal* state now depends on λ:

$$E_\lambda(\text{legal } x) = \mathrm{Cost}(x) + \lambda_{\text{cap}} \sum_v (\text{load}_v - C_v)^2$$

On Day-1 data the optimal split carries loads $14$ and $14$ against capacities
$15$ and $14$ — so the true optimum sits at penalty $\lambda_{\text{cap}} \cdot 1$,
not zero. Crank $\lambda_{\text{cap}}$ high enough and the landscape starts
preferring *worse routes with fuller trucks*. The penalty is fighting the
objective inside the feasible region — exactly where it should be silent.

What we actually want is the inequality penalty
$\max(0, \text{load}_v - C_v)^2$ — but $\max$ is not a quadratic function of
$x$, so it cannot go into a QUBO directly.

### 5.2 The classical trick, imported to quantum

Operations research has fixed this for a century: convert the inequality into
an equality by adding a **slack variable** — a nonnegative helper that absorbs
the unused capacity:

$$\sum_c d_c x_{c,v} \le C_v \quad \Longleftrightarrow \quad \sum_c d_c x_{c,v} + s_v = C_v, \qquad s_v \in \lbrace 0, 1, \dots, C_v \rbrace$$

Then penalize the *equality*:

$$P_v(x, s) = \lambda_{\text{cap}} \Big(\sum_c d_c x_{c,v} + s_v - C_v\Big)^2$$

Now for **every** legal load there exists a slack value ($s_v = C_v -
\text{load}_v$) making the penalty **exactly zero**, while every overload
($\text{load}_v > C_v$) leaves it positive for all $s_v \ge 0$. The penalty is
silent inside the feasible region and loud outside it. Underloaded trucks are
free again.

### 5.3 Paying for it in qubits

The QPU only has binary variables, so the integer $s_v$ must be **binary
encoded**:

$$s_v = \sum_{k=0}^{K-1} 2^k y_{v,k}, \qquad K = \lceil \log_2 (C_v + 1) \rceil$$

For $C_v = 15$: four bits ($8+4+2+1$) cover $0\ldots15$ exactly. For
$C_v = 14$: four bits overshoot to 15 — harmless (the extra slack states are
simply never the zero-penalty ones), or you can use the standard tightening
trick of giving the last bit the weight $C_v - (2^{K-1} - 1) = 7$ so the range
is exact. Expanding $P_v$ produces new pairwise terms — $y \cdot y$ within
the slack register and $x \cdot y$ cross terms coupling slack bits to
assignment bits — all legal QUBO.

**The bill:** $4$ slack qubits per vehicle $\times$ $2$ vehicles $= 8$ extra
qubits. Your problem grows from **16 to 24 qubits**, with wider coefficient
ranges (weights up to $2\lambda C_v \cdot 8$) that strain hardware precision,
and a deeper circuit that decoheres more. This is why we call it the **boss
fight**: the mathematically correct encoding is physically more expensive, and
whether correctness beats compactness *on a noisy 2026-era QPU* is a genuine
open engineering question. Winning teams will *measure* the trade-off, not
assume it.

### 5.4 Implementing it in this repo (legally)

You may not edit `problems/`, but the architecture was built for exactly this
move — subclass in your own file under `starter_kit/`:

1. Extend `VehicleRoutingProblem`; override `num_vars()` to return
   $16 + 8 = 24$ (slack bits take indices $16\ldots23$).
2. Write a `SlackCapacityConstraint(Constraint)` whose `penalty_terms()` emits
   the expansion of §5.2–5.3 and whose `violation()` still measures **real**
   overload $\sum_v \max(0, \text{load}_v - C_v)$ from the assignment bits
   only — the judge cares about trucks, not about your helper bits.
3. Override `constraints()` to swap the capacity constraint; keep `decode()`
   reading routes from the first 16 bits and ignoring the slack register.
4. Point your existing tuner and repair at the new class. Nothing else
   changes — same `LogisticsSolver`, same `evaluate()`, same hardware config.

Then prove it: on the same backend and shot budget, does the 24-qubit "clean
landscape" beat the 16-qubit "warty landscape" on `feasible_fraction` and
final cost? That comparison table is podium material.

---

## Further reading

- E. Farhi, J. Goldstone, S. Gutmann — *A Quantum Approximate Optimization
  Algorithm* (arXiv:1411.4028) — the original QAOA paper.
- A. Lucas — *Ising formulations of many NP problems* (arXiv:1302.5843) — the
  standard cookbook of QUBO/Ising encodings, including slack variables.
- F. Glover, G. Kochenberger, Y. Du — *A Tutorial on Formulating and Using
  QUBO Models* (arXiv:1811.11538) — penalty methods in depth.
- IQM Academy & `qiskit-iqm` documentation — what the hardware layer beneath
  `IQMCloudBackend` actually does.
