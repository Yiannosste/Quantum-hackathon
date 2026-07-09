"""LogisticsSolver: the single facade participants interact with.

API contract (see README for the full table):
    solve(lambdas, shots)    blocking; full pipeline, returns SolveResult
    submit(lambdas, shots)   non-blocking; returns SolveHandle
    gather(*handles)         blocking; resolves handles in order
    evaluate(lambdas)        blocking; returns EvalMetrics for outer-loop tuners
    validate(solution)       pure classical; same code the judges run

The pipeline behind solve():
    build_qubo -> circuits -> batch dispatch -> poll -> normalize
    -> decode -> repair -> score
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional

from qlkit.backends.base import Backend, JobStatus
from qlkit.core.metrics import EvalMetrics, ValidationReport
from qlkit.core.problem import ProblemDefinition
from qlkit.core.qubo import QUBO
from qlkit.core.solution import SampleSet, Solution
from qlkit.orchestration.circuit_factory import make_circuits
from qlkit.orchestration.dispatcher import BatchDispatcher, BudgetExceededError
from qlkit.orchestration.normalize import counts_to_sampleset
from qlkit.orchestration.tracker import JobLedger
from qlkit.repair.base import RepairHeuristic
from qlkit.validation.validator import validate as _validate

__all__ = [
    "SolverConfig",
    "SolveHandle",
    "SolveResult",
    "LogisticsSolver",
    "BudgetExceededError",
]


@dataclass(frozen=True)
class SolverConfig:
    shots: int = 1024
    circuits_per_solve: int = 1        # >1 = QAOA parameter sweep per solve
    batch_window_s: float = 0.2        # how long the dispatcher waits to fill a batch
    max_retries: int = 3
    poll_base_s: float = 0.25
    poll_cap_s: float = 30.0
    max_qpu_jobs: Optional[int] = 50   # circuit breaker; only enforced on hardware
    ledger_path: str = ".qlkit/jobs.json"
    verbose: bool = False


@dataclass
class SolveHandle:
    """A non-blocking solve in flight. Pass to gather() to collect."""

    lambdas: Dict[str, float]
    shots: int
    qubo: QUBO
    _future: object = field(repr=False, default=None)

    def done(self) -> bool:
        return self._future.done()


@dataclass
class SolveResult:
    samples: SampleSet
    metrics: EvalMetrics
    lambdas: Dict[str, float]
    qubo: QUBO

    @property
    def best_solution(self) -> Optional[Solution]:
        record = self.samples.best(feasible_only=True)
        return record.solution if record else None


class LogisticsSolver:
    def __init__(
        self,
        problem: ProblemDefinition,
        backend: Backend,
        repair: Optional[RepairHeuristic] = None,
        config: Optional[SolverConfig] = None,
    ):
        self.problem = problem
        self.backend = backend
        self.repair = repair
        self.config = config or SolverConfig()
        self._ledger = JobLedger(self.config.ledger_path)
        self._dispatcher = BatchDispatcher(
            backend,
            ledger=self._ledger,
            max_wait_s=self.config.batch_window_s,
            max_retries=self.config.max_retries,
            poll_base_s=self.config.poll_base_s,
            poll_cap_s=self.config.poll_cap_s,
            max_jobs=self.config.max_qpu_jobs if backend.capabilities.is_hardware else None,
            on_status=self._on_status if self.config.verbose else None,
        )

    # -- construction ------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        problem: ProblemDefinition,
        path: str | Path,
        repair: Optional[RepairHeuristic] = None,
    ) -> "LogisticsSolver":
        """The one-line backend switch:
        LogisticsSolver.from_config(problem, 'configs/iqm.json')"""
        from qlkit.config import load_backend, load_solver_config

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            problem,
            backend=load_backend(raw.get("backend", {})),
            repair=repair,
            config=load_solver_config(raw.get("solver", {})),
        )

    # -- Simulation & Submission Phase --------------------------------------

    def submit(
        self,
        lambdas: Optional[Dict[str, float]] = None,
        shots: Optional[int] = None,
    ) -> SolveHandle:
        """Non-blocking: fire off work, keep coding. The dispatcher batches
        concurrent submits into shared backend jobs."""
        resolved = {**self.problem.default_lambdas(), **(lambdas or {})}
        qubo = self.problem.build_qubo(resolved)
        shots = shots or self.config.shots
        circuits = make_circuits(qubo, self.config.circuits_per_solve)
        future = self._dispatcher.submit(circuits, shots)
        return SolveHandle(lambdas=resolved, shots=shots, qubo=qubo, _future=future)

    def gather(self, *handles: SolveHandle) -> List[SolveResult]:
        """Blocking; results match handle order. A failed handle raises when
        its turn comes — earlier results are still returned on the others."""
        return [self._finish(h) for h in handles]

    def solve(
        self,
        lambdas: Optional[Dict[str, float]] = None,
        shots: Optional[int] = None,
    ) -> SolveResult:
        """Blocking facade over submit()+gather(). Notebook-friendly."""
        return self._finish(self.submit(lambdas, shots))

    # -- Meta-Optimizer Support (the outer loop) -----------------------------

    def evaluate(
        self,
        lambdas: Optional[Dict[str, float]] = None,
        shots: Optional[int] = None,
    ) -> EvalMetrics:
        """The single method a Bayesian/RL lambda tuner calls per iteration."""
        return self.solve(lambdas, shots).metrics

    # -- Validation Phase ----------------------------------------------------

    def validate(self, solution: Solution) -> ValidationReport:
        return _validate(self.problem, solution)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self._dispatcher.shutdown()
        self.backend.close()

    def __enter__(self) -> "LogisticsSolver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- internals -----------------------------------------------------------

    def _finish(self, handle: SolveHandle) -> SolveResult:
        raw = handle._future.result()
        samples = counts_to_sampleset(
            raw.counts,
            self.problem,
            handle.qubo,
            repair=self.repair,
            metadata={"backend": self.backend.name, **raw.backend_meta},
        )
        return SolveResult(
            samples=samples,
            metrics=self._metrics(samples, handle),
            lambdas=handle.lambdas,
            qubo=handle.qubo,
        )

    def _metrics(self, samples: SampleSet, handle: SolveHandle) -> EvalMetrics:
        best_feasible = samples.best(feasible_only=True)
        representative = best_feasible or samples.best()
        return EvalMetrics(
            true_objective=representative.true_objective if representative else float("inf"),
            violations=dict(representative.violations) if representative else {},
            feasible_fraction=samples.feasible_fraction,
            best_energy=min((r.energy for r in samples.records), default=float("inf")),
            lambdas_used=dict(handle.lambdas),
            shots=samples.total_shots,
        )

    @staticmethod
    def _on_status(status: JobStatus) -> None:
        print(f"[qlkit] job status: {status.value}")
