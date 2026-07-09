"""qlkit — backend-agnostic quantum-classical hybrid optimization toolkit.

Layering (strict dependency direction, top depends on bottom):

    Problem layer      ProblemDefinition, Constraint   (participants implement)
    Orchestration      LogisticsSolver, dispatcher, tracker, normalize
    Execution          Backend: LocalSimulatorBackend | IQMCloudBackend | MockBackend

Participants interact almost exclusively with :class:`LogisticsSolver`.
"""

from qlkit.core.qubo import QUBO, QUBOBuilder
from qlkit.core.problem import Constraint, ProblemDefinition
from qlkit.core.solution import SampleRecord, SampleSet, Solution
from qlkit.core.metrics import EvalMetrics, ValidationReport
from qlkit.backends.base import (
    Backend,
    BackendCapabilities,
    BackendError,
    CircuitSpec,
    JobHandle,
    JobStatus,
    RawResult,
    TransientBackendError,
)
from qlkit.backends.local_simulator import LocalSimulatorBackend
from qlkit.backends.mock import MockBackend
from qlkit.repair.base import GreedyAssignmentRepair, RepairHeuristic
from qlkit.orchestration.solver import (
    BudgetExceededError,
    LogisticsSolver,
    SolveHandle,
    SolveResult,
    SolverConfig,
)
from qlkit.validation.validator import validate

__all__ = [
    "QUBO",
    "QUBOBuilder",
    "Constraint",
    "ProblemDefinition",
    "Solution",
    "SampleRecord",
    "SampleSet",
    "EvalMetrics",
    "ValidationReport",
    "Backend",
    "BackendCapabilities",
    "BackendError",
    "TransientBackendError",
    "CircuitSpec",
    "JobHandle",
    "JobStatus",
    "RawResult",
    "LocalSimulatorBackend",
    "MockBackend",
    "RepairHeuristic",
    "GreedyAssignmentRepair",
    "LogisticsSolver",
    "SolverConfig",
    "SolveHandle",
    "SolveResult",
    "BudgetExceededError",
    "validate",
]
