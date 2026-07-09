"""Config-file loading: the one-line backend switch.

Config files are plain JSON (no YAML dependency):

    {
      "backend": {"type": "local_simulator", "options": {"seed": 7}},
      "solver":  {"shots": 2048}
    }

Registering a custom backend:

    from qlkit.config import BACKEND_REGISTRY
    BACKEND_REGISTRY["my_backend"] = MyBackend
"""

from __future__ import annotations

from typing import Dict, Type

from qlkit.backends.base import Backend
from qlkit.backends.local_simulator import LocalSimulatorBackend
from qlkit.backends.mock import MockBackend
from qlkit.orchestration.solver import SolverConfig

BACKEND_REGISTRY: Dict[str, Type[Backend]] = {
    "local_simulator": LocalSimulatorBackend,
    "mock": MockBackend,
}


def _register_iqm() -> None:
    # Imported lazily so environments without qiskit-iqm still work; the
    # class itself raises a helpful error at construction time if deps are
    # missing.
    from qlkit.backends.iqm_cloud import IQMCloudBackend

    BACKEND_REGISTRY.setdefault("iqm_cloud", IQMCloudBackend)


def load_backend(cfg: dict) -> Backend:
    backend_type = cfg.get("type", "local_simulator")
    if backend_type == "iqm_cloud":
        _register_iqm()
    try:
        backend_cls = BACKEND_REGISTRY[backend_type]
    except KeyError:
        raise ValueError(
            f"Unknown backend type {backend_type!r}; known: {sorted(BACKEND_REGISTRY)}"
        ) from None
    return backend_cls(**cfg.get("options", {}))


def load_solver_config(cfg: dict) -> SolverConfig:
    valid = {f for f in SolverConfig.__dataclass_fields__}
    unknown = set(cfg) - valid
    if unknown:
        raise ValueError(f"Unknown solver config keys: {sorted(unknown)}")
    return SolverConfig(**cfg)
