"""QUBO -> CircuitSpec compilation.

Participants stay at the QUBO abstraction level; each backend interprets a
CircuitSpec natively (Boltzmann sampling locally, QAOA on IQM hardware).
``n_variations`` produces several specs with perturbed variational
parameters — sampled together in one batch, they give QAOA a cheap
parameter sweep without extra round trips.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from qlkit.backends.base import CircuitSpec
from qlkit.core.qubo import QUBO


def make_circuits(
    qubo: QUBO,
    n_variations: int = 1,
    params: Optional[Dict[str, Any]] = None,
) -> List[CircuitSpec]:
    base = dict(params or {})
    specs = []
    for k in range(n_variations):
        p = dict(base)
        p["variation"] = k
        if n_variations > 1:
            # Spread QAOA angles across variations; backends that don't use
            # angles (simulator, mock) simply ignore them.
            scale = 0.5 + k / max(1, n_variations - 1)
            p.setdefault("gammas", [0.8 * scale])
            p.setdefault("betas", [0.4 * scale])
        specs.append(CircuitSpec(qubo=qubo, params=p))
    return specs
