"""Re-export of the toolkit's MockBackend, plus the recipe for building
your own backend (useful for testing YOUR code against nasty conditions).

Any class satisfying the qlkit.Backend interface plugs into the solver:

    from qlkit.config import BACKEND_REGISTRY
    BACKEND_REGISTRY["chaos"] = MyChaosBackend       # usable from configs
    # or directly:
    solver = LogisticsSolver(problem, MyChaosBackend())

Useful MockBackend knobs for testing your error handling:
    MockBackend(fail_first_submits=2)   # first 2 submits raise a transient
                                        # error -> exercises retry/backoff
    MockBackend(queue_polls=5)          # 5 QUEUED polls before DONE
                                        # -> exercises the poll loop
    MockBackend(latency_s=1.0)          # slow submits -> exercises batching
"""

from qlkit import MockBackend

__all__ = ["MockBackend"]
