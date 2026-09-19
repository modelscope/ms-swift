# Copyright (c) ModelScope Contributors. All rights reserved.
import os

from .ep import ExpertParallel, expert_parallel
from .expert_optimizer import ExpertCPUOptimizer, ExpertOptimizerCallback

# ---------------------------------------------------------------------------
# Environment-variable bridge — centralised so the name is defined once.
# ---------------------------------------------------------------------------
SWIFT_EXPERT_PARALLEL_ENV = 'SWIFT_EXPERT_PARALLEL'


def set_expert_parallel_env(ep_size: int):
    """Set the ``SWIFT_EXPERT_PARALLEL`` env var when *ep_size* > 1.

    Called from ``TemplateArguments.__post_init__`` so that
    ``get_default_device_map()`` can steer model loading before the
    ``ExpertParallel`` singleton is initialised.
    """
    if ep_size > 1:
        os.environ[SWIFT_EXPERT_PARALLEL_ENV] = str(ep_size)


def is_expert_parallel_active() -> bool:
    """Return ``True`` when expert parallel has been requested via CLI."""
    return os.environ.get(SWIFT_EXPERT_PARALLEL_ENV, '0') != '0'