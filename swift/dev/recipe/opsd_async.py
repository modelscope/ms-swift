# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from swift.dev.recipe.gkd_async import StreamingGKDLoop
from swift.dev.recipe.run_opsd import OPSDLoop


class StreamingOPSDLoop(StreamingGKDLoop, OPSDLoop):
    """OPSD loop streaming per-sample over the on-policy regime (staleness 0 or 1), privileged teacher.

    Inherits ``StreamingGKDLoop``'s constructor and every streaming hook unchanged; only ``_RUN_ID`` is set
    so the in-process control-plane key is namespaced for OPSD.
    """

    _RUN_ID = 'run_opsd'
