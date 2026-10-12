# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from swift.dev.recipe.gkd_async import StreamingGKDLoop
from swift.dev.recipe.run_mopd import MOPDLoop


class StreamingMOPDLoop(StreamingGKDLoop, MOPDLoop):
    """MOPD loop streaming per-sample over the on-policy regime (staleness 0 or 1), multi-teacher signal.

    Inherits ``StreamingGKDLoop``'s constructor and every streaming hook unchanged; only ``_RUN_ID`` is set
    so the in-process control-plane key is namespaced for MOPD.
    """

    _RUN_ID = 'run_mopd'
