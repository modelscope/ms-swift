# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-sample streaming MOPD driver.

:class:`StreamingMOPDLoop` is :class:`~swift.dev.recipe.gkd_async.StreamingGKDLoop` recomposed with
:class:`~swift.dev.recipe.run_mopd.MOPDLoop`'s multi-teacher signal. Like OPSD, MOPD streams exactly as GKD
does -- every streaming hook and the mixin's per-sample driver are inherited from ``StreamingGKDLoop`` -- and
only the teacher view differs: ``MOPDLoop._teacher_kwargs`` loops the K frozen teachers over the student's
own on-policy features into K response-only ``teacher_logps`` channels (fused by ``MOPDLoss``), reached
through the MRO below. This file therefore adds no streaming logic of its own.

The base order ``(StreamingGKDLoop, MOPDLoop)`` linearises to
``StreamingMOPDLoop -> StreamingGKDLoop -> StreamingLoopMixin -> MOPDLoop -> OPSDLoop -> GKDLoop ->
GRPOLoop -> ...``: the streaming half resolves at ``StreamingGKDLoop``/``StreamingLoopMixin``,
``_teacher_kwargs`` at ``MOPDLoop``, ``_forward_kwargs`` at ``OPSDLoop`` and the consume/train skeleton at
``GKDLoop``. Each base is a distinct branch (``MOPDLoop`` does not inherit the streaming classes), so there
is no duplicate-base C3 conflict.
"""
from __future__ import annotations

from swift.dev.recipe.gkd_async import StreamingGKDLoop
from swift.dev.recipe.run_mopd import MOPDLoop


class StreamingMOPDLoop(StreamingGKDLoop, MOPDLoop):
    """MOPD loop streaming per-sample over the on-policy regime (staleness 0 or 1), multi-teacher signal.

    Inherits ``StreamingGKDLoop``'s constructor and every streaming hook unchanged; only ``_RUN_ID`` is set
    so the in-process control-plane key is namespaced for MOPD.
    """

    _RUN_ID = 'run_mopd'
