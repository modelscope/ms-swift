# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-sample streaming OPSD driver.

:class:`StreamingOPSDLoop` is :class:`~swift.dev.recipe.gkd_async.StreamingGKDLoop` recomposed with
:class:`~swift.dev.recipe.run_opsd.OPSDLoop`'s teacher signal. OPSD streams exactly like GKD -- the
per-sample disaggregated admission, as-completed collection, individual-sample assembly, real-optimizer-step
publish cadence and both publication mechanisms are all inherited from ``StreamingGKDLoop`` (which carries
the distill answers to the streaming mixin's hooks); the ONLY thing that changes is the teacher view, which
lives in ``OPSDLoop._teacher_kwargs``/``_forward_kwargs`` and is reached through the MRO below. So this file
adds no streaming logic of its own -- it just wires the mixin's driver onto the privileged-teacher subclass.

The base order ``(StreamingGKDLoop, OPSDLoop)`` linearises to
``StreamingOPSDLoop -> StreamingGKDLoop -> StreamingLoopMixin -> OPSDLoop -> GKDLoop -> GRPOLoop -> ...``:
the streaming half (``_drive``, the hooks, the pass-through ``_finalize_samples``) resolves at
``StreamingGKDLoop``/``StreamingLoopMixin``, while ``_teacher_kwargs``/``_forward_kwargs`` resolve at
``OPSDLoop`` and ``_consume_async_samples``/``_train_distill_rows`` at ``GKDLoop``. Each base is a distinct
branch (``OPSDLoop`` does not inherit the streaming classes), so there is no duplicate-base C3 conflict.
"""
from __future__ import annotations

from swift.dev.recipe.gkd_async import StreamingGKDLoop
from swift.dev.recipe.run_opsd import OPSDLoop


class StreamingOPSDLoop(StreamingGKDLoop, OPSDLoop):
    """OPSD loop streaming per-sample over the on-policy regime (staleness 0 or 1), privileged teacher.

    Inherits ``StreamingGKDLoop``'s constructor and every streaming hook unchanged; only ``_RUN_ID`` is set
    so the in-process control-plane key is namespaced for OPSD.
    """

    _RUN_ID = 'run_opsd'
