# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared row-error logging for the components that encode a row and can fail on it.

``SwiftDataset``, ``LazyLLMDataset`` and the ``Preprocessor`` family each catch a per-row encode error and
want to log the traceback -- but only a bounded number of times, so one broken dataset cannot flood the
logs. The budget logic is identical across them and only the warning text differs, so it lives here once.

What happens after the log differs by layer: the preprocessors drop the row or mark it unusable, while
the datasets let the error propagate so the reader above them (twinkle's ``RetryDataset``) can decide
whether to substitute.
"""
from __future__ import annotations

import traceback


def note_row_error(owner, message: str) -> None:
    """Log the in-flight exception's traceback, up to ``owner.traceback_limit`` times.

    Must be called from inside the ``except`` block so ``traceback.format_exc`` still sees the error.
    ``owner`` is any object carrying a ``traceback_limit`` (``None`` disables the budget, i.e. log every
    time) and a mutable ``_traceback_counter``.
    """
    if owner.traceback_limit is not None and owner._traceback_counter >= owner.traceback_limit:
        return
    from swift.dev.utils import get_logger
    logger = get_logger()
    logger.info(traceback.format_exc())
    logger.warning(message)
    owner._traceback_counter += 1
