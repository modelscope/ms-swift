# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import traceback


def note_row_error(owner, message: str) -> None:
    """Log the in-flight exception's traceback, up to ``owner.traceback_limit`` times.

    Call from inside the ``except`` block so ``traceback.format_exc`` still sees the error. ``owner``
    is any object carrying ``traceback_limit`` (``None`` disables the budget) and a mutable
    ``_traceback_counter``; what happens after the log differs by caller (drop or mark unusable).
    """
    if owner.traceback_limit is not None and owner._traceback_counter >= owner.traceback_limit:
        return
    from swift.dev.utils import get_logger
    logger = get_logger()
    logger.info(traceback.format_exc())
    logger.warning(message)
    owner._traceback_counter += 1
