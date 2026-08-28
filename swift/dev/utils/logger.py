# Copyright (c) ModelScope Contributors. All rights reserved.
"""dev's own logger, copied from ``swift.utils.logger`` so the dev stack is self-contained.

One deliberate difference from legacy: the logger is named ``swift.dev`` rather than ``swift``.
In a dev run legacy ``swift.utils.logger`` is still imported (for the handful of complex utils
dev keeps on legacy) and it attaches a ``StreamHandler`` to the global ``swift`` logger. If dev
attached to that same logger too, every record would be emitted twice. A distinct name with
``propagate=False`` keeps dev's handler off the shared ``swift`` logger entirely.
"""
import importlib.util
import logging
import os
from contextlib import contextmanager
from types import MethodType
from typing import Optional


# Avoid circular reference
def _is_local_master():
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    return local_rank in {-1, 0}


init_loggers = {}

logger_format = logging.Formatter('[%(levelname)s:%(name)s] %(message)s')

info_set = set()
warning_set = set()


def info_if(self, msg, cond, *args, **kwargs):
    if cond:
        with logger_context(self, logging.INFO):
            self.info(msg)


def warning_if(self, msg, cond, *args, **kwargs):
    if cond:
        with logger_context(self, logging.INFO):
            self.warning(msg)


def info_once(self, msg, *args, **kwargs):
    hash_id = kwargs.get('hash_id') or msg
    if hash_id in info_set:
        return
    info_set.add(hash_id)
    self.info(msg)


def warning_once(self, msg, *args, **kwargs):
    hash_id = kwargs.get('hash_id') or msg
    if hash_id in warning_set:
        return
    warning_set.add(hash_id)
    self.warning(msg)


def get_logger(log_file: Optional[str] = None, log_level: Optional[int] = None, file_mode: str = 'w'):
    """ Get logging logger

    Args:
        log_file: Log filename, if specified, file handler will be added to
            logger
        log_level: Logging level.
        file_mode: Specifies the mode to open the file, if filename is
            specified (if filemode is unspecified, it defaults to 'w').
    """
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_level = getattr(logging, log_level, logging.INFO)
    # Distinct from legacy's ``swift`` logger on purpose -- see the module docstring.
    logger_name = 'swift.dev'
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    if logger_name in init_loggers:
        add_file_handler_if_needed(logger, log_file, file_mode, log_level)
        return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    is_worker0 = _is_local_master()

    if is_worker0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    for handler in handlers:
        handler.setFormatter(logger_format)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if is_worker0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    init_loggers[logger_name] = True

    logger.info_once = MethodType(info_once, logger)
    logger.warning_once = MethodType(warning_once, logger)
    logger.info_if = MethodType(info_if, logger)
    logger.warning_if = MethodType(warning_if, logger)
    return logger


@contextmanager
def logger_context(logger, log_leval):
    origin_log_level = logger.level
    logger.setLevel(log_leval)
    try:
        yield
    finally:
        logger.setLevel(origin_log_level)


def add_file_handler_if_needed(logger, log_file, file_mode, log_level):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return

    if importlib.util.find_spec('torch') is not None:
        is_worker0 = int(os.getenv('LOCAL_RANK', -1)) in {-1, 0}
    else:
        is_worker0 = True

    if is_worker0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(logger_format)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
