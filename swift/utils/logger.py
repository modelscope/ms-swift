# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import logging
import os
from typing import Optional

from modelscope.utils.logger import get_logger as get_ms_logger


# Avoid circular reference
def _is_local_master():
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    return local_rank in {-1, 0}


init_loggers = {}

# old format
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_format = logging.Formatter('[%(levelname)s:%(name)s] %(message)s')


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
    logger_name = __name__.split('.')[0]
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

    return logger


logger = get_logger()
ms_logger = get_ms_logger()

logger.handlers[0].setFormatter(logger_format)
ms_logger.handlers[0].setFormatter(logger_format)
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
if _is_local_master():
    ms_logger.setLevel(log_level)
else:
    ms_logger.setLevel(logging.ERROR)


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
