# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import os

from swift.utils.logger import get_logger

logger = get_logger()

_DEFAULT_NPU_HCCL_CONNECT_TIMEOUT = '600'


def apply_patch() -> None:
    if 'HCCL_CONNECT_TIMEOUT' in os.environ:
        return

    os.environ['HCCL_CONNECT_TIMEOUT'] = _DEFAULT_NPU_HCCL_CONNECT_TIMEOUT
    logger.info(f'Set HCCL_CONNECT_TIMEOUT={_DEFAULT_NPU_HCCL_CONNECT_TIMEOUT} by default for NPU.')
