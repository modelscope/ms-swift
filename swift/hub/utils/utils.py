# Copyright (c) Alibaba, Inc. and its affiliates.

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from swift.hub.constants import (DEFAULT_MODELSCOPE_DOMAIN, DEFAULT_MODELSCOPE_GROUP, MODEL_ID_SEPARATOR,
                                 MODELSCOPE_SDK_DEBUG, MODELSCOPE_URL_SCHEME)
from swift.hub.errors import FileIntegrityError
from swift.utils.logger import get_logger

logger = get_logger()


def get_default_cache_dir():
    """
    default base dir: '~/.cache/modelscope'
    """
    default_cache_dir = Path.home().joinpath('.cache', 'modelscope')
    return default_cache_dir


def model_id_to_group_owner_name(model_id):
    if MODEL_ID_SEPARATOR in model_id:
        group_or_owner = model_id.split(MODEL_ID_SEPARATOR)[0]
        name = model_id.split(MODEL_ID_SEPARATOR)[1]
    else:
        group_or_owner = DEFAULT_MODELSCOPE_GROUP
        name = model_id
    return group_or_owner, name


def get_cache_dir(model_id: Optional[str] = None):
    """cache dir precedence:
        function parameter > environment > ~/.cache/modelscope/hub

    Args:
        model_id (str, optional): The model id.

    Returns:
        str: the model_id dir if model_id not None, otherwise cache root dir.
    """
    default_cache_dir = get_default_cache_dir()
    base_path = os.getenv('MODELSCOPE_CACHE', os.path.join(default_cache_dir, 'hub'))
    return base_path if model_id is None else os.path.join(base_path, model_id + '/')


def get_release_datetime():
    if MODELSCOPE_SDK_DEBUG in os.environ:
        rt = int(round(datetime.now().timestamp()))
    else:
        from swift import version
        rt = int(round(datetime.strptime(version.__release_datetime__, '%Y-%m-%d %H:%M:%S').timestamp()))
    return rt


def get_endpoint():
    modelscope_domain = os.getenv('MODELSCOPE_DOMAIN', DEFAULT_MODELSCOPE_DOMAIN)
    return MODELSCOPE_URL_SCHEME + modelscope_domain


def compute_hash(file_path):
    BUFFER_SIZE = 1024 * 64  # 64k buffer size
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(BUFFER_SIZE)
            if not data:
                break
            sha256_hash.update(data)
    return sha256_hash.hexdigest()


def file_integrity_validation(file_path, expected_sha256):
    """Validate the file hash is expected, if not, delete the file

    Args:
        file_path (str): The file to validate
        expected_sha256 (str): The expected sha256 hash

    Raises:
        FileIntegrityError: If file_path hash is not expected.

    """
    file_sha256 = compute_hash(file_path)
    if not file_sha256 == expected_sha256:
        os.remove(file_path)
        msg = 'File %s integrity check failed, the download may be incomplete, please try again.' % file_path
        logger.error(msg)
        raise FileIntegrityError(msg)
