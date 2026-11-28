# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import pytest
from datasets.utils.filelock import FileLock
from filelock import UnixFileLock
from unittest.mock import DEFAULT, patch

from swift.utils import torch_utils  # noqa: F401 (installs the filelock patch)


@pytest.mark.skipif(not issubclass(FileLock, UnixFileLock), reason='requires Unix file locks')
@pytest.mark.parametrize('error', [FileExistsError, FileNotFoundError])
def test_filelock_retries_open_race(tmp_path, error):
    lock = FileLock(str(tmp_path / 'builder.lock'), timeout=1)
    with patch.object(os, 'open', wraps=os.open, side_effect=[error, DEFAULT]):
        with lock:
            assert lock.is_locked
    assert not lock.is_locked
