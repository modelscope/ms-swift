# Copyright (c) ModelScope Contributors. All rights reserved.
"""Dev-side re-export of twinkle's storage media (local / oss:// / s3://)."""
from twinkle.utils.storage import (Storage, StorageEntry, WriteLease, is_remote_url, local_run_dir,  # noqa: F401
                                   to_local, write_dir)

__all__ = ['Storage', 'StorageEntry', 'WriteLease', 'is_remote_url', 'local_run_dir', 'to_local', 'write_dir']
