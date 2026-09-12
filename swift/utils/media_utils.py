# Copyright (c) ModelScope Contributors. All rights reserved.
import os

from .utils import get_env_args


class SafeMediaPath:
    """Guard a local media path that may have been chosen by an untrusted caller.

    A media field in a `swift deploy` request holds a URL, base64 data, or a plain string that the server
    looks up on disk. That last form lets a caller name any file the server can open: the file is read and
    handed to the model, which then describes its contents back to the caller, so a private image elsewhere
    on the host is readable through the API (and any other file is at least probed for its existence).

    `SWIFT_MEDIA_ALLOWED_DIRS=dir1,dir2` restricts local media to those directories. It is unset by default,
    because training and local inference legitimately load media from anywhere on disk; set it whenever
    callers are untrusted, pointing it at a directory holding nothing else -- an empty one refuses local
    paths outright, which is what a service reachable by others usually wants.
    """

    @classmethod
    def check(cls, path: str) -> str:
        """Return `path` unchanged, or raise a ValueError if reading it is not allowed."""
        allowed_dirs = get_env_args('swift_media_allowed_dirs', str, None)
        if not allowed_dirs:
            return path
        # Compare resolved paths so that neither `..` nor a symlink can step outside an allowed directory.
        real_path = os.path.realpath(path)
        for allowed_dir in allowed_dirs.split(','):
            allowed_dir = allowed_dir.strip()
            if not allowed_dir:
                continue
            allowed_dir = os.path.realpath(os.path.expanduser(allowed_dir))
            if real_path == allowed_dir or real_path.startswith(allowed_dir + os.sep):
                return path
        raise ValueError(f'Refusing to read the media file {path!r}: it is outside the directories listed in '
                         'the `SWIFT_MEDIA_ALLOWED_DIRS` environment variable.')
