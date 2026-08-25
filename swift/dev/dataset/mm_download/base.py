# Copyright (c) ModelScope Contributors. All rights reserved.
"""Fetch a remote media resource into a stable local cache directory, exactly once.

Many multimodal datasets ship only text and image *ids*; the pixels live in a separate archive that
has to be pulled and unpacked on first use. This module owns that fetch.

Design, following the ``base.py`` pattern used elsewhere in ``dev`` (a base class + a small registry +
a factory ``classmethod``): the three fetch modes are three :class:`MediaDownloader` subclasses rather
than legacy ``MediaResource``'s single ``_safe_download`` branching on a ``file_type`` string. Each
subclass only implements how it puts bytes on disk (:meth:`MediaDownloader.fetch`); the base owns the
parts that must be identical for all of them -- id/url resolution, DDP-safe locking, the cache check,
and completion.

Bug fixed versus legacy: **downloads are now atomic.** Legacy checked ``os.path.exists(final_folder)``
and then extracted straight into ``final_folder``; a crash midway through extraction left a partial
folder that every later run treated as a finished download, so the dataset was silently truncated and
the only recovery was to delete the cache by hand. Here every fetch lands in a sibling ``.tmp``
directory and is promoted to its final name with a single atomic ``os.rename`` only after it fully
succeeds, so an interrupted download leaves no folder to be mistaken for a complete one.
"""
from __future__ import annotations
import os
import shutil
from typing import Dict, List, Optional, Type, Union

from modelscope.hub.utils.utils import get_cache_dir

from swift.utils import get_logger

logger = get_logger()


class MediaDownloader:
    """Base downloader: resolve a resource, fetch it once under a lock, cache it atomically.

    Subclasses set :attr:`file_type` and implement :meth:`fetch`. Callers do not instantiate a
    subclass directly; they call :meth:`download`, which is the factory.
    """

    cache_dir = os.path.join(get_cache_dir(), 'media_resources')

    # Bare media-type ids that resolve to an archive on the shared sharegpt4v mirror; anything else
    # passed to `download` is treated as a literal URL.
    MEDIA_TYPES = {
        'llava', 'coco', 'sam', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2', 'share_textvqa', 'web-celebrity',
        'web-landmark', 'wikiart'
    }
    URL_PREFIX = 'https://www.modelscope.cn/api/v1/datasets/hjh0119/sharegpt4v-images/repo?Revision=master&FilePath='

    # Subclass fetch mode; also its registry key. `None` on the base keeps the base out of the registry.
    file_type: Optional[str] = None
    _REGISTRY: Dict[str, Type['MediaDownloader']] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.file_type is not None:
            MediaDownloader._REGISTRY[cls.file_type] = cls

    @classmethod
    def download(cls,
                 media_type_or_url: Union[str, List[str]],
                 local_alias: Optional[str] = None,
                 file_type: str = 'compressed') -> str:
        """Fetch a resource and return the local directory holding it.

        Args:
            media_type_or_url: A known media-type id (see :attr:`MEDIA_TYPES`), a literal archive URL,
                or -- for ``file_type='sharded'`` -- a list of URLs whose extracted contents are merged.
            local_alias: Local directory name under :attr:`cache_dir`. Optional only when
                ``media_type_or_url`` is a single string (the id/url is then reused as the name);
                required for a sharded list, since a list has no single name to fall back to.
            file_type: ``'compressed'`` (download and unpack an archive), ``'file'`` (download one
                file verbatim), ``'files'`` (download several loose files into one folder), or
                ``'sharded'`` (download and merge several archives).

        Returns:
            Absolute path of the directory containing the resource.
        """
        downloader_cls = cls._REGISTRY.get(file_type)
        if downloader_cls is None:
            raise ValueError(f'Unknown file_type `{file_type}`. Available: {sorted(cls._REGISTRY)}')
        return downloader_cls().run(media_type_or_url, local_alias)

    def run(self, media_type_or_url: Union[str, List[str]], local_alias: Optional[str]) -> str:
        source, alias, lock_key = self.resolve(media_type_or_url, local_alias)
        final_folder = os.path.join(self.cache_dir, alias)

        # Fast path: a completed download is a folder that exists, and -- because promotion is atomic
        # -- its mere existence now guarantees completeness, so no per-mode inner check is needed.
        if os.path.exists(final_folder):
            return final_folder

        with self.serialised(lock_key):
            # Re-check under the lock: another rank/process may have finished it while we waited.
            if os.path.exists(final_folder):
                return final_folder
            tmp_folder = f'{final_folder}.tmp'
            # A leftover .tmp is the fingerprint of a previously crashed attempt; clear it and retry
            # rather than resuming into a dir of unknown state.
            shutil.rmtree(tmp_folder, ignore_errors=True)
            os.makedirs(tmp_folder, exist_ok=True)

            logger.info(f'Downloading media resource `{source}` -> `{final_folder}`. '
                        'If this stalls, download it manually and extract it to that path.')
            self.fetch(source, tmp_folder)
            # Atomic promotion: an interrupted fetch above leaves only `tmp_folder`, never a
            # half-written `final_folder` that the fast path would trust.
            os.rename(tmp_folder, final_folder)
        return final_folder

    @staticmethod
    def serialised(key: str):
        """Let one rank download while the rest wait, then find the folder already there.

        ``sticky``, and that matters here beyond avoiding a redundant wait: the fast path above returns
        before this is ever entered, so ranks reach it asymmetrically. A round-based barrier -- which is
        what ``safe_ddp_context`` did here -- deadlocks on that, one rank waiting on peers that already
        went home. A sticky flag names the downloaded folder instead, so a rank that skipped the work
        owes nothing to anyone.
        """
        from twinkle.utils import processing_lock
        return processing_lock(key, sticky=True)

    def resolve(self, media_type_or_url: Union[str, List[str]],
                local_alias: Optional[str]) -> tuple:
        """Return ``(source, alias, lock_key)``: a media-type id maps to its archive URL here.

        ``lock_key`` is what serialises concurrent downloaders; for a sharded list it is the first
        URL, which is enough to name the one resource all its shards belong to.
        """
        if isinstance(media_type_or_url, str):
            alias = local_alias or media_type_or_url
            source = self.get_url(media_type_or_url) if media_type_or_url in self.MEDIA_TYPES else media_type_or_url
            return source, alias, source
        # A list (sharded): no single string to fall back to, so an alias must be supplied.
        assert local_alias, 'A `local_alias` is required when passing a list of URLs.'
        return media_type_or_url, local_alias, media_type_or_url[0]

    def fetch(self, source: Union[str, List[str]], dest_dir: str) -> None:
        """Fill ``dest_dir`` with the resource. Implemented per fetch mode by subclasses."""
        raise NotImplementedError

    # -- shared helpers --------------------------------------------------------------------------

    @classmethod
    def get_url(cls, media_type: str) -> str:
        """Archive URL for a known media-type id. ``ocr_vqa`` ships a tar; the rest ship zips."""
        extension = 'tar' if media_type == 'ocr_vqa' else 'zip'
        return f'{cls.URL_PREFIX}{media_type}.{extension}'

    @staticmethod
    def download_config():
        """A :class:`DownloadConfig` with a day-long timeout, for these large archives."""
        import aiohttp
        from datasets.download.download_manager import DownloadConfig
        config = DownloadConfig(cache_dir=MediaDownloader.cache_dir)
        config.storage_options = {'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=86400)}}
        return config

    @staticmethod
    def move_directory_contents(src_dir: str, dst_dir: str) -> None:
        """Merge everything under ``src_dir`` into ``dst_dir``, preserving sub-structure."""
        os.makedirs(dst_dir, exist_ok=True)
        for dirpath, _, filenames in os.walk(src_dir):
            relative_path = os.path.relpath(dirpath, src_dir)
            target_dir = os.path.join(dst_dir, relative_path)
            os.makedirs(target_dir, exist_ok=True)
            for filename in filenames:
                shutil.move(os.path.join(dirpath, filename), os.path.join(target_dir, filename))

    @staticmethod
    def safe_save(image, file_name: str, folder: str, image_format: str = 'JPEG') -> str:
        """Persist a PIL image under ``{cache_dir}/{folder}/{file_name}``, once.

        For datasets that carry images inline (as PIL objects) and only need them written to disk so a
        later stage can reference them by path. Skips the write if the file is already there.
        """
        folder = os.path.join(MediaDownloader.cache_dir, folder)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, file_name)
        if os.path.exists(path):
            return path
        image.save(path, format=image_format)
        return path


class CompressedDownloader(MediaDownloader):
    """Download one archive (zip/tar) and unpack it so its contents land in the resource folder."""

    file_type = 'compressed'

    def fetch(self, source: str, dest_dir: str) -> None:
        from datasets.download.download_manager import DownloadManager
        extracted = DownloadManager(download_config=self.download_config()).download_and_extract(source)
        self.move_directory_contents(str(extracted), dest_dir)


class FileDownloader(MediaDownloader):
    """Download a single file verbatim (no unpacking), keeping its original filename."""

    file_type = 'file'

    def fetch(self, source: str, dest_dir: str) -> None:
        from datasets.download.download_manager import DownloadManager
        downloaded = DownloadManager(download_config=self.download_config()).download(source)
        filename = source.split('/')[-1]
        shutil.move(str(downloaded), os.path.join(dest_dir, filename))


class ShardedDownloader(MediaDownloader):
    """Download several archives and merge their extracted contents into one resource folder."""

    file_type = 'sharded'

    def fetch(self, source: List[str], dest_dir: str) -> None:
        from datasets.download.download_manager import DownloadManager
        manager = DownloadManager(download_config=self.download_config())
        for url in source:
            extracted = manager.download_and_extract(url)
            self.move_directory_contents(str(extracted), dest_dir)


class FilesDownloader(MediaDownloader):
    """Download several plain files, unpacking none, into one resource folder.

    For a dataset that publishes its media as loose files rather than an archive. This has to be one
    resource rather than a ``'file'`` fetch per name: a fetched resource is a *folder*, and its
    existence is what marks the fetch complete -- so asking for the second file under the same folder
    name would be answered from the fast path, having downloaded only the first. Legacy did exactly
    that with MovieChat-1K's ~150 videos and kept only the first one.
    """

    file_type = 'files'

    def fetch(self, source: List[str], dest_dir: str) -> None:
        from datasets.download.download_manager import DownloadManager
        manager = DownloadManager(download_config=self.download_config())
        for url in source:
            downloaded = manager.download(url)
            shutil.move(str(downloaded), os.path.join(dest_dir, url.split('/')[-1]))
