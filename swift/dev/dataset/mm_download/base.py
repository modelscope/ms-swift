# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import os
import shutil
from typing import Dict, List, Optional, Type, Union

from modelscope.hub.utils.utils import get_cache_dir

from swift.dev.utils import get_logger

logger = get_logger()


class MediaDownloader:
    """Base downloader: resolve a resource, fetch it once under a lock, cache it atomically.

    Subclasses set ``file_type`` and implement ``fetch``. Callers do not instantiate a subclass
    directly; they call ``download``, which is the factory.
    """

    cache_dir = os.path.join(get_cache_dir(), 'media_resources')

    # Bare media-type ids resolving to an archive on the shared sharegpt4v mirror; anything else
    # passed to `download` is treated as a literal URL.
    MEDIA_TYPES = {
        'llava', 'coco', 'sam', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2', 'share_textvqa', 'web-celebrity',
        'web-landmark', 'wikiart'
    }
    URL_PREFIX = 'https://www.modelscope.cn/api/v1/datasets/hjh0119/sharegpt4v-images/repo?Revision=master&FilePath='

    # Subclass fetch mode; also its registry key. `None` on the base keeps the base out.
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

        ``media_type_or_url`` is a media-type id (see ``MEDIA_TYPES``), an archive URL, or -- for
        ``file_type='sharded'`` -- a list of URLs whose contents merge. ``local_alias`` names the
        directory under ``cache_dir`` (optional for a string, required for a list). ``file_type``:
        ``'compressed'`` (unpack) / ``'file'`` (one file) / ``'files'`` (loose files) / ``'sharded'``.
        """
        downloader_cls = cls._REGISTRY.get(file_type)
        if downloader_cls is None:
            raise ValueError(f'Unknown file_type `{file_type}`. Available: {sorted(cls._REGISTRY)}')
        return downloader_cls().run(media_type_or_url, local_alias)

    def run(self, media_type_or_url: Union[str, List[str]], local_alias: Optional[str]) -> str:
        source, alias, lock_key = self.resolve(media_type_or_url, local_alias)
        final_folder = os.path.join(self.cache_dir, alias)

        # Fast path: promotion is atomic, so the folder existing already guarantees completeness.
        if os.path.exists(final_folder):
            return final_folder

        with self.serialised(lock_key):
            # Re-check under the lock: another rank/process may have finished it while we waited.
            if os.path.exists(final_folder):
                return final_folder
            tmp_folder = f'{final_folder}.tmp'
            # A leftover .tmp is the fingerprint of a crashed attempt; clear and retry rather than
            # resuming into a directory of unknown state.
            shutil.rmtree(tmp_folder, ignore_errors=True)
            os.makedirs(tmp_folder, exist_ok=True)

            logger.info(f'Downloading media resource `{source}` -> `{final_folder}`. '
                        'If this stalls, download it manually and extract it to that path.')
            self.fetch(source, tmp_folder)
            # Atomic promotion: an interrupted fetch leaves only `tmp_folder`, never a half-written
            # `final_folder` that the fast path would trust.
            os.rename(tmp_folder, final_folder)
        return final_folder

    @staticmethod
    def serialised(key: str):
        """Let one rank download while the rest wait, then find the folder already there.

        ``sticky``: the fast path returns before this is entered, so ranks reach it
        asymmetrically -- a round-based barrier would deadlock on that. A sticky flag names the
        downloaded folder instead, so a rank that skipped the work owes nothing to anyone.
        """
        from twinkle.utils import processing_lock
        return processing_lock(key, sticky=True)

    def resolve(self, media_type_or_url: Union[str, List[str]],
                local_alias: Optional[str]) -> tuple:
        """Return ``(source, alias, lock_key)``; a media-type id maps to its archive URL here.

        ``lock_key`` serialises concurrent downloaders; for a sharded list it is the first URL,
        enough to name the one resource all its shards belong to.
        """
        if isinstance(media_type_or_url, str):
            alias = local_alias or media_type_or_url
            source = self.get_url(media_type_or_url) if media_type_or_url in self.MEDIA_TYPES else media_type_or_url
            return source, alias, source
        # A list (sharded): no single string to fall back on, so an alias must be supplied.
        assert local_alias, 'A `local_alias` is required when passing a list of URLs.'
        return media_type_or_url, local_alias, media_type_or_url[0]

    def fetch(self, source: Union[str, List[str]], dest_dir: str) -> None:
        """Fill ``dest_dir`` with the resource. Implemented per fetch mode by subclasses."""
        raise NotImplementedError

    @classmethod
    def get_url(cls, media_type: str) -> str:
        """Archive URL for a known media-type id. ``ocr_vqa`` ships a tar; the rest ship zips."""
        extension = 'tar' if media_type == 'ocr_vqa' else 'zip'
        return f'{cls.URL_PREFIX}{media_type}.{extension}'

    @staticmethod
    def download_config():
        """A ``DownloadConfig`` with a day-long timeout, for these large archives."""
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

        For datasets carrying images inline (as PIL objects) that only need them on disk so a later
        stage can reference them by path. Skips the write if the file is already there.
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

    One resource, not a ``'file'`` fetch per name: a fetched resource is a *folder* whose
    existence marks the fetch complete, so a second ``'file'`` fetch under the same folder name
    would be answered from the fast path having downloaded only the first file.
    """

    file_type = 'files'

    def fetch(self, source: List[str], dest_dir: str) -> None:
        from datasets.download.download_manager import DownloadManager
        manager = DownloadManager(download_config=self.download_config())
        for url in source:
            downloaded = manager.download(url)
            shutil.move(str(downloaded), os.path.join(dest_dir, url.split('/')[-1]))
