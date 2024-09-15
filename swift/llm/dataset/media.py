import hashlib
import os
import shutil
from typing import Any, Dict, Literal, Optional, Union, Callable

import numpy as np
from modelscope.hub.utils.utils import get_cache_dir

from swift.utils import get_logger

logger = get_logger()


class MediaProcessor:


class MediaResource:

    cache_dir = os.path.join(get_cache_dir(), 'media_resources')
    lock_dir = os.path.join(get_cache_dir(), 'lockers')

    media_type_urls = {
        'llava', 'coco', 'sam', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2', 'share_textvqa', 'web-celebrity',
        'web-landmark', 'wikiart'
    }

    URL_PREFIX = 'https://www.modelscope.cn/api/v1/datasets/hjh0119/sharegpt4v-images/repo?Revision=master&FilePath='

    @staticmethod
    def get_url(media_type):
        is_ocr_vqa = (media_type == 'ocr_vqa')
        extension = 'tar' if is_ocr_vqa else 'zip'
        return f'{MediaResource.URL_PREFIX}{media_type}.{extension}'

    @staticmethod
    def download(media_type_or_url: str, local_alias: Optional[str] = None):
        """Download and extract a resource from a http link.

        Args:
            media_type_or_url: `str`, Either belongs to the `media_type_urls` listed in the class field, or a
                remote url to download and extract. Be aware that, this media type or url
                needs to contain a zip or tar file.
            local_alias: `Options[str]`, The local alias name for the `media_type_or_url`. If the first arg is a
            media_type listed in this class, local_alias can leave None. else please pass in a name for the url.
            The local dir contains the extracted files will be: {cache_dir}/{local_alias}

        Returns:
            The local dir contains the extracted files.
        """
        from swift.utils import safe_ddp_context
        from datasets.utils.filelock import FileLock
        file_path = hashlib.md5(media_type_or_url.encode('utf-8')).hexdigest() + '.lock'
        file_path = os.path.join(MediaResource.lock_dir, file_path)
        os.makedirs(MediaResource.lock_dir, exist_ok=True)
        with safe_ddp_context():
            with FileLock(file_path):
                return MediaResource._safe_download(media_type=media_type_or_url, media_name=local_alias)

    @staticmethod
    def _safe_download(media_type, media_name=None):
        media_name = media_name or media_type
        if media_type in MediaResource.media_type_urls:
            media_type = MediaResource.get_url(media_type)

        from datasets.download.download_manager import DownloadManager, DownloadConfig
        final_folder = os.path.join(MediaResource.cache_dir, media_name)
        if os.path.exists(final_folder):
            return final_folder

        logger.info('# #################Resource downloading#################')
        logger.info('Downloading necessary resources...')
        logger.info(f'Resource package: {media_type}')
        logger.info(f'Extracting to local dir: {final_folder}')
        logger.info('If the downloading fails or lasts a long time, '
                    'you can manually download the resources and extracting to the local dir.')
        logger.info('Now begin.')
        local_dirs = DownloadManager(download_config=DownloadConfig(
            cache_dir=MediaResource.cache_dir)).download_and_extract(media_type)
        shutil.move(str(local_dirs), final_folder)
        logger.info('# #################Resource downloading finished#################')
        return final_folder

    @staticmethod
    def safe_save(image, file_name, folder, format='JPEG'):
        folder = os.path.join(MediaResource.cache_dir, folder)
        os.makedirs(folder, exist_ok=True)
        file = os.path.join(folder, file_name)
        if os.path.exists(file):
            return file
        image.save(file, format=format)
        return file
