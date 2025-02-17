# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from typing import List, Literal, Optional, Union

import aiohttp
from modelscope.hub.utils.utils import get_cache_dir

from swift.utils import get_logger, safe_ddp_context

logger = get_logger()


class MediaResource:
    """A class to manage the resource downloading."""

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
    def download(media_type_or_url: Union[str, List[str]],
                 local_alias: Optional[str] = None,
                 file_type: Literal['compressed', 'file', 'sharded'] = 'compressed'):
        """Download and extract a resource from a http link.

        Args:
            media_type_or_url: `str` or List or `str`, Either belongs to the `media_type_urls`
                listed in the class field, or a remote url to download and extract.
                Be aware that, this media type or url needs to contain a zip or tar file.
            local_alias: `Options[str]`, The local alias name for the `media_type_or_url`. If the first arg is a
                media_type listed in this class, local_alias can leave None. else please pass in a name for the url.
                The local dir contains the extracted files will be: {cache_dir}/{local_alias}
            file_type: The file type, if is a compressed file, un-compressed the file,
                if is an original file, only download it, if is a sharded file, download all files and extract.

        Returns:
            The local dir contains the extracted files.
        """
        media_file = media_type_or_url if isinstance(media_type_or_url, str) else media_type_or_url[0]
        with safe_ddp_context(hash_id=media_file):
            return MediaResource._safe_download(
                media_type=media_type_or_url, media_name=local_alias, file_type=file_type)

    @staticmethod
    def move_directory_contents(src_dir, dst_dir):
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        for dirpath, dirnames, filenames in os.walk(src_dir):
            relative_path = os.path.relpath(dirpath, src_dir)
            target_dir = os.path.join(dst_dir, relative_path)

            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            for file in filenames:
                src_file = os.path.join(dirpath, file)
                dst_file = os.path.join(target_dir, file)
                shutil.move(src_file, dst_file)

    @staticmethod
    def _safe_download(media_type: Union[str, List[str]],
                       media_name: Optional[str] = None,
                       file_type: Literal['compressed', 'file', 'sharded'] = 'compressed'):
        media_name = media_name or media_type
        assert isinstance(media_name, str), f'{media_name} is not a str'
        if isinstance(media_type, str) and media_type in MediaResource.media_type_urls:
            media_type = MediaResource.get_url(media_type)

        from datasets.download.download_manager import DownloadManager, DownloadConfig
        final_folder = os.path.join(MediaResource.cache_dir, media_name)

        if file_type == 'file':
            filename = media_type.split('/')[-1]
            final_path = os.path.join(final_folder, filename)
            if os.path.exists(final_path):  # if the download thing is a file but not folder,
                return final_folder  # check whether the file exists
            if not os.path.exists(final_folder):
                os.makedirs(final_folder)  # and make sure final_folder exists to contain it
        else:
            if os.path.exists(final_folder):
                return final_folder

        logger.info('# #################Resource downloading#################')
        logger.info('Downloading necessary resources...')
        logger.info(f'Resource package: {media_type}')
        logger.info(f'Extracting to local dir: {final_folder}')
        logger.info('If the downloading fails or lasts a long time, '
                    'you can manually download the resources and extracting to the local dir.')
        logger.info('Now begin.')
        download_config = DownloadConfig(cache_dir=MediaResource.cache_dir)
        download_config.storage_options = {'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}
        if file_type == 'file':
            filename = media_type.split('/')[-1]
            final_path = os.path.join(final_folder, filename)
            local_dirs = DownloadManager(download_config=download_config).download(media_type)
            shutil.move(str(local_dirs), final_path)
        elif file_type == 'compressed':
            local_dirs = DownloadManager(download_config=download_config).download_and_extract(media_type)
            shutil.move(str(local_dirs), final_folder)
        else:
            for media_url in media_type:
                local_dirs = DownloadManager(download_config=download_config).download_and_extract(media_url)
                MediaResource.move_directory_contents(str(local_dirs), final_folder)
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
