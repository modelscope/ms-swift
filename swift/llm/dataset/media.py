"""
模块功能
-------
本模块提供多媒体资源（图像/压缩包等）的下载、解压与本地缓存管理：
- 统一的资源下载入口 `MediaResource.download`，支持传入媒体类型或 URL；
- 支持三种文件类型：压缩包（自动解压）、原始文件（直接保存）、分片文件（多 URL 合并解压）；
- 使用 modelscope 的缓存目录组织本地资源；
- 在分布式/多进程环境下通过 `safe_ddp_context` 保证一次下载、多进程复用；
- 提供 `safe_save` 将图片安全保存到缓存的便捷方法。

快速示例
------
>>> # 1) 下载已预置的媒体类型（自动映射为 URL 并解压）
>>> root = MediaResource.download('coco')
>>> print(root)

>>> # 2) 下载任意 URL 的压缩包，指定本地别名
>>> root = MediaResource.download('https://example.com/data.zip', local_alias='my_data')

>>> # 3) 下载单个原始文件
>>> root = MediaResource.download('https://example.com/image.jpg', local_alias='imgs', file_type='file')
"""

# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明
import os  # 操作系统路径与目录操作
import shutil  # 文件移动与复制操作
from typing import List, Literal, Optional, Union  # 类型注解

import aiohttp  # HTTP 超时配置（供下载管理器使用）
from modelscope.hub.utils.utils import get_cache_dir  # ModelScope 缓存根目录

from swift.utils import get_logger, safe_ddp_context  # 日志器与分布式安全上下文

logger = get_logger()  # 模块级日志器


class MediaResource:
    """
    媒体资源管理类：提供资源下载/解压/缓存的统一接口。

    特性
    ----
    - 预置常见媒体类型到 URL 的映射，便于按关键字下载数据资源；
    - 自动选择下载目录并解压到缓存位置；
    - 支持分布式/多进程安全下载，避免重复工作。
    """

    cache_dir = os.path.join(get_cache_dir(), 'media_resources')  # 资源缓存根目录
    lock_dir = os.path.join(get_cache_dir(), 'lockers')  # 锁文件目录（如需扩展分布式锁）

    media_type_urls = {  # 预置可识别的媒体类型关键字集合
        'llava', 'coco', 'sam', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2', 'share_textvqa', 'web-celebrity',
        'web-landmark', 'wikiart'
    }

    URL_PREFIX = 'https://www.modelscope.cn/api/v1/datasets/hjh0119/sharegpt4v-images/repo?Revision=master&FilePath='  # URL 前缀模板

    @staticmethod
    def get_url(media_type):
        """
        将预置媒体类型转换为下载 URL。

        参数
        ----
        - media_type: 预置的媒体类型关键字（如 'coco'、'ocr_vqa'）。

        返回
        ----
        - str: 对应的压缩包下载链接（ocr_vqa 使用 tar，其余使用 zip）。

        示例
        ----
        >>> MediaResource.get_url('coco')
        'https://.../coco.zip'
        """
        is_ocr_vqa = (media_type == 'ocr_vqa')  # ocr_vqa 资源为 tar 包，其余为 zip 包
        extension = 'tar' if is_ocr_vqa else 'zip'  # 选择扩展名
        return f'{MediaResource.URL_PREFIX}{media_type}.{extension}'  # 拼接完整 URL

    @staticmethod
    def download(media_type_or_url: Union[str, List[str]],
                 local_alias: Optional[str] = None,
                 file_type: Literal['compressed', 'file', 'sharded'] = 'compressed'):
        """
        从 HTTP 链接下载资源并（可选）解压到本地缓存。

        参数
        ----
        - media_type_or_url: 可为字符串或字符串列表。若为预置媒体类型，将自动映射到 URL；若为 URL 则直接下载。
            注意：当 file_type='compressed' 时应指向压缩包；当 file_type='file' 时为原始文件；当 file_type='sharded' 时为多段压缩。
        - local_alias: 本地别名目录名；若传入的是预置媒体类型可留空，否则建议指定，最终目录为 `{cache_dir}/{local_alias}`。
        - file_type: 文件类型，compressed=压缩包并解压，file=原始文件仅下载，sharded=多 URL 下载并合并目录。

        返回
        ----
        - str: 包含资源文件的本地目录路径。

        示例
        ----
        >>> MediaResource.download('coco')
        '/.../media_resources/coco'
        """
        media_file = media_type_or_url if isinstance(media_type_or_url, str) else media_type_or_url[0]  # 选取用于加锁的标识
        with safe_ddp_context(hash_id=media_file):  # 分布式安全上下文，保证只下载一次
            return MediaResource._safe_download(
                media_type=media_type_or_url, media_name=local_alias, file_type=file_type)  # 执行安全下载

    @staticmethod
    def move_directory_contents(src_dir, dst_dir):
        """
        递归移动 `src_dir` 中的所有文件到 `dst_dir`，同时保留相对目录结构。

        参数
        ----
        - src_dir: 源目录（已解压目录）。
        - dst_dir: 目标目录。
        """
        if not os.path.exists(dst_dir):  # 目标目录不存在则创建
            os.makedirs(dst_dir)  # 创建目标目录

        for dirpath, dirnames, filenames in os.walk(src_dir):  # 遍历源目录
            relative_path = os.path.relpath(dirpath, src_dir)  # 计算相对路径
            target_dir = os.path.join(dst_dir, relative_path)  # 目标对应目录

            if not os.path.exists(target_dir):  # 若目标子目录不存在
                os.makedirs(target_dir)  # 创建它

            for file in filenames:  # 遍历当前目录的文件
                src_file = os.path.join(dirpath, file)  # 源文件路径
                dst_file = os.path.join(target_dir, file)  # 目标文件路径
                shutil.move(src_file, dst_file)  # 移动文件

    @staticmethod
    def _safe_download(media_type: Union[str, List[str]],
                       media_name: Optional[str] = None,
                       file_type: Literal['compressed', 'file', 'sharded'] = 'compressed'):
        """
        安全下载实现：根据不同 file_type 执行下载与解压逻辑。

        参数
        ----
        - media_type: 预置媒体类型关键字、URL 字符串，或分片 URL 列表。
        - media_name: 本地缓存目录名（默认使用 media_type）。
        - file_type: 'compressed' | 'file' | 'sharded'。

        返回
        ----
        - str: 最终缓存目录路径。
        """
        media_name = media_name or media_type  # 若未指定别名则使用原始标识
        assert isinstance(media_name, str), f'{media_name} is not a str'  # 目录名需为字符串
        if isinstance(media_type, str) and media_type in MediaResource.media_type_urls:  # 若传入为预置类型
            media_type = MediaResource.get_url(media_type)  # 转换为下载 URL

        from datasets.download.download_manager import DownloadManager, DownloadConfig  # 延迟导入下载器
        final_folder = os.path.join(MediaResource.cache_dir, media_name)  # 目标缓存目录

        if file_type == 'file':  # 单文件下载模式
            filename = media_type.split('/')[-1]  # 解析文件名
            final_path = os.path.join(final_folder, filename)  # 构造最终文件路径
            if os.path.exists(final_path):  # if the download thing is a file but not folder,
                return final_folder  # check whether the file exists  # 已存在则直接返回目录
            if not os.path.exists(final_folder):  # 目录不存在
                os.makedirs(final_folder)  # and make sure final_folder exists to contain it  # 创建目录
        else:  # 压缩包/分片模式
            if os.path.exists(final_folder):  # 若目标目录已存在（视为已完成）
                return final_folder  # 直接返回

        media_name = media_name or media_type
        assert isinstance(media_name, str), f''
        if isinstance(media_type, str) and media_type in MediaResource.media_type_urls:
            media_type = MediaResource.get_url(media_type)
        
        logger.info('# #################Resource downloading#################')  # 进度日志
        logger.info('Downloading necessary resources...')  # 提示开始下载
        logger.info(f'Resource package: {media_type}')  # 打印资源包信息
        logger.info(f'Extracting to local dir: {final_folder}')  # 打印解压目标目录
        logger.info('If the downloading fails or lasts a long time, '
                    'you can manually download the resources and extracting to the local dir.')  # 人工介入指引
        logger.info('Now begin.')  # 开始
        download_config = DownloadConfig(cache_dir=MediaResource.cache_dir)  # 配置缓存目录
        download_config.storage_options = {'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=86400)}}  # 设置长超时
        if file_type == 'file':  # 原始文件下载
            filename = media_type.split('/')[-1]  # 再取一次文件名（保证一致）
            final_path = os.path.join(final_folder, filename)  # 最终文件路径
            local_dirs = DownloadManager(download_config=download_config).download(media_type)  # 直接下载文件
            shutil.move(str(local_dirs), final_path)  # 移动到目标路径
        elif file_type == 'compressed':  # 压缩包下载并解压
            local_dirs = DownloadManager(download_config=download_config).download_and_extract(media_type)  # 下载并解压
            shutil.move(str(local_dirs), final_folder)  # 将解压产物移动到最终目录
        else:  # 'sharded' 分片压缩：逐个下载并合并
            for media_url in media_type:  # 遍历每个分片 URL
                local_dirs = DownloadManager(download_config=download_config).download_and_extract(media_url)  # 下载解压
                MediaResource.move_directory_contents(str(local_dirs), final_folder)  # 将内容合并到最终目录
        logger.info('# #################Resource downloading finished#################')  # 结束日志
        return final_folder  # 返回缓存目录

    @staticmethod
    def safe_save(image, file_name, folder, format='JPEG'):
        """
        将 PIL 图像对象安全保存到缓存，并返回保存路径。

        参数
        ----
        - image: PIL.Image 对象。
        - file_name: 文件名（包含扩展名）。
        - folder: 缓存子目录名。
        - format: 保存格式（默认 'JPEG'）。

        返回
        ----
        - str: 最终保存文件的绝对路径。

        示例
        ----
        >>> from PIL import Image
        >>> img = Image.new('RGB', (10, 10), color='white')
        >>> path = MediaResource.safe_save(img, 'x.jpg', 'demo')
        """
        folder = os.path.join(MediaResource.cache_dir, folder)  # 拼接缓存子目录路径
        os.makedirs(folder, exist_ok=True)  # 确保目录存在
        file = os.path.join(folder, file_name)  # 目标文件路径
        if os.path.exists(file):  # 若文件已存在
            return file  # 直接返回现有路径
        image.save(file, format=format)  # 保存图像到文件
        return file  # 返回保存路径
