import os
import shutil
import time
from typing import Literal, Union, List

import numpy as np

from swift.hub.utils.utils import get_cache_dir


class MediaTagReplacer:

    task_prompts = {
        'ref_grounding': {
            'en': [
                ('<object>', '<box>'),
                ('The positions of <object> is', '<box>'),
                ('Find the positions of <object>', '<box>'),
                ('Where is <object>', '<box>'),
                ('Find <object>', '<box>'),
                ('Show me <object>', '<box>'),
                ('Provide the bounding box coordinate of <object>', '<box>')
            ],
            'zh': [
                ('<object>', '<box>'),
                ('<object>的位置在图片中', '<box>'),
                ('<object>在图片中', '<box>'),
                ('<object>在', '<box>'),
                ('找到<object>的位置', '<box>'),
                ('<object>在哪里', '<box>'),
                ('提供<object>的坐标位置', '<box>')
            ]
        },
        'grounding_caption': {
            'en': [
                ('<box>', '<object>'),
                ('The object at position <box>', '<object>'),
                ('This <box> is', '<object>'),
                ('What is the thing at <box>', '<object>'),
                ('Describe <box>', '<object>'),
                ('<box> is', '<object>'),
                ('The bounding box coordinate <box> contains', '<object>'),
            ],
            'zh': [
                ('<box>', '<object>'),
                ('<box>是什么', '<object>'),
                ('<box>的位置包含', '<object>'),
                ('描述<box>', '<object>'),
                ('<box>中是', '<object>'),
                ('坐标<box>描述了什么', '<object>'),
                ('描述<box>中的事物', '<object>'),
            ]
        },
    }

    def __init__(self,
                 media_type: Literal['image', 'audio', 'video'],
                 media_tag=None,
                 task_type: Literal['caption_with_grounding', 'ref_grounding',
                 'grounding_caption', 'ocr', 'vqa'] = 'vqa'):
        self.media_type = media_type
        self.task_type = task_type
        self.media_tag = media_tag or '<unused_tag>'
        self.tag_pairs = {
            'image': ('<img>', '</img>'),
            'audio': ('<audio>', '</audio>'),
            'video': ('<video>', '</video>'),
        }

        self.standard_tags = {
            'image': '<image>',
            'audio': '<audio>',
            'video': '<video>',
        }

        self.media_keys = {
            'audio': 'audios',
            'image': 'images',
            'video': 'videos',
        }

    def replace_tag(self, text, url_or_base64):
        standard_tag = self.standard_tags[self.media_type]
        tag_pair = self.tag_pairs[self.media_type]
        return text.replace(standard_tag, f'{tag_pair[0]}{url_or_base64}{tag_pair[1]}', count=1)

    def split_tag(self, text: str):
        tag_pair = self.tag_pairs[self.media_type]
        if tag_pair[0] not in text or tag_pair[1] not in text:
            return text, None

        head, left = text.split(tag_pair[0], maxsplit=1)
        url_or_base64, tail = left.split(tag_pair[1], maxsplit=1)
        return f'{head}{self.standard_tags[self.media_type]}{tail}', url_or_base64

    def merge(self, text: str, medias: List):
        if not self.media_type or not medias or not isinstance(medias[0], str):
            return text
        if self.media_tag in text:
            assert text.count(self.media_tag) == len(medias)
        else:
            text = ''.join([self.media_tag] * len(medias)) + text
        for media in medias:
            text = self.replace_tag(text, media)
        return text

    def split(self, text: str):
        if not self.media_type:
            return text, None
        medias = []
        while True:
            text, media = self.split_tag(text)
            if media is None:
                break
            else:
                medias.append(media)
        return text, medias

    def __call__(self, d: dict, medias: Union[tuple, list], objects: List = None):
        """Format the query/response/history with medias

        Args:
            d: A dict contains history/query/response
            medias: A list of medias(one round, multiple medias),
                    a single media(one round, one media), or a tuple of media list(multiple rounds)
            objects: A list of object-bbox pairs(one round), or a tuple of object-bbox lists(multiple rounds)
        """
        if not self.media_type:
            return
        
        media_cnt = len(medias) if isinstance(medias, (tuple, list)) else 1 if medias else 0

        history = d.get('history') or []
        query = d.get('query')
        response = d.get('response')
        if self.task_type == 'caption_with_grounding':
            pass
        elif self.task_type in ('ref_grounding', 'grounding_caption'):
            lang = np.random.choice(['en', 'zh'], p=[0.8, 0.2])
            query, response = np.random.choice(self.task_prompts[self.task_type][lang])
        elif self.task_type == 'ocr':
            raise NotImplemented
        else:
            pass
        standard_tag = self.standard_tags[self.media_type]

        all_queries = ''.join([h[0] for h in history]) + query
        if self.media_tag in all_queries:
            assert all_queries.count(self.media_tag) == media_cnt
            for h in history:
                h[0] = h[0].replace(self.media_tag, standard_tag)

            query = query.replace(self.media_tag, standard_tag)
        else:
            if history:
                history[0][0] = ''.join([standard_tag]*media_cnt)+history[0][0]
            else:
                query = ''.join([standard_tag]*media_cnt)+query

        if 'history' in d:
            d['history'] = history
        d['query'] = query
        if 'response' in d:
            d['response'] = response
        if self.media_type:
            d[self.media_keys[self.media_type]] = medias


class MediaCache:

    cache_dir = os.path.join(get_cache_dir(), 'media_resources')

    media_type_urls = {
        'llava': 'data/llava/llava_pretrain/images',
        'coco': 'data/coco',
        'sam': 'data/sam/images',
        'gqa': 'data/gqa',
        'ocr_vqa': '.',
        'textvqa': 'textvqa',
        'VG_100K': 'vg',
        'VG_100K_2': 'vg',
        'share_textvqa': '.',
        'web-celebrity': '.',
        'web-landmark': '.',
        'wikiart': '.'
    }

    URL_PREFIX = 'https://www.modelscope.cn/api/v1/datasets/hjh0119/sharegpt4v-images/repo?Revision=master&FilePath='

    @staticmethod
    def get_url(media_type):
        is_ocr_vqa = (media_type == 'ocr_vqa')
        extension = 'tar' if is_ocr_vqa else 'zip'
        return f'{MediaCache.URL_PREFIX}{media_type}.{extension}'

    @staticmethod
    def download(media_type, media_name=None):
        from swift.utils import safe_ddp_context
        with safe_ddp_context():
            return MediaCache._safe_download(media_type=media_type, media_name=media_name)

    @staticmethod
    def _safe_download(media_type, media_name=None):
        media_name = media_name or media_type
        if media_type in MediaCache.media_type_urls:
            media_type = MediaCache.get_url(media_type)

        from datasets.download.download_manager import DownloadManager, DownloadConfig
        final_folder = os.path.join(MediaCache.cache_dir, media_name)
        if os.path.exists(final_folder):
            return final_folder
        local_dirs = DownloadManager(download_config=DownloadConfig(cache_dir=MediaCache.cache_dir)).download_and_extract(media_type)
        shutil.move(str(local_dirs), final_folder)
        return final_folder


