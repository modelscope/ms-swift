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

    @staticmethod
    def ref_tag(ref: str):
        return ref

    @staticmethod
    def bbox_tag(bboxes: List):
        bbox_str = ''
        for bbox in bboxes:
            bbox_str += f'({bbox[0]},{bbox[1]}),({bbox[2]},{bbox[3]})'

    def __init__(self,
                 media_type: Literal['image', 'audio', 'video'],
                 media_tag=None,
                 task_type: Literal['caption_with_grounding', 'ref_grounding',
                 'grounding_caption', 'ocr', 'vqa'] = 'vqa',
                 ref_tag_func=None,
                 bbox_tag_func=None):
        self.media_type = media_type
        self.task_type = task_type
        self.media_tag = media_tag or '<unused_tag>'
        self.ref_tag = ref_tag_func
        self.bbox_tag = bbox_tag_func
        if not self.ref_tag:
            self.ref_tag = self.ref_tag
        if not self.bbox_tag:
            self.bbox_tag = self.bbox_tag
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

    def replace_bbox(self, text, bbox_pair):
        if '<object>' in text:
            assert text.count('<object>') == 1
            text = text.replace('<object>', self.ref_tag(bbox_pair[0]))
        elif '<box>' in text:
            assert text.count('<box>') == 1
            text = text.replace('<box>', self.bbox_tag(bbox_pair[1]))

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
        if not self.media_type or not medias:
            return

        if not isinstance(medias, (list, tuple)):
            medias = [medias]

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
        if isinstance(medias, list):
            all_queries = ''.join([h[0] for h in history]) + query
            if self.media_tag in all_queries:
                media_round = []
                assert all_queries.count(self.media_tag) == len(medias)
                for h in history:
                    h[0] = h[0].replace(self.media_tag, standard_tag)
                    tags_cnt = h[0].count(standard_tag)
                    media_round.append(medias[:tags_cnt])
                    medias = medias[tags_cnt:]
                media_round.append(medias)
            else:
                media_round = [medias] + [[]] * len(history)
            medias = media_round

        assert len(medias) == len(history) + 1
        # for round, media in zip(history, medias[:-1]):
        #     round[0] = self.merge(round[0], media)
        # query = self.merge(query, medias[-1])
        medias = [m for m in medias if m]
        medias = medias if not isinstance(medias[0], list) else medias[0]
        medias = medias if len(medias) > 1 else medias[0]

        if objects:
            if isinstance(objects, list):
                objects = [objects] + [[]] * len(history)
            for h, object in zip(history, objects[:-1]):
                h[0] = self.replace_bbox(h[0], object)
                h[1] = self.replace_bbox(h[1], object)
            query = self.replace_bbox(query, objects[-1])
            response = self.replace_bbox(response, objects[-1])

        if history:
            d['history'] = history
        d['query'] = query
        if 'response' in d:
            d['response'] = response
        if medias:
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
            media_type = MediaCache.media_type_urls[media_type]

        from datasets.download.download_manager import DownloadManager

        if os.path.exists(media_type):
            return media_type

        media_folder = os.path.join(MediaCache.cache_dir, media_name + '_temp')
        final_folder = os.path.join(MediaCache.cache_dir, media_name)
        if os.path.exists(final_folder):
            return final_folder
        shutil.rmtree(media_folder, ignore_errors=True)
        media_file = os.path.join(media_folder, media_type[media_type.rfind('='):])
        DownloadManager().download_and_extract(media_type)
        shutil.rmtree(media_file, ignore_errors=True)
        shutil.move(media_folder, final_folder)
        return final_folder
