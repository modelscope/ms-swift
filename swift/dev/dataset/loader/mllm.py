# Copyright (c) ModelScope Contributors. All rights reserved.
"""Multimodal dataset registrations, the counterpart of ``llm.py`` for datasets that carry media.

The three tiers of ``llm.py`` still apply (declaration only / plus ``columns`` / a
:class:`Preprocessor` subclass), but a multimodal dataset has one more thing to say: **where its media
actually is**. That is what groups the sections here, in order of what it costs:

1. The row already holds a usable reference -- a URL, an absolute path, or the bytes themselves. The
   only work is naming the column ``images`` / ``videos`` / ``audios``.
2. The row holds a path *relative* to an archive published alongside the dataset. The archive is
   fetched once in :meth:`Preprocessor.prepare_dataset`, each row's path is joined onto it, and rows
   whose file is absent are dropped -- a caption prompt with no image teaches nothing.

One recurring legacy defect to know about when reading these: a preprocessor that builds its output
row from scratch (``return {'messages': ...}``) silently drops the media column, because nothing
re-attaches it afterwards. Legacy shipped four such datasets, all tagged multi-modal while producing
text-only rows. Where this file departs from legacy to keep the media, the site says so.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from swift.utils import get_hf_endpoint, use_hf_hub

from .base import DatasetLoader, SubsetMeta, register_dataset
from ..mm_download import MediaDownloader
from ..preprocessor import Preprocessor


# ============================================================================================
# 1. The media is usable straight from the row -- nothing to download.
# ============================================================================================


@register_dataset
class RLAIFVLoader(DatasetLoader):
    dataset_type = 'RLAIF-V-Dataset'
    datasets = [('swift/RLAIF-V-Dataset', 'openbmb/RLAIF-V-Dataset')]
    columns = {'question': 'query', 'chosen': 'response', 'rejected': 'rejected_response'}
    tags = ['rlhf', 'dpo', 'multi-modal', 'en']


class FixedQueryPreprocessor(Preprocessor):
    """A dataset with no question column: the task is fixed, so the query is a constant.

    Common in caption and classification dumps, which ship an image and its label and leave the
    prompt to the trainer.
    """

    query: str = ''

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        row['query'] = self.query
        return super().preprocess(row)


class GPT4vPreprocessor(FixedQueryPreprocessor):
    columns = {'link': 'images', 'caption': 'response'}
    query = 'What is the caption of this image?'


@register_dataset
class GPT4vLoader(DatasetLoader):
    dataset_type = 'gpt4v-dataset'
    datasets = [('swift/gpt4v-dataset', 'laion/gpt4v-dataset')]
    preprocessor = GPT4vPreprocessor
    tags = ['en', 'caption', 'multi-modal', 'quality']
    huge_dataset = True


class GarbagePreprocessor(FixedQueryPreprocessor):
    # `label` rather than `response`: the rows are a class index, so this is a classification dataset
    # whose query happens to be phrased as a sentence.
    columns = {'category': 'label', 'image:FILE': 'images'}
    query = 'Task: Classify household waste.'


@register_dataset
class Garbage265Loader(DatasetLoader):
    dataset_type = 'garbage265'
    datasets = ['tany0699/garbage265']
    preprocessor = GarbagePreprocessor
    tags = ['cls', '🔥', 'multi-modal']


class COCO2014CaptionPreprocessor(FixedQueryPreprocessor):
    query = 'please describe the image.'

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        # Some rows hold several captions joined by `&&`; the first is the one written for the image,
        # the rest are alternatives, and a response cannot be all of them at once.
        caption = row['caption']
        row['response'] = caption.split('&&')[0] if '&&' in caption else caption
        return super().preprocess(row)


@register_dataset
class COCO2014CaptionLoader(DatasetLoader):
    dataset_type = 'coco-en-mini'
    datasets = ['modelscope/coco_2014_caption']
    subsets = [
        SubsetMeta('coco_2014_caption', name='train', split=['train']),
        SubsetMeta('coco_2014_caption', name='validation', split=['validation']),
    ]
    preprocessor = COCO2014CaptionPreprocessor
    tags = ['chat', 'multi-modal', 'vision', '🔥']


class AIShell1Preprocessor(FixedQueryPreprocessor):
    columns = {'Audio:FILE': 'audios'}
    query = '语音转文本'

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        # The transcript is stored space-separated per character, which is a tokenisation of the
        # label rather than the text to produce.
        row['response'] = row['Text:LABEL'].replace(' ', '')
        return super().preprocess(row)


@register_dataset
class AIShell1Loader(DatasetLoader):
    dataset_type = 'aishell1-zh'
    datasets = ['speech_asr/speech_asr_aishell1_trainsets']
    subsets = [
        SubsetMeta('default', name='train', split=['train']),
        SubsetMeta('default', name='validation', split=['validation']),
        SubsetMeta('default', name='test', split=['test']),
    ]
    preprocessor = AIShell1Preprocessor
    tags = ['chat', 'multi-modal', 'audio']


class SA1BCaptionPreprocessor(Preprocessor):
    """An image and one caption of it, turned into a caption request in Chinese.

    The dataset has no question column, and unlike :class:`FixedQueryPreprocessor` legacy varies the
    wording per row, so the prompt is drawn from a handful of phrasings. The draw uses this
    preprocessor's seeded generator: legacy drew from the global ``np.random``, which made the
    resulting dataset differ between two runs over the same input.
    """

    CAPTION_PROMPTS = ['图片中展示了什么', '讲述一下图片中内容', '告诉我里面有什么', '图片内容是啥']

    def caption(self, row: Dict[str, Any]) -> Optional[str]:
        """The caption this dataset's rows hide, wherever it keeps it."""
        raise NotImplementedError

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        caption = self.caption(row)
        if not caption:
            return None
        return super().preprocess({
            'query': str(self.random_state.choice(self.CAPTION_PROMPTS)),
            'response': caption,
            # Legacy returned only `messages`, dropping the image column it had just renamed -- so
            # these two datasets, both tagged multi-modal, yielded rows asking what is in a picture
            # that is not there. Carried through instead.
            'images': row['images'],
        })


class SA1BPairedCaptionPreprocessor(SA1BCaptionPreprocessor):
    columns = {'opensource_url': 'images'}

    def caption(self, row: Dict[str, Any]) -> Optional[str]:
        return row['global_caption']


class SA1BDenseCaptionPreprocessor(SA1BCaptionPreprocessor):
    columns = {'url': 'images'}

    def caption(self, row: Dict[str, Any]) -> Optional[str]:
        # `cap_seg` is a dict dumped into a string, holding the whole-image caption alongside
        # per-segment ones; only the whole-image caption answers the prompt.
        cap_seg = self.converter.parse_literal(row['cap_seg'])
        return cap_seg.get('global_caption') if isinstance(cap_seg, dict) else None


@register_dataset
class SA1BPairedCaptionLoader(DatasetLoader):
    dataset_type = 'sa1b-paired-caption'
    datasets = ['Tongyi-DataEngine/SA1B-Paired-Captions-Images']
    preprocessor = SA1BPairedCaptionPreprocessor
    tags = ['zh', 'multi-modal', 'vqa']


@register_dataset
class SA1BDenseCaptionLoader(DatasetLoader):
    dataset_type = 'sa1b-dense-caption'
    datasets = ['Tongyi-DataEngine/SA1B-Dense-Caption']
    preprocessor = SA1BDenseCaptionPreprocessor
    tags = ['zh', 'multi-modal', 'vqa']
    huge_dataset = True


class PixelProsePreprocessor(SA1BCaptionPreprocessor):
    CAPTION_PROMPTS = [
        'Give the description of this image.', 'Describe this picture', 'What is the proper title of this image?'
    ]
    columns = {'url': 'images'}

    def caption(self, row: Dict[str, Any]) -> Optional[str]:
        # The captions were generated by a VLM that prefixed each one with its own preamble.
        caption = row['vlm_caption']
        prefix = 'This image displays:'
        return caption[len(prefix):].strip() if caption.startswith(prefix) else caption


@register_dataset
class PixelProseLoader(DatasetLoader):
    dataset_type = 'pixelprose'
    datasets = [('swift/pixelprose', 'tomg-group-umd/pixelprose')]
    preprocessor = PixelProsePreprocessor
    split = ['train', 'cc12m', 'commonpool', 'redcaps']
    tags = ['caption', 'multi-modal', 'vision']
    huge_dataset = True


# ============================================================================================
# 2. The media lives in an archive next to the dataset: fetch it once, then resolve each row's
#    relative path against it.
# ============================================================================================


class ArchiveMediaPreprocessor(Preprocessor):
    """A dataset whose media has to be fetched before any row can be read.

    Holds only the fetching: :meth:`media_url` says what to get, and the archive lands in
    :attr:`media_dir` once for the whole dataset.
    """

    # Local directory name for the fetched archive. Shared by datasets that reference the same one.
    media_alias: str = ''
    # `'compressed'` for a single archive, `'sharded'` when :meth:`media_url` returns a list.
    media_file_type: str = 'compressed'

    def media_url(self):
        """The archive to fetch: a URL, or a list of them for a sharded resource."""
        raise NotImplementedError

    def prepare_dataset(self, dataset):
        self.media_dir = MediaDownloader.download(self.media_url(), self.media_alias, self.media_file_type)
        return dataset


class ArchiveImagePreprocessor(ArchiveMediaPreprocessor):
    """The common case: the row names an image file inside the fetched archive.

    :meth:`resolve` turns one row's reference into a local path and the base owns what every such
    dataset repeats -- drop the row when its file is not in the archive. That is not defensiveness:
    these archives are published separately from the rows and routinely cover only part of them.
    """

    def resolve(self, row: Dict[str, Any]) -> Optional[Any]:
        """The row's images as local paths, or ``None`` to drop the row."""
        raise NotImplementedError

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        images = self.resolve(row)
        if not images:
            return None
        row['images'] = images
        return super().preprocess(row)


class ShareGPT4oPreprocessor(ArchiveImagePreprocessor):
    media_alias = 'sharegpt_4o_images'

    def media_url(self) -> str:
        if use_hf_hub():
            return f'{get_hf_endpoint()}/datasets/OpenGVLab/ShareGPT-4o/blob/main/images.zip'
        return ('https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/ShareGPT-4o/repo?'
                'Revision=master&FilePath=images.zip')

    # The archive was packed from an absolute path on the uploader's cluster, so the images sit this
    # far down inside it.
    IMAGE_SUBDIR = ('mnt', 'petrelfs', 'wangwenhai', 'workspace_cef', '4o', 'image')

    def resolve(self, row: Dict[str, Any]) -> Optional[Any]:
        image = row.get('images')
        if not image:
            return None
        image = os.path.join(self.media_dir, *self.IMAGE_SUBDIR, image)
        return [image] if os.path.exists(image) else None


@register_dataset
class ShareGPT4oLoader(DatasetLoader):
    dataset_type = 'sharegpt-4o-image'
    datasets = [('AI-ModelScope/ShareGPT-4o', 'OpenGVLab/ShareGPT-4o')]
    subsets = ['image_caption']
    split = ['images']
    preprocessor = ShareGPT4oPreprocessor
    tags = ['vqa', 'multi-modal']


class MantisPreprocessor(ArchiveImagePreprocessor):
    """Mantis publishes one image archive per subset, so the subset is part of what to fetch."""

    subset: str = ''

    @classmethod
    def for_subset(cls, subset: str) -> type:
        """A subclass bound to one subset.

        The subset cannot be a constructor argument: a loader declares its preprocessor as a class,
        so there is no call site to pass it at.
        """
        return type(f'Mantis{subset}Preprocessor', (cls, ), {'subset': subset, 'media_alias': f'mantis_{subset}'})

    def media_url(self) -> str:
        if use_hf_hub():
            return (f'{get_hf_endpoint()}/datasets/TIGER-Lab/Mantis-Instruct/'
                    f'resolve/main/{self.subset}/train_images.zip')
        return (f'https://www.modelscope.cn/api/v1/datasets/swift/Mantis-Instruct/repo?Revision='
                f'master&FilePath={self.subset}/train_images.zip')

    def resolve(self, row: Dict[str, Any]) -> Optional[Any]:
        images = [os.path.join(self.media_dir, image['path']) for image in row['images']]
        # All or nothing: a multi-image row is a comparison ("spot the difference"), so a subset of
        # its images does not make a smaller version of the same question.
        return images if all(os.path.exists(image) for image in images) else None


@register_dataset
class MantisInstructLoader(DatasetLoader):
    dataset_type = 'mantis-instruct'
    datasets = ['swift/Mantis-Instruct']
    subsets = [
        SubsetMeta(subset, preprocessor=MantisPreprocessor.for_subset(subset), split=['train']) for subset in [
            'birds-to-words', 'chartqa', 'coinstruct', 'contrastive_caption', 'docvqa', 'dreamsim', 'dvqa', 'iconqa',
            'imagecode', 'llava_665k_multi', 'lrv_multi', 'multi_vqa', 'nextqa', 'nlvr2', 'spot-the-diff', 'star',
            'visual_story_telling'
        ]
    ]
    tags = ['chat', 'multi-modal', 'vision']


class LLaVADataPreprocessor(Preprocessor):
    """Rows pointing into the six public image collections LLaVA was assembled from.

    Each path starts with the name of the collection it came from, so the prefix says which archive
    to look in -- and what to strip before joining, which is not always the prefix itself: the two
    Visual Genome archives are named after the collection but their paths are written under ``vg/``.
    """

    # path prefix -> (archive to fetch, what to strip from the path before joining)
    MEDIA_PREFIXES = {
        'coco/': ('coco', 'coco/'),
        'gqa/': ('gqa', 'gqa/'),
        'ocr_vqa/': ('ocr_vqa', ''),
        'textvqa/': ('textvqa', 'textvqa/'),
        'VG_100K/': ('VG_100K', 'vg/'),
        'VG_100K_2/': ('VG_100K_2', 'vg/'),
    }

    def prepare_dataset(self, dataset):
        self.media_dirs = {archive: MediaDownloader.download(archive) for archive, _ in self.MEDIA_PREFIXES.values()}
        return dataset

    def resolve_image(self, image: str) -> str:
        for prefix, (archive, strip) in self.MEDIA_PREFIXES.items():
            if prefix in image:
                return os.path.join(self.media_dirs[archive], image.replace(strip, '') if strip else image)
        return image

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        if not row.get('images'):
            return None
        images = [self.resolve_image(image['path']) for image in row['images']]
        # Legacy returned `{'images': None}` for a row whose files are missing, which is not a row
        # with no images but a row with nothing at all -- no messages either. Dropped instead.
        if not all(os.path.exists(image) for image in images):
            return None
        row['images'] = images
        return super().preprocess(row)


@register_dataset
class LLaVADataLoader(DatasetLoader):
    dataset_type = 'llava-data-instruct'
    datasets = [('swift/llava-data', 'TIGER-Lab/llava-data')]
    subsets = ['llava_instruct']
    preprocessor = LLaVADataPreprocessor
    tags = ['sft', 'multi-modal', 'quality']


class EgoSchemaPreprocessor(ArchiveMediaPreprocessor):
    """Multiple-choice questions about first-person videos, answered by option letter.

    The videos are published as five archives. Legacy fetched them in a loop under one shared
    directory name, and since a fetch is skipped when that directory already exists, only the first
    archive ever landed -- every row belonging to the other four was then dropped as missing media.
    Fetched as one sharded resource here, which is what the five archives are.
    """

    media_alias = 'egoschema'
    media_file_type = 'sharded'
    OPTION_LETTERS = ('A', 'B', 'C', 'D', 'E')

    def media_url(self) -> list:
        if use_hf_hub():
            return [f'{get_hf_endpoint()}/datasets/lmms-lab/egoschema/resolve/main/videos_chunked_0{i}.zip'
                    for i in range(1, 6)]
        return [
            f'https://modelscope.cn/datasets/AI-ModelScope/egoschema/resolve/master/videos_chunked_0{i}.zip'
            for i in range(1, 6)
        ]

    def video_path(self, row: Dict[str, Any]) -> Optional[str]:
        path = os.path.join(self.media_dir, 'videos', f"{row['video_idx']}.mp4")
        return path if os.path.exists(path) else None

    def answer(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """The answer as the letter of the chosen option, which is how the question is phrased."""
        return {'response': self.OPTION_LETTERS[int(row['response'])]}

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        video = self.video_path(row)
        if not video:
            return None
        converted = {'query': row['query'] + '\n' + '\n'.join(row['option']), 'videos': [video]}
        converted.update(self.answer(row))
        return super().preprocess(converted)


class EgoSchemaClsPreprocessor(EgoSchemaPreprocessor):

    def answer(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # The classification variant keeps the option index, for a head that scores options rather
        # than generating the letter.
        return {'label': int(row['response'])}


@register_dataset
class EgoSchemaLoader(DatasetLoader):
    dataset_type = 'egoschema'
    datasets = [('AI-ModelScope/egoschema', 'lmms-lab/egoschema')]
    subsets = [
        SubsetMeta('Subset', name='default', preprocessor=EgoSchemaPreprocessor),
        SubsetMeta('Subset', name='cls', preprocessor=EgoSchemaClsPreprocessor),
    ]
    split = ['test']
    tags = ['chat', 'multi-modal', 'video']
