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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


# ============================================================================================
# 1 (continued). More datasets whose media needs no fetching -- what these have to say is about
#     their *text*: a question column that has to be picked from several, an answer assembled from
#     a rationale and a choice, or no question at all.
# ============================================================================================


@register_dataset
class OKVQALoader(DatasetLoader):
    dataset_type = 'OK-VQA_train'
    datasets = [('swift/OK-VQA_train', 'Multimodal-Fatima/OK-VQA_train')]
    tags = ['multi-modal', 'en', 'vqa', 'quality']


@register_dataset
class AOKVQALoader(DatasetLoader):
    dataset_type = 'A-OKVQA'
    datasets = [('swift/A-OKVQA', 'HuggingFaceM4/A-OKVQA')]
    # The answer is the *rationale*, not the one-word `answer` column: this dataset exists to teach
    # the reasoning, and several rationales are offered per question (the first is used).
    columns = {'rationales': 'response'}
    split = ['train', 'validation']
    tags = ['multi-modal', 'en', 'vqa', 'quality']


class QARolePreprocessor(Preprocessor):
    """A dialogue whose turns are labelled ``question`` / ``answer``.

    Pinning the two roles is not the same as adding them as aliases: a dataset that calls its user
    turn ``question`` is saying nothing about what ``user`` would mean there, so the whole family of
    user-ish role names is replaced rather than extended -- which is what ``user_role`` does.
    """

    converter_kwargs = {'user_role': 'question', 'assistant_role': 'answer'}


@register_dataset
class LnqaLoader(DatasetLoader):
    dataset_type = 'lnqa'
    datasets = [('swift/lnqa', 'vikhyatk/lnqa')]
    preprocessor = QARolePreprocessor
    split = ['train', 'validation']
    huge_dataset = True
    tags = ['multi-modal', 'en', 'ocr-vqa', 'quality']


class MideficsPreprocessor(QARolePreprocessor):
    """The same two roles, one level deeper: the column holds ``{'data': [...]}``."""

    converter_kwargs = {'user_role': 'question', 'assistant_role': 'answer', 'inner_key': 'data'}


@register_dataset
class MideficsLoader(DatasetLoader):
    dataset_type = 'MideficsDataset'
    datasets = [('swift/MideficsDataset', 'WinterSchool/MideficsDataset')]
    preprocessor = MideficsPreprocessor
    tags = ['medical', 'en', 'vqa']


class OcrvqaPreprocessor(Preprocessor):
    """Several questions asked of one book cover; one of them becomes the example.

    The pick is drawn from :attr:`Preprocessor.random_state`, which is seeded -- legacy used the
    global numpy generator, so which question each row contributed depended on everything else that
    had drawn from it, and two runs over the same dataset disagreed.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        questions, answers = row['questions'], row['answers']
        if not questions:
            return None
        index = self.random_state.choice(len(questions))
        return super().preprocess({'query': questions[index], 'response': answers[index], 'images': row['image']})


@register_dataset
class OcrvqaLoader(DatasetLoader):
    dataset_type = 'OCR-VQA'
    datasets = [('swift/OCR-VQA', 'howard-hou/OCR-VQA')]
    preprocessor = OcrvqaPreprocessor
    split = ['train', 'validation']
    tags = ['multi-modal', 'en', 'ocr-vqa']


class ScienceQAPreprocessor(Preprocessor):
    """A multiple-choice science question answered by working through it first.

    The answer column is an *index* into ``choices``, and the explanation is a separate column, so the
    response has to be assembled: the reasoning, then the option it leads to.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        answer = row['choices'][row['answer']]
        return super().preprocess({
            'query': row['question'],
            'response': f"{row['solution']}\nSo the final answer is: {answer}",
            'images': row['image'],
        })


@register_dataset
class ScienceQALoader(DatasetLoader):
    dataset_type = 'ScienceQA'
    datasets = [('swift/ScienceQA', 'derek-thomas/ScienceQA')]
    preprocessor = ScienceQAPreprocessor
    split = ['train', 'validation']
    tags = ['multi-modal', 'science', 'vqa', 'quality']


class LatexOcrPreprocessor(FixedQueryPreprocessor):
    query = 'Using LaTeX to perform OCR on the image.'


@register_dataset
class LatexOcrLoader(DatasetLoader):
    dataset_type = 'LaTeX_OCR'
    datasets = [('AI-ModelScope/LaTeX_OCR', 'linxy/LaTeX_OCR')]
    subsets = ['default', 'human_handwrite', 'human_handwrite_print', 'synthetic_handwrite', 'small']
    preprocessor = LatexOcrPreprocessor
    split = ['train', 'validation', 'test']
    tags = ['chat', 'ocr', 'multi-modal', 'vision']


class CaptchaPreprocessor(FixedQueryPreprocessor):
    columns = {'solution': 'response'}
    query = 'recognize the content.'


@register_dataset
class CaptchaImagesLoader(DatasetLoader):
    dataset_type = 'captcha-images'
    datasets = [('AI-ModelScope/captcha-images', None)]
    preprocessor = CaptchaPreprocessor
    split = ['train', 'validation']
    tags = ['chat', 'multi-modal', 'vision']


class ClevrPreprocessor(Preprocessor):
    """Counting questions for GRPO: the question is extended to demand a specific answer format,
    because the reward function parses the tags rather than the prose.

    The answer is also kept under ``solution``, which is where that reward function reads the reference
    from. Legacy kept it by a different route -- a ``__#solution`` column added before its ``map`` and
    stripped inside it, so the name survived the rename to ``response`` -- which dev does not have.
    Written onto the converted row, since setting it beforehand would make the alias pass fight over
    which column is the answer.
    """

    instruction = (' Output the thinking process in <think> </think> and '
                   'final answer (number) in <answer> </answer> tags.')

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        row['query'] = row.get('query', '') + self.instruction
        response = row.get('response')
        converted = super().preprocess(row)
        if converted is None:
            return None
        converted['solution'] = response
        return converted


@register_dataset
class ClevrCogenLoader(DatasetLoader):
    dataset_type = 'clevr_cogen_a_train'
    datasets = [('AI-ModelScope/clevr_cogen_a_train', 'leonardPKU/clevr_cogen_a_train')]
    preprocessor = ClevrPreprocessor
    tags = ['qa', 'math', 'vision', 'grpo']


class Voc2007MultilabelPreprocessor(Preprocessor):
    """Multi-label classification: the label is the *set* of classes present, as their indices.

    The row stores a one-hot vector, so the label is where it is set -- and stays a list of ints
    rather than being written out, since there is no single answer to spell.
    """

    columns = {'webp': 'images'}
    CLASS_NAMES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                   'tvmonitor')

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        row['query'] = f'多标签分类，类别包括：{list(self.CLASS_NAMES)}'
        row['label'] = [index for index, present in enumerate(row['npy']) if present == 1]
        return super().preprocess(row)


@register_dataset
class Voc2007MultilabelLoader(DatasetLoader):
    dataset_type = 'wds_voc2007_multilabel'
    datasets = ['clip-benchmark/wds_voc2007_multilabel']
    preprocessor = Voc2007MultilabelPreprocessor
    tags = ['multilabel', 'multi-modal']


class Geometry3KPreprocessor(Preprocessor):
    """A geometry problem whose answer is kept a second time under ``solution``.

    That is where a GRPO reward function reads the reference from, while ``messages`` is what training
    compares against. Written onto the row conversion *returns*: ``solution`` is an alias of
    ``response``, so setting it beforehand would make the alias pass fight over which is the answer.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        response = self.standardise(row).get('response')
        converted = super().preprocess(row)
        if converted is None:
            return None
        converted['solution'] = response
        return converted


@register_dataset
class Geometry3KLoader(DatasetLoader):
    dataset_type = 'geometry3k'
    datasets = [(None, 'hiyouga/geometry3k')]
    subsets = [SubsetMeta('default', name=name, split=[name]) for name in ('train', 'validation', 'test')]
    preprocessor = Geometry3KPreprocessor
    tags = ['multi-modal', 'en', 'math']


class LLaVAMixSFTPreprocessor(Preprocessor):
    """A dialogue whose turns hold *content parts* rather than a string.

    Each turn is a list of ``{'type': 'text'|'image', ...}`` fragments; flattening them means keeping
    the order, with an image fragment becoming the ``<image>`` marker that stands in for it.
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        messages = []
        for message in row['messages']:
            content = ''
            for part in message['content']:
                content += part['text'] if part['type'] == 'text' else '<image>'
            messages.append({'role': message['role'], 'content': content})
        row['messages'] = messages
        return super().preprocess(row)


@register_dataset
class LLaVAMixSFTLoader(DatasetLoader):
    dataset_type = 'llava-instruct-mix-vsft'
    datasets = [('swift/llava-instruct-mix-vsft', 'HuggingFaceH4/llava-instruct-mix-vsft')]
    preprocessor = LLaVAMixSFTPreprocessor
    split = ['test']
    tags = ['multi-modal', 'en', 'vqa', 'quality']


class TextCapsPreprocessor(FixedQueryPreprocessor):
    """Captions of images containing text. Offered as three tasks over the same rows.

    The image is referenced by a local path that may not be there -- this dataset is distributed as
    rows plus a separately downloaded image set -- so every view starts by checking the file.
    """

    columns = {'reference_strs': 'response'}
    query = 'What is the caption of this image?'

    @staticmethod
    def image_path(row: Dict[str, Any]) -> Optional[str]:
        """The row's image as a local path, or ``None`` when there is not one to check."""
        images = row.get('images')
        if isinstance(images, list):
            images = images[0] if images else None
        path = images.get('path') if isinstance(images, dict) else images
        return path if isinstance(path, str) and os.path.exists(path) else None

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.image_path(self.standardise(row)) is None:
            return None
        return super().preprocess(row)


class TextCapsEmbPreprocessor(TextCapsPreprocessor):
    """The embedding view: image and caption are the pair to be pulled together, so there is no answer
    turn -- the user turn is the bare ``<image>`` marker and the caption is the positive."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        if self.image_path(row) is None:
            return None
        return {
            'messages': [{
                'role': 'user',
                'content': '<image>'
            }],
            'positive_messages': [[{
                'role': 'user',
                'content': row['response'][0]
            }]],
            'images': row['images'],
        }


class TextCapsReRankPreprocessor(TextCapsEmbPreprocessor):
    """The reranking view: the same positive, plus captions of *other* images as negatives.

    The negatives have to come from somewhere, and this dataset ships none -- so they are sampled from
    the other rows' captions, which is what makes them hard: they describe real images of the same
    kind. Captions belonging to this row are excluded, since a correct caption is not a negative.

    The pool is built once in :meth:`prepare_dataset`, which is also what bounds the cost: reading one
    column of the dataset, not one pass per row.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._pool: List[str] = []
        # Matches the collator's own cap, so a batch is not built and then truncated.
        self.negatives_per_sample = int(os.environ.get('MAX_NEGATIVE_SAMPLES', 7))

    def prepare_dataset(self, dataset):
        column = self.columns.get('reference_strs', 'reference_strs')
        source = column if column in dataset.features else 'response'
        pool: List[str] = []
        for captions in (dataset[source] if source in dataset.features else []):
            captions = captions if isinstance(captions, (list, tuple)) else [captions]
            pool += [caption for caption in captions if isinstance(caption, str)]
        self._pool = pool
        return dataset

    def sample_negatives(self, own: Sequence[str]) -> List[str]:
        negatives: List[str] = []
        if not self._pool:
            return negatives
        for index in self.random_state.permutation(len(self._pool)):
            caption = self._pool[index]
            if caption not in own:
                negatives.append(caption)
            if len(negatives) >= self.negatives_per_sample:
                break
        return negatives

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        converted = super().preprocess(row)
        if converted is None:
            return None
        captions = self.standardise(row)['response']
        converted['negative_messages'] = [[{
            'role': 'user',
            'content': caption
        }] for caption in self.sample_negatives(captions)]
        return converted


@register_dataset
class TextCapsLoader(DatasetLoader):
    dataset_type = 'TextCaps'
    datasets = [('swift/TextCaps', 'HuggingFaceM4/TextCaps')]
    subsets = [
        SubsetMeta('default', preprocessor=TextCapsPreprocessor),
        SubsetMeta('default', name='emb', preprocessor=TextCapsEmbPreprocessor),
        SubsetMeta('default', name='rerank', preprocessor=TextCapsReRankPreprocessor),
    ]
    split = ['train', 'validation']
    huge_dataset = True
    tags = ['multi-modal', 'en', 'caption', 'quality']


# ============================================================================================
# 2 (continued). More archive-backed datasets.
# ============================================================================================


class ShareGPT4VPreprocessor(Preprocessor):
    """Captions written for images drawn from seven public collections.

    Like :class:`LLaVADataPreprocessor`, a path's leading component says which archive holds it -- but
    here the rewrite is not always a strip: three collections are named after themselves while their
    paths are written under ``data/``, so each prefix carries the replacement to apply.

    Which archives are needed depends on the subset, and only the needed ones are fetched: the ``-PT``
    subset draws from three of the seven, and each is a multi-gigabyte download.
    """

    # path prefix -> (archive, text to replace, replacement)
    MEDIA_PREFIXES = {
        'coco/': ('coco', 'coco/', ''),
        'sam/': ('sam', 'sam/images/', ''),
        'llava/': ('llava', 'llava/llava_pretrain/images/', ''),
        'wikiart/': ('wikiart', 'wikiart/images/', 'data/wikiart/images/'),
        'share_textvqa/': ('share_textvqa', 'share_textvqa/images/', 'data/share_textvqa/images/'),
        'web-celebrity/': ('web-celebrity', 'web-celebrity/images/', 'data/web-celebrity/images/'),
        'web-landmark/': ('web-landmark', 'web-landmark/images/', 'data/web-landmark/images/'),
    }
    archives: Sequence[str] = ('coco', 'sam', 'llava', 'wikiart', 'share_textvqa', 'web-celebrity', 'web-landmark')

    def prepare_dataset(self, dataset):
        self.media_dirs = {archive: MediaDownloader.download(archive) for archive in self.archives}
        return dataset

    def resolve_image(self, image: str) -> Optional[str]:
        for prefix, (archive, old, new) in self.MEDIA_PREFIXES.items():
            if prefix not in image:
                continue
            # A row pointing at a collection this subset does not fetch is dropped rather than
            # resolved against a directory that was never downloaded.
            if archive not in self.media_dirs:
                return None
            return os.path.join(self.media_dirs[archive], image.replace(old, new))
        return None

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        image = row.get('images')
        if not isinstance(image, str):
            return None
        image = self.resolve_image(image)
        if not image or not os.path.exists(image):
            return None
        row['images'] = [image]
        return super().preprocess(row)


class ShareGPT4VPTPreprocessor(ShareGPT4VPreprocessor):
    archives = ('coco', 'sam', 'llava')


@register_dataset
class ShareGPT4VLoader(DatasetLoader):
    dataset_type = 'ShareGPT4V'
    datasets = [('AI-ModelScope/ShareGPT4V', None)]
    subsets = [
        SubsetMeta('ShareGPT4V', preprocessor=ShareGPT4VPreprocessor),
        SubsetMeta('ShareGPT4V-PT', preprocessor=ShareGPT4VPTPreprocessor),
    ]
    huge_dataset = True
    tags = ['chat', 'multi-modal', 'vision']


class LLaVAInstructPreprocessor(LLaVADataPreprocessor):
    """The same six collections as :class:`LLaVADataPreprocessor`, one image per row as a bare path."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        image = row.get('images')
        if not isinstance(image, str):
            return None
        image = self.resolve_image(image)
        if not os.path.exists(image):
            return None
        row['images'] = [image]
        # Skips the parent's own body, which expects the column to hold image dicts.
        return Preprocessor.preprocess(self, row)


@register_dataset
class LLaVAInstruct150KLoader(DatasetLoader):
    dataset_type = 'LLaVA-Instruct-150K'
    datasets = [('AI-ModelScope/LLaVA-Instruct-150K', None)]
    ms_revision = 'd5db3806e395c60496630a206c336932e85a2d00'
    preprocessor = LLaVAInstructPreprocessor
    tags = ['chat', 'multi-modal', 'vision']


class LLaVAPretrainPreprocessor(ArchiveImagePreprocessor):
    """Short captions for the LLaVA pre-training images, published as one archive beside the rows.

    Legacy returned ``{'images': file_path}`` here -- a row with no ``messages`` at all, so the caption
    it was fetched for was thrown away. The row is kept whole instead.
    """

    media_alias = 'llava_pretrain'

    def media_url(self) -> str:
        if use_hf_hub():
            return f'{get_hf_endpoint()}/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip'
        return ('https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/LLaVA-Pretrain/repo?'
                'Revision=master&FilePath=images.zip')

    def resolve(self, row: Dict[str, Any]) -> Optional[Any]:
        image = row.get('images')
        if not isinstance(image, str):
            return None
        image = os.path.join(self.media_dir, image)
        return [image] if os.path.exists(image) else None


@register_dataset
class LLaVAPretrainLoader(DatasetLoader):
    dataset_type = 'LLaVA-Pretrain'
    datasets = [('AI-ModelScope/LLaVA-Pretrain', 'liuhaotian/LLaVA-Pretrain')]
    ms_revision = 'e3a3f0bfaad05e90e46745152a32bf944e0f4a63'
    preprocessor = LLaVAPretrainPreprocessor
    huge_dataset = True
    tags = ['chat', 'multi-modal', 'quality']


class GQAPreprocessor(ArchiveImagePreprocessor):
    """Compositional questions about Visual Genome scenes; the answer is the full sentence form.

    Legacy wrote ``if os.path.join(...)`` where it meant ``os.path.exists``: a joined path is a
    non-empty string, so the guard was always true and rows whose image was absent were kept.
    """

    media_alias = 'gqa'

    def media_url(self) -> str:
        return MediaDownloader.get_url('gqa')

    def resolve(self, row: Dict[str, Any]) -> Optional[Any]:
        image = os.path.join(self.media_dir, 'images', row['imageId'] + '.jpg')
        return [image] if os.path.exists(image) else None

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        images = self.resolve(row)
        if not images:
            return None
        return Preprocessor.preprocess(self, {
            'query': row['question'],
            'response': row['fullAnswer'],
            'images': images,
        })


@register_dataset
class GQALoader(DatasetLoader):
    dataset_type = 'GQA'
    datasets = [(None, 'lmms-lab/GQA')]
    preprocessor = GQAPreprocessor
    split = ['train_all_instructions']
    huge_dataset = True
    tags = ['multi-modal', 'en', 'vqa', 'quality']


class Qwen3TTSPreprocessor(ArchiveMediaPreprocessor):
    """Speech synthesis: the text is what to say, and a reference clip fixes the voice to say it in.

    There is no question -- the assistant turn is the whole example -- and two audio columns, the
    target and the reference, both named relative to the fetched archive.
    """

    columns = {'ref_audio': 'ref_audios'}
    media_alias = 'qwen3_tts_furina'

    def media_url(self) -> str:
        return 'https://modelscope.cn/datasets/qsdong/Qwen3-1.7-TTS-SFT-Furina/resolve/master/Furina.zip'

    def resolve_audio(self, audio: Any) -> Any:
        if isinstance(audio, str) and not os.path.isabs(audio):
            return os.path.join(self.media_dir, audio)
        return audio

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        row['audios'] = self.resolve_audio(row.get('audios'))
        ref_audios = self.resolve_audio(row.get('ref_audios'))
        if isinstance(ref_audios, str):
            ref_audios = [ref_audios]
        row['ref_audios'] = ref_audios
        return super().preprocess(row)


@register_dataset
class Qwen3TTSFurinaLoader(DatasetLoader):
    dataset_type = 'Qwen3-1.7-TTS-SFT-Furina'
    datasets = [('qsdong/Qwen3-1.7-TTS-SFT-Furina', None)]
    preprocessor = Qwen3TTSPreprocessor
    tags = ['chat', 'multi-modal', 'audio', 'tts']


# ============================================================================================
# 3. Grounding: rows carrying boxes, where the prompt itself is part of the task.
#
# A box dataset can be read as two opposite tasks -- name a thing and be told where it is
# (`grounding`), or be given a place and say what is there (`caption`) -- so the phrasing is drawn
# from a table per row rather than fixed, to keep the model from learning one wording.
# ============================================================================================


class GroundingPreprocessor(Preprocessor):
    """Rows with boxes, phrased as one of the two grounding tasks.

    The wording is drawn per row, and from :attr:`Preprocessor.random_state` rather than legacy's
    global numpy generator -- so a rerun of the same dataset asks the same questions.

    Boxes travel in the standard ``objects`` column, which the template turns into whatever coordinate
    notation the model was trained on; ``bbox_type`` says what the numbers currently mean.
    """

    task_type: str = 'grounding'
    # Chinese is a fifth of the prompts, matching legacy's mix: enough that the ability transfers,
    # not so much that it dominates a dataset whose captions are English.
    LANGUAGE_WEIGHTS = (0.8, 0.2)
    PROMPTS = {
        'grounding': {
            'en': [('<ref-object>', '<bbox>'), ('The positions of <ref-object> is', '<bbox>'),
                   ('Find the positions of <ref-object>', '<bbox>'), ('Where is <ref-object>', '<bbox>'),
                   ('Find <ref-object>', '<bbox>'), ('Show me <ref-object>', '<bbox>'),
                   ('Detect <ref-object>', '<bbox>'), ('Locate <ref-object>', '<bbox>'),
                   ('Tell me the location of <ref-object>', '<bbox>'),
                   ('Give the location of <ref-object>', '<bbox>'),
                   ('Provide the bounding box coordinate of <ref-object>', '<bbox>')],
            'zh': [('<ref-object>', '<bbox>'), ('<ref-object>的位置在图片中', '<bbox>'), ('<ref-object>在图片中', '<bbox>'),
                   ('<ref-object>在', '<bbox>'), ('找到<ref-object>的位置', '<bbox>'), ('<ref-object>在哪里', '<bbox>'),
                   ('提供<ref-object>的坐标位置', '<bbox>')],
        },
        'caption': {
            'en': [('<bbox>', '<ref-object>'), ('The object at position <bbox>', '<ref-object>'),
                   ('This <bbox> is', '<ref-object>'), ('What is the object at <bbox>', '<ref-object>'),
                   ('Describe <bbox>', '<ref-object>'), ('<bbox> is', '<ref-object>'),
                   ('The bounding box coordinate <bbox> contains', '<ref-object>')],
            'zh': [('<bbox>', '<ref-object>'), ('<bbox>是什么', '<ref-object>'), ('<bbox>的位置包含', '<ref-object>'),
                   ('描述<bbox>', '<ref-object>'), ('<bbox>中是', '<ref-object>'), ('坐标<bbox>描述了什么', '<ref-object>'),
                   ('描述<bbox>中的事物', '<ref-object>')],
        },
    }

    def grounding_prompt(self) -> Tuple[str, str]:
        """One ``(query, response)`` pair for this task, drawn from the table."""
        language = self.random_state.choice(['en', 'zh'], p=list(self.LANGUAGE_WEIGHTS))
        prompts = self.PROMPTS[self.task_type][language]
        return prompts[self.random_state.choice(len(prompts))]

    @classmethod
    def for_task(cls, task_type: str, **attrs) -> type:
        """A subclass bound to one task, since a loader declares a preprocessor class not an instance."""
        return type(f'{cls.__name__}_{task_type}', (cls, ), {'task_type': task_type, **attrs})


class RefCOCOPreprocessor(GroundingPreprocessor):
    """One region of a COCO image with the phrase that refers to it.

    The images are the COCO 2014 train set, published as one archive; the rows name paths under
    ``coco/train2014`` while the archive holds ``train2014`` directly.
    """

    COCO2014_URL = ('https://www.modelscope.cn/api/v1/datasets/we_dont_produce_water/'
                    'coco_res/repo?Revision=master&FilePath=coco_2014.zip')

    def prepare_dataset(self, dataset):
        self.media_dir = MediaDownloader.download(self.COCO2014_URL, 'coco2014')
        return dataset

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        image = os.path.join(self.media_dir, row['image_path'].replace('coco/train2014', 'train2014'))
        if not os.path.exists(image):
            return None
        query, response = self.grounding_prompt()
        # Rounded to whole pixels: the source stores them as floats, and a fractional pixel is not a
        # meaningful coordinate to ask a model to produce.
        bbox = [round(float(value)) for value in row['bbox']]
        return super().preprocess({
            'query': query,
            'response': response,
            'images': [image],
            'objects': {
                'ref': [row['captions'][0]],
                'bbox': [bbox]
            },
        })


@register_dataset
class RefCOCOLoader(DatasetLoader):
    dataset_type = 'refcoco'
    datasets = [('swift/refcoco', 'jxu124/refcoco')]
    subsets = [
        SubsetMeta('default', name='caption', preprocessor=RefCOCOPreprocessor.for_task('caption')),
        SubsetMeta('default', name='grounding', preprocessor=RefCOCOPreprocessor.for_task('grounding')),
    ]
    split = ['train', 'validation']
    tags = ['multi-modal', 'en', 'grounding']


@register_dataset
class RefCOCOgLoader(RefCOCOLoader):
    dataset_type = 'refcocog'
    datasets = [('swift/refcocog', 'jxu124/refcocog')]


class GritPreprocessor(GroundingPreprocessor):
    """Web captions with spans of the caption annotated with the box they refer to.

    Each annotation is ``(start, end, x1, y1, x2, y2, confidence)``: a slice of the caption text plus
    where in the image it points. Overlapping spans are dropped rather than resolved -- one word
    cannot refer to two boxes at once, and picking either would invent an annotation.

    The ``vqa`` task asks for the caption itself, so it is the one view here that does not draw a
    prompt from the grounding table.
    """

    columns = {'url': 'images'}

    @staticmethod
    def has_overlap(spans: List[Any]) -> bool:
        return any(spans[i][0] < spans[i - 1][1] for i in range(1, len(spans)))

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        caption = row['caption']
        annotations = row['ref_exps']
        if not annotations:
            return None
        objects = {'ref': [], 'bbox': [], 'bbox_type': 'norm1'}
        spans = []
        for annotation in annotations:
            start, end = annotation[0], annotation[1]
            spans.append([start, end])
            objects['ref'].append(caption[int(start):int(end)])
            objects['bbox'].append(annotation[2:6])
        spans.sort(key=lambda span: (span[0], span[1]))
        if self.has_overlap(spans):
            return None

        if self.task_type in ('grounding', 'caption'):
            query, response = self.grounding_prompt()
        else:
            query, response = 'what is the proper caption of this image?', caption
        return super().preprocess({
            'query': query,
            'response': response,
            'images': row['images'],
            'objects': objects,
        })


@register_dataset
class GritLoader(DatasetLoader):
    dataset_type = 'GRIT'
    datasets = [('swift/GRIT', 'zzliang/GRIT')]
    subsets = [
        SubsetMeta('default', name=task, preprocessor=GritPreprocessor.for_task(task))
        for task in ('caption', 'grounding', 'vqa')
    ]
    huge_dataset = True
    tags = ['multi-modal', 'en', 'caption-grounding', 'vqa', 'quality']


class CocoDetectionPreprocessor(Preprocessor):
    """Object detection as generation: one ``<ref-object><bbox>`` line per object in the image.

    The row stores class *indices*, so the names have to be looked up -- and they go in ``objects.ref``
    rather than in the answer text, because the answer is the markers and the template fills them in.
    """

    query = 'Task: Object Detection'
    CATEGORIES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                  'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                  'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                  'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                  'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                  'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                  'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        objects = row['objects']
        # The indices are consumed here, not passed on: `ref` is what the template reads.
        objects['ref'] = [self.CATEGORIES[category] for category in objects.pop('category')]
        row['query'] = self.query
        row['response'] = '\n'.join(['<ref-object><bbox>'] * len(objects['ref']))
        return super().preprocess(row)


@register_dataset
class CocoDetectionLoader(DatasetLoader):
    dataset_type = 'coco'
    datasets = [('AI-ModelScope/coco', 'detection-datasets/coco')]
    preprocessor = CocoDetectionPreprocessor
    huge_dataset = True
    tags = ['multi-modal', 'en', 'vqa', 'quality']


@register_dataset
class M3ITLoader(DatasetLoader):
    """49 task-specific dumps sharing one column layout, so the whole family is a rename."""

    dataset_type = 'M3IT'
    datasets = [('AI-ModelScope/M3IT', None)]
    subsets = [
        'coco', 'vqa-v2', 'shapes', 'shapes-rephrased', 'coco-goi-rephrased', 'snli-ve', 'snli-ve-rephrased', 'okvqa',
        'a-okvqa', 'viquae', 'textcap', 'docvqa', 'science-qa', 'imagenet', 'imagenet-open-ended', 'imagenet-rephrased',
        'coco-goi', 'clevr', 'clevr-rephrased', 'nlvr', 'coco-itm', 'coco-itm-rephrased', 'vsr', 'vsr-rephrased',
        'mocheg', 'mocheg-rephrased', 'coco-text', 'fm-iqa', 'activitynet-qa', 'msrvtt', 'ss', 'coco-cn', 'refcoco',
        'refcoco-rephrased', 'multi30k', 'image-paragraph-captioning', 'visual-dialog', 'visual-dialog-rephrased',
        'iqa', 'vcr', 'visual-mrc', 'ivqa', 'msrvtt-qa', 'msvd-qa', 'gqa', 'text-vqa', 'ocr-vqa', 'st-vqa',
        'flickr8k-cn'
    ]
    # The task description is a *system* prompt here: it is constant per subset and says what to do,
    # not what this row asks.
    columns = {
        'instruction': 'system',
        'inputs': 'query',
        'image_base64_str': 'images',
        'outputs': 'response',
    }
    huge_dataset = True
    tags = ['chat', 'multi-modal', 'vision']


# ============================================================================================
# 4. Video. The same archive story as images, with one difference worth its own section: a video
#    corpus is published in *many* parts, so what to fetch is a list and how it is packaged (one
#    archive per part, or loose files) varies per dataset.
# ============================================================================================


class VideoChatGPTPreprocessor(ArchiveMediaPreprocessor):
    """Questions about test videos, up to three per row, of which the first asked one is used.

    The columns are ``query`` / ``question_1`` / ``question_2``, and an unasked one holds the *string*
    ``'None'`` rather than a null -- so it has to be compared as text.

    The set of available videos is read once here. Legacy listed the directory inside ``preprocess``,
    i.e. once per row.
    """

    media_alias = 'video_chatgpt'
    QUERY_KEYS = ('query', 'question_1', 'question_2')

    def media_url(self) -> str:
        if use_hf_hub():
            return f'{get_hf_endpoint()}/datasets/lmms-lab/VideoChatGPT/resolve/main/videos.zip'
        return 'https://modelscope.cn/datasets/swift/VideoChatGPT/resolve/master/videos.zip'

    def prepare_dataset(self, dataset):
        dataset = super().prepare_dataset(dataset)
        self.video_dir = os.path.join(self.media_dir, 'Test_Videos')
        self.video_names = {
            name[:-len('.mp4')]
            for name in os.listdir(self.video_dir) if name.endswith('.mp4')
        }
        return dataset

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        if row['video_name'] not in self.video_names:
            return None
        for key in self.QUERY_KEYS:
            query = row.get(key)
            if query is None or query == 'None':
                continue
            row['query'] = query
            row['videos'] = [os.path.join(self.video_dir, f"{row['video_name']}.mp4")]
            return super().preprocess(row)
        # Every question column was empty, so there is nothing being asked about the video.
        return None


@register_dataset
class VideoChatGPTLoader(DatasetLoader):
    dataset_type = 'VideoChatGPT'
    datasets = [('swift/VideoChatGPT', 'lmms-lab/VideoChatGPT')]
    subsets = ['Generic', 'Temporal', 'Consistency']
    preprocessor = VideoChatGPTPreprocessor
    split = ['test']
    tags = ['chat', 'multi-modal', 'video', '🔥']


class MovieChat1KPreprocessor(ArchiveMediaPreprocessor):
    """Long-video question answering. The videos are published as loose ``.mp4`` files, not an archive.

    The file names are not derivable from the rows -- they follow several unrelated series -- so the
    list is spelled out. Fetched as one ``'files'`` resource, which is what fixes legacy's outcome of
    downloading the first video and dropping every row belonging to the other ~150.
    """

    media_alias = 'moviechat_1k_test'
    media_file_type = 'files'
    # (prefix, count): `AWA-1.mp4` ... `AWA-9.mp4`, and a bare-numbered series either side of a gap.
    VIDEO_SERIES = (('', range(1, 10)), ('', range(201, 240)), ('AWA-', range(1, 10)), ('AWB-', range(1, 16)),
                    ('AWC-', range(1, 11)), ('AWD-', range(1, 8)), ('AWE-', range(1, 7)), ('AWG-', range(1, 12)),
                    ('AWH-', range(1, 8)), ('BWA-', range(1, 7)), ('BWB-', range(1, 7)), ('BWD-', range(1, 6)),
                    ('BWE-', range(1, 6)), ('BWG-', range(1, 6)), ('BWH-', range(1, 6)), ('TFS-', range(1, 13)),
                    ('UWA-', range(1, 5)))
    # Out of sequence in its own series, hence listed apart rather than by widening the range.
    EXTRA_VIDEOS = ('UWA-6.mp4', )

    def media_url(self) -> List[str]:
        names = [f'{prefix}{index}.mp4' for prefix, indices in self.VIDEO_SERIES for index in indices]
        names += list(self.EXTRA_VIDEOS)
        if use_hf_hub():
            prefix = f'{get_hf_endpoint()}/datasets/Enxin/MovieChat-1K-test/resolve/main/videos/'
        else:
            prefix = 'https://modelscope.cn/datasets/AI-ModelScope/MovieChat-1K-test/resolve/master/videos/'
        return [prefix + name for name in names]

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        video = os.path.join(self.media_dir, row['info']['video_path'])
        if not os.path.exists(video):
            return None
        # `global` holds the questions asked of the whole video, as opposed to the per-clip ones; the
        # first is used.
        question = row['global'][0]
        return super().preprocess({
            'query': question['question'],
            'response': question['answer'],
            'videos': [video],
        })


@register_dataset
class MovieChat1KLoader(DatasetLoader):
    dataset_type = 'MovieChat-1K-test'
    datasets = [('AI-ModelScope/MovieChat-1K-test', 'Enxin/MovieChat-1K-test')]
    preprocessor = MovieChat1KPreprocessor
    tags = ['chat', 'multi-modal', 'video']


class LLaVAVideo178KPreprocessor(ArchiveMediaPreprocessor):
    """Video instruction data, one sharded archive set per duration bucket.

    Which bucket is being loaded decides what to fetch, and the buckets are large enough that fetching
    the others would be a serious waste -- so the subset is bound into the preprocessor class.
    """

    media_file_type = 'sharded'
    # subset -> how many `_videos_{i}.tar.gz` parts it was split into.
    SHARD_COUNTS = {
        '0_30_s_academic_v0_1': 8,
        '0_30_s_youtube_v0_1': 19,
        '1_2_m_academic_v0_1': 14,
        '1_2_m_youtube_v0_1': 50,
        '2_3_m_academic_v0_1': 18,
        '2_3_m_youtube_v0_1': 98,
        '30_60_s_academic_v0_1': 10,
        '30_60_s_youtube_v0_1': 13,
    }
    subset: str = ''

    @classmethod
    def for_subset(cls, subset: str) -> type:
        return type(f'LLaVAVideo178K_{subset}_Preprocessor', (cls, ), {
            'subset': subset,
            'media_alias': f'llava_video_178k_{subset}'
        })

    def media_url(self) -> List[str]:
        if use_hf_hub():
            prefix = f'{get_hf_endpoint()}/datasets/lmms-lab/LLaVA-Video-178K/resolve/main/'
        else:
            prefix = 'https://www.modelscope.cn/datasets/lmms-lab/LLaVA-Video-178K/resolve/master/'
        return [
            f'{prefix}{self.subset}/{self.subset}_videos_{index}.tar.gz'
            for index in range(1, self.SHARD_COUNTS[self.subset] + 1)
        ]

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = self.standardise(row)
        video = os.path.join(self.media_dir, str(row['videos']))
        if not os.path.exists(video):
            return None
        return super().preprocess({'messages': row['messages'], 'videos': [video]})


@register_dataset
class LLaVAVideo178KLoader(DatasetLoader):
    dataset_type = 'LLaVA-Video-178K'
    datasets = [(None, 'lmms-lab/LLaVA-Video-178K')]
    # The three question styles are published as *splits*, not subsets, so every bucket loads all
    # three.
    subsets = [
        SubsetMeta(
            subset,
            split=['caption', 'open_ended', 'multi_choice'],
            preprocessor=LLaVAVideo178KPreprocessor.for_subset(subset))
        for subset in LLaVAVideo178KPreprocessor.SHARD_COUNTS
    ]
    tags = ['chat', 'multi-modal', 'video']


# ============================================================================================
# 5. One dataset whose rows are not examples: consecutive rows are *steps of one episode*, and the
#    example is the episode. Regrouping happens in `prepare_dataset`, which is the hook that may
#    return a different dataset -- there is no per-row transform that could do it.
# ============================================================================================


class Mind2WebPreprocessor(Preprocessor):
    """Web-browsing episodes: each source row is one action, and a full episode is the training example.

    Rows arrive in order and ``target_action_index == '0'`` marks the start of a new episode, so the
    regrouping is a fold over the dataset -- which no row transform can express, hence the reshape in
    :meth:`prepare_dataset` and a pass-through row transform afterwards.

    The screenshot is carried inline as a PIL image and is written to disk here, because what the rest
    of the pipeline passes around is a path.
    """

    TOOLS = [{
        'function': {
            'name': 'CLICK',
            'desc': 'Choose and click an element in the web page',
            'parameter': [{
                'element': 'string, the element in the web page to click'
            }]
        }
    }, {
        'function': {
            'name': 'TYPE',
            'desc': 'Input some text into a web element like <input> or <textbox>',
            'parameter': [{
                'element': 'string, the element in the web page to input to',
                'content': 'string, what content to input into the textbox element'
            }]
        }
    }, {
        'function': {
            'name': 'SELECT',
            'desc': 'Select an element from a combobox',
            'parameter': [{
                'element': 'string, the combobox or dropdown in the web page on which the select happens',
                'content': 'string, which choices to choose'
            }]
        }
    }]

    @classmethod
    def build_turn(cls, row: Dict[str, Any]) -> Tuple[str, str, str]:
        """One action as ``(query, response, screenshot_path)``.

        The target is written as ``a -> b: c``: what to do is the last arrow-separated part, where to
        do it is the first, and anything after a colon is the content to type.
        """
        screenshot = MediaDownloader.safe_save(row['screenshot'], row['action_uid'] + '.jpg', 'mind2web')
        query = f"The snapshot of screen:<image>\nThe html source code:{row['cleaned_html']}\n"

        parts = row['target_action_reprs'].split('->')
        action, where = parts[-1], (parts[0] if len(parts) > 1 else '')
        what = ''
        if ':' in action:
            action, what = action.split(':', 1)
        response = f'Action: {action.strip()}\nAction Input: {where.strip()},{what.strip()}'
        return query, response, screenshot

    @classmethod
    def episodes(cls, dataset):
        """Fold consecutive action rows into one standard row per episode."""
        messages: List[Dict[str, str]] = []
        images: List[str] = []

        def episode():
            return {'messages': list(messages), 'images': list(images), 'tools': cls.TOOLS}

        for row in dataset:
            query, response, screenshot = cls.build_turn(row)
            if row['target_action_index'] == '0':
                if messages:
                    yield episode()
                messages, images = [], []
                # The first action is the only one told what the episode is for.
                query = query + '\n' + row['confirmed_task']
            messages += [{'role': 'user', 'content': query}, {'role': 'assistant', 'content': response}]
            images.append(screenshot)
        if messages:
            yield episode()

    def prepare_dataset(self, dataset):
        from datasets import IterableDataset as HfIterableDataset
        if isinstance(dataset, HfIterableDataset):
            return HfIterableDataset.from_generator(self.episodes, gen_kwargs={'dataset': dataset})
        from datasets import Dataset as HfDataset
        return HfDataset.from_list(list(self.episodes(dataset)))

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # `prepare_dataset` already produced standard rows; conversion would only re-detect a format
        # for a dialogue that is already built.
        return row


@register_dataset
class Mind2WebLoader(DatasetLoader):
    dataset_type = 'Multimodal-Mind2Web'
    datasets = [('swift/Multimodal-Mind2Web', 'osunlp/Multimodal-Mind2Web')]
    preprocessor = Mind2WebPreprocessor
    tags = ['agent', 'multi-modal']


