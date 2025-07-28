# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import os
from typing import Any, Dict, Optional

import numpy as np
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from tqdm import tqdm

from swift.utils import get_hf_endpoint, use_hf_hub
from ..media import MediaResource
from ..preprocessor import GroundingMixin, MessagesPreprocessor, ResponsePreprocessor, RowPreprocessor
from ..register import DatasetMeta, SubsetDataset, register_dataset


class ShareGPT4oPreprocessor(MessagesPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = super().preprocess(row)
        image = row['images']
        if not image:
            return
        image = os.path.join(self.prefix_path, image)
        if not os.path.exists(image):
            return
        row['images'] = [image]
        return row

    def prepare_dataset(self, dataset):
        if not use_hf_hub():
            url = ('https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/ShareGPT-4o/repo?'
                   'Revision=master&FilePath=images.zip')
        else:
            url = f'{get_hf_endpoint()}/datasets/OpenGVLab/ShareGPT-4o/blob/main/images.zip'
        local_dir = MediaResource.download(url, 'sharegpt_4o_images')
        self.prefix_path = os.path.join(local_dir, 'mnt', 'petrelfs', 'wangwenhai', 'workspace_cef', '4o', 'image')
        return super().prepare_dataset(dataset)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/ShareGPT-4o',
        hf_dataset_id='OpenGVLab/ShareGPT-4o',
        preprocess_func=ShareGPT4oPreprocessor(),
        subsets=['image_caption'],
        split=['images'],
        tags=['vqa', 'multi-modal'],
    ))


class GPT4vDataset(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row['query'] = 'What is the caption of this image?'
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/gpt4v-dataset',
        hf_dataset_id='laion/gpt4v-dataset',
        preprocess_func=GPT4vDataset(columns={
            'link': 'images',
            'caption': 'response'
        }),
        split=['train'],
        tags=['en', 'caption', 'multi-modal', 'quality'],
        huge_dataset=True,
    ))

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/RLAIF-V-Dataset',
        hf_dataset_id='openbmb/RLAIF-V-Dataset',
        preprocess_func=ResponsePreprocessor(columns={
            'question': 'query',
            'chosen': 'response',
            'rejected': 'rejected_response'
        }),
        tags=['rlhf', 'dpo', 'multi-modal', 'en'],
    ))


class GarbagePreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row['query'] = 'Task: Classify household waste.'
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='tany0699/garbage265',
        preprocess_func=GarbagePreprocessor(columns={
            'category': 'label',
            'image:FILE': 'images'
        }),
        tags=['cls', '🔥', 'multi-modal'],
    ))


class SA1BPairedCaptionPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ['图片中展示了什么', '讲述一下图片中内容', '告诉我里面有什么', '图片内容是啥']
        response = row['global_caption']
        query = np.random.choice(prompt)
        return {
            'messages': [{
                'role': 'user',
                'content': query,
            }, {
                'role': 'assistant',
                'content': response,
            }]
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='Tongyi-DataEngine/SA1B-Paired-Captions-Images',
        preprocess_func=SA1BPairedCaptionPreprocessor(columns={
            'opensource_url': 'images',
        }),
        tags=['zh', 'multi-modal', 'vqa'],
    ))


class SA1BDenseCaptionPreprocessor(RowPreprocessor):
    column_mapping = {
        'url': 'images',
    }

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ['图片中展示了什么', '讲述一下图片中内容', '告诉我里面有什么', '图片内容是啥']
        response = ast.literal_eval(row['cap_seg'])
        response = response.get('global_caption')
        query = np.random.choice(prompt)
        return {
            'messages': [{
                'role': 'user',
                'content': query,
            }, {
                'role': 'assistant',
                'content': response,
            }]
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='Tongyi-DataEngine/SA1B-Dense-Caption',
        preprocess_func=SA1BDenseCaptionPreprocessor(columns={
            'url': 'images',
        }),
        tags=['zh', 'multi-modal', 'vqa'],
        huge_dataset=True,
    ))


class COCO2014Preprocess(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        caption = row['caption']
        if '&&' in caption:
            caption = caption.split('&&')[0]
        row['query'] = 'please describe the image.'
        row['response'] = caption

        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='modelscope/coco_2014_caption',
        preprocess_func=COCO2014Preprocess(),
        subsets=[
            SubsetDataset('train', 'coco_2014_caption', ['train']),
            SubsetDataset('validation', 'coco_2014_caption', ['validation']),
        ],
        tags=['chat', 'multi-modal', 'vision', '🔥'],
    ))


class MantisPreprocessor(MessagesPreprocessor):

    def __init__(self, *, subset: str, columns: Optional[Dict[str, str]] = None) -> None:
        self.subset = subset
        super().__init__(columns=columns)

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        if not use_hf_hub():
            url = (f'https://www.modelscope.cn/api/v1/datasets/swift/Mantis-Instruct/repo?Revision='
                   f'master&FilePath={self.subset}/train_images.zip')  # noqa
        else:
            url = (f'{get_hf_endpoint()}/datasets/TIGER-Lab/Mantis-Instruct/'
                   f'resolve/main/{self.subset}/train_images.zip')
        self.local_dir = MediaResource.download(url, f'mantis_{self.subset}')
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        images = [os.path.join(self.local_dir, p['path']) for p in row['images']]
        if not all([os.path.exists(d) for d in images]):
            images = []

        if not images:
            return
        row['images'] = images
        return super().preprocess(row)


mantis_subsets_name = [
    'birds-to-words', 'chartqa', 'coinstruct', 'contrastive_caption', 'docvqa', 'dreamsim', 'dvqa', 'iconqa',
    'imagecode', 'llava_665k_multi', 'lrv_multi', 'multi_vqa', 'nextqa', 'nlvr2', 'spot-the-diff', 'star',
    'visual_story_telling'
]

_mantis_subsets = []
for subset in mantis_subsets_name:
    _subset = SubsetDataset(subset=subset, split=['train'], preprocess_func=MantisPreprocessor(subset=subset))
    _mantis_subsets.append(_subset)

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/Mantis-Instruct',
        subsets=_mantis_subsets,
        tags=['chat', 'multi-modal', 'vision'],
    ))


class LLaVADataPreprocessor(MessagesPreprocessor):

    def prepare_dataset(self, dataset):
        self.all_folders = {}
        for media_type in ['coco', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2']:
            self.all_folders[media_type] = MediaResource.download(media_type)
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not row['images']:
            return
        row = super().preprocess(row)
        images = [p['path'] for p in row['images']]
        new_images = []
        for image in images:
            if 'coco/' in image:
                image = os.path.join(self.all_folders['coco'], image.replace('coco/', ''))
            elif 'gqa/' in image:
                image = os.path.join(self.all_folders['gqa'], image.replace('gqa/', ''))
            elif 'ocr_vqa/' in image:
                image = os.path.join(self.all_folders['ocr_vqa'], image)
            elif 'textvqa/' in image:
                image = os.path.join(self.all_folders['textvqa'], image.replace('textvqa/', ''))
            elif 'VG_100K/' in image:
                image = os.path.join(self.all_folders['VG_100K'], image.replace('vg/', ''))
            elif 'VG_100K_2/' in image:
                image = os.path.join(self.all_folders['VG_100K_2'], image.replace('vg/', ''))
            new_images.append(image)
        if all(os.path.exists(image) for image in new_images):
            row['images'] = new_images
        else:
            return {'images': None}
        return row


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/llava-data',
        hf_dataset_id='TIGER-Lab/llava-data',
        subsets=['llava_instruct'],
        preprocess_func=LLaVADataPreprocessor(),
        tags=['sft', 'multi-modal', 'quality'],
    ))


class PixelProsePreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        caption_prompt = [
            'Give the description of this image.', 'Describe this picture', 'What is the proper title of this image?'
        ]
        vlm_caption = row['vlm_caption']
        if vlm_caption.startswith('This image displays:'):
            vlm_caption = vlm_caption[len('This image displays:'):].strip()
        return {
            'messages': [{
                'role': 'user',
                'content': np.random.choice(caption_prompt)
            }, {
                'role': 'assistant',
                'content': vlm_caption
            }],
            'images':
            row['url']
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/pixelprose',
        hf_dataset_id='tomg-group-umd/pixelprose',
        preprocess_func=PixelProsePreprocessor(),
        split=['train', 'cc12m', 'commonpool', 'redcaps'],
        tags=['caption', 'multi-modal', 'vision'],
        huge_dataset=True,
    ))


class AIShell1Preprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row['query'] = '语音转文本'
        row['response'] = row['Text:LABEL'].replace(' ', '')
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='speech_asr/speech_asr_aishell1_trainsets',
        subsets=[
            SubsetDataset('train', split=['train']),
            SubsetDataset('validation', split=['validation']),
            SubsetDataset('test', split=['test']),
        ],
        preprocess_func=AIShell1Preprocessor(columns={'Audio:FILE': 'audios'}),
        tags=['chat', 'multi-modal', 'audio'],
    ))


class EmoSchemaPreprocessor(ResponsePreprocessor):

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        for i in range(1, 6):
            if not use_hf_hub():
                url = f'https://modelscope.cn/datasets/AI-ModelScope/egoschema/resolve/master/videos_chunked_0{i}.zip'
            else:
                url = f'{get_hf_endpoint()}/datasets/lmms-lab/egoschema/resolve/main/videos_chunked_0{i}.zip'
            local_dir = MediaResource.download(url, 'egoschema')

        self.local_dir = os.path.join(local_dir, 'videos')
        self.mp4_set = [file[:-4] for file in os.listdir(self.local_dir) if file.endswith('mp4')]
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if row['video_idx'] not in self.mp4_set:
            return None
        transfer_to_option = {
            '0': 'A',
            '1': 'B',
            '2': 'C',
            '3': 'D',
            '4': 'E',
        }
        row = {
            'query': row['query'] + '\n' + '\n'.join(row['option']),
            'response': transfer_to_option[row['response']],
            'videos': [os.path.join(self.local_dir, f"{row['video_idx']}.mp4")],
        }
        return super().preprocess(row)


class EmoSchemaClsPreprocessor(EmoSchemaPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if row['video_idx'] not in self.mp4_set:
            return None
        row = {
            'query': row['query'] + '\n' + '\n'.join(row['option']),
            'label': int(row['response']),
            'videos': [os.path.join(self.local_dir, f"{row['video_idx']}.mp4")],
        }
        return ResponsePreprocessor.preprocess(self, row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/egoschema',
        hf_dataset_id='lmms-lab/egoschema',
        subsets=[
            SubsetDataset('default', 'Subset', preprocess_func=EmoSchemaPreprocessor()),
            SubsetDataset('cls', 'Subset', preprocess_func=EmoSchemaClsPreprocessor())
        ],
        split=['test'],
        tags=['chat', 'multi-modal', 'video'],
    ))


def _generate_url_list(_url, _range):
    lst = []
    for i in range(1, (_range + 1)):
        lst.append(_url.replace('{}', str(i)))
    return lst


class LLaVAVideo178KPreprocessor(MessagesPreprocessor):

    def __init__(self, *, subset: str, columns: Optional[Dict[str, str]] = None) -> None:
        self.subset = subset
        super().__init__(columns=columns)

    url_prefix = 'https://www.modelscope.cn/datasets/lmms-lab/LLaVA-Video-178K/resolve/master/'
    if use_hf_hub():
        url_prefix = f'{get_hf_endpoint()}/datasets/lmms-lab/LLaVA-Video-178K/resolve/main/'

    video_resources = {
        '0_30_s_academic_v0_1':
        _generate_url_list(
            url_prefix + '0_30_s_academic_v0_1/0_30_s_academic_v0_1_videos_{}.tar.gz',
            8,
        ),
        '0_30_s_youtube_v0_1':
        _generate_url_list(
            url_prefix + '0_30_s_youtube_v0_1/0_30_s_youtube_v0_1_videos_{}.tar.gz',
            19,
        ),
        '1_2_m_academic_v0_1':
        _generate_url_list(
            url_prefix + '1_2_m_academic_v0_1/1_2_m_academic_v0_1_videos_{}.tar.gz',
            14,
        ),
        '1_2_m_youtube_v0_1':
        _generate_url_list(
            url_prefix + '1_2_m_youtube_v0_1/1_2_m_youtube_v0_1_videos_{}.tar.gz',
            50,
        ),
        '2_3_m_academic_v0_1':
        _generate_url_list(
            url_prefix + '2_3_m_academic_v0_1/2_3_m_academic_v0_1_videos_{}.tar.gz',
            18,
        ),
        '2_3_m_youtube_v0_1':
        _generate_url_list(
            url_prefix + '2_3_m_youtube_v0_1/2_3_m_youtube_v0_1_videos_{}.tar.gz',
            98,
        ),
        '30_60_s_academic_v0_1':
        _generate_url_list(
            url_prefix + '30_60_s_academic_v0_1/30_60_s_academic_v0_1_videos_{}.tar.gz',
            10,
        ),
        '30_60_s_youtube_v0_1':
        _generate_url_list(
            url_prefix + '30_60_s_youtube_v0_1/30_60_s_youtube_v0_1_videos_{}.tar.gz',
            13,
        ),
    }

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        urls = self.video_resources[self.subset]
        self.local_dir = MediaResource.download(urls, f'llava_video_178k_{self.subset}', file_type='sharded')
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        file_path = os.path.join(self.local_dir, f"{row['videos']}")
        if not os.path.exists(file_path):
            return None
        return super().preprocess({'messages': row['messages'], 'videos': file_path})


llava_video_subsets = []
for subset in [
        '0_30_s_academic_v0_1',
        '0_30_s_youtube_v0_1',
        '1_2_m_academic_v0_1',
        '1_2_m_youtube_v0_1',
        '2_3_m_academic_v0_1',
        '2_3_m_youtube_v0_1',
        '30_60_s_academic_v0_1',
        '30_60_s_youtube_v0_1',
]:
    subset = SubsetDataset(
        subset=subset,
        split=['caption', 'open_ended', 'multi_choice'],
        preprocess_func=LLaVAVideo178KPreprocessor(subset=subset),
    )
    llava_video_subsets.append(subset)

register_dataset(
    DatasetMeta(
        hf_dataset_id='lmms-lab/LLaVA-Video-178K', subsets=llava_video_subsets, tags=['chat', 'multi-modal', 'video']))


class MovieChat1KPreprocessor(ResponsePreprocessor):

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        mp4_set = [f'{i}.mp4' for i in range(1, 10)] + \
                  [f'{i}.mp4' for i in range(201, 240)] + \
                  [f'AWA-{i}.mp4' for i in range(1, 10)] + \
                  [f'AWB-{i}.mp4' for i in range(1, 16)] + \
                  [f'AWC-{i}.mp4' for i in range(1, 11)] + \
                  [f'AWD-{i}.mp4' for i in range(1, 8)] + \
                  [f'AWE-{i}.mp4' for i in range(1, 7)] + \
                  [f'AWG-{i}.mp4' for i in range(1, 12)] + \
                  [f'AWH-{i}.mp4' for i in range(1, 8)] + \
                  [f'BWA-{i}.mp4' for i in range(1, 7)] + \
                  [f'BWB-{i}.mp4' for i in range(1, 7)] + \
                  [f'BWD-{i}.mp4' for i in range(1, 6)] + \
                  [f'BWE-{i}.mp4' for i in range(1, 6)] + \
                  [f'BWG-{i}.mp4' for i in range(1, 6)] + \
                  [f'BWH-{i}.mp4' for i in range(1, 6)] + \
                  [f'TFS-{i}.mp4' for i in range(1, 13)] + \
                  [f'UWA-{i}.mp4' for i in range(1, 5)] + ['UWA-6.mp4']
        for file in mp4_set:
            if not use_hf_hub():
                url = f'https://modelscope.cn/datasets/AI-ModelScope/MovieChat-1K-test/resolve/master/videos/{file}'
            else:
                url = f'{get_hf_endpoint()}/datasets/Enxin/MovieChat-1K-test/resolve/main/videos/{file}'
            self.local_dir = MediaResource.download(url, 'moviechat_1k_test', file_type='file')
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        file_path = os.path.join(self.local_dir, f"{row['info']['video_path']}")
        if not os.path.exists(file_path):
            return None
        return super().preprocess({
            'query': row['global'][0]['question'],
            'response': row['global'][0]['answer'],
            'videos': file_path,
        })


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/MovieChat-1K-test',
        hf_dataset_id='Enxin/MovieChat-1K-test',
        preprocess_func=MovieChat1KPreprocessor(),
        split=['train'],
        tags=['chat', 'multi-modal', 'video']))


class VideoChatGPTPreprocessor(ResponsePreprocessor):

    def prepare_dataset(self, dataset: HfDataset) -> HfDataset:
        if not use_hf_hub():
            url = 'https://modelscope.cn/datasets/swift/VideoChatGPT/resolve/master/videos.zip'
        else:
            url = f'{get_hf_endpoint()}/datasets/lmms-lab/VideoChatGPT/resolve/main/videos.zip'
        local_dir = MediaResource.download(url, 'video_chatgpt')
        self.local_dir = os.path.join(local_dir, 'Test_Videos')
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # only `.mp4`
        mp4_set = [file[:-4] for file in os.listdir(self.local_dir) if file.endswith('mp4')]
        if row['video_name'] not in mp4_set:
            return
        row['videos'] = os.path.join(self.local_dir, f"{row['video_name']}.mp4")
        for key in ['query', 'question_1', 'question_2']:
            query = row.get(key)
            if query is None or query == 'None':
                continue
            row['query'] = query
            return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/VideoChatGPT',
        hf_dataset_id='lmms-lab/VideoChatGPT',
        subsets=['Generic', 'Temporal', 'Consistency'],
        preprocess_func=VideoChatGPTPreprocessor(),
        split=['test'],
        tags=['chat', 'multi-modal', 'video', '🔥'],
    ))


def preprocess_mind2web(dataset, **kwargs):

    def preprocess_row(row: Dict[str, Any]) -> Dict[str, Any]:
        raw_html = row['cleaned_html']
        screenshot = row['screenshot']
        row['screenshot'] = MediaResource.safe_save(screenshot, row['action_uid'] + '.jpg', 'mind2web')
        action = row['target_action_reprs']
        actions = action.split('->')
        row['query'] = f'The snapshot of screen:<image>\nThe html source code:{raw_html}\n'
        action = actions[-1]
        where = actions[0] if len(actions) > 1 else ''
        what = ''
        if ':' in action:
            action, what = action[:action.find(':')], action[action.find(':') + 1:]
        row['response'] = f'Action: {action.strip()}\nAction Input: {where.strip()}{"," + what.strip()}'
        return row

    conversations = []
    tools = [{
        'function': {
            'name': 'CLICK',
            'desc': 'Choose and click an element in the web page',
            'parameter': [{
                'element': 'string, the element in the web page to click'
            }]
        }
    }, {
        'function': {
            'name':
            'TYPE',
            'desc':
            'Input some text into a web element like <input> or <textbox>',
            'parameter': [{
                'element': 'string, the element in the web page to input to',
                'content': 'string, what content to input into the textbox element'
            }]
        }
    }, {
        'function': {
            'name':
            'SELECT',
            'desc':
            'Select an element from a combobox',
            'parameter': [{
                'element': 'string, the combobox or dropdown in the web page on which the select happens',
                'content': 'string, which choices to choose'
            }]
        }
    }]

    def history_to_messages(history):
        messages = []
        for h in history:
            messages.append({'role': 'user', 'content': h[0]})
            messages.append({'role': 'assistant', 'content': h[1]})
        return messages

    if isinstance(dataset, HfIterableDataset):

        def generate_example(dataset):
            history = []
            images = []
            for row in dataset:
                target_action_index = row['target_action_index']
                row = preprocess_row(row)
                query = row['query']
                if target_action_index == '0':
                    if history:
                        yield {'messages': history_to_messages(history), 'images': images, 'tools': tools}
                        images = []
                        history = []
                    query = query + '\n' + row['confirmed_task']
                history.append([query, row['response']])
                images.append(row['screenshot'])

            if history:
                yield {'messages': history_to_messages(history), 'images': images, 'tools': tools}

        return HfIterableDataset.from_generator(generate_example, gen_kwargs={'dataset': dataset})

    history = []
    images = []
    for row in tqdm(dataset):
        target_action_index = row['target_action_index']
        row = preprocess_row(row)
        query = row['query']
        if target_action_index == '0':
            if history:
                conversations.append({'messages': history_to_messages(history), 'images': images, 'tools': tools})
                images = []
                history = []
            query = query + '\n' + row['confirmed_task']
        history.append([query, row['response']])
        images.append(row['screenshot'])

    if history:
        conversations.append({'messages': history_to_messages(history), 'images': images, 'tools': tools})

    return HfDataset.from_list(conversations)


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/Multimodal-Mind2Web',
        hf_dataset_id='osunlp/Multimodal-Mind2Web',
        preprocess_func=preprocess_mind2web,
        tags=['agent', 'multi-modal']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/M3IT',
        subsets=[
            'coco', 'vqa-v2', 'shapes', 'shapes-rephrased', 'coco-goi-rephrased', 'snli-ve', 'snli-ve-rephrased',
            'okvqa', 'a-okvqa', 'viquae', 'textcap', 'docvqa', 'science-qa', 'imagenet', 'imagenet-open-ended',
            'imagenet-rephrased', 'coco-goi', 'clevr', 'clevr-rephrased', 'nlvr', 'coco-itm', 'coco-itm-rephrased',
            'vsr', 'vsr-rephrased', 'mocheg', 'mocheg-rephrased', 'coco-text', 'fm-iqa', 'activitynet-qa', 'msrvtt',
            'ss', 'coco-cn', 'refcoco', 'refcoco-rephrased', 'multi30k', 'image-paragraph-captioning', 'visual-dialog',
            'visual-dialog-rephrased', 'iqa', 'vcr', 'visual-mrc', 'ivqa', 'msrvtt-qa', 'msvd-qa', 'gqa', 'text-vqa',
            'ocr-vqa', 'st-vqa', 'flickr8k-cn'
        ],
        preprocess_func=ResponsePreprocessor(columns={
            'instruction': 'system',
            'inputs': 'query',
            'image_base64_str': 'images',
            'outputs': 'response'
        }),
        split=['train'],
        huge_dataset=True,
        tags=['chat', 'multi-modal', 'vision']))


class ShareGPT4VPreprocessor(MessagesPreprocessor):

    def prepare_dataset(self, dataset):
        split = ['ShareGPT4V', 'ShareGPT4V-PT'] if dataset.config_name is None else dataset.config_name
        IMAGE_DATASET_REQUIREMENTS = {
            'ShareGPT4V': ['coco', 'sam', 'llava', 'wikiart', 'share_textvqa', 'web-celebrity', 'web-landmark'],
            'ShareGPT4V-PT': ['coco', 'sam', 'llava']
        }

        if isinstance(split, str):
            split = [split]
        self.all_folders = {}
        for sp in split:
            for media_type in IMAGE_DATASET_REQUIREMENTS[sp]:
                self.all_folders[media_type] = MediaResource.download(media_type)
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        image = row['image']
        row.update(super().preprocess(row))
        if 'coco/' in image:
            image = os.path.join(self.all_folders['coco'], image.replace('coco/', ''))
        elif 'sam/' in image:
            image = os.path.join(self.all_folders['sam'], image.replace('sam/images/', ''))
        elif 'llava/' in image:
            image = os.path.join(self.all_folders['llava'], image.replace('llava/llava_pretrain/images/', ''))
        elif 'wikiart/' in image:
            image = os.path.join(self.all_folders['wikiart'], image.replace('wikiart/images/', 'data/wikiart/images/'))
        elif 'share_textvqa/' in image:
            image = os.path.join(self.all_folders['share_textvqa'],
                                 image.replace('share_textvqa/images/', 'data/share_textvqa/images/'))
        elif 'web-celebrity/' in image:
            image = os.path.join(self.all_folders['web-celebrity'],
                                 image.replace('web-celebrity/images/', 'data/web-celebrity/images/'))
        elif 'web-landmark/' in image:
            image = os.path.join(self.all_folders['web-landmark'],
                                 image.replace('web-landmark/images/', 'data/web-landmark/images/'))
        if os.path.exists(image):
            row['images'] = image
        else:
            return
        return row


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/ShareGPT4V',
        subsets=['ShareGPT4V', 'ShareGPT4V-PT'],
        preprocess_func=ShareGPT4VPreprocessor(),
        huge_dataset=True,
        tags=['chat', 'multi-modal', 'vision']))


class TextCapsPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row['query'] = 'What is the caption of this image?'
        if not os.path.exists(row['images']['path']):
            return None
        return super().preprocess(row)


class TextCapsEmbPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row['query'] = ''
        if not os.path.exists(row['images']['path']):
            return None
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/TextCaps',
        hf_dataset_id='HuggingFaceM4/TextCaps',
        subsets=[
            SubsetDataset(
                name='default',
                preprocess_func=TextCapsPreprocessor(columns={'reference_strs': 'response'}),
                split=['train', 'validation'],
            ),
            SubsetDataset(
                name='emb',
                preprocess_func=TextCapsEmbPreprocessor(columns={'reference_strs': 'response'}),
                split=['train', 'validation'],
            ),
        ],
        huge_dataset=True,
        tags=['multi-modal', 'en', 'caption', 'quality']))


class RefCOCOPreprocessor(ResponsePreprocessor, GroundingMixin):
    task_type = 'caption'

    def __init__(self, task_type, **kwargs):
        self.task_type = task_type
        super().__init__(**kwargs)

    def prepare_dataset(self, dataset):
        self.cache_dir = MediaResource.download(
            'https://www.modelscope.cn/api/v1/datasets/we_dont_produce_water/'
            'coco_res/repo?Revision=master&FilePath=coco_2014.zip', 'coco2014')
        return dataset

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        caption = row['captions'][0]
        bbox = row['bbox']
        image_path = os.path.join(self.cache_dir, row['image_path'].replace('coco/train2014', 'train2014'))
        if not os.path.exists(image_path):
            return

        for i in range(len(bbox)):
            bbox[i] = round(float(bbox[i]))
        res = {}

        objects = {
            'ref': [caption],
            'bbox': [bbox],
        }
        res['query'], res['response'] = self.construct_grounding_prompt()
        res['images'] = [image_path]
        res['objects'] = objects
        return super().preprocess(res)


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/refcoco',
        hf_dataset_id='jxu124/refcoco',
        subsets=[
            SubsetDataset(
                name='caption',
                preprocess_func=RefCOCOPreprocessor('caption'),
            ),
            SubsetDataset(
                name='grounding',
                preprocess_func=RefCOCOPreprocessor('grounding'),
            )
        ],
        split=['train', 'validation'],
        tags=['multi-modal', 'en', 'grounding']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/refcocog',
        hf_dataset_id='jxu124/refcocog',
        subsets=[
            SubsetDataset(
                name='caption',
                preprocess_func=RefCOCOPreprocessor('caption'),
            ),
            SubsetDataset(
                name='grounding',
                preprocess_func=RefCOCOPreprocessor('grounding'),
            )
        ],
        split=['train', 'validation'],
        tags=['multi-modal', 'en', 'grounding']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/lnqa',
        hf_dataset_id='vikhyatk/lnqa',
        preprocess_func=MessagesPreprocessor(user_role='question', assistant_role='answer'),
        split=['train', 'validation'],
        huge_dataset=True,
        tags=['multi-modal', 'en', 'ocr-vqa', 'quality']))


class LLaVAInstructPreprocessor(MessagesPreprocessor):

    def prepare_dataset(self, dataset):
        self.all_folders = {}
        for media_type in ['coco', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2']:
            self.all_folders[media_type] = MediaResource.download(media_type)
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        image = row['images']
        if 'coco/' in image:
            image = os.path.join(self.all_folders['coco'], image.replace('coco/', ''))
        elif 'gqa/' in image:
            image = os.path.join(self.all_folders['gqa'], image.replace('gqa/', ''))
        elif 'ocr_vqa/' in image:
            image = os.path.join(self.all_folders['ocr_vqa'], image)
        elif 'textvqa/' in image:
            image = os.path.join(self.all_folders['textvqa'], image.replace('textvqa/', ''))
        elif 'VG_100K/' in image:
            image = os.path.join(self.all_folders['VG_100K'], image.replace('vg/', ''))
        elif 'VG_100K_2/' in image:
            image = os.path.join(self.all_folders['VG_100K_2'], image.replace('vg/', ''))
        if os.path.exists(image):
            row['images'] = image
        else:
            return

        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/LLaVA-Instruct-150K',
        ms_revision='d5db3806e395c60496630a206c336932e85a2d00',
        preprocess_func=LLaVAInstructPreprocessor(),
        split=['train'],
        tags=['chat', 'multi-modal', 'vision']))


class LLaVAPretrainPreprocessor(MessagesPreprocessor):

    def prepare_dataset(self, dataset):
        if not use_hf_hub():
            url = ('https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/LLaVA-Pretrain/repo?'
                   'Revision=master&FilePath=images.zip')
        else:
            url = f'{get_hf_endpoint()}/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip'
        self.media_dir = MediaResource.download(
            url,
            # noqa
            'llava_pretrain')
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row.update(super().preprocess(row))
        if row['image']:
            file_path = os.path.join(self.media_dir, row['image'])
            if os.path.exists(file_path):
                return {'images': file_path}
            else:
                return
        else:
            return


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/LLaVA-Pretrain',
        ms_revision='e3a3f0bfaad05e90e46745152a32bf944e0f4a63',
        hf_dataset_id='liuhaotian/LLaVA-Pretrain',
        preprocess_func=LLaVAPretrainPreprocessor(),
        huge_dataset=True,
        tags=['chat', 'multi-modal', 'quality']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/MideficsDataset',
        hf_dataset_id='WinterSchool/MideficsDataset',
        preprocess_func=MessagesPreprocessor(inner_key='data', user_role='question', assistant_role='answer'),
        tags=['medical', 'en', 'vqa']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/OK-VQA_train',
        hf_dataset_id='Multimodal-Fatima/OK-VQA_train',
        preprocess_func=ResponsePreprocessor(),
        tags=['multi-modal', 'en', 'vqa', 'quality']))

register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/A-OKVQA',
        hf_dataset_id='HuggingFaceM4/A-OKVQA',
        split=['train', 'validation'],
        preprocess_func=ResponsePreprocessor(columns={'rationales': 'response'}),
        tags=['multi-modal', 'en', 'vqa', 'quality']))


class OcrvqaPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        idx = np.random.choice(range(len(row['questions'])))
        query = row['questions'][idx]
        response = row['answers'][idx]
        return {
            'messages': [{
                'role': 'user',
                'content': query
            }, {
                'role': 'assistant',
                'content': response
            }],
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/OCR-VQA',
        hf_dataset_id='howard-hou/OCR-VQA',
        split=['train', 'validation'],
        preprocess_func=OcrvqaPreprocessor(),
        tags=['multi-modal', 'en', 'ocr-vqa']))


class ScienceQAPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row['question']
        response = row['choices'][row['answer']]
        solution = row['solution']
        response = f'{solution}\nSo the final answer is: {response}'
        return {'messages': [{'role': 'user', 'content': query}, {'role': 'assistant', 'content': response}]}


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/ScienceQA',
        hf_dataset_id='derek-thomas/ScienceQA',
        split=['train', 'validation'],
        preprocess_func=ScienceQAPreprocessor(),
        tags=['multi-modal', 'science', 'vqa', 'quality']))


class GritPreprocessor(RowPreprocessor, GroundingMixin):

    def __init__(self, task_type, **kwargs):
        self.task_type = task_type
        super().__init__(**kwargs)

    @staticmethod
    def has_overlap(start_ends):
        for i in range(1, len(start_ends)):
            if start_ends[i][0] < start_ends[i - 1][1]:
                return True
        return False

    @staticmethod
    def replace_intervals_with_tags(response, start_ends):
        result = []
        last_end = 0
        for start, end in start_ends:
            result.append(response[int(last_end):int(start)])
            result.append('<ref-object><bbox>')
            last_end = end
        result.append(response[int(last_end):])
        return ''.join(result)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        images = row['images']
        caption = row['caption']
        ref_exps = row['ref_exps']
        objects = {'ref': [], 'bbox': [], 'bbox_type': 'norm1'}
        start_end_pairs = []
        for ref_exp in ref_exps:
            start = ref_exp[0]
            end = ref_exp[1]
            # conf = ref_exp[6] TODO filter low confidence rows?
            start_end_pairs.append(ref_exp[0:2])

            object_part = caption[int(start):int(end)]
            objects['ref'].append(object_part)
            objects['bbox'].append(ref_exp[2:6])

        start_end_pairs.sort(key=lambda x: (x[0], x[1]))
        if self.has_overlap(start_end_pairs) or not ref_exps:
            return

        if self.task_type in ('grounding', 'caption'):
            query, response = self.construct_grounding_prompt()
        else:
            query = 'what is the proper caption of this image?'
            response = caption
        return {
            'messages': [{
                'role': 'user',
                'content': query
            }, {
                'role': 'assistant',
                'content': response
            }],
            'images': images,
            'objects': objects
        }


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/GRIT',
        hf_dataset_id='zzliang/GRIT',
        subsets=[
            SubsetDataset(
                name='caption',
                preprocess_func=GritPreprocessor('caption', columns={'url': 'images'}),
            ),
            SubsetDataset(
                name='grounding',
                preprocess_func=GritPreprocessor('grounding', columns={'url': 'images'}),
            ),
            SubsetDataset(
                name='vqa',
                preprocess_func=GritPreprocessor('vqa', columns={'url': 'images'}),
            )
        ],
        huge_dataset=True,
        tags=['multi-modal', 'en', 'caption-grounding', 'vqa', 'quality']))


class GQAPreprocessor(RowPreprocessor):

    def prepare_dataset(self, dataset):
        self.local_cache = MediaResource.download('gqa')
        return super().prepare_dataset(dataset)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if os.path.join(self.local_cache, 'images', row['imageId'] + '.jpg'):
            return {
                'messages': [{
                    'role': 'user',
                    'content': row['question']
                }, {
                    'role': 'assistant',
                    'content': row['fullAnswer']
                }],
                'images':
                os.path.join(self.local_cache, 'images', row['imageId'] + '.jpg'),
            }
        else:
            return


register_dataset(
    DatasetMeta(
        hf_dataset_id='lmms-lab/GQA',
        split=['train_all_instructions'],
        preprocess_func=GQAPreprocessor(),
        huge_dataset=True,
        tags=['multi-modal', 'en', 'vqa', 'quality']))


class CocoPreprocessor(ResponsePreprocessor):
    category = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row['query'] = 'Task: Object Detection'
        objects = row['objects']
        objects['ref'] = [self.category[c] for c in objects['category']]
        row['response'] = '\n'.join(['<ref-object><bbox>'] * len(objects['ref']))
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/coco',
        hf_dataset_id='detection-datasets/coco',
        preprocess_func=CocoPreprocessor(),
        huge_dataset=True,
        tags=['multi-modal', 'en', 'vqa', 'quality']))


class LLaVAMixSFTPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        messages = row['messages']
        rounds = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            text = ''
            for index in content:
                if index['type'] == 'text':
                    text += index['text']
                elif index['type'] == 'image':
                    text += '<image>'

            rounds.append({'role': role, 'content': text})

        return {'messages': rounds}


register_dataset(
    DatasetMeta(
        ms_dataset_id='swift/llava-instruct-mix-vsft',
        hf_dataset_id='HuggingFaceH4/llava-instruct-mix-vsft',
        split=['test'],
        preprocess_func=LLaVAMixSFTPreprocessor(),
        tags=['multi-modal', 'en', 'vqa', 'quality']))


class LatexocrPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row['query'] = 'Using LaTeX to perform OCR on the image.'
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/LaTeX_OCR',
        hf_dataset_id='linxy/LaTeX_OCR',
        subsets=['default', 'human_handwrite', 'human_handwrite_print', 'synthetic_handwrite', 'small'],
        preprocess_func=LatexocrPreprocessor(),
        split=['train', 'validation', 'test'],
        tags=['chat', 'ocr', 'multi-modal', 'vision'],
    ))


class CapchaImagesPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row['query'] = 'recognize the content.'
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='AI-ModelScope/captcha-images',
        split=['train', 'validation'],
        preprocess_func=CapchaImagesPreprocessor(columns={'solution': 'response'}),
        tags=['chat', 'multi-modal', 'vision']))


class ClevrPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row.get('query', '')
        query = (f'{query} Output the thinking process in <think> </think> and '
                 'final answer (number) in <answer> </answer> tags.')
        row.update({'query': query})
        return super().preprocess(row)


register_dataset(
    DatasetMeta(
        ms_dataset_id='okwinds/clevr_cogen_a_train',
        hf_dataset_id='leonardPKU/clevr_cogen_a_train',
        preprocess_func=ClevrPreprocessor(),
        tags=['qa', 'math', 'vision', 'grpo']))
