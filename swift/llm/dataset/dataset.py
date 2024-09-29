# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import itertools
import json
import os
import re
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from datasets import concatenate_datasets, interleave_datasets
from numpy.random import RandomState
from tqdm.auto import tqdm
from transformers.utils import strtobool

from swift.llm.dataset.preprocess import (AlpacaPreprocessor, ClsPreprocessor, ComposePreprocessor,
                                          ConversationsPreprocessor,
                                          ListPreprocessor, PreprocessFunc, RenameColumnsPreprocessor,
                                          TextGenerationPreprocessor, RowPreprocessor)
from swift.utils import get_logger
from .loader import HubDatasetLoader, DatasetLoader
from .media import MediaResource
from .register import register_dataset

dataset_enable_cache = strtobool(os.environ.get('DATASET_ENABLE_CACHE', 'False'))

DATASET_TYPE = Union[HfDataset, HfIterableDataset]

standard_keys = {
    'messages', 'rejected_response', 'images', 'objects',
    'videos', 'audios', 'tools', 'label'
}


SubsetSplit = Union[str, Tuple[str, str], List[str]]

logger = get_logger()


class DatasetName:
    # general
    ms_bench = 'ms-bench'  # used for mixed training
    alpaca_en = 'alpaca-en'
    alpaca_zh = 'alpaca-zh'
    multi_alpaca = 'multi-alpaca'
    instinwild = 'instinwild'
    cot_en = 'cot-en'
    cot_zh = 'cot-zh'
    instruct_en = 'instruct-en'
    firefly_zh = 'firefly-zh'
    gpt4all_en = 'gpt4all-en'
    sharegpt = 'sharegpt'
    tulu_v2_sft_mixture = 'tulu-v2-sft-mixture'
    wikipedia_zh = 'wikipedia-zh'
    open_orca = 'open-orca'
    sharegpt_gpt4 = 'sharegpt-gpt4'
    deepctrl_sft = 'deepctrl-sft'
    coig_cqia = 'coig-cqia'
    ruozhiba = 'ruozhiba'
    long_alpaca_12k = 'long-alpaca-12k'
    lmsys_chat_1m = 'lmsys-chat-1m'
    # agent
    ms_agent = 'ms-agent'
    ms_agent_for_agentfabric = 'ms-agent-for-agentfabric'
    ms_agent_multirole = 'ms-agent-multirole'
    toolbench_for_alpha_umi = 'toolbench-for-alpha-umi'
    damo_agent_zh = 'damo-agent-zh'
    damo_agent_zh_mini = 'damo-agent-zh-mini'
    agent_instruct_all_en = 'agent-instruct-all-en'
    msagent_pro = 'msagent-pro'
    toolbench = 'toolbench'

    # coding
    code_alpaca_en = 'code-alpaca-en'
    leetcode_python_en = 'leetcode-python-en'
    codefuse_python_en = 'codefuse-python-en'
    codefuse_evol_instruction_zh = 'codefuse-evol-instruction-zh'
    # medical
    medical_en = 'medical-en'
    medical_zh = 'medical-zh'
    disc_med_sft_zh = 'disc-med-sft-zh'
    # law
    lawyer_llama_zh = 'lawyer-llama-zh'
    tigerbot_law_zh = 'tigerbot-law-zh'
    disc_law_sft_zh = 'disc-law-sft-zh'
    # math
    blossom_math_zh = 'blossom-math-zh'
    school_math_zh = 'school-math-zh'
    open_platypus_en = 'open-platypus-en'
    # sql
    text2sql_en = 'text2sql-en'
    sql_create_context_en = 'sql-create-context-en'
    synthetic_text_to_sql = 'synthetic-text-to-sql'
    # text-generation
    advertise_gen_zh = 'advertise-gen-zh'
    dureader_robust_zh = 'dureader-robust-zh'
    # classification
    cmnli_zh = 'cmnli-zh'
    jd_sentiment_zh = 'jd-sentiment-zh'
    hc3_zh = 'hc3-zh'
    hc3_en = 'hc3-en'
    dolly_15k = 'dolly-15k'
    zhihu_kol = 'zhihu-kol'
    zhihu_kol_filtered = 'zhihu-kol-filtered'
    # other
    finance_en = 'finance-en'
    poetry_zh = 'poetry-zh'
    webnovel_zh = 'webnovel-zh'
    generated_chat_zh = 'generated-chat-zh'
    self_cognition = 'self-cognition'
    swift_mix = 'swift-mix'

    # example dataset for specific model
    cls_fudan_news_zh = 'cls-fudan-news-zh'  # seqgpt-560m
    ner_java_zh = 'ner-jave-zh'  # seqgpt-560m

    # multi-modal
    # <img></img>
    coco_en = 'coco-en'
    coco_en_mini = 'coco-en-mini'
    # images
    coco_en_2 = 'coco-en-2'
    coco_en_2_mini = 'coco-en-2-mini'
    capcha_images = 'capcha-images'
    latex_ocr_print = 'latex-ocr-print'
    latex_ocr_handwrite = 'latex-ocr-handwrite'
    # for qwen-audio
    aishell1_zh = 'aishell1-zh'
    aishell1_zh_mini = 'aishell1-zh-mini'
    # for video
    video_chatgpt = 'video-chatgpt'

    # rlhf
    hh_rlhf = 'hh-rlhf'
    hh_rlhf_cn = 'hh-rlhf-cn'
    orpo_dpo_mix_40k = 'orpo-dpo-mix-40k'
    stack_exchange_paired = 'stack-exchange-paired'
    shareai_llama3_dpo_zh_en_emoji = 'shareai-llama3-dpo-zh-en-emoji'
    ultrafeedback_kto = 'ultrafeedback-kto'

    # visual rlhf
    rlaif_v = 'rlaif-v'

    # for awq
    pileval = 'pileval'

    mantis_instruct = 'mantis-instruct'
    llava_data_instruct = 'llava-data-instruct'
    midefics = 'midefics'
    gqa = 'gqa'
    text_caps = 'text-caps'
    refcoco_unofficial_caption = 'refcoco-unofficial-caption'
    refcoco_unofficial_grounding = 'refcoco-unofficial-grounding'
    refcocog_unofficial_caption = 'refcocog-unofficial-caption'
    refcocog_unofficial_grounding = 'refcocog-unofficial-grounding'
    a_okvqa = 'a-okvqa'
    okvqa = 'okvqa'
    ocr_vqa = 'ocr-vqa'
    grit = 'grit'
    llava_instruct_mix = 'llava-instruct-mix'
    gpt4v_dataset = 'gpt4v-dataset'
    lnqa = 'lnqa'
    science_qa = 'science-qa'
    guanaco = 'guanaco'
    mind2web = 'mind2web'
    sharegpt_4o_image = 'sharegpt-4o-image'
    pixelprose = 'pixelprose'

    m3it = 'm3it'
    # additional images
    sharegpt4v = 'sharegpt4v'

    llava_instruct_150k = 'llava-instruct-150k'
    llava_pretrain = 'llava-pretrain'

    sa1b_dense_caption = 'sa1b-dense-caption'
    sa1b_paired_caption = 'sa1b-paired-caption'

    @classmethod
    def get_dataset_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__') or k == 'get_dataset_name_list':
                continue
            res.append(cls.__dict__[k])
        return res


def _concat_inst_inp_alpaca_zh(inst: str, inp: str) -> str:
    if inp.startswith('输入：'):
        inp = inp[3:]
    return f'{inst}\n{inp}'


register_dataset(
    DatasetName.alpaca_zh,
    'AI-ModelScope/alpaca-gpt4-data-zh',
    None,
    AlpacaPreprocessor(concat_inst_inp=_concat_inst_inp_alpaca_zh),
    HubDatasetLoader.dataset_get_function,
    tags=['chat', 'general', '🔥'],
    hf_dataset_id='llm-wizard/alpaca-gpt4-data-zh')


class ShareGPT4oPreprocessor(RowPreprocessor):

    column_mapping: Dict[str, str] = {}
    modals: List[str] = ['image']
    modal_tags: List[str] = {'image': '<image>'}
    modal_keys: List[str] = {'image': 'image'}
    task_type: Optional[str] = 'vqa'

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        image = row['image']
        if not image:
            return self.empty_row()
        image = os.path.join(self.prefix_path, image)
        if not os.path.exists(image):
            return self.empty_row()
        row = ConversationsPreprocessor(
                user_role='human', assistant_role='gpt', media_type='image', error_strategy='delete',
                modals=['image'], modal_keys={'image': 'image'}).preprocess(row)
        row['image'] = [image]
        return row

    def prepare_downloading(self, dataset):
        url = 'https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/ShareGPT-4o/repo?Revision=master&FilePath=images.zip'
        local_dir = MediaResource.download(url, 'sharegpt_4o_images')
        self.prefix_path = os.path.join(local_dir, 'mnt', 'petrelfs', 'wangwenhai', 'workspace_cef', '4o', 'image')



class GPT4vDataset(RowPreprocessor):

    modals = ['image']

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'messages': [
                {'role': 'user', 'content': 'What is the caption of this image?'},
                {'role': 'assistant', 'content': row['caption']}
            ],
            'images': row['link']
        }


register_dataset(
    DatasetName.gpt4v_dataset,
    'swift/gpt4v-dataset', ['default'],
    GPT4vDataset(),
    HubDatasetLoader.dataset_get_function,
    split=['train'],
    tags=["en", "caption", "multi-modal", "quality"],
    hf_dataset_id='laion/gpt4v-dataset')


register_dataset(
    DatasetName.sharegpt_4o_image,
    'AI-ModelScope/ShareGPT-4o', ['image_caption'],
    ShareGPT4oPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['images'],
    tags=['vqa', 'multi-modal'],
    hf_dataset_id='OpenGVLab/ShareGPT-4o')


class SA1BPairedCaptionPreprocessor(RowPreprocessor):

    column_mapping = {
            'opensource_url': 'images',
        }

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ['图片中展示了什么', '讲述一下图片中内容', '告诉我里面有什么', '图片内容是啥']
        response = row['global_caption']
        query = np.random.choice(prompt)
        return {
            'messages': [
                {
                    'role': 'user',
                    'content': query,
                },
                {
                    'query': 'assistant',
                    'content': response,
                }
            ]
        }


register_dataset(
    DatasetName.sa1b_paired_caption,
    'Tongyi-DataEngine/SA1B-Paired-Captions-Images',
    None,
    SA1BPairedCaptionPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['train'],
    huge_dataset=True,
    tags=['zh', 'multi-modal', 'vqa'])


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
            'messages': [
                {
                    'role': 'user',
                    'content': query,
                },
                {
                    'query': 'assistant',
                    'content': response,
                }
            ]
        }


register_dataset(
    DatasetName.sa1b_dense_caption,
    'Tongyi-DataEngine/SA1B-Dense-Caption',
    None,
    SA1BDenseCaptionPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['train'],
    huge_dataset=True,
    tags=['zh', 'multi-modal', 'vqa'])


class COCO2014Preprocess(RowPreprocessor):

    modals = ['image']
    modal_keys = {'image': 'image'}

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = 'please describe the image.'
        image_key = 'image'
        response_key = 'caption'
        query_format = f'<img>{{image_path}}</img>{prompt}'
        if '&&' in row[response_key]:
            row[response_key] = row[response_key].split('&&')[0]

        return {
            'messages': [
                {
                    'role': 'user',
                    'content': query_format.format(image_path=row[image_key]['path']),
                },
                {
                    'query': 'assistant',
                    'content': row[response_key],
                }
            ]
        }

    def __call__(self, dataset, **kwargs):
        from datasets import Image
        dataset = super(COCO2014Preprocess, self).__call__(dataset.cast_column('image', Image(decode=False)), **kwargs)
        return dataset.remove_columns(['images'])


register_dataset(
    DatasetName.coco_en,
    'modelscope/coco_2014_caption', ['coco_2014_caption'],
    COCO2014Preprocess(),
    HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    tags=['chat', 'multi-modal', 'vision'],
    is_main=False)

register_dataset(
    DatasetName.coco_en_mini,
    'modelscope/coco_2014_caption', ['coco_2014_caption'],
    COCO2014Preprocess(),
    HubDatasetLoader.dataset_get_function,
    split=['validation'],
    tags=['chat', 'multi-modal', 'vision', '🔥'],
    is_main=False)


def preprocess_mantis_image(dataset, subset):
    url = f'https://www.modelscope.cn/api/v1/datasets/swift/Mantis-Instruct/repo?Revision=master&FilePath={subset}/train_images.zip'  # noqa
    local_dir = MediaResource.download(url, f'mantis_{subset}')

    def preprocess_row(row):
        images = [os.path.join(local_dir, p['path']) for p in row['images']]
        if all([os.path.exists(d) for d in images]):
            return {'images': images}
        else:
            return {'images': []}

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess_row, **kwargs).filter(lambda row: row['images'])


def get_mantis_dataset(dataset_id: str,
                       subsets: Optional[List[str]],
                       preprocess_func: PreprocessFunc,
                       split: List[str],
                       dataset_sample: int = -1,
                       *,
                       random_state: Optional[RandomState] = None,
                       dataset_test_ratio: float = 0.,
                       remove_useless_columns: bool = True,
                       use_hf: bool = False,
                       **kwargs) -> Tuple[HfDataset, Optional[HfDataset]]:
    streaming = kwargs.get('streaming', False)
    if subsets is None:
        subsets = []
    assert len(split) > 0
    if len(subsets) == 0:
        subset_split_list = split
    else:
        subset_split_list = list(itertools.product(subsets, split))
    all_datasets = []
    for subset in subset_split_list:
        dataset = HubDatasetLoader.load_dataset_from_hub(dataset_id, [subset], use_hf, streaming=streaming)
        dataset = preprocess_mantis_image(dataset, subset=subset[0])
        all_datasets.append(dataset)
        break
    if len(all_datasets) > 1:
        dataset = concatenate_datasets(all_datasets) if not streaming else interleave_datasets(all_datasets)
    else:
        dataset = all_datasets[0]
    return HubDatasetLoader.post_preprocess(dataset, dataset_sample, random_state, preprocess_func, dataset_test_ratio,
                            remove_useless_columns, **kwargs)


register_dataset(
    DatasetName.mantis_instruct,
    'swift/Mantis-Instruct', [
        'birds-to-words', 'chartqa', 'coinstruct', 'contrastive_caption', 'docvqa', 'dreamsim', 'dvqa', 'iconqa',
        'imagecode', 'llava_665k_multi', 'lrv_multi', 'multi_vqa', 'nextqa', 'nlvr2', 'spot-the-diff', 'star',
        'visual_story_telling'
    ],
    ConversationsPreprocessor(
        user_role='user',
        assistant_role='assistant',
        conversations_key='conversation',
        from_key='role',
        value_key='content',
        media_type='image',
        media_key='images',
        error_strategy='delete'),
    get_mantis_dataset,
    split=['train'],
    tags=['chat', 'multi-modal', 'vision', 'quality'],
    hf_dataset_id='TIGER-Lab/Mantis-Instruct')


class LLaVADataPreprocessor(RowPreprocessor):

    modals = ['image']

    def prepare_downloading(self, dataset):
        self.all_folders = {}
        for media_type in ['coco', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2']:
            self.all_folders[media_type] = MediaResource.download(media_type)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if not row['images']:
            return {}
        
        row.update(ConversationsPreprocessor(
            user_role='user',
            assistant_role='assistant',
            conversations_key='conversation',
            from_key='role',
            value_key='content').preprocess(row))
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
            row['images'] = []
        return row


register_dataset(
    DatasetName.llava_data_instruct,
    'swift/llava-data', ['llava_instruct'],
    LLaVADataPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['train'],
    tags=['sft', 'multi-modal', 'quality'],
    hf_dataset_id='TIGER-Lab/llava-data')


class COCOEn2Preprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = 'please describe the image.'
        image_key = 'image'
        response_key = 'caption'
        images = [row[image_key]['path']]
        if '&&' in row[response_key]:
            row[response_key] = row[response_key].split('&&')[0]
        response = row[response_key]
        return {'messages': [
            {
                'role': 'user',
                'content': query,
            },
            {
                'role': 'assistant',
                'content': response,
            }
        ], 'images': images}

    def __call__(self, dataset, **kwargs):
        from datasets import Image
        dataset = dataset.cast_column('image', Image(decode=False))
        return super().__call__(dataset)


register_dataset(
    DatasetName.coco_en_2,
    'modelscope/coco_2014_caption', ['coco_2014_caption'],
    COCOEn2Preprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    tags=['chat', 'multi-modal', 'vision'],
    is_main=False)

register_dataset(
    DatasetName.coco_en_2_mini,
    'modelscope/coco_2014_caption', ['coco_2014_caption'],
    COCOEn2Preprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['validation'],
    tags=['chat', 'multi-modal', 'vision', '🔥'],
    is_main=False)


class PixelProsePreprocessor(RowPreprocessor):

    models = ['image']

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        caption_prompt = [
            'Give the description of this image.', 'Describe this picture', 'What is the proper title of this image?'
        ]
        vlm_caption = row['vlm_caption']
        if vlm_caption.startswith('This image displays:'):
            vlm_caption = vlm_caption[len('This image displays:'):].strip()
        return {
            'messages': [
                {'role': 'user', 'content': np.random.choice(caption_prompt)},
                {'role': 'assistant', 'content': vlm_caption}
            ],
            'images': row['url']
        }


register_dataset(
    DatasetName.pixelprose,
    'swift/pixelprose',
    None,
    PixelProsePreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['train', 'cc12m', 'commonpool', 'redcaps'],
    hf_dataset_id='tomg-group-umd/pixelprose',
    tags=['caption', 'multi-modal', 'vision'],
    huge_dataset=True,
    is_main=False)


class AIShell1Preprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = '语音转文本'
        audio_key = 'Audio:FILE'
        response_key = 'Text:LABEL'
        query_format = f'<audio>{{audio_path}}</audio>{prompt}'
        return {
            'messages': [
                {'role': 'user', 'content': query_format.format(audio_path=row[audio_key])},
                {'role': 'assistant', 'content': row[response_key].replace(' ', '')}
            ],
        }


register_dataset(
    DatasetName.aishell1_zh,
    'speech_asr/speech_asr_aishell1_trainsets',
    None,
    AIShell1Preprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['train', 'validation', 'test'],
    tags=['chat', 'multi-modal', 'audio'])

register_dataset(
    DatasetName.aishell1_zh_mini,
    'speech_asr/speech_asr_aishell1_trainsets',
    None,
    AIShell1Preprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['validation', 'test'],
    tags=['chat', 'multi-modal', 'audio', '🔥'],
    is_main=False)


class VideoChatGPTPreprocessor(RowPreprocessor):

    modals = ['video']

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        url = 'https://modelscope.cn/datasets/swift/VideoChatGPT/resolve/master/videos.zip'
        local_dir = MediaResource.download(url, 'video_chatgpt')
        local_dir = os.path.join(local_dir, 'Test_Videos')
        # only `.mp4`
        mp4_set = [file[:-4] for file in os.listdir(local_dir) if file.endswith('mp4')]
        if row['video_name'] not in mp4_set:
            return self.empty_row()
        return {
            'messages': [
                {'role': 'user', 'content': row['question'] or row['question_1'] or row['question_2']},
                {'role': 'assistant', 'content': row['answer']}
            ],
            'videos': [os.path.join(local_dir, f"{row['video_name']}.mp4")],
        }


register_dataset(
    DatasetName.video_chatgpt,
    'swift/VideoChatGPT', ['Generic', 'Temporal', 'Consistency'],
    VideoChatGPTPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['test'],
    hf_dataset_id='lmms-lab/VideoChatGPT',
    tags=['chat', 'multi-modal', 'video', '🔥'])


class LongAlpacaPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        response = row['response']
        if response and response.startswith('Answer:'):
            response = response[len('Answer:') + 1:].strip()
            row['response'] = response
        return row


register_dataset(
    DatasetName.long_alpaca_12k,
    'AI-ModelScope/LongAlpaca-12k',
    None,
    ComposePreprocessor([
        AlpacaPreprocessor(),
        LongAlpacaPreprocessor()
    ]),
    HubDatasetLoader.dataset_get_function,
    tags=['longlora', 'QA'],
    hf_dataset_id='Yukang/LongAlpaca-12k')


class RuozhibaPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        title = row['title'] if row.get('title', None) is not None else row['content']
        abs = row['abs'] if 'abs' in row else None
        if abs and abs != title:
            title = title + '，' + abs

        pattern = r'\d+[\.,\s,\、](.+)'
        match = re.search(pattern, title)
        if match:
            title = match.group(1)
        if title:
            return {'messages': {'role': 'assistant', 'content': title}}
        else:
            return self.empty_row()


register_dataset(
    DatasetName.ruozhiba,
    'AI-ModelScope/ruozhiba', ['post-annual', 'title-good', 'title-norm'],
    RuozhibaPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    tags=['pretrain', '🔥'])


def _repair_ms_bench(conversations: str) -> Optional[List[Dict[str, str]]]:
    if isinstance(conversations, str):
        conversations = ast.literal_eval(conversations)
    default_system = 'You are a helpful assistant.'
    conversations: List[Dict[str, str]]
    if conversations[0]['from'] == 'system' and conversations[0]['value'] == default_system:
        conversations.pop(0)
    # skip MOSS
    for c in conversations:
        value = c['value'].lower()
        if 'moss' in value or 'human:' in value or 'assistant:' in value or 'user:' in value:
            return
    return conversations


register_dataset(
    DatasetName.ms_bench,
    'iic/ms_bench',
    None,
    ConversationsPreprocessor(repair_conversations=_repair_ms_bench, error_strategy='delete'),
    HubDatasetLoader.dataset_get_function,
    tags=['chat', 'general', 'multi-round', '🔥'])


def _repair_agent_conversations(conversations: str, use_mini: bool) -> Optional[List[Dict[str, str]]]:
    if use_mini:
        pattern = r'\d\. {"plugin_name": "(.+?)"'
    else:
        pattern = r'\d\. {"(?:plugin_)?name": "(.+?)"'

    idx = conversations.find(r"'from': 'user")
    if idx == -1:
        return
    # remove dirty data
    find_list = re.findall(pattern, conversations[:idx])
    if len(set(find_list)) <= 1:
        return
    if isinstance(conversations, str):
        conversations = ast.literal_eval(conversations)
    if len(conversations) == 1:
        return
    return conversations


register_dataset(
    DatasetName.damo_agent_zh_mini,
    'damo/MSAgent-Bench',
    None,
    ConversationsPreprocessor(
        repair_conversations=partial(_repair_agent_conversations, use_mini=True), error_strategy='delete'),
    HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    tags=['chat', 'agent', 'multi-round'],
    is_main=False)
register_dataset(
    DatasetName.damo_agent_zh,
    'damo/MSAgent-Bench',
    None,
    ConversationsPreprocessor(
        repair_conversations=partial(_repair_agent_conversations, use_mini=False), error_strategy='delete'),
    HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    tags=['chat', 'agent', 'multi-round'])

advertise_gen_prompt = """Task: Generating advertisements based on keywords.
Keywords: {query}
Advertisements:"""
register_dataset(
    DatasetName.advertise_gen_zh,
    'lvjianjin/AdvertiseGen',
    None,
    TextGenerationPreprocessor(prompt=advertise_gen_prompt, query_key='content', response_key='summary'),
    HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    tags=['text-generation', '🔥'],
    hf_dataset_id='shibing624/AdvertiseGen')


class FireflyPreprocessor(RowPreprocessor):

    _firefly_kind_list = {
        'ProseGeneration', 'MRC', 'JinYongGeneration', 'TextCorrection', 'ClassicalChinese', 'BELLE', 'StoryGeneration',
        'Couplet', 'Cot', 'Dictionary', 'Translation', 'Program', 'SentimentAnalyze', 'OpenQA', 'AncientPoem',
        'TextMatching', 'NLI', 'Summary', 'KeywordRecognition', 'ProductDesc', 'LyricGeneration', 'Composition',
        'MusicComment', 'NER'
    }

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if row['kind'] not in FireflyPreprocessor._firefly_kind_list:
            return self.empty_row()
        return {
            'messages': [
                {'role': 'user', 'content': row['input']},
                {'role': 'assistant', 'content': row['target']}
            ]
        }


@register_dataset(
    DatasetName.firefly_zh,
    'AI-ModelScope/firefly-train-1.1M',
    None,
    FireflyPreprocessor(),
    tags=['chat', 'general'],
    hf_dataset_id='YeungNLP/firefly-train-1.1M')
def get_firefly_zh_dataset(dataset_id: str, _, preprocess_func: PreprocessFunc, *args, **kwargs) -> HfDataset:
    file = 'firefly-train-1.1M.jsonl'
    dataset_dir = DatasetLoader.download_dataset(dataset_id, [file])
    fpath = os.path.join(dataset_dir, file)
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()
        text = text.replace('}{', '},{')
        text = f'[{text}]'
        dataset = json.loads(text)
    return preprocess_func(dataset, **kwargs)


register_dataset(
    DatasetName.cmnli_zh,
    'modelscope/clue', ['cmnli'],
    ClsPreprocessor(['neutral', 'entailment', 'contradiction'], 'Natural Language Inference', True),
    HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    tags=['text-generation', 'classification'],
    hf_dataset_id='clue')

register_dataset(
    DatasetName.jd_sentiment_zh,
    'DAMO_NLP/jd',
    None,
    ClsPreprocessor(['negative', 'positive'], 'Sentiment Classification', False),
    HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    tags=['text-generation', 'classification', '🔥'])


class DureaderPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = """Task: Question Generation
Context: {context}
Answer: {answer}
Question:"""
        answer, context = row['text1'].split('[SEP]')
        return {'messages': {
            {'role': 'user', 'content': prompt.format(context=context, answer=answer)},
            {'role': 'assistant', 'content': row['text2']}
        }}


register_dataset(
    DatasetName.dureader_robust_zh,
    'modelscope/DuReader_robust-QG',
    None,
    DureaderPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['train', 'validation', 'test'],
    tags=['text-generation', '🔥'])


class HHRLHFPreprocessor(RowPreprocessor):

    def empty_row(self):
        row = super().empty_row()
        row['rejected_response'] = None

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        import re
        chosen = row['chosen'].strip()
        rejected = row['rejected'].strip()
        parts_chosen = [s.strip() for s in re.split('\n\nHuman:|\n\nAssistant:|\n\nHum:', chosen)]
        parts_rejected = [s.strip() for s in re.split('\n\nHuman:|\n\nAssistant:|\n\nHum:', rejected)]
        if parts_chosen[0].startswith('Human:'):
            assert parts_rejected[0].startswith('Human:')
            parts_chosen[0] = parts_chosen[0][6:].strip()
            parts_rejected[0] = parts_rejected[0][6:].strip()
        history = []
        idx, s1, s2 = None, None, None
        for idx, (s1, s2) in enumerate(zip(parts_chosen, parts_rejected)):
            if s1 == s2:
                if idx % 2 == 0:
                    history.append([s1, None])
                else:
                    history[-1][-1] = s1
            else:
                break

        if idx % 2 == 0:
            return self.empty_row()

        messages = []
        for h in history:
            messages.append({'role': 'user', 'content': h[0]})
            messages.append({'role': 'assistant', 'content': h[1]})

        messages[-1]['content'] = s1
        return {
            'messages': messages,
            'rejected_response': s2,
        }


register_dataset(
    DatasetName.hh_rlhf,
    'AI-ModelScope/hh-rlhf', ['harmless-base', 'helpful-base', 'helpful-online', 'helpful-rejection-sampled'],
    HHRLHFPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['train', 'test'],
    tags=['rlhf', 'dpo', 'pairwise'])


class HHRLHFCNPreprocessor(RowPreprocessor):

    def empty_row(self):
        row = super().empty_row()
        row['rejected_response'] = None

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        history = []
        try:
            if isinstance(row['context'], str):
                row['context'] = ast.literal_eval(row['context'])
            if isinstance(row['chosen'], str):
                row['chosen'] = ast.literal_eval(row['chosen'])
            if isinstance(row['rejected'], str):
                row['rejected'] = ast.literal_eval(row['rejected'])
            for idx, h in enumerate(row['context']):
                if idx % 2 == 0 and h['role'] != 'human':
                    raise ValueError()
                if idx % 2 != 0 and h['role'] != 'assistant':
                    raise ValueError()
                if idx % 2 == 0:
                    history.append([h['text'], None])
                else:
                    history[-1][-1] = h['text']

            if history[-1][-1] is not None:
                raise ValueError()
        except:  # noqa
            return self.empty_row()
        else:
            messages = []
            for h in history:
                messages.append({'role': 'user', 'content': h[0]})
                messages.append({'role': 'assistant', 'content': h[1]})

            messages[-1]['content'] = row['chosen']['text']
            return {
                'messages': messages,
                'rejected_response': row['rejected']['text'],
            }

    def filter_valid_row(self, row):
        try:
            if isinstance(row['context'], str):
                row['context'] = ast.literal_eval(row['context'])
            if isinstance(row['chosen'], str):
                row['chosen'] = ast.literal_eval(row['chosen'])
            if isinstance(row['rejected'], str):
                row['rejected'] = ast.literal_eval(row['rejected'])
            return True
        except:  # noqa
            return False

    def __call__(self, dataset, **kwargs):
        dataset = dataset.filter(self.filter_valid_row, **kwargs)
        return super(HHRLHFCNPreprocessor, self).__call__(dataset, **kwargs)


register_dataset(
    DatasetName.hh_rlhf_cn,
    'AI-ModelScope/hh_rlhf_cn',
    ['hh_rlhf', 'harmless_base_cn', 'harmless_base_en', 'helpful_base_cn', 'helpful_base_en'],
    HHRLHFCNPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['train', 'test'],
    tags=['rlhf', 'dpo', 'pairwise', '🔥'])


class M3ITPreprocessor(RowPreprocessor):

    column_mapping = {'instruction': 'system', 'inputs': 'query', 'image_base64_str': 'images', 'outputs': 'response'}


register_dataset(
    DatasetName.m3it,
    'AI-ModelScope/M3IT',  # error: 'vist' , 'iqa-rephrased ', 'mmchat' / test: 'winoground','chinese-food'
    [
        'coco', 'vqa-v2', 'shapes', 'shapes-rephrased', 'coco-goi-rephrased', 'snli-ve', 'snli-ve-rephrased', 'okvqa',
        'a-okvqa', 'viquae', 'textcap', 'docvqa', 'science-qa', 'imagenet', 'imagenet-open-ended', 'imagenet-rephrased',
        'coco-goi', 'clevr', 'clevr-rephrased', 'nlvr', 'coco-itm', 'coco-itm-rephrased', 'vsr', 'vsr-rephrased',
        'mocheg', 'mocheg-rephrased', 'coco-text', 'fm-iqa', 'activitynet-qa', 'msrvtt', 'ss', 'coco-cn', 'refcoco',
        'refcoco-rephrased', 'multi30k', 'image-paragraph-captioning', 'visual-dialog', 'visual-dialog-rephrased',
        'iqa', 'vcr', 'visual-mrc', 'ivqa', 'msrvtt-qa', 'msvd-qa', 'gqa', 'text-vqa', 'ocr-vqa', 'st-vqa',
        'flickr8k-cn'
    ],
    M3ITPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['train'],
    huge_dataset=True,
    tags=['chat', 'multi-modal', 'vision'])


class ShareGPT4VPreprocessor(RowPreprocessor):

    modals = ['image']

    def prepare_downloading(self, dataset):
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

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        image = row['image']
        row.update(ConversationsPreprocessor(
                user_role='human', assistant_role='gpt', error_strategy='delete').preprocess(row))
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
            row['images'] = None
        return row


register_dataset(
    DatasetName.sharegpt4v,
    'AI-ModelScope/ShareGPT4V', ['ShareGPT4V', 'ShareGPT4V-PT'],
    ShareGPT4VPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['train'],
    huge_dataset=True,
    tags=['chat', 'multi-modal', 'vision'])


class TextCapsPreprocessor(RowPreprocessor):

    modals = ['image']
    modal_keys = {'image': 'image'}

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image = row['image']
            query = 'What is the caption of this image?'
            response = row['reference_strs']
            return {
                'messages': [
                    {'role': 'user', 'content': query},
                    {'role': 'assistant', 'content': response[np.random.choice(range(len(response)))]},
                ],
                'image': image
            }
        except Exception:
            return self.empty_row()


register_dataset(
    DatasetName.text_caps,
    'swift/TextCaps', [],
    preprocess_func=TextCapsPreprocessor(),
    get_function=HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    hf_dataset_id='HuggingFaceM4/TextCaps',
    huge_dataset=True,
    tags=['multi-modal', 'en', 'caption', 'quality'])


class RefCOCOCaptionPreprocessor(RowPreprocessor):

    task_type = 'caption'
    modals = ['image']

    def prepare_downloading(self, dataset):
        self.cache_dir = MediaResource.download(
            'https://www.modelscope.cn/api/v1/datasets/we_dont_produce_water/'
            'coco_res/repo?Revision=master&FilePath=coco_2014.zip', 'coco2014')

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        caption = row['captions'][0]
        bbox = row['bbox']
        image_path = os.path.join(self.cache_dir, row['image_path'].replace('coco/train2014', 'train2014'))
        for i in range(len(bbox)):
            bbox[i] = round(float(bbox[i]))
        res = {}

        objects = [{
            'caption': caption,
            'bbox': bbox,
            'bbox_type': 'real',
            'image': 0,
        }]
        res['images'] = [image_path]
        res['objects'] = json.dumps(objects, ensure_ascii=False)
        return res

    def filter(self, row: Dict[str, Any]) -> Dict[str, Any]:
        image_path = row['images'][0]
        return os.path.exists(image_path)


register_dataset(
    DatasetName.refcoco_unofficial_caption,
    'swift/refcoco', [],
    preprocess_func=RefCOCOCaptionPreprocessor(),
    get_function=HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    hf_dataset_id='jxu124/refcoco',
    tags=['multi-modal', 'en', 'caption'])

register_dataset(
    DatasetName.refcocog_unofficial_caption,
    'swift/refcocog', [],
    preprocess_func=RefCOCOCaptionPreprocessor(),
    get_function=HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    hf_dataset_id='jxu124/refcocog',
    tags=['multi-modal', 'en', 'caption'])


class RefCOCOGroundingPreprocessor(RefCOCOCaptionPreprocessor):
    task_type = 'grounding'


register_dataset(
    DatasetName.refcoco_unofficial_grounding,
    'swift/refcoco', [],
    preprocess_func=RefCOCOGroundingPreprocessor(),
    get_function=HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    hf_dataset_id='jxu124/refcoco',
    tags=['multi-modal', 'en', 'grounding'])

register_dataset(
    DatasetName.refcocog_unofficial_grounding,
    'swift/refcocog', [],
    preprocess_func=RefCOCOGroundingPreprocessor(),
    get_function=HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    hf_dataset_id='jxu124/refcocog',
    tags=['multi-modal', 'en', 'grounding'])

register_dataset(
    DatasetName.lnqa,
    'swift/lnqa', [],
    preprocess_func=ListPreprocessor(query_key='question', response_key='answer', media_type='image'),
    get_function=HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    hf_dataset_id='vikhyatk/lnqa',
    huge_dataset=True,
    tags=['multi-modal', 'en', 'ocr-vqa', 'quality'])


class LLaVAInstructPreprocessor(RowPreprocessor):

    modals = ['image']

    def prepare_downloading(self, dataset):
        self.all_folders = {}
        for media_type in ['coco', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2']:
            self.all_folders[media_type] = MediaResource.download(media_type)

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row.update(ConversationsPreprocessor(
            user_role='human', assistant_role='gpt', media_type='image', media_key='images', error_strategy='delete').preprocess(row))
        image = row['image']
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
            row['images'] = None
        return row

    def filter(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return super().filter(row) and row.get('images')


register_dataset(
    DatasetName.llava_instruct_150k,
    'AI-ModelScope/LLaVA-Instruct-150K',
    None,
    LLaVAInstructPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['train'],
    revision='d5db3806e395c60496630a206c336932e85a2d00',
    tags=['chat', 'multi-modal', 'vision'])


def repair_conversations(s: Union[str, Any]) -> Any:
    if isinstance(s, str):
        s = s.replace('}\n {', '},{')
        s = s.replace('}\n{', '},{')
        s = s.replace('}{', '},{')
        s = s.replace('}\n  {', '},{')
        return ast.literal_eval(s)
    return s


register_dataset(
    DatasetName.lmsys_chat_1m,
    'AI-ModelScope/lmsys-chat-1m',
    None,
    ConversationsPreprocessor(
        user_role='user',
        assistant_role='assistant',
        conversations_key='conversation',
        from_key='role',
        value_key='content',
        error_strategy='delete',
        repair_conversations=repair_conversations),
    HubDatasetLoader.dataset_get_function,
    hf_dataset_id='lmsys/lmsys-chat-1m',
    tags=['chat', 'em'])


class LLaVAPretrainPreprocessor(RowPreprocessor):

    modals = ['image']
    modal_keys = {'image': 'image'}

    def prepare_downloading(self, dataset):
        self.media_dir = MediaResource.download(
            'https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/LLaVA-Pretrain/repo?Revision=master&FilePath=images.zip',
            # noqa
            'llava_pretrain')

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row.update(ConversationsPreprocessor(
                user_role='human', assistant_role='gpt', media_type='image', error_strategy='delete').preprocess(row))   
        if row['image']:
            file_path = os.path.join(self.media_dir, row['image'])
            if os.path.exists(file_path):
                return {'image': file_path}
            else:
                return {'image': ''}
        else:
            return {'image': ''}

    def filter(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return row.get('image')


register_dataset(
    DatasetName.llava_pretrain,
    'AI-ModelScope/LLaVA-Pretrain', ['default'],
    LLaVAPretrainPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['train'],
    hf_dataset_id='liuhaotian/LLaVA-Pretrain',
    huge_dataset=True,
    revision='e3a3f0bfaad05e90e46745152a32bf944e0f4a63',
    tags=['vqa', 'multi-modal', 'quality'])


class ShareAIDPOPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'messages': [
                {'role': 'user', 'content': row['question']},
                {'role': 'assistant', 'content': row['answer_zh']},
            ],
            'rejected_response': row['answer_en'],
        }


register_dataset(
    DatasetName.shareai_llama3_dpo_zh_en_emoji,
    'hjh0119/shareAI-Llama3-DPO-zh-en-emoji', ['default'],
    ShareAIDPOPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    tags=['rlhf', 'dpo', 'pairwise'])


class UltraFeedbackKTOPreprocessor(RowPreprocessor):
    column_mapping = {'prompt': 'query', 'completion': 'response'}


register_dataset(
    DatasetName.ultrafeedback_kto,
    'AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto', ['default'],
    UltraFeedbackKTOPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    remove_useless_columns=False,
    tags=['rlhf', 'kto'])


class ZhihuKOLPreprocessor(RowPreprocessor):
    column_mapping = {'INSTRUCTION': 'query', 'RESPONSE': 'response'}


register_dataset(
    DatasetName.zhihu_kol_filtered,
    'OmniData/Zhihu-KOL-More-Than-100-Upvotes', ['default'],
    ZhihuKOLPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    hf_dataset_id='bzb2023/Zhihu-KOL-More-Than-100-Upvotes',
    tags=['zhihu', 'qa'])

register_dataset(
    DatasetName.zhihu_kol,
    'OmniData/Zhihu-KOL', ['default'],
    ZhihuKOLPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    hf_dataset_id='wangrui6/Zhihu-KOL',
    huge_dataset=True,
    tags=['zhihu', 'qa'])


class GuanacoPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        from swift.utils.utils import split_str_parts_by
        instruction = row['instruction']
        input = row['input']
        output = row['output']
        history = []
        if instruction:
            parts = split_str_parts_by(
                instruction, ['User:', 'User：', 'Assistant：', 'Assistant:', 'Asssistent:', 'Assistent:', 'Assistenz:'])
            for idx, part in enumerate(parts):
                if idx % 2 == 0:
                    if 'user' not in part['key'].lower():
                        return {'query': '', 'history': [], 'response': ''}
                    history.append([part['content'], None])
                else:
                    if 'assist' not in part['key'].lower() and 'asssist' not in part['key'].lower():
                        return {'query': '', 'history': [], 'response': ''}
                    history[-1][-1] = part['content']
        if input.startswith('User:'):
            input = input[len('User:'):].strip()
        if any([not h[0] or not h[1] for h in history]):
            return self.empty_row()

        messages = []
        for h in history:
            messages.append({'role': 'user', 'content': h[0]})
            messages.append({'role': 'assistant', 'content': h[1]})
        messages.append({'role': 'user', 'content': input})
        messages.append({'role': 'assistant', 'content': output})
        return {
            'messages': messages,
        }


register_dataset(
    DatasetName.guanaco,
    'AI-ModelScope/GuanacoDataset', ['default'],
    GuanacoPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    hf_dataset_id='JosephusCheung/GuanacoDataset',
    tags=['chat', 'zh'])


class Dolly15kPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        instruction = row['instruction']
        context = row['context']
        response = row['response']
        query = ''
        if context:
            query = 'Here gives some useful information:\n'
            query += context
            query += '\n'
        query += instruction
        return {
            'messages': [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': response}
            ],
        }


register_dataset(
    DatasetName.dolly_15k,
    'AI-ModelScope/databricks-dolly-15k', ['default'],
    Dolly15kPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    hf_dataset_id='databricks/databricks-dolly-15k',
    tags=['multi-task', 'en', 'quality'])


register_dataset(
    DatasetName.midefics,
    'swift/MideficsDataset', [],
    ListPreprocessor(
        conversations_key='conversation',
        query_key='question',
        response_key='answer',
        inner_key='data',
        modals=['image'],
        modal_keys=['image']),
    HubDatasetLoader.dataset_get_function,
    hf_dataset_id='WinterSchool/MideficsDataset',
    tags=['medical', 'en', 'vqa'])


class OkvqaPreprocessor(RowPreprocessor):

    column_mapping = {'image': 'images'}

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row['question']
        response = np.random.choice(row['answers'])
        return {
            'messages': [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': response}
            ],
        }


register_dataset(
    DatasetName.okvqa,
    'swift/OK-VQA_train', [],
    preprocess_func=OkvqaPreprocessor(),
    get_function=HubDatasetLoader.dataset_get_function,
    split=['train'],
    hf_dataset_id='Multimodal-Fatima/OK-VQA_train',
    tags=['multi-modal', 'en', 'vqa', 'quality'])


class AOkvqaPreprocessor(RowPreprocessor):

    column_mapping = {'image': 'images'}

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row['question']
        response = np.random.choice(row['rationales'])
        return {
            'messages': [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': response}
            ],
        }


register_dataset(
    DatasetName.a_okvqa,
    'swift/A-OKVQA', [],
    preprocess_func=AOkvqaPreprocessor(),
    get_function=HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    hf_dataset_id='HuggingFaceM4/A-OKVQA',
    tags=['multi-modal', 'en', 'vqa', 'quality'])


class OcrvqaPreprocessor(RowPreprocessor):

    modals = ['image']
    modal_keys = {'image': 'image'}
    column_mapping = {'image': 'images'}

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        idx = np.random.choice(range(len(row['questions'])))
        query = row['questions'][idx]
        response = row['answers'][idx]
        return {
            'messages': [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': response}
            ],
        }


register_dataset(
    DatasetName.ocr_vqa,
    'swift/OCR-VQA', [],
    preprocess_func=OcrvqaPreprocessor(),
    get_function=HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    hf_dataset_id='howard-hou/OCR-VQA',
    tags=['multi-modal', 'en', 'ocr-vqa'])


class ScienceQAPreprocessor(RowPreprocessor):
    modals = ['image']
    modal_keys = {'image': 'image'}
    column_mapping = {'image': 'images'}

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row['question']
        response = row['choices'][row['answer']]
        solution = row['solution']
        response = f'{solution}\nSo the final answer is: {response}'
        return {
            'messages': [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': response}
            ]
        }


register_dataset(
    DatasetName.science_qa,
    'swift/ScienceQA', [],
    preprocess_func=ScienceQAPreprocessor(),
    get_function=HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    hf_dataset_id='derek-thomas/ScienceQA',
    tags=['multi-modal', 'science', 'vqa', 'quality'])


class GritPreprocessor(RowPreprocessor):

    modals = ['image']

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

    def empty_row(self):
        row = super().empty_row()
        row['objects'] = None
        return row

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        images = row['url']
        caption = row['caption']
        ref_exps = row['ref_exps']
        objects = []
        start_end_pairs = []
        for ref_exp in ref_exps:
            start = ref_exp[0]
            end = ref_exp[1]
            # conf = ref_exp[6] TODO filter low confidence rows?
            start_end_pairs.append(ref_exp[0:2])

            object_part = caption[int(start):int(end)]
            objects.append({'caption': object_part, 'bbox': ref_exp[2:6], 'bbox_type': 'real', 'image': 0})

        start_end_pairs.sort(key=lambda x: (x[0], x[1]))
        if self.has_overlap(start_end_pairs):
            return self.empty_row()

        response = self.replace_intervals_with_tags(caption, start_end_pairs)
        return {
            'messages': [
                {'role': 'user', 'content': 'what is the proper caption of this image?'},
                {'role': 'assistant', 'content': response}
            ],
            'images': images,
            'objects': json.dumps(objects or [], ensure_ascii=False)
        }

    def filter(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return super().filter(row) and row.get('objects')


register_dataset(
    DatasetName.grit,
    'swift/GRIT', [],
    preprocess_func=GritPreprocessor(),
    get_function=HubDatasetLoader.dataset_get_function,
    split=['train'],
    hf_dataset_id='zzliang/GRIT',
    huge_dataset=True,
    tags=['multi-modal', 'en', 'caption-grounding', 'quality'])


class GQAPreprocessor(RowPreprocessor):

    modals = ['image']

    def prepare_downloading(self, dataset):
        self.local_cache = MediaResource.download('gqa')

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if os.path.join(self.local_cache, 'images', row['imageId'] + '.jpg'):
            return {
                'messages': [
                    {'role': 'user', 'content': row['question']},
                    {'role': 'assistant', 'content': row['fullAnswer']}
                ],
                'images': os.path.join(self.local_cache, 'images', row['imageId'] + '.jpg'),
            }
        else:
            return self.empty_row()


register_dataset(
    DatasetName.gqa,
    None, ['train_all_instructions'],
    GQAPreprocessor(),
    get_function=HubDatasetLoader.dataset_get_function,
    hf_dataset_id='lmms-lab/GQA',
    huge_dataset=True,
    tags=['multi-modal', 'en', 'vqa', 'quality'])


class LLaVAMixSFTPreprocessor(RowPreprocessor):

    modals = ['image']

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
    DatasetName.llava_instruct_mix,
    'swift/llava-instruct-mix-vsft', [],
    LLaVAMixSFTPreprocessor(),
    get_function=HubDatasetLoader.dataset_get_function,
    split=['test'],
    hf_dataset_id='HuggingFaceH4/llava-instruct-mix-vsft',
    tags=['multi-modal', 'en', 'vqa', 'quality'])


class OrpoDPOMix40kPreprocessor(RowPreprocessor):

    def empty_row(self):
        row = self.empty_row()
        row['rejected_response'] = None
        return row

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        chosen_history = row['chosen']
        rejected_history = row['rejected']
        history = []
        query = None
        response = None
        rejected_response = None
        try:
            for i, (chosen, rejected) in enumerate(zip(chosen_history, rejected_history)):
                role = chosen['role']
                content = chosen['content']
                rejected_role = rejected['role']
                rejected_content = rejected['content']
                assert role == rejected_role
                if i % 2 == 0:
                    assert role == 'user'
                else:
                    assert role == 'assistant'

                if content != rejected_content:
                    assert role == 'assistant'
                    response = content
                    rejected_response = rejected_content
                    query = history.pop(-1)[0]
                else:
                    if role == 'user':
                        history.append([content, None])
                    else:
                        history[-1][-1] = content

        except (AssertionError, IndexError) as e:
            logger.warning(e)
            return self.empty_row()

        messages = []
        for h in history:
            messages.append({'role': 'user', 'content': h[0]})
            messages.append({'role': 'assistant', 'content': h[1]})

        messages.append({'role': 'user', 'content': query})
        messages.append({'role': 'assistant', 'content': response})
        return {
            'messages': messages,
            'rejected_response': rejected_response,
        }

    def filter(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return super().filter(row) and row['source'] != 'toxic-dpo-v0.2'


register_dataset(
    DatasetName.orpo_dpo_mix_40k,
    'AI-ModelScope/orpo-dpo-mix-40k', ['default'],
    OrpoDPOMix40kPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    hf_dataset_id='mlabonne/orpo-dpo-mix-40k',
    tags=['dpo', 'orpo', 'en', 'quality'])


class SyntheticText2SqlPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        sql_prompt = row['sql_prompt']
        sql_context = row['sql_context']
        sql = row['sql']
        sql_explanation = row['sql_explanation']
        query = f'Sql Table information:\n{sql_context}\n{sql_prompt}'
        response = f'Let\'s think step by step:\n{sql_explanation}\nSo the final sql is:\n{sql}'
        return {
            'messages': [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': response}
            ],
        }


register_dataset(
    DatasetName.synthetic_text_to_sql,
    'AI-ModelScope/synthetic_text_to_sql', ['default'],
    SyntheticText2SqlPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    hf_dataset_id='gretelai/synthetic_text_to_sql',
    tags=['nl2sql', 'en'])

register_dataset(
    DatasetName.sharegpt,
    'swift/sharegpt', ['common-zh', 'computer-zh', 'unknow-zh', 'common-en', 'computer-en'],
    ListPreprocessor(user_key='human', assistant_key='assistant'),
    HubDatasetLoader.dataset_get_function,
    tags=['chat', 'general', 'multi-round'])


class LatexocrPreprocessor(RowPreprocessor):

    modals = ['image']
    modal_keys = {'image': 'image'}
    column_mapping = {'image': 'images'}

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'messages': [
                {'role': 'user', 'content': 'Using LaTeX to perform OCR on the image.'},
                {'role': 'assistant', 'content': row['text']}
            ]
        }


register_dataset(
    DatasetName.latex_ocr_print,
    'AI-ModelScope/LaTeX_OCR',
    ['full'],
    LatexocrPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['validation', 'test'],  # There are some problems in the training dataset.
    hf_dataset_id='linxy/LaTeX_OCR',
    tags=['chat', 'ocr', 'multi-modal', 'vision'])

register_dataset(
    DatasetName.latex_ocr_handwrite,
    'AI-ModelScope/LaTeX_OCR', ['synthetic_handwrite'],
    LatexocrPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['train', 'validation', 'test'],
    hf_dataset_id='linxy/LaTeX_OCR',
    tags=['chat', 'ocr', 'multi-modal', 'vision'])


class CapchaImagesPreprocessor(RowPreprocessor):
    modals = ['image']
    modal_keys = {'image': 'image'}

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = 'recognize the content.'
        response_key = 'solution'
        return {
            'messages': [
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': row[response_key]}
            ],
        }


register_dataset(
    DatasetName.capcha_images,
    'AI-ModelScope/captcha-images',
    None,
    CapchaImagesPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    split=['train', 'validation'],
    tags=['chat', 'multi-modal', 'vision'])


def _repair_toolbench(conversations: List[Dict[str, str]]) -> List[Dict[str, str]]:
    assert len(conversations) == 2
    if conversations[1]['from'] in {'caller', 'conclusion'}:
        conversations[1]['from'] = 'assistant'
    return conversations


register_dataset(
    DatasetName.toolbench_for_alpha_umi,
    'shenweizhou/alpha-umi-toolbench-processed-v2', ['backbone', 'caller', 'planner', 'summarizer'],
    # TODO
    ConversationsPreprocessor(system_role='system', repair_conversations=_repair_toolbench),
    HubDatasetLoader.dataset_get_function,
    tags=['chat', 'agent', '🔥'],
    huge_dataset=True)


class BlossomMathPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        output, answer = row['output'], row['answer']
        return {
            'messages': [
                {'role': 'user', 'content': row['input']},
                {'role': 'assistant', 'content': f'{output}\n\nAnswer: {answer}'}
            ],
        }


register_dataset(
    DatasetName.blossom_math_zh,
    'AI-ModelScope/blossom-math-v2',
    None,
    BlossomMathPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    tags=['chat', 'math', '🔥'],
    hf_dataset_id='Azure99/blossom-math-v2')

register_dataset(
    DatasetName.sql_create_context_en,
    'AI-ModelScope/sql-create-context',
    None,
    ComposePreprocessor([
        RenameColumnsPreprocessor({
            'question': 'instruction',
            'context': 'input',
            'answer': 'output'
        }),
        AlpacaPreprocessor(),
    ]),
    HubDatasetLoader.dataset_get_function,
    tags=['chat', 'sql', '🔥'],
    hf_dataset_id='b-mc2/sql-create-context')


class TigerBotLawPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        prompt = """{type}
{title}
"""
        cur_prompt = prompt.format(type=row['type'], title=row['title'])
        for i in range(1, 4):
            chapter = row[f'chapter{i}']
            if chapter is not None:
                cur_prompt += f'{chapter}'
        cur_prompt += f'{row["content"]}'
        return {
            'messages': [
                {'role': 'assistant', 'content': cur_prompt}
            ],
        }


register_dataset(
    DatasetName.tigerbot_law_zh,
    'AI-ModelScope/tigerbot-law-plugin',
    None,
    TigerBotLawPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    tags=['text-generation', 'law', 'pretrained'],
    hf_dataset_id='TigerResearch/tigerbot-law-plugin')


class LeetcodePythonPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        code_with_problem = row['code_with_problem']
        idx = code_with_problem.find('```python')
        problem = code_with_problem[:idx]
        if problem.startswith('# '):
            problem = problem[2:]
        code = code_with_problem[idx:].strip()
        explanation = row['explanation_only']
        return {
            'messages': [
                {'role': 'user', 'content': problem},
                {'role': 'assistant', 'content': f'{code}\n\n{explanation}'}
            ],
        }


register_dataset(
    DatasetName.leetcode_python_en,
    'AI-ModelScope/leetcode-solutions-python',
    None,
    LeetcodePythonPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    tags=['chat', 'coding', '🔥'])


def _repair_conversations_agent_instruct(s: str) -> List[Dict[str, Any]]:
    s = s.replace('}\n {', '},\n {')
    if isinstance(s, str):
        s = ast.literal_eval(s)
    return s


register_dataset(
    DatasetName.agent_instruct_all_en,
    'huangjintao/AgentInstruct_copy', ['alfworld', 'db', 'kg', 'mind2web', 'os', 'webshop'],
    ConversationsPreprocessor(user_role='human', assistant_role='gpt',
                              repair_conversations=_repair_conversations_agent_instruct),
    HubDatasetLoader.dataset_get_function,
    tags=['chat', 'agent', 'multi-round'])


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
                'content': 'string, what content to input into the textbox elment'
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
                        yield {
                            'messages': history_to_messages(history),
                            'images': images,
                            'tools': tools
                        }
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
                conversations.append({
                    'messages': history_to_messages(history),
                    'images': images,
                    'tools': tools
                })
                images = []
                history = []
            query = query + '\n' + row['confirmed_task']
        history.append([query, row['response']])
        images.append(row['screenshot'])

    if history:
        conversations.append({'messages': history_to_messages(history), 'images': images, 'tools': tools})

    return HfDataset.from_list(conversations)


register_dataset(
    DatasetName.mind2web,
    'swift/Multimodal-Mind2Web', [],
    preprocess_mind2web,
    HubDatasetLoader.dataset_get_function,
    hf_dataset_id='osunlp/Multimodal-Mind2Web',
    tags=['agent', 'multi-modal'])


class MultiRoleAgentPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        conv = row['conversations']
        res_prompt = """\n\n【注意事项】\n1. 这是聊天室，不要发送私信给任何人\n2. 仅代表你个人说话,不要扮演其他人，
        只根据对话历史进行回复\n3. 长话短说，不要说太多话，不要超过50字 """
        history_prompt = '\n\n【chat history】'
        conv_prompt = '\n {name}:{content}'
        query, response = '', conv[-1]['value']
        system = conv[0]['value'] if conv[0]['from'] != 'user' else ''
        if conv[0]['from'] == 'user':
            query = conv[0]['value']
        elif 'next_speakers:' not in system:
            if '【注意事项】' not in system and system:
                system += res_prompt
            system += history_prompt
            system += ''.join([conv_prompt.format(name=c['from'], content=c['value']) for c in conv[1:-1]])

        return {
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': query},
                {'role': 'assistant', 'content': response}
            ],
        }


register_dataset(
    DatasetName.ms_agent_multirole,
    'iic/MSAgent-MultiRole',
    None,
    MultiRoleAgentPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    tags=['chat', 'agent', 'multi-round', 'role-play', 'multi-agent'])


register_dataset(
    DatasetName.toolbench,
    'swift/ToolBench',
    None,
    ConversationsPreprocessor(
        from_key='from',
        value_key='value',
    ),
    HubDatasetLoader.dataset_get_function,
    remove_useless_columns=False,
    tags=['chat', 'agent', 'multi-round'])


def _preprocess_hc3(dataset: DATASET_TYPE) -> DATASET_TYPE:
    prompt = """Classification Task: Are the following responses from a human or from ChatGPT?
Question: {question}
Answer: {answer}
Category: Human, ChatGPT
Output:"""
    if isinstance(dataset, HfIterableDataset):

        def generate_example(dataset):
            for example in dataset:
                question = example['question']
                # TODO
                for h in example['human_answers']:
                    yield {'query': prompt.format(question=question, answer=h), 'response': 'Human'}
                for c in example['chatgpt_answers']:
                    yield {'query': prompt.format(question=question, answer=c), 'response': 'ChatGPT'}

        return HfIterableDataset.from_generator(generate_example, gen_kwargs={'dataset': dataset})

    query = []
    response = []
    for d in dataset:
        question = d['question']
        for h in d['human_answers']:
            query.append(prompt.format(question=question, answer=h))
            response.append('Human')
        for c in d['chatgpt_answers']:
            query.append(prompt.format(question=question, answer=c))
            response.append('ChatGPT')
    return HfDataset.from_dict({'query': query, 'response': response})


register_dataset(
    DatasetName.hc3_zh,
    'simpleai/HC3-Chinese', ['baike', 'open_qa', 'nlpcc_dbqa', 'finance', 'medicine', 'law', 'psychology'],
    _preprocess_hc3,
    HubDatasetLoader.dataset_get_function,
    tags=['text-generation', 'classification', '🔥'],
    hf_dataset_id='Hello-SimpleAI/HC3-Chinese')

register_dataset(
    DatasetName.hc3_en,
    'simpleai/HC3', ['finance', 'medicine'],
    _preprocess_hc3,
    HubDatasetLoader.dataset_get_function,
    tags=['text-generation', 'classification', '🔥'],
    hf_dataset_id='Hello-SimpleAI/HC3')

NoneType = type(None)


class RLAIFVPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'images': row['image'],
            'messages': [
                {'role': 'user', 'content': row['question']},
                {'role': 'assistant', 'content': row['chosen']},
            ],
            'rejected_response': row['rejected']
        }


register_dataset(
    DatasetName.rlaif_v,
    'swift/RLAIF-V-Dataset', ['default'],
    RLAIFVPreprocessor(),
    HubDatasetLoader.dataset_get_function,
    tags=['rlhf', 'dpo', 'multi-modal', 'en'],
    hf_dataset_id='openbmb/RLAIF-V-Dataset')
