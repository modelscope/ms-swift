# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import itertools
import os
import re
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import json
import numpy as np
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from datasets import concatenate_datasets, interleave_datasets
from datasets import load_dataset as load_hf_dataset
from numpy.random import RandomState
from pandas import DataFrame
from tqdm.auto import tqdm
from transformers.utils import strtobool

from swift.utils import get_logger, get_seed, is_dist, is_local_master
from swift.llm.dataset.media import MediaCache, MediaTag
from swift.llm.dataset.preprocess import (AlpacaPreprocessor, ClsPreprocessor, ComposePreprocessor, ConversationsPreprocessor,
                                          ListPreprocessor, PreprocessFunc, RenameColumnsPreprocessor, SmartPreprocessor,
                                          TextGenerationPreprocessor, preprocess_sharegpt)
from swift.llm.utils.utils import download_dataset

dataset_enable_cache = strtobool(os.environ.get('DATASET_ENABLE_CACHE', 'False'))

DATASET_TYPE = Union[HfDataset, HfIterableDataset]


standard_keys = {
    'query', 'query_role', 'response', 'rejected_response', 'system', 'history', 'history_roles', 'images', 'objects',
    'videos', 'audios', 'tools', 'label'
}


SubsetSplit = Union[str, Tuple[str, str], List[str]]
DATASET_MAPPING: Dict[str, Dict[str, Any]] = {}

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
    get_dataset_from_repo,
    tags=['chat', 'general', '🔥'],
    hf_dataset_id='llm-wizard/alpaca-gpt4-data-zh')


def preprocess_sharegpt_4o_images(dataset: DATASET_TYPE):
    url = 'https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/ShareGPT-4o/repo?Revision=master&FilePath=images.zip'
    local_dir = MediaCache.download(url, 'sharegpt_4o_images')
    prefix_path = os.path.join(local_dir, 'mnt', 'petrelfs', 'wangwenhai', 'workspace_cef', '4o', 'image')

    def preprocess_row(row):
        image = row['image']
        if not image:
            return {'image': []}
        image = os.path.join(prefix_path, image)
        if not os.path.exists(image):
            return {'image': [], 'conversations': []}
        return {'image': [image]}

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    dataset = dataset.map(preprocess_row, **kwargs).filter(lambda row: row['conversations'])
    return ConversationsPreprocessor(
        user_role='human', assistant_role='gpt', media_type='image', error_strategy='delete')(
            dataset)


register_dataset(
    DatasetName.sharegpt_4o_image,
    'AI-ModelScope/ShareGPT-4o', ['image_caption'],
    preprocess_sharegpt_4o_images,
    get_dataset_from_repo,
    split=['images'],
    tags=['vqa', 'multi-modal'],
    hf_dataset_id='OpenGVLab/ShareGPT-4o')


def preprocess_sa1b_paired_caption(dataset: DATASET_TYPE):

    prompt = ['图片中展示了什么', '讲述一下图片中内容', '告诉我里面有什么', '图片内容是啥']

    def preprocess_row(row):
        response = row['global_caption']
        query = np.random.choice(prompt)
        return {
            'query': query,
            'response': response,
        }

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess_row, **kwargs).rename_column('opensource_url', 'images')


register_dataset(
    DatasetName.sa1b_paired_caption,
    'Tongyi-DataEngine/SA1B-Paired-Captions-Images',
    None,
    preprocess_sa1b_paired_caption,
    get_dataset_from_repo,
    split=['train'],
    huge_dataset=True,
    tags=['zh', 'multi-modal', 'vqa'])


def preprocess_sa1b_dense_caption(dataset: DATASET_TYPE):

    prompt = ['图片中展示了什么', '讲述一下图片中内容', '告诉我里面有什么', '图片内容是啥']

    def preprocess_row(row):
        response = ast.literal_eval(row['cap_seg'])
        response = response.get('global_caption')
        query = np.random.choice(prompt)
        return {
            'query': query,
            'response': response,
        }

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess_row, **kwargs).filter(lambda row: row.get('response')).rename_column('url', 'images')


register_dataset(
    DatasetName.sa1b_dense_caption,
    'Tongyi-DataEngine/SA1B-Dense-Caption',
    None,
    preprocess_sa1b_dense_caption,
    get_dataset_from_repo,
    split=['train'],
    huge_dataset=True,
    tags=['zh', 'multi-modal', 'vqa'])


def _preprocess_vision_dataset(dataset: DATASET_TYPE) -> DATASET_TYPE:
    from datasets import Image
    prompt = 'please describe the image.'
    image_key = 'image'
    response_key = 'caption'
    dataset = dataset.cast_column('image', Image(decode=False))
    query_format = f'<img>{{image_path}}</img>{prompt}'

    def _process(d):
        if '&&' in d[response_key]:
            d[response_key] = d[response_key].split('&&')[0]

        return {'query': query_format.format(image_path=d[image_key]['path']), 'response': d[response_key]}

    return dataset.map(_process)


def preprocess_mantis_image(dataset, subset):
    url = f'https://www.modelscope.cn/api/v1/datasets/swift/Mantis-Instruct/repo?Revision=master&FilePath={subset}/train_images.zip'  # noqa
    local_dir = MediaCache.download(url, f'mantis_{subset}')

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
        dataset = load_ms_dataset(dataset_id, [subset], use_hf, streaming=streaming)
        dataset = preprocess_mantis_image(dataset, subset=subset[0])
        all_datasets.append(dataset)
        break
    if len(all_datasets) > 1:
        dataset = concatenate_datasets(all_datasets) if not streaming else interleave_datasets(all_datasets)
    else:
        dataset = all_datasets[0]
    return _post_preprocess(dataset, dataset_sample, random_state, preprocess_func, dataset_test_ratio,
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


def preprocess_llava_data(dataset: DATASET_TYPE) -> DATASET_TYPE:

    all_folders = {}
    for media_type in ['coco', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2']:
        all_folders[media_type] = MediaCache.download(media_type)

    def preprocess_image(example, all_folders):
        if not example['images']:
            return {}
        images = [p['path'] for p in example['images']]
        new_images = []
        for image in images:
            if 'coco/' in image:
                image = os.path.join(all_folders['coco'], image.replace('coco/', ''))
            elif 'gqa/' in image:
                image = os.path.join(all_folders['gqa'], image.replace('gqa/', ''))
            elif 'ocr_vqa/' in image:
                image = os.path.join(all_folders['ocr_vqa'], image)
            elif 'textvqa/' in image:
                image = os.path.join(all_folders['textvqa'], image.replace('textvqa/', ''))
            elif 'VG_100K/' in image:
                image = os.path.join(all_folders['VG_100K'], image.replace('vg/', ''))
            elif 'VG_100K_2/' in image:
                image = os.path.join(all_folders['VG_100K_2'], image.replace('vg/', ''))
            new_images.append(image)
        if all(os.path.exists(image) for image in new_images):
            example['images'] = new_images
        else:
            example['images'] = []
        return example

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    dataset = dataset.map(partial(preprocess_image, all_folders=all_folders),
                          **kwargs).filter(lambda row: row['images'])
    return ConversationsPreprocessor(
        user_role='user',
        assistant_role='assistant',
        conversations_key='conversation',
        from_key='role',
        value_key='content',
        media_type='image',
        media_key='images')(
            dataset)


register_dataset(
    DatasetName.llava_data_instruct,
    'swift/llava-data', ['llava_instruct'],
    preprocess_llava_data,
    get_dataset_from_repo,
    split=['train'],
    tags=['sft', 'multi-modal', 'quality'],
    hf_dataset_id='TIGER-Lab/llava-data')

register_dataset(
    DatasetName.coco_en,
    'modelscope/coco_2014_caption', ['coco_2014_caption'],
    _preprocess_vision_dataset,
    get_dataset_from_repo,
    split=['train', 'validation'],
    tags=['chat', 'multi-modal', 'vision'],
    is_main=False)

register_dataset(
    DatasetName.coco_en_mini,
    'modelscope/coco_2014_caption', ['coco_2014_caption'],
    _preprocess_vision_dataset,
    get_dataset_from_repo,
    split=['validation'],
    tags=['chat', 'multi-modal', 'vision', '🔥'],
    is_main=False)


def _preprocess_vision_dataset2(dataset: DATASET_TYPE) -> DATASET_TYPE:
    from datasets import Image
    query = 'please describe the image.'
    image_key = 'image'
    response_key = 'caption'
    dataset = dataset.cast_column('image', Image(decode=False))

    def _process(d):
        images = [d[image_key]['path']]
        if '&&' in d[response_key]:
            d[response_key] = d[response_key].split('&&')[0]
        response = d[response_key]
        return {'query': query, 'response': response, 'images': images}

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(_process, **kwargs)


register_dataset(
    DatasetName.coco_en_2,
    'modelscope/coco_2014_caption', ['coco_2014_caption'],
    _preprocess_vision_dataset2,
    get_dataset_from_repo,
    split=['train', 'validation'],
    tags=['chat', 'multi-modal', 'vision'],
    is_main=False)

register_dataset(
    DatasetName.coco_en_2_mini,
    'modelscope/coco_2014_caption', ['coco_2014_caption'],
    _preprocess_vision_dataset2,
    get_dataset_from_repo,
    split=['validation'],
    tags=['chat', 'multi-modal', 'vision', '🔥'],
    is_main=False)


def _preprocess_pixelprose(dataset: DATASET_TYPE):

    caption_prompt = [
        'Give the description of this image.', 'Describe this picture', 'What is the proper title of this image?'
    ]

    def preprocess(row):
        vlm_caption = row['vlm_caption']
        if vlm_caption.startswith('This image displays:'):
            vlm_caption = vlm_caption[len('This image displays:'):].strip()
        return {
            'response': vlm_caption,
            'images': row['url'],
            'query': np.random.choice(caption_prompt),
        }

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess, **kwargs)


register_dataset(
    DatasetName.pixelprose,
    'swift/pixelprose',
    None,
    _preprocess_pixelprose,
    get_dataset_from_repo,
    split=['train', 'cc12m', 'commonpool', 'redcaps'],
    hf_dataset_id='tomg-group-umd/pixelprose',
    tags=['caption', 'multi-modal', 'vision'],
    huge_dataset=True,
    is_main=False)


def _preprocess_aishell1_dataset(dataset: DATASET_TYPE) -> DATASET_TYPE:
    prompt = '语音转文本'
    audio_key = 'Audio:FILE'
    response_key = 'Text:LABEL'
    query_format = f'<audio>{{audio_path}}</audio>{prompt}'

    def _process(d):
        return {'query': query_format.format(audio_path=d[audio_key]), 'response': d[response_key].replace(' ', '')}

    return dataset.map(_process)


register_dataset(
    DatasetName.aishell1_zh,
    'speech_asr/speech_asr_aishell1_trainsets',
    None,
    _preprocess_aishell1_dataset,
    get_dataset_from_repo,
    split=['train', 'validation', 'test'],
    tags=['chat', 'multi-modal', 'audio'])

register_dataset(
    DatasetName.aishell1_zh_mini,
    'speech_asr/speech_asr_aishell1_trainsets',
    None,
    _preprocess_aishell1_dataset,
    get_dataset_from_repo,
    split=['validation', 'test'],
    tags=['chat', 'multi-modal', 'audio', '🔥'],
    is_main=False)


def _preprocess_video_chatgpt(dataset: DATASET_TYPE) -> DATASET_TYPE:
    url = 'https://modelscope.cn/datasets/swift/VideoChatGPT/resolve/master/videos.zip'
    local_dir = MediaCache.download(url, 'video_chatgpt')
    local_dir = os.path.join(local_dir, 'Test_Videos')
    # only `.mp4`
    mp4_set = [file[:-4] for file in os.listdir(local_dir) if file.endswith('mp4')]

    def _process(d):
        if d['video_name'] not in mp4_set:
            return {'query': None, 'response': None, 'videos': None}
        return {
            'query': d['question'] or d['question_1'] or d['question_2'],
            'response': d['answer'],
            'videos': [os.path.join(local_dir, f"{d['video_name']}.mp4")]
        }

    return dataset.map(_process).filter(lambda row: row['query'] is not None)


register_dataset(
    DatasetName.video_chatgpt,
    'swift/VideoChatGPT', ['Generic', 'Temporal', 'Consistency'],
    _preprocess_video_chatgpt,
    get_dataset_from_repo,
    split=['test'],
    hf_dataset_id='lmms-lab/VideoChatGPT',
    tags=['chat', 'multi-modal', 'video', '🔥'])


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


def long_alpaca_preprocessor(dataset: DATASET_TYPE):

    def map_row(row):
        response = row['response']
        if response and response.startswith('Answer:'):
            response = response[len('Answer:') + 1:].strip()
            row['response'] = response
        return response

    dataset = AlpacaPreprocessor()(dataset)
    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(map_row, **kwargs)


register_dataset(
    DatasetName.long_alpaca_12k,
    'AI-ModelScope/LongAlpaca-12k',
    None,
    long_alpaca_preprocessor,
    get_dataset_from_repo,
    tags=['longlora', 'QA'],
    hf_dataset_id='Yukang/LongAlpaca-12k')


def _preprocess_ruozhiba(dataset: DATASET_TYPE):

    def map_row(row):
        title = row['title'] if row.get('title', None) is not None else row['content']
        abs = row['abs'] if 'abs' in row else None
        if abs and abs != title:
            title = title + '，' + abs

        pattern = r'\d+[\.,\s,\、](.+)'
        match = re.search(pattern, title)
        if match:
            title = match.group(1)
        return {'response': title}

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(map_row, **kwargs).filter(lambda row: row['response'])


register_dataset(
    DatasetName.ruozhiba,
    'AI-ModelScope/ruozhiba', ['post-annual', 'title-good', 'title-norm'],
    _preprocess_ruozhiba,
    get_dataset_from_repo,
    tags=['pretrain', '🔥'])

register_dataset(
    DatasetName.ms_bench,
    'iic/ms_bench',
    None,
    ConversationsPreprocessor(repair_conversations=_repair_ms_bench, error_strategy='delete'),
    get_dataset_from_repo,
    tags=['chat', 'general', 'multi-round', '🔥'])

register_dataset(
    DatasetName.damo_agent_zh_mini,
    'damo/MSAgent-Bench',
    None,
    ConversationsPreprocessor(
        repair_conversations=partial(_repair_agent_conversations, use_mini=True), error_strategy='delete'),
    get_dataset_from_repo,
    split=['train', 'validation'],
    tags=['chat', 'agent', 'multi-round'],
    is_main=False)
register_dataset(
    DatasetName.damo_agent_zh,
    'damo/MSAgent-Bench',
    None,
    ConversationsPreprocessor(
        repair_conversations=partial(_repair_agent_conversations, use_mini=False), error_strategy='delete'),
    get_dataset_from_repo,
    split=['train', 'validation'],
    tags=['chat', 'agent', 'multi-round'])

advertise_gen_prompt = """Task: Generating advertisements based on keywords.
Keywords: {query}
Advertisements:"""
register_dataset(
    DatasetName.advertise_gen_zh,
    'lvjianjin/AdvertiseGen',
    None,
    TextGenerationPreprocessor(advertise_gen_prompt, 'content', 'summary'),
    get_dataset_from_repo,
    split=['train', 'validation'],
    tags=['text-generation', '🔥'],
    hf_dataset_id='shibing624/AdvertiseGen')

_firefly_kind_list = [
    'ProseGeneration', 'MRC', 'JinYongGeneration', 'TextCorrection', 'ClassicalChinese', 'BELLE', 'StoryGeneration',
    'Couplet', 'Cot', 'Dictionary', 'Translation', 'Program', 'SentimentAnalyze', 'OpenQA', 'AncientPoem',
    'TextMatching', 'NLI', 'Summary', 'KeywordRecognition', 'ProductDesc', 'LyricGeneration', 'Composition',
    'MusicComment', 'NER'
]


def _preprocess_firefly(dataset: DATASET_TYPE, kind_list: List[str]) -> DATASET_TYPE:
    kind_set = set(kind_list)

    def _process(d):
        if d['kind'] not in kind_set:
            return {'query': None, 'response': None}
        return {'query': d['input'], 'response': d['target']}

    return dataset.map(_process).filter(lambda row: row['query'])


@register_dataset(
    DatasetName.firefly_zh,
    'AI-ModelScope/firefly-train-1.1M',
    None,
    _preprocess_firefly,
    tags=['chat', 'general'],
    hf_dataset_id='YeungNLP/firefly-train-1.1M',
    function_kwargs={'kind_list': _firefly_kind_list})
def get_firefly_zh_dataset(dataset_id: str, _, preprocess_func: PreprocessFunc, *args, **kwargs) -> HfDataset:
    kind_list = kwargs['kind_list']
    file = 'firefly-train-1.1M.jsonl'
    dataset_dir = download_dataset(dataset_id, [file])
    fpath = os.path.join(dataset_dir, file)
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()
        text = text.replace('}{', '},{')
        text = f'[{text}]'
        dataset = json.loads(text)
    return preprocess_func(dataset, kind_list)


register_dataset(
    DatasetName.cmnli_zh,
    'modelscope/clue', ['cmnli'],
    ClsPreprocessor(['neutral', 'entailment', 'contradiction'], 'Natural Language Inference', True),
    get_dataset_from_repo,
    split=['train', 'validation'],
    tags=['text-generation', 'classification'],
    hf_dataset_id='clue')

register_dataset(
    DatasetName.jd_sentiment_zh,
    'DAMO_NLP/jd',
    None,
    ClsPreprocessor(['negative', 'positive'], 'Sentiment Classification', False),
    get_dataset_from_repo,
    split=['train', 'validation'],
    tags=['text-generation', 'classification', '🔥'])


def _preprocess_dureader_robust(dataset: DATASET_TYPE) -> DATASET_TYPE:
    prompt = """Task: Question Generation
Context: {context}
Answer: {answer}
Question:"""

    def _process(d):
        answer, context = d['text1'].split('[SEP]')
        return {'query': prompt.format(context=context, answer=answer), 'response': d['text2']}

    return dataset.map(_process)


register_dataset(
    DatasetName.dureader_robust_zh,
    'modelscope/DuReader_robust-QG',
    None,
    _preprocess_dureader_robust,
    get_dataset_from_repo,
    split=['train', 'validation', 'test'],
    tags=['text-generation', '🔥'])


def process_hh_rlhf(dataset: DATASET_TYPE):

    def reorganize_row(row):
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
            return {
                'query': None,
                'response': None,
                'rejected_response': None,
                'history': None,
            }
        query = history[-1][0]
        history = history[:-1]
        response = s1
        rejected_response = s2
        return {
            'query': query,
            'response': response,
            'rejected_response': rejected_response,
            'history': history,
        }

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(reorganize_row, **kwargs).filter(lambda row: row['query'] is not None)


register_dataset(
    DatasetName.hh_rlhf,
    'AI-ModelScope/hh-rlhf', ['harmless-base', 'helpful-base', 'helpful-online', 'helpful-rejection-sampled'],
    process_hh_rlhf,
    get_dataset_from_repo,
    split=['train', 'test'],
    tags=['rlhf', 'dpo', 'pairwise'])


def process_hh_rlhf_cn(dataset: DATASET_TYPE):

    def reorganize_row(row):
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
            query = history[-1][0]
            history = history[:-1]
            response = row['chosen']['text']
            rejected_response = row['rejected']['text']
        except:  # noqa
            return {
                'query': '',
                'response': '',
                'rejected_response': '',
                'history': [],
            }
        return {
            'query': query,
            'response': response,
            'rejected_response': rejected_response,
            'history': history,
        }

    def row_can_be_parsed(row):
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

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.filter(row_can_be_parsed).map(reorganize_row, **kwargs).filter(lambda row: row['query'])


register_dataset(
    DatasetName.hh_rlhf_cn,
    'AI-ModelScope/hh_rlhf_cn',
    ['hh_rlhf', 'harmless_base_cn', 'harmless_base_en', 'helpful_base_cn', 'helpful_base_en'],
    process_hh_rlhf_cn,
    get_dataset_from_repo,
    split=['train', 'test'],
    tags=['rlhf', 'dpo', 'pairwise', '🔥'])


def _preprocess_m3it(dataset: DATASET_TYPE) -> DATASET_TYPE:
    column_mapping = {'instruction': 'system', 'inputs': 'query', 'image_base64_str': 'images', 'outputs': 'response'}
    dataset = dataset.rename_columns(column_mapping)
    return dataset


def _preprocess_sharegpt4v(dataset: DATASET_TYPE) -> DATASET_TYPE:
    split = ['ShareGPT4V', 'ShareGPT4V-PT'] if dataset.config_name is None else dataset.config_name
    IMAGE_DATASET_REQUIREMENTS = {
        'ShareGPT4V': ['coco', 'sam', 'llava', 'wikiart', 'share_textvqa', 'web-celebrity', 'web-landmark'],
        'ShareGPT4V-PT': ['coco', 'sam', 'llava']
    }

    if isinstance(split, str):
        split = [split]
    all_folders = {}
    for sp in split:
        for media_type in IMAGE_DATASET_REQUIREMENTS[sp]:
            all_folders[media_type] = MediaCache.download(media_type)

    def preprocess_image(example, all_folders):
        image = example['image']
        if 'coco/' in image:
            image = os.path.join(all_folders['coco'], image.replace('coco/', ''))
        elif 'sam/' in image:
            image = os.path.join(all_folders['sam'], image.replace('sam/images/', ''))
        elif 'llava/' in image:
            image = os.path.join(all_folders['llava'], image.replace('llava/llava_pretrain/images/', ''))
        elif 'wikiart/' in image:
            image = os.path.join(all_folders['wikiart'], image.replace('wikiart/images/', 'data/wikiart/images/'))
        elif 'share_textvqa/' in image:
            image = os.path.join(all_folders['share_textvqa'],
                                 image.replace('share_textvqa/images/', 'data/share_textvqa/images/'))
        elif 'web-celebrity/' in image:
            image = os.path.join(all_folders['web-celebrity'],
                                 image.replace('web-celebrity/images/', 'data/web-celebrity/images/'))
        elif 'web-landmark/' in image:
            image = os.path.join(all_folders['web-landmark'],
                                 image.replace('web-landmark/images/', 'data/web-landmark/images/'))
        if os.path.exists(image):
            example['images'] = image
        else:
            example['images'] = None
        return example

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    dataset = dataset.map(partial(preprocess_image, all_folders=all_folders),
                          **kwargs).filter(lambda example: example['images'] is not None)
    processer = ConversationsPreprocessor(
        user_role='human', assistant_role='gpt', media_type='image', media_key='images', error_strategy='delete')
    return processer(dataset)


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
    _preprocess_m3it,
    get_dataset_from_repo,
    split=['train'],
    huge_dataset=True,
    tags=['chat', 'multi-modal', 'vision'])

register_dataset(
    DatasetName.sharegpt4v,
    'AI-ModelScope/ShareGPT4V', ['ShareGPT4V', 'ShareGPT4V-PT'],
    _preprocess_sharegpt4v,
    get_dataset_from_repo,
    split=['train'],
    huge_dataset=True,
    tags=['chat', 'multi-modal', 'vision'])


def preprocess_text_caps(dataset: DATASET_TYPE):

    def preprocess(row):
        try:
            image = row['image']
            response = np.random.choice(row['reference_strs'])
            return {'response': response, 'image': image}
        except Exception:
            return {'response': '', 'image': None}

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess, **kwargs).filter(lambda row: row.get('response')).rename_columns({'image': 'images'})


def preprocess_refcoco_unofficial_caption(dataset: DATASET_TYPE):

    cache_dir = MediaCache.download(
        'https://www.modelscope.cn/api/v1/datasets/we_dont_produce_water/'
        'coco_res/repo?Revision=master&FilePath=coco_2014.zip', 'coco2014')

    def preprocess(row):
        caption = row['captions'][0]
        bbox = row['bbox']
        image_path = os.path.join(cache_dir, row['image_path'].replace('coco/train2014', 'train2014'))
        media_tag = MediaTag(media_type='image', task_type='grounding_caption')
        for i in range(len(bbox)):
            bbox[i] = round(float(bbox[i]))
        res = {}

        objects = [{
            'caption': caption,
            'bbox': bbox,
            'bbox_type': 'real',
            'image': 0,
        }]
        media_tag(res, [image_path])
        res['images'] = [image_path]
        res['objects'] = json.dumps(objects, ensure_ascii=False)
        if not os.path.exists(image_path):
            res['response'] = ''
        return res

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess, **kwargs).filter(lambda row: row.get('response'))


register_dataset(
    DatasetName.refcoco_unofficial_caption,
    'swift/refcoco', [],
    preprocess_func=preprocess_refcoco_unofficial_caption,
    get_function=get_dataset_from_repo,
    split=['train', 'validation'],
    hf_dataset_id='jxu124/refcoco',
    tags=['multi-modal', 'en', 'caption'])

register_dataset(
    DatasetName.refcocog_unofficial_caption,
    'swift/refcocog', [],
    preprocess_func=preprocess_refcoco_unofficial_caption,
    get_function=get_dataset_from_repo,
    split=['train', 'validation'],
    hf_dataset_id='jxu124/refcocog',
    tags=['multi-modal', 'en', 'caption'])


def preprocess_refcoco_unofficial_grounding(dataset: DATASET_TYPE):

    cache_dir = MediaCache.download(
        'https://www.modelscope.cn/api/v1/datasets/we_dont_produce_water/'
        'coco_res/repo?Revision=master&FilePath=coco_2014.zip', 'coco2014')

    def preprocess(row):
        caption = row['captions'][0]
        bbox = row['bbox']
        image_path = os.path.join(cache_dir, row['image_path'].replace('coco/train2014', 'train2014'))
        media_tag = MediaTag(media_type='image', task_type='ref_grounding')
        for i in range(len(bbox)):
            bbox[i] = round(float(bbox[i]))
        res = {}

        objects = [{
            'caption': caption,
            'bbox': bbox,
            'bbox_type': 'real',
            'image': 0,
        }]
        media_tag(res, [image_path])
        res['images'] = [image_path]
        res['objects'] = json.dumps(objects, ensure_ascii=False)
        if not os.path.exists(image_path):
            res['response'] = ''
        return res

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess, **kwargs).filter(lambda row: row.get('response'))


register_dataset(
    DatasetName.refcoco_unofficial_grounding,
    'swift/refcoco', [],
    preprocess_func=preprocess_refcoco_unofficial_grounding,
    get_function=get_dataset_from_repo,
    split=['train', 'validation'],
    hf_dataset_id='jxu124/refcoco',
    tags=['multi-modal', 'en', 'grounding'])

register_dataset(
    DatasetName.refcocog_unofficial_grounding,
    'swift/refcocog', [],
    preprocess_func=preprocess_refcoco_unofficial_grounding,
    get_function=get_dataset_from_repo,
    split=['train', 'validation'],
    hf_dataset_id='jxu124/refcocog',
    tags=['multi-modal', 'en', 'grounding'])

register_dataset(
    DatasetName.text_caps,
    'swift/TextCaps', [],
    preprocess_func=preprocess_text_caps,
    get_function=get_dataset_from_repo,
    split=['train', 'val'],
    hf_dataset_id='HuggingFaceM4/TextCaps',
    huge_dataset=True,
    tags=['multi-modal', 'en', 'caption', 'quality'])

register_dataset(
    DatasetName.lnqa,
    'swift/lnqa', [],
    preprocess_func=ListPreprocessor(query_key='question', response_key='answer', media_type='image'),
    get_function=get_dataset_from_repo,
    split=['train', 'validation'],
    hf_dataset_id='vikhyatk/lnqa',
    huge_dataset=True,
    tags=['multi-modal', 'en', 'ocr-vqa', 'quality'])


def _preprocess_llava_instruct_images(dataset: DATASET_TYPE) -> DATASET_TYPE:
    all_folders = {}
    for media_type in ['coco', 'gqa', 'ocr_vqa', 'textvqa', 'VG_100K', 'VG_100K_2']:
        all_folders[media_type] = MediaCache.download(media_type)

    def preprocess_image(example, all_folders):
        image = example['image']
        if 'coco/' in image:
            image = os.path.join(all_folders['coco'], image.replace('coco/', ''))
        elif 'gqa/' in image:
            image = os.path.join(all_folders['gqa'], image.replace('gqa/', ''))
        elif 'ocr_vqa/' in image:
            image = os.path.join(all_folders['ocr_vqa'], image)
        elif 'textvqa/' in image:
            image = os.path.join(all_folders['textvqa'], image.replace('textvqa/', ''))
        elif 'VG_100K/' in image:
            image = os.path.join(all_folders['VG_100K'], image.replace('vg/', ''))
        elif 'VG_100K_2/' in image:
            image = os.path.join(all_folders['VG_100K_2'], image.replace('vg/', ''))
        if os.path.exists(image):
            example['images'] = image
        else:
            example['images'] = None
        return example

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    dataset = dataset.map(partial(preprocess_image, all_folders=all_folders),
                          **kwargs).filter(lambda example: example['images'] is not None)
    processer = ConversationsPreprocessor(
        user_role='human', assistant_role='gpt', media_type='image', media_key='images', error_strategy='delete')
    return processer(dataset)


register_dataset(
    DatasetName.llava_instruct_150k,
    'AI-ModelScope/LLaVA-Instruct-150K',
    None,
    _preprocess_llava_instruct_images,
    get_dataset_from_repo,
    split=['train'],
    revision='d5db3806e395c60496630a206c336932e85a2d00',
    tags=['chat', 'multi-modal', 'vision'])


def preprocess_lmsys_chat(dataset):

    def repair_conversations(s: Union[str, Any]) -> Any:
        if isinstance(s, str):
            s = s.replace('}\n {', '},{')
            s = s.replace('}\n{', '},{')
            s = s.replace('}{', '},{')
            s = s.replace('}\n  {', '},{')
            return ast.literal_eval(s)
        return s

    return ConversationsPreprocessor(
        user_role='user',
        assistant_role='assistant',
        conversations_key='conversation',
        from_key='role',
        value_key='content',
        error_strategy='delete',
        repair_conversations=repair_conversations)(
            dataset)


register_dataset(
    DatasetName.lmsys_chat_1m,
    'AI-ModelScope/lmsys-chat-1m',
    None,
    preprocess_lmsys_chat,
    get_dataset_from_repo,
    hf_dataset_id='lmsys/lmsys-chat-1m',
    tags=['chat', 'em'])


def _preprocess_llava_pretrain(dataset: DATASET_TYPE):
    media_dir = MediaCache.download(
        'https://www.modelscope.cn/api/v1/datasets/AI-ModelScope/LLaVA-Pretrain/repo?Revision=master&FilePath=images.zip',  # noqa
        'llava_pretrain')

    def preprocess(row):
        if row['image']:
            file_path = os.path.join(media_dir, row['image'])
            if os.path.exists(file_path):
                return {'image': file_path}
            else:
                return {'image': ''}
        else:
            return {'image': ''}

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    dataset = dataset.map(preprocess, **kwargs).filter(lambda row: row['image'])
    return ConversationsPreprocessor(
        user_role='human', assistant_role='gpt', media_type='image', error_strategy='delete')(
            dataset)


register_dataset(
    DatasetName.llava_pretrain,
    'AI-ModelScope/LLaVA-Pretrain', ['default'],
    _preprocess_llava_pretrain,
    get_dataset_from_repo,
    split=['train'],
    hf_dataset_id='liuhaotian/LLaVA-Pretrain',
    huge_dataset=True,
    revision='e3a3f0bfaad05e90e46745152a32bf944e0f4a63',
    tags=['vqa', 'multi-modal', 'quality'])


def process_shareai_dpo(dataset: DATASET_TYPE):

    def reorganize_row(row):
        return {
            'query': row['question'],
            'response': row['answer_zh'],
            'rejected_response': row['answer_en'],
        }

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(reorganize_row, **kwargs)


def process_ultrafeedback_kto(dataset: DATASET_TYPE):

    new_column_names = {'prompt': 'query', 'completion': 'response'}

    return dataset.rename_columns(new_column_names)


register_dataset(
    DatasetName.ultrafeedback_kto,
    'AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto', ['default'],
    process_ultrafeedback_kto,
    get_dataset_from_repo,
    remove_useless_columns=False,
    tags=['rlhf', 'kto'])


def process_zhihu_kol(dataset: DATASET_TYPE):

    def reorganize_row(row):
        return {
            'query': row['INSTRUCTION'],
            'response': row['RESPONSE'],
        }

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(reorganize_row, **kwargs)


register_dataset(
    DatasetName.zhihu_kol_filtered,
    'OmniData/Zhihu-KOL-More-Than-100-Upvotes', ['default'],
    process_zhihu_kol,
    get_dataset_from_repo,
    hf_dataset_id='bzb2023/Zhihu-KOL-More-Than-100-Upvotes',
    tags=['zhihu', 'qa'])

register_dataset(
    DatasetName.zhihu_kol,
    'OmniData/Zhihu-KOL', ['default'],
    process_zhihu_kol,
    get_dataset_from_repo,
    hf_dataset_id='wangrui6/Zhihu-KOL',
    huge_dataset=True,
    tags=['zhihu', 'qa'])


def preprocess_guanaco(dataset: DATASET_TYPE):
    from swift.utils.utils import split_str_parts_by

    def preprocess_row(row):
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
            return {'query': '', 'history': [], 'response': ''}
        return {
            'history': history,
            'query': input,
            'response': output,
        }

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess_row, **kwargs).filter(lambda row: row['query'] and row['response'])


register_dataset(
    DatasetName.guanaco,
    'AI-ModelScope/GuanacoDataset', ['default'],
    preprocess_guanaco,
    get_dataset_from_repo,
    hf_dataset_id='JosephusCheung/GuanacoDataset',
    tags=['chat', 'zh'])


def preprocess_dolly_15k(dataset: DATASET_TYPE):

    def preprocess_row(row):
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
            'query': query,
            'response': response,
        }

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess_row, **kwargs)


register_dataset(
    DatasetName.dolly_15k,
    'AI-ModelScope/databricks-dolly-15k', ['default'],
    preprocess_dolly_15k,
    get_dataset_from_repo,
    hf_dataset_id='databricks/databricks-dolly-15k',
    tags=['multi-task', 'en', 'quality'])

register_dataset(
    DatasetName.shareai_llama3_dpo_zh_en_emoji,
    'hjh0119/shareAI-Llama3-DPO-zh-en-emoji', ['default'],
    process_shareai_dpo,
    get_dataset_from_repo,
    tags=['rlhf', 'dpo', 'pairwise'])

register_dataset(
    DatasetName.midefics,
    'swift/MideficsDataset', [],
    ListPreprocessor(
        conversations_key='conversation',
        query_key='question',
        response_key='answer',
        inner_key='data',
        media_type='image'),
    get_dataset_from_repo,
    hf_dataset_id='WinterSchool/MideficsDataset',
    tags=['medical', 'en', 'vqa'])


def preprocess_okvqa(dataset: DATASET_TYPE):

    def preprocess(row):
        query = row['question']
        response = np.random.choice(row['answers'])
        return {
            'response': response,
            'query': query,
        }

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess, **kwargs).rename_column('image', 'images')


register_dataset(
    DatasetName.okvqa,
    'swift/OK-VQA_train', [],
    preprocess_func=preprocess_okvqa,
    get_function=get_dataset_from_repo,
    split=['train'],
    hf_dataset_id='Multimodal-Fatima/OK-VQA_train',
    tags=['multi-modal', 'en', 'vqa', 'quality'])


def preprocess_a_okvqa(dataset: DATASET_TYPE):

    def preprocess(row):
        query = row['question']
        response = np.random.choice(row['rationales'])
        return {
            'response': response,
            'query': query,
        }

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess, **kwargs).rename_column('image', 'images')


register_dataset(
    DatasetName.a_okvqa,
    'swift/A-OKVQA', [],
    preprocess_func=preprocess_a_okvqa,
    get_function=get_dataset_from_repo,
    split=['train', 'validation'],
    hf_dataset_id='HuggingFaceM4/A-OKVQA',
    tags=['multi-modal', 'en', 'vqa', 'quality'])


def preprocess_ocr_vqa(dataset: DATASET_TYPE):

    def preprocess(row):
        idx = np.random.choice(range(len(row['questions'])))
        query = row['questions'][idx]
        response = row['answers'][idx]
        return {
            'response': response,
            'query': query,
        }

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess, **kwargs).rename_column('image', 'images')


register_dataset(
    DatasetName.ocr_vqa,
    'swift/OCR-VQA', [],
    preprocess_func=preprocess_ocr_vqa,
    get_function=get_dataset_from_repo,
    split=['train', 'validation'],
    hf_dataset_id='howard-hou/OCR-VQA',
    tags=['multi-modal', 'en', 'ocr-vqa'])


def preprocess_science_qa(dataset: DATASET_TYPE):

    def preprocess_row(row):
        query = row['question']
        response = row['choices'][row['answer']]
        solution = row['solution']
        return {'query': query, 'response': f'{solution}\nSo the final answer is: {response}'}

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess_row, **kwargs).filter(lambda row: row['image']).rename_columns({'image': 'images'})


register_dataset(
    DatasetName.science_qa,
    'swift/ScienceQA', [],
    preprocess_func=preprocess_science_qa,
    get_function=get_dataset_from_repo,
    split=['train', 'validation'],
    hf_dataset_id='derek-thomas/ScienceQA',
    tags=['multi-modal', 'science', 'vqa', 'quality'])


def preprocess_grit(dataset: DATASET_TYPE):

    def has_overlap(start_ends):
        for i in range(1, len(start_ends)):
            if start_ends[i][0] < start_ends[i - 1][1]:
                return True
        return False

    def replace_intervals_with_tags(response, start_ends):
        result = []
        last_end = 0
        for start, end in start_ends:
            result.append(response[int(last_end):int(start)])
            result.append('<ref-object><bbox>')
            last_end = end
        result.append(response[int(last_end):])
        return ''.join(result)

    def preprocess_row(row):
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
        if has_overlap(start_end_pairs):
            return {'images': None, 'response': '', 'objects': None}

        response = replace_intervals_with_tags(caption, start_end_pairs)

        return {'images': images, 'response': response, 'objects': json.dumps(objects or [], ensure_ascii=False)}

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess_row, **kwargs).filter(lambda row: row['objects'])


register_dataset(
    DatasetName.grit,
    'swift/GRIT', [],
    preprocess_func=preprocess_grit,
    get_function=get_dataset_from_repo,
    split=['train'],
    hf_dataset_id='zzliang/GRIT',
    huge_dataset=True,
    tags=['multi-modal', 'en', 'caption-grounding', 'quality'])


def preprocess_gqa(dataset: DATASET_TYPE):
    local_cache = MediaCache.download('gqa')

    def preprocess_row(row):
        if os.path.join(local_cache, 'images', row['imageId'] + '.jpg'):
            return {
                'query': row['question'],
                'response': row['fullAnswer'],
                'images': os.path.join(local_cache, 'images', row['imageId'] + '.jpg'),
            }
        else:
            return {'query': '', 'response': '', 'images': ''}

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess_row, **kwargs).filter(lambda row: row['query'])


register_dataset(
    DatasetName.gqa,
    None, ['train_all_instructions'],
    preprocess_gqa,
    get_function=get_dataset_from_repo,
    hf_dataset_id='lmms-lab/GQA',
    huge_dataset=True,
    tags=['multi-modal', 'en', 'vqa', 'quality'])


def preprocess_llava_mix_sft(dataset: DATASET_TYPE):

    def preprocess_row(row):
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

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    dataset = dataset.map(preprocess_row, **kwargs).map(
        ConversationsPreprocessor(
            user_role='user',
            assistant_role='assistant',
            conversations_key='messages',
            from_key='role',
            value_key='content',
            media_key='images',
            media_type='image',
        ).preprocess, **kwargs)
    return dataset


register_dataset(
    DatasetName.llava_instruct_mix,
    'swift/llava-instruct-mix-vsft', [],
    preprocess_llava_mix_sft,
    get_function=get_dataset_from_repo,
    split=['test'],
    hf_dataset_id='HuggingFaceH4/llava-instruct-mix-vsft',
    tags=['multi-modal', 'en', 'vqa', 'quality'])


def orpo_dpo_mix_40k_preprocessor(dataset: DATASET_TYPE):

    def preprocess(row):
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

        return {
            'history': history,
            'query': query,
            'response': response,
            'rejected_response': rejected_response,
        }

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess,
                       **kwargs).filter(lambda r: r['source'] != 'toxic-dpo-v0.2' and r['query'] is not None)


register_dataset(
    DatasetName.orpo_dpo_mix_40k,
    'AI-ModelScope/orpo-dpo-mix-40k', ['default'],
    orpo_dpo_mix_40k_preprocessor,
    get_dataset_from_repo,
    hf_dataset_id='mlabonne/orpo-dpo-mix-40k',
    tags=['dpo', 'orpo', 'en', 'quality'])


def synthetic_text_to_sql_preprocesser(dataset: DATASET_TYPE):

    def preprocess(row):
        sql_prompt = row['sql_prompt']
        sql_context = row['sql_context']
        sql = row['sql']
        sql_explanation = row['sql_explanation']
        query = f'Sql Table information:\n{sql_context}\n{sql_prompt}'
        response = f'Let\'s think step by step:\n{sql_explanation}\nSo the final sql is:\n{sql}'
        return {
            'query': query,
            'response': response,
        }

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(preprocess, **kwargs)


register_dataset(
    DatasetName.synthetic_text_to_sql,
    'AI-ModelScope/synthetic_text_to_sql', ['default'],
    synthetic_text_to_sql_preprocesser,
    get_dataset_from_repo,
    hf_dataset_id='gretelai/synthetic_text_to_sql',
    tags=['nl2sql', 'en'])

register_dataset(
    DatasetName.sharegpt,
    'swift/sharegpt', ['common-zh', 'computer-zh', 'unknow-zh', 'common-en', 'computer-en'],
    preprocess_sharegpt,
    get_dataset_from_repo,
    tags=['chat', 'general', 'multi-round'])


def _preprocess_latex_ocr_dataset(dataset: DATASET_TYPE) -> DATASET_TYPE:
    prompt = 'Using LaTeX to perform OCR on the image.'

    def _process(d):
        return {'query': prompt, 'response': d['text']}

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(_process, **kwargs).rename_column('image', 'images')


register_dataset(
    DatasetName.latex_ocr_print,
    'AI-ModelScope/LaTeX_OCR',
    ['full'],
    _preprocess_latex_ocr_dataset,
    get_dataset_from_repo,
    split=['validation', 'test'],  # There are some problems in the training dataset.
    hf_dataset_id='linxy/LaTeX_OCR',
    tags=['chat', 'ocr', 'multi-modal', 'vision'])

register_dataset(
    DatasetName.latex_ocr_handwrite,
    'AI-ModelScope/LaTeX_OCR', ['synthetic_handwrite'],
    _preprocess_latex_ocr_dataset,
    get_dataset_from_repo,
    split=['train', 'validation', 'test'],
    hf_dataset_id='linxy/LaTeX_OCR',
    tags=['chat', 'ocr', 'multi-modal', 'vision'])


def _preprocess_capcha_images(dataset: DATASET_TYPE) -> DATASET_TYPE:
    query = 'recognize the content.'
    response_key = 'solution'

    def _process(d):
        return {'query': query, 'response': d[response_key]}

    return dataset.map(_process).rename_column('image', 'images')


register_dataset(
    DatasetName.capcha_images,
    'AI-ModelScope/captcha-images',
    None,
    _preprocess_capcha_images,
    get_dataset_from_repo,
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
    ConversationsPreprocessor('system', system_role='-', repair_conversations=_repair_toolbench),
    get_dataset_from_repo,
    tags=['chat', 'agent', '🔥'],
    huge_dataset=True)


def _preprocess_blossom_math(dataset: DATASET_TYPE) -> DATASET_TYPE:

    def _process(d):
        output, answer = d['output'], d['answer']
        return {'query': d['input'], 'response': f'{output}\n\nAnswer: {answer}'}

    return dataset.map(_process)


register_dataset(
    DatasetName.blossom_math_zh,
    'AI-ModelScope/blossom-math-v2',
    None,
    _preprocess_blossom_math,
    get_dataset_from_repo,
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
    get_dataset_from_repo,
    tags=['chat', 'sql', '🔥'],
    hf_dataset_id='b-mc2/sql-create-context')


def _preprocess_tigerbot_law(dataset: DATASET_TYPE) -> DATASET_TYPE:
    prompt = """{type}
{title}
"""

    def _process(d):
        cur_prompt = prompt.format(type=d['type'], title=d['title'])
        for i in range(1, 4):
            chapter = d[f'chapter{i}']
            if chapter is not None:
                cur_prompt += f'{chapter}'
        cur_prompt += f'{d["content"]}'
        return {'response': cur_prompt}

    return dataset.map(_process)


register_dataset(
    DatasetName.tigerbot_law_zh,
    'AI-ModelScope/tigerbot-law-plugin',
    None,
    _preprocess_tigerbot_law,
    get_dataset_from_repo,
    tags=['text-generation', 'law', 'pretrained'],
    hf_dataset_id='TigerResearch/tigerbot-law-plugin')


def _preprocess_leetcode_python(dataset: DATASET_TYPE) -> DATASET_TYPE:

    def _process(d):
        code_with_problem = d['code_with_problem']
        idx = code_with_problem.find('```python')
        problem = code_with_problem[:idx]
        if problem.startswith('# '):
            problem = problem[2:]
        code = code_with_problem[idx:].strip()
        explanation = d['explanation_only']
        return {'query': problem, 'response': f'{code}\n\n{explanation}'}

    return dataset.map(_process)


register_dataset(
    DatasetName.leetcode_python_en,
    'AI-ModelScope/leetcode-solutions-python',
    None,
    _preprocess_leetcode_python,
    get_dataset_from_repo,
    tags=['chat', 'coding', '🔥'])


def _repair_conversations_agent_instruct(s: str) -> List[Dict[str, Any]]:
    s = s.replace('}\n {', '},\n {')
    if isinstance(s, str):
        s = ast.literal_eval(s)
    return s


register_dataset(
    DatasetName.agent_instruct_all_en,
    'huangjintao/AgentInstruct_copy', ['alfworld', 'db', 'kg', 'mind2web', 'os', 'webshop'],
    ConversationsPreprocessor('human', 'gpt', repair_conversations=_repair_conversations_agent_instruct),
    get_dataset_from_repo,
    tags=['chat', 'agent', 'multi-round'])


def preprocess_mind2web(dataset):

    def preprocess_row(row: Dict[str, Any]) -> Dict[str, Any]:
        raw_html = row['cleaned_html']
        screenshot = row['screenshot']
        row['screenshot'] = MediaCache.safe_save(screenshot, row['action_uid'] + '.jpg', 'mind2web')
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
        'api': 'CLICK',
        'desc': 'Choose and click an element in the web page',
        'parameter': [{
            'element': 'string, the element in the web page to click'
        }]
    }, {
        'api':
        'TYPE',
        'desc':
        'Input some text into a web element like <input> or <textbox>',
        'parameter': [{
            'element': 'string, the element in the web page to input to',
            'content': 'string, what content to input into the textbox elment'
        }]
    }, {
        'api':
        'SELECT',
        'desc':
        'Select an element from a combobox',
        'parameter': [{
            'element': 'string, the combobox or dropdown in the web page on which the select happens',
            'content': 'string, which choices to choose'
        }]
    }]
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
                        query, response = history.pop(-1)
                        yield {
                            'history': history,
                            'query': query,
                            'response': response,
                            'images': images,
                            'tools': tools
                        }
                        images = []
                        history = []
                    query = query + '\n' + row['confirmed_task']
                history.append([query, row['response']])
                images.append([row['screenshot']])

            if history:
                query, response = history.pop(-1)
                yield {'history': history, 'query': query, 'response': response, 'images': images, 'tools': tools}

        return HfIterableDataset.from_generator(generate_example, gen_kwargs={'dataset': dataset})

    history = []
    images = []
    for row in tqdm(dataset):
        target_action_index = row['target_action_index']
        row = preprocess_row(row)
        query = row['query']
        if target_action_index == '0':
            if history:
                query, response = history.pop(-1)
                conversations.append({
                    'history': history,
                    'query': query,
                    'response': response,
                    'images': images,
                    'tools': tools
                })
                images = []
                history = []
            query = query + '\n' + row['confirmed_task']
        history.append([query, row['response']])
        images.append([row['screenshot']])

    if history:
        query, response = history.pop(-1)
        conversations.append({'history': history, 'query': query, 'response': response, 'images': images})

    return HfDataset.from_list(conversations)


register_dataset(
    DatasetName.mind2web,
    'swift/Multimodal-Mind2Web', [],
    preprocess_mind2web,
    get_dataset_from_repo,
    hf_dataset_id='osunlp/Multimodal-Mind2Web',
    tags=['agent', 'multi-modal'])


def _preprocess_msagent_multirole_dataset(dataset: DATASET_TYPE) -> DATASET_TYPE:
    res_prompt = """\n\n【注意事项】\n1. 这是聊天室，不要发送私信给任何人\n2. 仅代表你个人说话,不要扮演其他人，
    只根据对话历史进行回复\n3. 长话短说，不要说太多话，不要超过50字 """
    history_prompt = '\n\n【chat history】'
    conv_prompt = '\n {name}:{content}'

    def process_conversation(conv):
        query, response = '', conv[-1]['value']
        system = conv[0]['value'] if conv[0]['from'] != 'user' else ''
        if conv[0]['from'] == 'user':
            query = conv[0]['value']
        elif 'next_speakers:' not in system:
            if '【注意事项】' not in system and system:
                system += res_prompt
            system += history_prompt
            system += ''.join([conv_prompt.format(name=c['from'], content=c['value']) for c in conv[1:-1]])

        return system, query, response

    def _process(d):
        sys, qry, resp = process_conversation(d['conversations'])
        return {'system': sys, 'query': qry, 'response': resp}

    return dataset.map(_process)


register_dataset(
    DatasetName.ms_agent_multirole,
    'iic/MSAgent-MultiRole',
    None,
    _preprocess_msagent_multirole_dataset,
    get_dataset_from_repo,
    tags=['chat', 'agent', 'multi-round', 'role-play', 'multi-agent'])


def _preprocess_toolbench(dataset: DATASET_TYPE) -> DATASET_TYPE:

    def reorganize_row(row):
        convs = row['conversations']
        history = []
        history_roles = []
        for idx in range(1, len(convs) - 2, 2):
            history.append((convs[idx]['value'], convs[idx + 1]['value']))
            history_roles.append((convs[idx]['from'], convs[idx + 1]['from']))

        return {
            'history': history,
            'history_roles': history_roles,
            'query': convs[-2]['value'],
            'query_role': convs[-2]['from'],
            'response': convs[-1]['value']
        }

    kwargs = {}
    if not isinstance(dataset, HfIterableDataset):
        kwargs['load_from_cache_file'] = dataset_enable_cache
    return dataset.map(reorganize_row, **kwargs)


register_dataset(
    DatasetName.toolbench,
    'swift/ToolBench',
    None,
    _preprocess_toolbench,
    get_dataset_from_repo,
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
    get_dataset_from_repo,
    tags=['text-generation', 'classification', '🔥'],
    hf_dataset_id='Hello-SimpleAI/HC3-Chinese')

register_dataset(
    DatasetName.hc3_en,
    'simpleai/HC3', ['finance', 'medicine'],
    _preprocess_hc3,
    get_dataset_from_repo,
    tags=['text-generation', 'classification', '🔥'],
    hf_dataset_id='Hello-SimpleAI/HC3')

NoneType = type(None)


def process_rlaif_v(dataset: DATASET_TYPE):

    new_column_names = {'image': 'images', 'question': 'query', 'chosen': 'response', 'rejected': 'rejected_response'}

    return dataset.rename_columns(new_column_names)


register_dataset(
    DatasetName.rlaif_v,
    'swift/RLAIF-V-Dataset', ['default'],
    process_rlaif_v,
    get_dataset_from_repo,
    tags=['rlhf', 'dpo', 'multi-modal', 'en'],
    hf_dataset_id='openbmb/RLAIF-V-Dataset')
