# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import os
import re
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import json
import numpy as np
import pandas as pd
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets, load_dataset
from numpy.random import RandomState
from pandas import DataFrame
from tqdm.auto import tqdm
from transformers.utils import strtobool

from swift.utils import (get_logger, get_seed, read_from_jsonl,
                         transform_jsonl_to_df)
from .preprocess import (AlpacaPreprocessor, ClsPreprocessor,
                         ComposePreprocessor, ConversationsPreprocessor,
                         PreprocessFunc, RenameColumnsPreprocessor,
                         SmartPreprocessor, TextGenerationPreprocessor)
from .template import History
from .utils import dataset_map, download_dataset


def _remove_useless_columns(dataset: HfDataset) -> HfDataset:
    k_list = []
    for k in dataset.features.keys():
        if k in {
                'query', 'response', 'rejected_response', 'system', 'history',
                'images'
        }:
            k_list.append(k)
    dataset = dataset.select_columns(k_list)
    return dataset


GetDatasetFunction = Callable[[], Union[HfDataset, Tuple[HfDataset,
                                                         Optional[HfDataset]]]]
SubsetSplit = Union[str, Tuple[str, str], List[str]]
DATASET_MAPPING: Dict[str, Dict[str, Any]] = {}

logger = get_logger()


class DatasetName:
    # general
    ms_bench = 'ms-bench'  # used for mixed training
    ms_bench_mini = 'ms-bench-mini'
    alpaca_en = 'alpaca-en'
    alpaca_zh = 'alpaca-zh'
    multi_alpaca_all = 'multi-alpaca-all'
    instinwild_en = 'instinwild-en'
    instinwild_zh = 'instinwild-zh'
    cot_en = 'cot-en'
    cot_zh = 'cot-zh'
    firefly_all_zh = 'firefly-all-zh'
    instruct_en = 'instruct-en'
    gpt4all_en = 'gpt4all-en'
    sharegpt_en = 'sharegpt-en'
    sharegpt_zh = 'sharegpt-zh'
    tulu_v2_sft_mixture = 'tulu-v2-sft-mixture'
    wikipedia_zh = 'wikipedia-zh'
    open_orca = 'open-orca'
    open_orca_gpt4 = 'open-orca-gpt4'
    sharegpt_gpt4 = 'sharegpt-gpt4'
    sharegpt_gpt4_mini = 'sharegpt-gpt4-mini'
    # agent
    ms_agent = 'ms-agent'
    ms_agent_for_agentfabric_default = 'ms-agent-for-agentfabric-default'
    ms_agent_for_agentfabric_addition = 'ms-agent-for-agentfabric-addition'
    ms_agent_multirole = 'ms-agent-multirole'
    damo_agent_zh = 'damo-agent-zh'
    damo_agent_mini_zh = 'damo-agent-mini-zh'
    agent_instruct_all_en = 'agent-instruct-all-en'
    # coding
    code_alpaca_en = 'code-alpaca-en'
    leetcode_python_en = 'leetcode-python-en'
    codefuse_python_en = 'codefuse-python-en'
    codefuse_evol_instruction_zh = 'codefuse-evol-instruction-zh'
    # medical
    medical_en = 'medical-en'
    medical_zh = 'medical-zh'
    medical_mini_zh = 'medical-mini-zh'
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
    # text-generation
    advertise_gen_zh = 'advertise-gen-zh'
    dureader_robust_zh = 'dureader-robust-zh'
    # classification
    cmnli_zh = 'cmnli-zh'
    cmnli_mini_zh = 'cmnli-mini-zh'
    jd_sentiment_zh = 'jd-sentiment-zh'
    hc3_zh = 'hc3-zh'
    hc3_en = 'hc3-en'
    # other
    finance_en = 'finance-en'
    poetry_zh = 'poetry-zh'
    webnovel_zh = 'webnovel-zh'
    generated_chat_zh = 'generated-chat-zh'
    # example dataset for specific model
    cls_fudan_news_zh = 'cls-fudan-news-zh'  # seqgpt-560m
    ner_java_zh = 'ner-jave-zh'  # seqgpt-560m
    long_alpaca_12k = 'long-alpaca-12k'

    # multi-modal
    # for qwen-vl
    coco_en = 'coco-en'
    coco_mini_en = 'coco-mini-en'
    # for yi-vl, cogagnet
    coco_mini_en_2 = 'coco-mini-en-2'
    capcha_images = 'capcha-images'
    # for qwen-audio
    aishell1_zh = 'aishell1-zh'
    aishell1_mini_zh = 'aishell1-mini-zh'

    # dpo/hfrl dataset
    hh_rlhf_harmless_base = 'hh-rlhf-harmless-base'
    hh_rlhf_helpful_base = 'hh-rlhf-helpful-base'
    hh_rlhf_helpful_online = 'hh-rlhf-helpful-online'
    hh_rlhf_helpful_rejection_sampled = 'hh-rlhf-helpful-rejection-sampled'
    hh_rlhf_red_team_attempts = 'hh-rlhf-red-team-attempts'
    hh_rlhf_cn = 'hh-rlhf-cn'
    hh_rlhf_cn_harmless_base_cn = 'hh-rlhf-cn-harmless-base-cn'
    hh_rlhf_cn_helpful_base_cn = 'hh-rlhf-cn-helpful-base-cn'
    hh_rlhf_cn_harmless_base_en = 'hh-rlhf-cn-harmless-base-en'
    hh_rlhf_cn_helpful_base_en = 'hh-rlhf-cn-helpful-base-en'
    stack_exchange_paired = 'stack-exchange-paired'

    # for awq
    pileval = 'pileval'

    # COIG-CQIA
    coig_cqia_chinese_traditional = 'coig-cqia-chinese-traditional'
    coig_cqia_coig_pc = 'coig-cqia-coig-pc'
    coig_cqia_exam = 'coig-cqia-exam'
    coig_cqia_finance = 'coig-cqia-finance'
    coig_cqia_douban = 'coig-cqia-douban'
    coig_cqia_human_value = 'coig-cqia-human-value'
    coig_cqia_logi_qa = 'coig-cqia-logi-qa'
    coig_cqia_ruozhiba = 'coig-cqia-ruozhiba'
    coig_cqia_segmentfault = 'coig-cqia-segmentfault'
    coig_cqia_wiki = 'coig-cqia-wiki'
    coig_cqia_wikihow = 'coig-cqia-wikihow'
    coig_cqia_xhs = 'coig-cqia-xhs'
    coig_cqia_zhihu = 'coig-cqia-zhihu'

    # ruozhiba
    ruozhiba_post_annual = 'ruozhiba-post-annual'
    ruozhiba_title_good = 'ruozhiba-title-good'
    ruozhiba_title_norm = 'ruozhiba-title-norm'

    @classmethod
    def get_dataset_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__') or k == 'get_dataset_name_list':
                continue
            res.append(cls.__dict__[k])
        return res


def register_dataset(
        dataset_name: str,
        dataset_id_or_path: str,
        train_subset_split_list: Optional[List[SubsetSplit]] = None,
        val_subset_split_list: Optional[List[SubsetSplit]] = None,
        preprocess_func: Optional[PreprocessFunc] = None,
        get_function: Optional[GetDatasetFunction] = None,
        *,
        hf_dataset_id: Optional[str] = None,
        function_kwargs: Optional[Dict[str, Any]] = None,
        exists_ok: bool = False,
        **kwargs
) -> Optional[Callable[[GetDatasetFunction], GetDatasetFunction]]:
    if preprocess_func is None:
        preprocess_func = SmartPreprocessor()
    if not exists_ok and dataset_name in DATASET_MAPPING:
        raise ValueError(
            f'The `{dataset_name}` has already been registered in the DATASET_MAPPING.'
        )
    if train_subset_split_list is None:
        train_subset_split_list = []
    if val_subset_split_list is None:
        val_subset_split_list = []
    if function_kwargs is None:
        function_kwargs = {}

    dataset_info = {
        'dataset_id_or_path': dataset_id_or_path,
        'train_subset_split_list': train_subset_split_list,
        'val_subset_split_list': val_subset_split_list,
        'preprocess_func': preprocess_func,
        'hf_dataset_id': hf_dataset_id,
        **kwargs
    }
    if get_function is not None:
        if len(function_kwargs) > 0:
            get_function = partial(get_function, **function_kwargs)
        dataset_info['get_function'] = get_function
        DATASET_MAPPING[dataset_name] = dataset_info
        return

    def _register_dataset(
            get_function: GetDatasetFunction) -> GetDatasetFunction:
        _old_get_function = get_function
        if len(function_kwargs) > 0:
            get_function = partial(get_function, **function_kwargs)
        dataset_info['get_function'] = get_function
        DATASET_MAPPING[dataset_name] = dataset_info
        return _old_get_function

    return _register_dataset


def load_ms_dataset(
        dataset_id: str,
        subset_split_list: Optional[List[SubsetSplit]]) -> Optional[HfDataset]:
    from modelscope import MsDataset
    if subset_split_list is None or len(subset_split_list) == 0:
        return None
    dataset_list = []
    for subset_split in subset_split_list:
        if isinstance(subset_split, str):
            subset_split = ('default', subset_split)
        assert len(subset_split) == 2
        subset_name, split = subset_split
        dataset = MsDataset.load(
            dataset_id, subset_name=subset_name, split=split)
        if hasattr(dataset, 'to_hf_dataset'):
            dataset = dataset.to_hf_dataset()
        dataset_list.append(dataset)
    return concatenate_datasets(dataset_list)


def load_hf_dataset(
        dataset_id: str,
        subset_split_list: Optional[List[SubsetSplit]]) -> Optional[HfDataset]:
    if subset_split_list is None or len(subset_split_list) == 0:
        return None
    dataset_list = []
    for subset_split in subset_split_list:
        if isinstance(subset_split, str):
            subset_split = (None, subset_split)
        assert len(subset_split) == 2
        subset_name, split = subset_split
        dataset = load_dataset(dataset_id, name=subset_name, split=split)
        dataset_list.append(dataset)
    return concatenate_datasets(dataset_list)


@register_dataset(
    DatasetName.text2sql_en,
    'AI-ModelScope/texttosqlv2_25000_v2', ['train'],
    tags=['chat', 'sql'],
    hf_dataset_id='Clinton/texttosqlv2_25000_v2')
@register_dataset(
    DatasetName.school_math_zh,
    'AI-ModelScope/school_math_0.25M', ['train'],
    tags=['chat', 'math'],
    hf_dataset_id='BelleGroup/school_math_0.25M')
@register_dataset(
    DatasetName.gpt4all_en,
    'wyj123456/GPT4all', ['train'],
    tags=['chat', 'general'])
@register_dataset(
    DatasetName.cot_zh, 'YorickHe/CoT_zh', ['train'], tags=['chat', 'general'])
@register_dataset(
    DatasetName.cot_en, 'YorickHe/CoT', ['train'], tags=['chat', 'general'])
@register_dataset(
    DatasetName.instinwild_en,
    'wyj123456/instinwild', [('subset', 'train')],
    tags=['chat', 'general'])
@register_dataset(
    DatasetName.instinwild_zh,
    'wyj123456/instinwild', ['train'],
    tags=['chat', 'general'])
@register_dataset(
    DatasetName.code_alpaca_en,
    'wyj123456/code_alpaca_en', ['train'],
    tags=['chat', 'coding'],
    hf_dataset_id='sahil2801/CodeAlpaca-20k')
@register_dataset(
    DatasetName.finance_en,
    'wyj123456/finance_en', ['train'],
    tags=['chat', 'financial'],
    hf_dataset_id='ssbuild/alpaca_finance_en')
@register_dataset(
    DatasetName.alpaca_en,
    'AI-ModelScope/alpaca-gpt4-data-en', ['train'],
    tags=['chat', 'general', '🔥'],
    hf_dataset_id='vicgalle/alpaca-gpt4')
@register_dataset(
    DatasetName.coig_cqia_chinese_traditional,
    'AI-ModelScope/COIG-CQIA', [('chinese_traditional', 'train')],
    tags=['general', '🔥'])
@register_dataset(
    DatasetName.coig_cqia_coig_pc,
    'AI-ModelScope/COIG-CQIA', [('coig_pc', 'train')],
    tags=['general', '🔥'])
@register_dataset(
    DatasetName.coig_cqia_exam,
    'AI-ModelScope/COIG-CQIA', [('exam', 'train')],
    tags=['general', '🔥'])
@register_dataset(
    DatasetName.coig_cqia_finance,
    'AI-ModelScope/COIG-CQIA', [('finance', 'train')],
    tags=['general', '🔥'])
@register_dataset(
    DatasetName.coig_cqia_douban,
    'AI-ModelScope/COIG-CQIA', [('douban', 'train')],
    tags=['general', '🔥'])
@register_dataset(
    DatasetName.coig_cqia_human_value,
    'AI-ModelScope/COIG-CQIA', [('human_value', 'train')],
    tags=['general', '🔥'])
@register_dataset(
    DatasetName.coig_cqia_logi_qa,
    'AI-ModelScope/COIG-CQIA', [('logi_qa', 'train')],
    tags=['general', '🔥'])
@register_dataset(
    DatasetName.coig_cqia_ruozhiba,
    'AI-ModelScope/COIG-CQIA', [('ruozhiba', 'train')],
    tags=['general', '🔥'])
@register_dataset(
    DatasetName.coig_cqia_segmentfault,
    'AI-ModelScope/COIG-CQIA', [('segmentfault', 'train')],
    tags=['general', '🔥'])
@register_dataset(
    DatasetName.coig_cqia_wiki,
    'AI-ModelScope/COIG-CQIA', [('wiki', 'train')],
    tags=['general', '🔥'])
@register_dataset(
    DatasetName.coig_cqia_wikihow,
    'AI-ModelScope/COIG-CQIA', [('wikihow', 'train')],
    tags=['general', '🔥'])
@register_dataset(
    DatasetName.coig_cqia_xhs,
    'AI-ModelScope/COIG-CQIA', [('xhs', 'train')],
    tags=['general', '🔥'])
@register_dataset(
    DatasetName.coig_cqia_zhihu,
    'AI-ModelScope/COIG-CQIA', [('zhihu', 'train')],
    tags=['general', '🔥'])
@register_dataset(
    DatasetName.ms_agent_for_agentfabric_default,
    'AI-ModelScope/ms_agent_for_agentfabric', [('default', 'train')],
    tags=['chat', 'agent', 'multi-round'])
@register_dataset(
    DatasetName.ms_agent_for_agentfabric_addition,
    'AI-ModelScope/ms_agent_for_agentfabric', [('addition', 'train')],
    tags=['chat', 'agent', 'multi-round'])
def get_dataset_from_repo(
        dataset_id: str,
        train_subset_split_list: List[SubsetSplit],
        val_subset_split_list: Optional[List[SubsetSplit]],
        preprocess_func: PreprocessFunc,
        remove_useless_columns: bool = True,
        train_dataset_sample: int = -1,
        val_dataset_sample: int = -1,
        use_hf: bool = False) -> Tuple[HfDataset, Optional[HfDataset]]:
    dataset_list = []
    _iter = zip([train_subset_split_list, val_subset_split_list],
                [train_dataset_sample, val_dataset_sample])
    for subset_split_list, dataset_sample in _iter:
        if use_hf:
            dataset = load_hf_dataset(dataset_id, subset_split_list)
        else:
            dataset = load_ms_dataset(dataset_id, subset_split_list)
        if dataset is not None:
            if dataset_sample > 0 and len(dataset) > dataset_sample:
                random_state = np.random.RandomState(42)
                idxs = random_state.permutation(dataset_sample)
                dataset = dataset.select(idxs)
            dataset = preprocess_func(dataset)
            if remove_useless_columns:
                dataset = _remove_useless_columns(dataset)
        dataset_list.append(dataset)
    return tuple(dataset_list)


_multi_alpaca_subset_list = [
    'ar', 'de', 'es', 'fr', 'id', 'ja', 'ko', 'pt', 'ru', 'th', 'vi'
]

register_dataset(
    DatasetName.multi_alpaca_all,
    'damo/nlp_polylm_multialpaca_sft',
    [(subset, 'train') for subset in _multi_alpaca_subset_list],
    None,
    None,
    get_dataset_from_repo,
    tags=['chat', 'general', 'multilingual'],
    help="""language_list
    Language-key	Language	# examples
    ar	Arabic	14,671
    de	German	9,515
    es	Spanish	9,958
    fr	France	11,332
    id	Indonesian	12,117
    ja	Japanese	10,191
    ko	Korean	14,402
    pt	Portuguese	10,825
    ru	Russian	14,286
    th	Thai	11,496
    vi	Vietnamese	13,908
""")


def _concat_inst_inp_alpaca_zh(inst: str, inp: str) -> str:
    if inp.startswith('输入：'):
        inp = inp[3:]
    return f'{inst}\n{inp}'


register_dataset(
    DatasetName.alpaca_zh,
    'AI-ModelScope/alpaca-gpt4-data-zh', ['train'],
    None,
    AlpacaPreprocessor(concat_inst_inp=_concat_inst_inp_alpaca_zh),
    get_dataset_from_repo,
    tags=['chat', 'general', '🔥'],
    hf_dataset_id='c-s-ale/alpaca-gpt4-data-zh')


def _preprocess_vision_dataset(dataset: HfDataset) -> HfDataset:
    prompt = 'please describe the image.'
    image_key = 'image'
    response_key = 'caption'

    dataset._info.features._column_requires_decoding['image'] = False
    query_format = f'Picture 1:<img>{{image_path}}</img>\n{prompt}'
    query = []
    response = []
    for d in tqdm(dataset):
        query.append(query_format.format(image_path=d[image_key]['path']))
        if '&&' in d[response_key]:
            d[response_key] = d[response_key].split('&&')[0]
        response.append(d[response_key])
    dataset = HfDataset.from_dict({'query': query, 'response': response})
    return dataset


register_dataset(
    DatasetName.coco_en,
    'modelscope/coco_2014_caption', [('coco_2014_caption', 'train')],
    [('coco_2014_caption', 'validation')],
    _preprocess_vision_dataset,
    get_dataset_from_repo,
    tags=['chat', 'multi-modal', 'vision'])

register_dataset(
    DatasetName.coco_mini_en,
    'modelscope/coco_2014_caption', [('coco_2014_caption', 'train')],
    [('coco_2014_caption', 'validation')],
    _preprocess_vision_dataset,
    get_dataset_from_repo,
    function_kwargs={
        'train_dataset_sample': 20000,
        'val_dataset_sample': 200
    },
    tags=['chat', 'multi-modal', 'vision', '🔥'])


def _preprocess_vision_dataset2(dataset: HfDataset) -> HfDataset:
    query = 'please describe the image.'
    image_key = 'image'
    response_key = 'caption'

    dataset._info.features._column_requires_decoding['image'] = False
    response = []
    images = []
    for d in tqdm(dataset):
        images.append([d[image_key]['path']])
        if '&&' in d[response_key]:
            d[response_key] = d[response_key].split('&&')[0]
        response.append(d[response_key])
    return HfDataset.from_dict({
        'query': [query] * len(response),
        'response': response,
        'images': images
    })


register_dataset(
    DatasetName.coco_mini_en_2,
    'modelscope/coco_2014_caption', [('coco_2014_caption', 'train')],
    [('coco_2014_caption', 'validation')],
    _preprocess_vision_dataset2,
    get_dataset_from_repo,
    function_kwargs={
        'train_dataset_sample': 20000,
        'val_dataset_sample': 200
    },
    tags=['chat', 'multi-modal', 'vision', '🔥'])


def _preprocess_aishell1_dataset(dataset: HfDataset) -> HfDataset:
    prompt = '语音转文本'
    audio_key = 'Audio:FILE'
    response_key = 'Text:LABEL'
    query_format = f'Audio 1:<audio>{{audio_path}}</audio>\n{prompt}'
    query = []
    response = []
    for d in tqdm(dataset):
        query.append(query_format.format(audio_path=d[audio_key]))
        response.append(d[response_key].replace(' ', ''))
    dataset = HfDataset.from_dict({'query': query, 'response': response})
    return dataset


register_dataset(
    DatasetName.aishell1_zh,
    'speech_asr/speech_asr_aishell1_trainsets', ['train', 'validation'],
    ['test'],
    _preprocess_aishell1_dataset,
    get_dataset_from_repo,
    tags=['chat', 'multi-modal', 'audio'])

register_dataset(
    DatasetName.aishell1_mini_zh,
    'speech_asr/speech_asr_aishell1_trainsets', ['validation'], ['test'],
    _preprocess_aishell1_dataset,
    get_dataset_from_repo,
    function_kwargs={'val_dataset_sample': 200},
    tags=['chat', 'multi-modal', 'audio', '🔥'])


def _repair_agent_conversations(conversations: str,
                                use_mini: bool) -> Dict[str, str]:
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
    conversations = ast.literal_eval(conversations)
    if len(conversations) == 1:
        return
    return conversations


def _repair_ms_bench(conversations: str) -> Dict[str, str]:
    conversations = ast.literal_eval(conversations)
    default_system = 'You are a helpful assistant.'
    if conversations[0]['from'] == 'system' and conversations[0][
            'value'] == default_system:
        conversations.pop(0)
    # skip MOSS
    for c in conversations:
        value = c['value'].lower()
        if 'moss' in value or 'human:' in value or 'assistant:' in value:
            return
    return conversations


def long_alpaca_preprocessor(dataset: HfDataset):

    def map_row(row):
        response = row['response']
        if response and response.startswith('Answer:'):
            response = response[len('Answer:') + 1:].strip()
        return {'query': row['query'], 'response': response}

    return dataset.rename_columns({'instruction': 'query', 'output': 'response'}) \
        .remove_columns(['input', 'file']).map(map_row).filter(lambda row: row['response'] is not None)


register_dataset(
    DatasetName.long_alpaca_12k,
    'AI-ModelScope/LongAlpaca-12k', ['train'], [],
    long_alpaca_preprocessor,
    get_dataset_from_repo,
    tags=['longlora', 'QA'],
    hf_dataset_id='Yukang/LongAlpaca-12k')


def _preprocess_ruozhiba(dataset: HfDataset):

    def map_row(row):
        title = row['title'] if 'title' in row else row['content']
        abs = row['abs'] if 'abs' in row else None
        if abs and abs != title:
            title = title + '，' + abs

        pattern = r'\d+[\.,\s,\、](.+)'
        match = re.search(pattern, title)
        if match:
            title = match.group(1)
        return {'response': title}

    return dataset.map(map_row).filter(lambda row: row['response'])


register_dataset(
    DatasetName.ruozhiba_post_annual,
    'AI-ModelScope/ruozhiba', [('post-annual', 'train')], [],
    _preprocess_ruozhiba,
    get_dataset_from_repo,
    tags=['pretrain', '🔥'])
register_dataset(
    DatasetName.ruozhiba_title_good,
    'AI-ModelScope/ruozhiba', [('title-good', 'train')], [],
    _preprocess_ruozhiba,
    get_dataset_from_repo,
    tags=['pretrain', '🔥'])
register_dataset(
    DatasetName.ruozhiba_title_norm,
    'AI-ModelScope/ruozhiba', [('title-norm', 'train')], [],
    _preprocess_ruozhiba,
    get_dataset_from_repo,
    tags=['pretrain', '🔥'])

register_dataset(
    DatasetName.ms_bench,
    'iic/ms_bench', ['train'], [],
    ConversationsPreprocessor(
        repair_conversations=_repair_ms_bench, error_strategy='delete'),
    get_dataset_from_repo,
    tags=['chat', 'general', 'multi-round', '🔥'])

register_dataset(
    DatasetName.ms_bench_mini,
    'iic/ms_bench', ['train'], [],
    ConversationsPreprocessor(
        repair_conversations=_repair_ms_bench, error_strategy='delete'),
    get_dataset_from_repo,
    function_kwargs={'train_dataset_sample': 20000},
    tags=['chat', 'general', 'multi-round', '🔥'])

register_dataset(
    DatasetName.ms_agent,
    'iic/ms_agent', ['train'], [],
    ConversationsPreprocessor(error_strategy='delete'),
    get_dataset_from_repo,
    tags=['chat', 'agent', 'multi-round', '🔥'])

register_dataset(
    DatasetName.damo_agent_mini_zh,
    'damo/MSAgent-Bench', ['train'], ['validation'],
    ConversationsPreprocessor(
        repair_conversations=partial(
            _repair_agent_conversations, use_mini=True)),
    get_dataset_from_repo,
    tags=['chat', 'agent', 'multi-round'])
register_dataset(
    DatasetName.damo_agent_zh,
    'damo/MSAgent-Bench', ['train'], ['validation'],
    ConversationsPreprocessor(
        repair_conversations=partial(
            _repair_agent_conversations, use_mini=False)),
    get_dataset_from_repo,
    tags=['chat', 'agent', 'multi-round'])

advertise_gen_prompt = """Task: Generating advertisements based on keywords.
Keywords: {query}
Advertisements:"""
register_dataset(
    DatasetName.advertise_gen_zh,
    'lvjianjin/AdvertiseGen', ['train'], ['validation'],
    TextGenerationPreprocessor(advertise_gen_prompt, 'content', 'summary'),
    get_dataset_from_repo,
    tags=['text-generation', '🔥'],
    hf_dataset_id='shibing624/AdvertiseGen')

_firefly_kind_list = [
    'ProseGeneration', 'MRC', 'JinYongGeneration', 'TextCorrection',
    'ClassicalChinese', 'BELLE', 'StoryGeneration', 'Couplet', 'Cot',
    'Dictionary', 'Translation', 'Program', 'SentimentAnalyze', 'OpenQA',
    'AncientPoem', 'TextMatching', 'NLI', 'Summary', 'KeywordRecognition',
    'ProductDesc', 'LyricGeneration', 'Composition', 'MusicComment', 'NER'
]


def _preprocess_firefly(dataset: List[Dict[str, str]],
                        kind_list: List[str]) -> HfDataset:
    kind_set = set(kind_list)
    query: List[str] = []
    response: List[str] = []
    for d in tqdm(dataset):
        if d['kind'] not in kind_set:
            continue
        query.append(d['input'])
        response.append(d['target'])

    return HfDataset.from_dict({
        'query': query,
        'response': response,
    })


@register_dataset(
    DatasetName.firefly_all_zh,
    'wyj123456/firefly',
    preprocess_func=_preprocess_firefly,
    tags=['chat', 'general'],
    function_kwargs={'kind_list': _firefly_kind_list})
def get_firefly_zh_dataset(dataset_id: str, preprocess_func,
                           kind_list: List[str], **kwargs) -> HfDataset:
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
    DatasetName.poetry_zh,
    'modelscope/chinese-poetry-collection', ['train'], ['test'],
    RenameColumnsPreprocessor({'text1': 'response'}),
    get_dataset_from_repo,
    tags=['text-generation', 'poetry'])

register_dataset(
    DatasetName.instruct_en,
    'wyj123456/instruct', ['train'],
    None,
    RenameColumnsPreprocessor({
        'prompt': 'query',
        'completion': 'response'
    }),
    get_dataset_from_repo,
    tags=['chat', 'general'])

register_dataset(
    DatasetName.cmnli_zh,
    'clue', [('cmnli', 'train')], [('cmnli', 'validation')],
    ClsPreprocessor(['neutral', 'entailment', 'contradiction'],
                    'Natural Language Inference', True),
    get_dataset_from_repo,
    tags=['text-generation', 'classification'],
    hf_dataset_id='clue')

register_dataset(
    DatasetName.cmnli_mini_zh,
    'clue', [('cmnli', 'train')], [('cmnli', 'validation')],
    ClsPreprocessor(['neutral', 'entailment', 'contradiction'],
                    'Natural Language Inference', True),
    get_dataset_from_repo,
    function_kwargs={
        'train_dataset_sample': 20000,
        'val_dataset_sample': 200
    },
    tags=['text-generation', 'classification', '🔥'],
    hf_dataset_id='clue')

register_dataset(
    DatasetName.jd_sentiment_zh,
    'DAMO_NLP/jd', ['train'], ['validation'],
    ClsPreprocessor(['negative', 'positive'], 'Sentiment Classification',
                    False),
    get_dataset_from_repo,
    tags=['text-generation', 'classification', '🔥'])


def _preprocess_dureader_robust(dataset: HfDataset) -> HfDataset:
    prompt = """Task: Question Generation
Context: {context}
Answer: {answer}
Question:"""
    query = []
    response = []
    for d in dataset:
        answer, context = d['text1'].split('[SEP]')
        q = prompt.format(context=context, answer=answer)
        query.append(q)
        response.append(d['text2'])
    return HfDataset.from_dict({'query': query, 'response': response})


register_dataset(
    DatasetName.dureader_robust_zh,
    'modelscope/DuReader_robust-QG', ['train', 'validation'], ['test'],
    _preprocess_dureader_robust,
    get_dataset_from_repo,
    tags=['text-generation', '🔥'])

register_dataset(
    DatasetName.medical_en,
    'huangjintao/medical_zh', [('en', 'train'), ('en', 'val')],
    [('en', 'test')],
    RenameColumnsPreprocessor({
        'input': 'query',
        'output': 'response'
    }),
    get_dataset_from_repo,
    tags=['chat', 'medical'])

register_dataset(
    DatasetName.stack_exchange_paired,
    'AI-ModelScope/stack-exchange-paired', [('default', 'train')],
    None,
    RenameColumnsPreprocessor({
        'question': 'query',
        'response_j': 'response',
        'response_k': 'rejected_response',
    }),
    get_dataset_from_repo,
    tags=['hfrl', 'dpo', 'pairwise'])


def process_hh_rlhf(dataset):

    def extract_anthropic_prompt(prompt_and_response):
        """Extract the anthropic prompt from a prompt and response pair."""
        search_term = '\n\nAssistant:'
        search_term_idx = prompt_and_response.rfind(search_term)
        assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
        return prompt_and_response[:search_term_idx + len(search_term)]

    def reorganize_row_simple(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample['chosen'])
        return {
            'query': prompt,
            'response': sample['chosen'][len(prompt):],
            'rejected_response': sample['rejected'][len(prompt):],
        }

    def reorganize_row(row):
        import re
        chosen = row['chosen'].strip()
        rejected = row['rejected'].strip()
        parts_chosen = [
            s.strip()
            for s in re.split('\n\nHuman:|\n\nAssistant:|\n\nHum:', chosen)
        ]
        parts_rejected = [
            s.strip()
            for s in re.split('\n\nHuman:|\n\nAssistant:|\n\nHum:', rejected)
        ]
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

    return dataset.map(reorganize_row).filter(
        lambda row: row['query'] is not None)


register_dataset(
    DatasetName.hh_rlhf_harmless_base,
    'AI-ModelScope/hh-rlhf', [('harmless-base', 'train')],
    [('harmless-base', 'test')],
    process_hh_rlhf,
    get_dataset_from_repo,
    tags=['rlhf', 'dpo', 'pairwise'])

register_dataset(
    DatasetName.hh_rlhf_helpful_base,
    'AI-ModelScope/hh-rlhf', [('helpful-base', 'train')],
    [('helpful-base', 'test')],
    process_hh_rlhf,
    get_dataset_from_repo,
    tags=['rlhf', 'dpo', 'pairwise'])

register_dataset(
    DatasetName.hh_rlhf_helpful_online,
    'AI-ModelScope/hh-rlhf', [('helpful-online', 'train')],
    [('helpful-online', 'test')],
    process_hh_rlhf,
    get_dataset_from_repo,
    tags=['rlhf', 'dpo', 'pairwise'])

register_dataset(
    DatasetName.hh_rlhf_helpful_rejection_sampled,
    'AI-ModelScope/hh-rlhf', [('helpful-rejection-sampled', 'train')],
    [('helpful-rejection-sampled', 'test')],
    process_hh_rlhf,
    get_dataset_from_repo,
    tags=['rlhf', 'dpo', 'pairwise'])

register_dataset(
    DatasetName.hh_rlhf_red_team_attempts,
    'AI-ModelScope/hh-rlhf', [('red-team-attempts', 'train')],
    [('red-team-attempts', 'test')],
    process_hh_rlhf,
    get_dataset_from_repo,
    tags=['rlhf', 'dpo', 'pairwise'])


def process_hh_rlhf_cn(dataset):

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

    return dataset.filter(row_can_be_parsed).map(reorganize_row).filter(
        lambda row: row['query'])


register_dataset(
    DatasetName.hh_rlhf_cn,
    'AI-ModelScope/hh_rlhf_cn', [('hh_rlhf', 'train')], [('hh_rlhf', 'test')],
    process_hh_rlhf_cn,
    get_dataset_from_repo,
    tags=['rlhf', 'dpo', 'pairwise', '🔥'])

register_dataset(
    DatasetName.hh_rlhf_cn_harmless_base_cn,
    'AI-ModelScope/hh_rlhf_cn', [('harmless_base_cn', 'train')],
    [('harmless_base_cn', 'test')],
    process_hh_rlhf_cn,
    get_dataset_from_repo,
    tags=['rlhf', 'dpo', 'pairwise'])

register_dataset(
    DatasetName.hh_rlhf_cn_harmless_base_en,
    'AI-ModelScope/hh_rlhf_cn', [('harmless_base_en', 'train')],
    [('harmless_base_en', 'test')],
    process_hh_rlhf_cn,
    get_dataset_from_repo,
    tags=['rlhf', 'dpo', 'pairwise'])

register_dataset(
    DatasetName.hh_rlhf_cn_helpful_base_cn,
    'AI-ModelScope/hh_rlhf_cn', [('helpful_base_cn', 'train')],
    [('helpful_base_cn', 'test')],
    process_hh_rlhf_cn,
    get_dataset_from_repo,
    tags=['rlhf', 'dpo', 'pairwise'])

register_dataset(
    DatasetName.hh_rlhf_cn_helpful_base_en,
    'AI-ModelScope/hh_rlhf_cn', [('helpful_base_en', 'train')],
    [('helpful_base_en', 'test')],
    process_hh_rlhf_cn,
    get_dataset_from_repo,
    tags=['rlhf', 'dpo', 'pairwise'])

register_dataset(
    DatasetName.medical_zh,
    'huangjintao/medical_zh', [('zh', 'train'), ('zh', 'val')],
    [('zh', 'test')],
    RenameColumnsPreprocessor({
        'instruction': 'query',
        'output': 'response'
    }),
    get_dataset_from_repo,
    tags=['chat', 'medical'])

register_dataset(
    DatasetName.medical_mini_zh,
    'huangjintao/medical_zh', [('zh', 'train'), ('zh', 'val')],
    [('zh', 'test')],
    RenameColumnsPreprocessor({
        'instruction': 'query',
        'output': 'response'
    }),
    get_dataset_from_repo,
    function_kwargs={'train_dataset_sample': 50000},
    tags=['chat', 'medical'])


def _preprocess_sharegpt(dataset: HfDataset) -> HfDataset:
    query = []
    response = []
    history: List[History] = []
    for d in tqdm(dataset):
        conversation = ast.literal_eval(d['conversation'])
        query.append(conversation[-1]['human'])
        response.append(conversation[-1]['assistant'])
        h = []
        for c in conversation[:-1]:
            h.append([c['human'], c['assistant']])
        history.append(h)
    return HfDataset.from_dict({
        'query': query,
        'response': response,
        'history': history
    })


_sharegpt_zh_subset_list = ['common-zh', 'computer-zh', 'unknow-zh']

_sharegpt_en_subset_list = ['common-en', 'computer-en']

register_dataset(
    DatasetName.sharegpt_zh,
    'huangjintao/sharegpt',
    [(subset, 'train') for subset in _sharegpt_zh_subset_list],
    None,
    _preprocess_sharegpt,
    get_dataset_from_repo,
    tags=['chat', 'general', 'multi-round'])

register_dataset(
    DatasetName.sharegpt_en,
    'huangjintao/sharegpt',
    [(subset, 'train') for subset in _sharegpt_en_subset_list],
    None,
    _preprocess_sharegpt,
    get_dataset_from_repo,
    tags=['chat', 'general', 'multi-round'])


def _preprocess_capcha_images(dataset: HfDataset) -> HfDataset:
    query = 'recognize the content.'
    image_key = 'image'
    response_key = 'solution'

    response = []
    images = []
    for d in tqdm(dataset):
        images.append(d[image_key])
        response.append(d[response_key])
    dataset = HfDataset.from_dict({
        'query': [query] * len(response),
        'response': response,
        'images': images
    })
    dataset._info.features._column_requires_decoding['images'] = True
    return dataset


register_dataset(
    DatasetName.capcha_images,
    'AI-ModelScope/captcha-images', [('default', 'train')],
    [('default', 'validation')],
    _preprocess_capcha_images,
    get_dataset_from_repo,
    tags=['chat', 'multi-modal', 'vision'])

register_dataset(
    DatasetName.cls_fudan_news_zh,
    'damo/zh_cls_fudan-news', ['train'],
    None,
    RenameColumnsPreprocessor({
        'prompt': 'query',
        'answer': 'response'
    }),
    get_dataset_from_repo,
    tags=['chat', 'classification'])

register_dataset(
    DatasetName.ner_java_zh,
    'damo/zh_ner-JAVE', ['train'],
    None,
    RenameColumnsPreprocessor({
        'prompt': 'query',
        'answer': 'response'
    }),
    get_dataset_from_repo,
    tags=['chat', 'ner'])

register_dataset(
    DatasetName.codefuse_python_en,
    'codefuse-ai/CodeExercise-Python-27k', ['train'],
    None,
    ConversationsPreprocessor(
        'human',
        'bot',
        conversations_key='chat_rounds',
        from_key='role',
        value_key='content'),
    get_dataset_from_repo,
    tags=['chat', 'coding', '🔥'])


def _preprocess_blossom_math(dataset: HfDataset) -> HfDataset:
    response = []
    for d in tqdm(dataset):
        output, answer = d['output'], d['answer']
        response.append(f'{output}\n\nAnswer: {answer}')
    return HfDataset.from_dict({
        'query': dataset['input'],
        'response': response
    })


register_dataset(
    DatasetName.blossom_math_zh,
    'AI-ModelScope/blossom-math-v2', ['train'],
    None,
    _preprocess_blossom_math,
    get_dataset_from_repo,
    tags=['chat', 'math', '🔥'],
    hf_dataset_id='Azure99/blossom-math-v2')

register_dataset(
    DatasetName.sql_create_context_en,
    'AI-ModelScope/sql-create-context', ['train'],
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

register_dataset(
    DatasetName.lawyer_llama_zh,
    'AI-ModelScope/lawyer_llama_data', ['train'],
    None,
    RenameColumnsPreprocessor({
        'instruction': 'query',
        'output': 'response',
        'history': '_'
    }),
    get_dataset_from_repo,
    tags=['chat', 'law'],
    hf_dataset_id='Skepsun/lawyer_llama_data')


def _preprocess_tigerbot_law(dataset: HfDataset) -> HfDataset:
    prompt = """{type}
{title}
"""
    response = []
    for d in tqdm(dataset):
        cur_prompt = prompt.format(type=d['type'], title=d['title'])
        for i in range(1, 4):
            chapter = d[f'chapter{i}']
            if chapter is not None:
                cur_prompt += f'{chapter}'
        cur_prompt += f'{d["content"]}'
        response.append(cur_prompt)
    return HfDataset.from_dict({
        'response': response,
    })


register_dataset(
    DatasetName.tigerbot_law_zh,
    'AI-ModelScope/tigerbot-law-plugin', ['train'],
    None,
    _preprocess_tigerbot_law,
    get_dataset_from_repo,
    tags=['text-generation', 'law', 'pretrained'],
    hf_dataset_id='TigerResearch/tigerbot-law-plugin')


def _preprocess_leetcode_python(dataset: HfDataset) -> HfDataset:
    query = []
    response = []
    for d in dataset:
        code_with_problem = d['code_with_problem']
        idx = code_with_problem.find('```python')
        idx2 = code_with_problem.rfind('```python')
        assert idx == idx2
        problem = code_with_problem[:idx]
        if problem.startswith('# '):
            problem = problem[2:]
        code = code_with_problem[idx:].strip()
        explanation = d['explanation_only']
        query.append(problem)
        response.append(f'{code}\n\n{explanation}')
    return HfDataset.from_dict({'query': query, 'response': response})


register_dataset(
    DatasetName.leetcode_python_en,
    'AI-ModelScope/leetcode-solutions-python', ['train'],
    None,
    _preprocess_leetcode_python,
    get_dataset_from_repo,
    tags=['chat', 'coding', '🔥'])

_agent_instruct_subset_list = [
    'alfworld', 'db', 'kg', 'mind2web', 'os', 'webshop'
]


def _repair_conversations_agent_instruct(s: str) -> str:
    s = s.replace('}\n {', '},\n {')
    return ast.literal_eval(s)


register_dataset(
    DatasetName.agent_instruct_all_en,
    'huangjintao/AgentInstruct_copy',
    [(subset, 'train') for subset in _agent_instruct_subset_list],
    None,
    ConversationsPreprocessor(
        'human',
        'gpt',
        repair_conversations=_repair_conversations_agent_instruct),
    get_dataset_from_repo,
    tags=['chat', 'agent', 'multi-round'])


def _preprocess_msagent_multirole_dataset(dataset: HfDataset) -> HfDataset:
    res_prompt = """\n\n【注意事项】\n1. 这是聊天室，不要发送私信给任何人\n2. 仅代表你个人说话,不要扮演其他人，
    只根据对话历史进行回复\n3. 长话短说，不要说太多话，不要超过50字 """
    history_prompt = '\n\n【chat history】'
    conv_prompt = '\n {name}:{content}'
    query = []
    response = []

    for d in dataset:
        conv = d['conversations']
        system = conv[0]['value']
        if '【注意事项】' not in system:
            system += res_prompt
        system += history_prompt
        response.append(conv[-1]['value'])
        for i in range(1, len(conv) - 1):
            system += conv_prompt.format(
                name=conv[i]['from'], content=conv[i]['value'])
        query.append(system)
    return HfDataset.from_dict({'query': query, 'response': response})


register_dataset(
    DatasetName.ms_agent_multirole,
    'iic/MSAgent-MultiRole', [('default', 'train')],
    None,
    _preprocess_msagent_multirole_dataset,
    get_dataset_from_repo,
    tags=['chat', 'agent', 'multi-round', 'role-play', 'multi-agent'])

register_dataset(
    DatasetName.codefuse_evol_instruction_zh,
    'codefuse-ai/Evol-instruction-66k', ['train'],
    None,
    RenameColumnsPreprocessor({
        'instruction': 'query',
        'output': 'response'
    }),
    get_dataset_from_repo,
    tags=['chat', 'coding', '🔥'])

hc3_chinese_subset = [
    'baike', 'open_qa', 'nlpcc_dbqa', 'finance', 'medicine', 'law',
    'psychology'
]


def _preprocess_hc3(dataset: HfDataset) -> HfDataset:
    prompt = """Classification Task: Are the following responses from a human or from ChatGPT?
Question: {question}
Answer: {answer}
Category: Human, ChatGPT
Output:"""
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
    'simpleai/HC3-Chinese',
    [[subset, 'train'] for subset in hc3_chinese_subset], [],
    _preprocess_hc3,
    get_dataset_from_repo,
    tags=['text-generation', 'classification', '🔥'],
    hf_dataset_id='Hello-SimpleAI/HC3-Chinese')

register_dataset(
    DatasetName.hc3_en,
    'simpleai/HC3', [[subset, 'train'] for subset in ['finance', 'medicine']],
    [],
    _preprocess_hc3,
    get_dataset_from_repo,
    tags=['text-generation', 'classification', '🔥'],
    hf_dataset_id='Hello-SimpleAI/HC3')

register_dataset(
    DatasetName.tulu_v2_sft_mixture,
    'AI-ModelScope/tulu-v2-sft-mixture', ['train'], [],
    None,
    get_dataset_from_repo,
    tags=['chat', 'multilingual', 'general', 'multi-round'],
    hf_dataset_id='allenai/tulu-v2-sft-mixture')
register_dataset(
    DatasetName.webnovel_zh,
    'AI-ModelScope/webnovel_cn', ['train'], [],
    None,
    get_dataset_from_repo,
    tags=['chat', 'novel'],
    hf_dataset_id='zxbsmk/webnovel_cn')
register_dataset(
    DatasetName.generated_chat_zh,
    'AI-ModelScope/generated_chat_0.4M', ['train'], [],
    None,
    get_dataset_from_repo,
    tags=['chat', 'character-dialogue'],
    hf_dataset_id='BelleGroup/generated_chat_0.4M')
register_dataset(
    DatasetName.wikipedia_zh,
    'AI-ModelScope/wikipedia-cn-20230720-filtered', ['train'],
    None,
    RenameColumnsPreprocessor({'completion': 'response'}),
    get_dataset_from_repo,
    tags=['text-generation', 'general', 'pretrained'],
    hf_dataset_id='pleisto/wikipedia-cn-20230720-filtered')
register_dataset(
    DatasetName.open_platypus_en,
    'AI-ModelScope/Open-Platypus', ['train'],
    None,
    None,
    get_dataset_from_repo,
    tags=['chat', 'math'],
    hf_dataset_id='garage-bAInd/Open-Platypus')
register_dataset(
    DatasetName.open_orca_gpt4,
    'AI-ModelScope/OpenOrca', ['train'],
    None,
    RenameColumnsPreprocessor({'question': 'query'}),
    get_dataset_from_repo,
    tags=['chat', 'multilingual', 'general'])
register_dataset(
    DatasetName.open_orca,
    'AI-ModelScope/OpenOrca', [['3_5M', 'train']],
    None,
    RenameColumnsPreprocessor({'question': 'query'}),
    get_dataset_from_repo,
    tags=['chat', 'multilingual', 'general'])

register_dataset(
    DatasetName.sharegpt_gpt4_mini,
    'AI-ModelScope/sharegpt_gpt4', ['train'],
    None,
    ConversationsPreprocessor('human', 'gpt', error_strategy='delete'),
    get_dataset_from_repo,
    tags=['chat', 'multilingual', 'general', 'multi-round', 'gpt4', '🔥'])
register_dataset(
    DatasetName.sharegpt_gpt4,
    'AI-ModelScope/sharegpt_gpt4',
    [[subset, 'train']
     for subset in ['default', 'V3_format', 'zh_38K_format']],
    None,
    ConversationsPreprocessor('human', 'gpt', error_strategy='delete'),
    get_dataset_from_repo,
    tags=['chat', 'multilingual', 'general', 'multi-round'])

register_dataset(
    DatasetName.disc_med_sft_zh,
    'AI-ModelScope/DISC-Med-SFT', ['train'],
    None,
    ConversationsPreprocessor(
        conversations_key='conversation',
        from_key='role',
        value_key='content',
        error_strategy='delete'),
    get_dataset_from_repo,
    tags=['chat', 'medical', '🔥'],
    hf_dataset_id='Flmc/DISC-Med-SFT')

register_dataset(
    DatasetName.disc_law_sft_zh,
    'AI-ModelScope/DISC-Law-SFT', ['train'],
    None,
    RenameColumnsPreprocessor({
        'input': 'query',
        'output': 'response'
    }),
    get_dataset_from_repo,
    tags=['chat', 'law', '🔥'],
    hf_dataset_id='ShengbinYue/DISC-Law-SFT')

register_dataset(
    DatasetName.pileval,
    'huangjintao/pile-val-backup', ['validation'],
    None,
    RenameColumnsPreprocessor({
        'text': 'response',
    }),
    get_dataset_from_repo,
    tags=['text-generation', 'awq'],
    hf_dataset_id='mit-han-lab/pile-val-backup')


def add_self_cognition_dataset(
        train_dataset: HfDataset, dataset_sample: int,
        model_name: Tuple[str, Optional[str]],
        model_author: Tuple[str, Optional[str]]) -> HfDataset:
    assert model_name[0] is not None
    assert model_author[0] is not None
    if model_name[1] is None:
        model_name = (model_name[0], model_name[0])
    if model_author[1] is None:
        model_author = (model_author[0], model_author[0])
    dataset_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'data',
        'self_cognition.jsonl')
    assert os.path.exists(dataset_path)
    dataset = load_dataset_from_local([dataset_path], SmartPreprocessor())
    response = []
    for d in dataset:
        if d['tag'] == 'zh':
            model_n, model_a = model_name[0], model_author[0]
        else:
            model_n, model_a = model_name[1], model_author[1]

        r = d['response'].replace('{{NAME}}',
                                  model_n).replace('{{AUTHOR}}', model_a)
        response.append(r)
    dataset = dataset.remove_columns('response').add_column(
        'response', response).remove_columns('tag')

    random_state = RandomState(42)
    idx = random_state.permutation(len(dataset))[:dataset_sample]
    dataset_sample -= len(idx)
    if dataset_sample > 0:
        idx2 = random_state.choice(len(dataset), dataset_sample)
        idx = np.concatenate([idx, idx2], axis=0)
    dataset = dataset.select(idx)
    if train_dataset is None:
        return dataset
    else:
        return concatenate_datasets([train_dataset, dataset])


NoneType = type(None)


def _check_dataset(
    dataset: Optional[None],
    check_dataset_strategy: Literal['none', 'discard', 'error', 'warning']
) -> HfDataset:
    if check_dataset_strategy == 'none' or dataset is None:
        return dataset
    idx_list = []
    has_query = 'query' in dataset.features
    has_history = 'history' in dataset.features
    has_system = 'system' in dataset.features
    is_modified = False
    for i, d in enumerate(tqdm(dataset)):
        if not isinstance(d['response'], str):
            is_modified = True
            if check_dataset_strategy == 'discard':
                continue
            elif check_dataset_strategy == 'warning':
                logger.warning(f"d['response']: {d['response']}, i: {i}")
                continue
            else:
                raise ValueError(f"d['response']: {d['response']}, i: {i}")
        if has_query and not isinstance(d['query'], (str, NoneType)):
            is_modified = True
            if check_dataset_strategy == 'discard':
                continue
            elif check_dataset_strategy == 'warning':
                logger.warning(f"d['query']: {d['query']}, i: {i}")
                continue
            else:
                raise ValueError(f"d['query']: {d['query']}, i: {i}")
        if has_history and not isinstance(d['history'], (list, NoneType)):
            is_modified = True
            if check_dataset_strategy == 'discard':
                continue
            elif check_dataset_strategy == 'warning':
                logger.warning(f"d['history']: {d['history']}, i: {i}")
                continue
            else:
                raise ValueError(f"d['history']: {d['history']}, i: {i}")
        if has_system and not isinstance(d['system'], (str, NoneType)):
            is_modified = True
            if check_dataset_strategy == 'discard':
                continue
            elif check_dataset_strategy == 'warning':
                logger.warning(f"d['system']: {d['system']}, i: {i}")
                continue
            else:
                raise ValueError(f"d['system']: {d['system']}, i: {i}")
        idx_list.append(i)
    if is_modified:
        dataset = dataset.select(idx_list)
    assert len(dataset) > 0
    return dataset


def get_dataset(
    dataset_name_list: Union[List[str], str],
    dataset_test_ratio: float = 0.,
    dataset_seed: Union[RandomState, int] = 42,
    check_dataset_strategy: Literal['none', 'discard', 'error',
                                    'warning'] = 'none',
) -> Tuple[HfDataset, Optional[HfDataset]]:
    """Returns train_dataset and val_dataset"""
    if isinstance(dataset_name_list, str):
        dataset_name_list = [dataset_name_list]
    train_dataset_list: List[HfDataset] = []
    val_dataset_list: List[HfDataset] = []
    random_state = dataset_seed
    if isinstance(dataset_seed, int):
        random_state = RandomState(dataset_seed)
    for dataset_name in dataset_name_list:
        dataset_info = DATASET_MAPPING[dataset_name]
        use_hf = strtobool(os.environ.get('USE_HF', 'False'))
        dataset_str_f = 'Downloading the dataset from {hub}, dataset_id: {dataset_id}'
        if use_hf:
            dataset_id_or_path = dataset_info['hf_dataset_id']
            dataset_str = dataset_str_f.format(
                hub='HuggingFace', dataset_id=dataset_id_or_path)
        else:
            dataset_id_or_path = dataset_info['dataset_id_or_path']
            dataset_str = dataset_str_f.format(
                hub='ModelScope', dataset_id=dataset_id_or_path)
        logger.info(dataset_str)
        assert dataset_id_or_path is not None, (
            f'dataset_name: {dataset_name}, use_hf: {use_hf}, '
            f'dataset_id_or_path: {dataset_id_or_path}.')
        get_function: GetDatasetFunction = dataset_info['get_function']
        dataset = get_function(
            dataset_id_or_path,
            train_subset_split_list=dataset_info['train_subset_split_list'],
            val_subset_split_list=dataset_info['val_subset_split_list'],
            preprocess_func=dataset_info['preprocess_func'],
            use_hf=use_hf)
        train_d: HfDataset
        if isinstance(dataset, (list, tuple)):
            train_d, val_d = dataset
        else:
            train_d, val_d = dataset, None
        assert train_d is not None or val_d is not None
        if val_d is None and dataset_test_ratio > 0:
            dataset_dict = train_d.train_test_split(
                dataset_test_ratio, seed=get_seed(random_state))
            train_d, val_d = dataset_dict['train'], dataset_dict['test']
        if train_d is not None:
            train_dataset_list.append(train_d)
        if val_d is not None:
            val_dataset_list.append(val_d)

    train_dataset = None
    if len(train_dataset_list) > 0:
        train_dataset = concatenate_datasets(train_dataset_list)
    val_dataset = None
    if len(val_dataset_list) > 0:
        val_dataset = concatenate_datasets(val_dataset_list)
    if check_dataset_strategy != 'none':
        logger.info('check dataset...')
        logger.info(f"check_dataset_strategy: '{check_dataset_strategy}'")
    train_dataset = _check_dataset(train_dataset, check_dataset_strategy)
    val_dataset = _check_dataset(val_dataset, check_dataset_strategy)
    return train_dataset, val_dataset


def load_dataset_from_local(
        dataset_path_list: Optional[Union[str, List[str]]],
        preprocess_func: PreprocessFunc) -> Optional[HfDataset]:
    if isinstance(dataset_path_list, str):
        dataset_path_list = [dataset_path_list]
    if dataset_path_list is None or len(dataset_path_list) == 0:
        return None
    assert isinstance(dataset_path_list, (list, tuple))

    dataset_list = []
    for dataset_path in dataset_path_list:
        assert isinstance(dataset_path, str)
        df: DataFrame
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path, na_filter=False)
        elif dataset_path.endswith('.jsonl'):
            df = transform_jsonl_to_df(read_from_jsonl(dataset_path))
        elif dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                obj_list = json.load(f)
            df = transform_jsonl_to_df(obj_list)
        else:
            raise ValueError(
                'The custom dataset only supports CSV, JSONL or JSON format. You can refer to the link '
                '`https://github.com/modelscope/swift/blob/main/docs/source/LLM/自定义与拓展.md#注册数据集的方式` '
                'for more information.')
        dataset = HfDataset.from_dict(df.to_dict(orient='list'))
        dataset_list.append(preprocess_func(dataset))
    return concatenate_datasets(dataset_list)


def get_custom_dataset(_: str, train_dataset_path_list: Union[str, List[str]],
                       val_dataset_path_list: Optional[Union[str, List[str]]],
                       preprocess_func: PreprocessFunc,
                       **kwargs) -> Tuple[HfDataset, Optional[HfDataset]]:
    train_dataset = load_dataset_from_local(train_dataset_path_list,
                                            preprocess_func)
    val_dataset = load_dataset_from_local(val_dataset_path_list,
                                          preprocess_func)
    return train_dataset, val_dataset
