# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import itertools
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

from swift.utils import (get_logger, is_dist, is_local_master, read_from_jsonl,
                         transform_jsonl_to_df)
from .preprocess import (AlpacaPreprocessor, ClsPreprocessor,
                         ComposePreprocessor, ConversationsPreprocessor,
                         PreprocessFunc, RenameColumnsPreprocessor,
                         SmartPreprocessor, TextGenerationPreprocessor)
from .template import History
from .utils import download_dataset


def _remove_useless_columns(dataset: HfDataset) -> HfDataset:
    k_list = []
    for k in dataset.features.keys():
        if k in {'query', 'response', 'rejected_response', 'system', 'history', 'images'}:
            k_list.append(k)
    dataset = dataset.select_columns(k_list)
    return dataset


GetDatasetFunction = Callable[[], Union[HfDataset, Tuple[HfDataset, Optional[HfDataset]]]]
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

    # agent
    ms_agent = 'ms-agent'
    ms_agent_for_agentfabric = 'ms-agent-for-agentfabric'
    ms_agent_multirole = 'ms-agent-multirole'
    alpha_umi_toolbench = 'alpha-umi-toolbench'
    damo_agent_zh = 'damo-agent-zh'
    damo_agent_zh_mini = 'damo-agent-zh-mini'
    agent_instruct_all_en = 'agent-instruct-all-en'

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
    # text-generation
    advertise_gen_zh = 'advertise-gen-zh'
    dureader_robust_zh = 'dureader-robust-zh'
    # classification
    cmnli_zh = 'cmnli-zh'
    jd_sentiment_zh = 'jd-sentiment-zh'
    hc3_zh = 'hc3-zh'
    hc3_en = 'hc3-en'
    # other
    finance_en = 'finance-en'
    poetry_zh = 'poetry-zh'
    webnovel_zh = 'webnovel-zh'
    generated_chat_zh = 'generated-chat-zh'
    self_cognition = 'self-cognition'

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
    # for qwen-audio
    aishell1_zh = 'aishell1-zh'
    aishell1_zh_mini = 'aishell1-zh-mini'

    # dpo/hfrl dataset
    hh_rlhf = 'hh-rlhf'
    hh_rlhf_cn = 'hh-rlhf-cn'
    stack_exchange_paired = 'stack-exchange-paired'

    # for awq
    pileval = 'pileval'

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
        dataset_id_or_path: Optional[str] = None,
        subsets: Optional[List[str]] = None,
        preprocess_func: Optional[PreprocessFunc] = None,
        get_function: Optional[GetDatasetFunction] = None,
        *,
        train_split: Optional[List[str]] = None,
        val_split: Optional[List[str]] = None,
        hf_dataset_id: Optional[str] = None,
        function_kwargs: Optional[Dict[str, Any]] = None,
        exist_ok: bool = False,
        is_local: bool = False,
        **kwargs
) -> Optional[Callable[[GetDatasetFunction], GetDatasetFunction]]:
    if preprocess_func is None:
        preprocess_func = SmartPreprocessor()
    if not exist_ok and dataset_name in DATASET_MAPPING:
        raise ValueError(
            f'The `{dataset_name}` has already been registered in the DATASET_MAPPING.'
        )
    if subsets is None:
        subsets = []
    if train_split is None:
        train_split = ['train']
    if val_split is None:
        val_split = []
    if function_kwargs is None:
        function_kwargs = {}

    dataset_info = {
        'dataset_id_or_path': dataset_id_or_path,
        'subsets': subsets,
        'preprocess_func': preprocess_func,
        'train_split': train_split,
        'val_split': val_split,
        'hf_dataset_id': hf_dataset_id,
        'is_local': is_local,
        **kwargs
    }
    if get_function is not None:
        if len(function_kwargs) > 0:
            get_function = partial(get_function, **function_kwargs)
        dataset_info['get_function'] = get_function
        DATASET_MAPPING[dataset_name] = dataset_info
        return

    def _register_dataset(get_function: GetDatasetFunction) -> GetDatasetFunction:
        _old_get_function = get_function
        if len(function_kwargs) > 0:
            get_function = partial(get_function, **function_kwargs)
        dataset_info['get_function'] = get_function
        DATASET_MAPPING[dataset_name] = dataset_info
        return _old_get_function

    return _register_dataset


def register_local_dataset(
        dataset_name: str,
        train_dataset_path: Optional[List[str]] = None,
        val_dataset_path: Optional[List[str]] = None,
        # Convert relative path to absolute path,
        base_dir: Optional[str] = None,
        **kwargs) -> None:
    if train_dataset_path is None:
        train_dataset_path = []
    elif isinstance(train_dataset_path, str):
        train_dataset_path = [train_dataset_path]
    if val_dataset_path is None:
        val_dataset_path = []
    elif isinstance(val_dataset_path, str):
        val_dataset_path = [val_dataset_path]
    assert len(train_dataset_path) > 0 or len(val_dataset_path) > 0
    for dataset_path in [train_dataset_path, val_dataset_path]:
        for i, path in enumerate(dataset_path):
            if not os.path.isabs(path):
                dataset_path[i] = os.path.join(base_dir, dataset_path[i])

    register_dataset(
        dataset_name,
        get_function=get_local_dataset,
        train_split=train_dataset_path,
        val_split=val_dataset_path,
        exist_ok=True,
        is_local=True,
        **kwargs)


def register_dataset_info(dataset_name: str, d_info: Dict[str, Any]) -> None:
    preprocess_func = None
    if 'columns' in d_info:
        preprocess_func = RenameColumnsPreprocessor(d_info['columns'])
        d_info.pop('columns')
    elif 'conversations' in d_info:
        preprocess_func = ConversationsPreprocessor(**d_info['conversations'])
        d_info.pop('conversations')
    dataset_id = d_info.pop('dataset_id', None)
    subsets = d_info.pop('subsets', None)
    register_dataset(
        dataset_name,
        dataset_id,
        subsets,
        preprocess_func,
        get_dataset_from_repo,
        **d_info,
        exist_ok=True)


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
        if is_dist() and not is_local_master():
            force_redownload = False
        else:
            force_redownload = strtobool(os.environ.get('FORCE_REDOWNLOAD', 'False'))
        download_mode = 'force_redownload' if force_redownload else 'reuse_dataset_if_exists'
        dataset = MsDataset.load(dataset_id, subset_name=subset_name, split=split, download_mode=download_mode)
        if hasattr(dataset, 'to_hf_dataset'):
            dataset = dataset.to_hf_dataset()
        dataset_list.append(dataset)
    return concatenate_datasets(dataset_list)


def load_hf_dataset(dataset_id: str, subset_split_list: Optional[List[SubsetSplit]]) -> Optional[HfDataset]:
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


def sample_dataset(dataset: HfDataset,
                   dataset_sample: int,
                   random_state: Optional[RandomState] = None) -> HfDataset:
    if dataset_sample in {None, -1, len(dataset)}:
        return dataset
    if random_state is None:
        random_state = RandomState()
    # Sample the part that exceeds the length of the dataset.
    idx = random_state.permutation(len(dataset))[:dataset_sample]
    dataset_sample -= len(idx)
    if dataset_sample > 0:
        idx2 = random_state.choice(len(dataset), dataset_sample)
        idx = np.concatenate([idx, idx2], axis=0)
    dataset = dataset.select(idx)
    return dataset


def _post_preprocess(
    dataset_list: List[Optional[HfDataset]],
    train_sample: int,
    val_sample: int,
    random_state: Optional[RandomState] = None,
    preprocess_func: Optional[PreprocessFunc] = None,
    dataset_test_ratio: float = 0.,
    remove_useless_columns: bool = True,
) -> Tuple[HfDataset, Optional[HfDataset]]:
    assert len(dataset_list) == 2
    train_dataset, val_dataset = dataset_list
    if train_dataset is None:
        assert val_dataset is not None
    if val_dataset is None:
        if val_sample == -1 and dataset_test_ratio > 0:
            assert 0 < dataset_test_ratio < 1
            assert train_sample != 0
            _train_len = len(train_dataset)
            if train_sample != -1:
                _train_len = min(_train_len, train_sample)
            val_sample = max(int(_train_len * dataset_test_ratio), 1)
        if val_sample > 0:
            assert isinstance(val_sample, int)
            train_dataset, val_dataset = train_dataset.train_test_split(
                test_size=val_sample).values()

    if val_dataset is not None and val_sample >= 0:
        assert val_sample <= len(val_dataset), (
            f'val_sample: {val_sample}, val_dataset: {val_dataset}')
    res: List[HfDataset] = []
    for dataset, dataset_sample in zip([train_dataset, val_dataset],
                                       [train_sample, val_sample]):
        if dataset is None or dataset_sample == 0:
            res.append(None)
            continue
        assert dataset_sample != 0, f'dataset: {dataset}, dataset_sample: {dataset_sample}'
        dataset = sample_dataset(dataset, dataset_sample, random_state)
        if preprocess_func is not None:
            dataset = preprocess_func(dataset)
        if remove_useless_columns:
            dataset = _remove_useless_columns(dataset)
        res.append(dataset)
    return tuple(res)


def get_dataset_from_repo(
        dataset_id: str,
        subsets: Optional[List[str]],
        preprocess_func: PreprocessFunc,
        train_split: List[str],
        val_split: List[str],
        train_sample: int = -1,
        val_sample: int = -1,
        *,
        random_state: Optional[RandomState] = None,
        dataset_test_ratio: float = 0.,
        remove_useless_columns: bool = True,
        use_hf: bool = False) -> Tuple[HfDataset, Optional[HfDataset]]:
    dataset_list = []
    if subsets is None:
        subsets = []
    for split in [train_split, val_split]:
        if len(split) == 0:
            dataset_list.append(None)
            continue
        if len(subsets) == 0:
            subset_split_list = split
        else:
            subset_split_list = list(itertools.product(subsets, split))
        if use_hf:
            dataset = load_hf_dataset(dataset_id, subset_split_list)
        else:
            dataset = load_ms_dataset(dataset_id, subset_split_list)
        dataset_list.append(dataset)
    return _post_preprocess(dataset_list, train_sample, val_sample,
                            random_state, preprocess_func, dataset_test_ratio,
                            remove_useless_columns)


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
    'modelscope/coco_2014_caption', ['coco_2014_caption'],
    _preprocess_vision_dataset,
    get_dataset_from_repo,
    val_split=['validation'],
    tags=['chat', 'multi-modal', 'vision'],
    is_main=False)

register_dataset(
    DatasetName.coco_en_mini,
    'modelscope/coco_2014_caption', ['coco_2014_caption'],
    _preprocess_vision_dataset,
    get_dataset_from_repo,
    train_split=['validation'],
    tags=['chat', 'multi-modal', 'vision', '🔥'],
    is_main=False)


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
    return HfDataset.from_dict({'query': [query] * len(response), 'response': response, 'images': images})


register_dataset(
    DatasetName.coco_en_2,
    'modelscope/coco_2014_caption', ['coco_2014_caption'],
    _preprocess_vision_dataset2,
    get_dataset_from_repo,
    val_split=['validation'],
    tags=['chat', 'multi-modal', 'vision'],
    is_main=False)

register_dataset(
    DatasetName.coco_en_2_mini,
    'modelscope/coco_2014_caption', ['coco_2014_caption'],
    _preprocess_vision_dataset2,
    get_dataset_from_repo,
    train_split=['validation'],
    tags=['chat', 'multi-modal', 'vision', '🔥'],
    is_main=False)


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
    'speech_asr/speech_asr_aishell1_trainsets',
    None,
    _preprocess_aishell1_dataset,
    get_dataset_from_repo,
    train_split=['train', 'validation'],
    val_split=['test'],
    tags=['chat', 'multi-modal', 'audio'])

register_dataset(
    DatasetName.aishell1_zh_mini,
    'speech_asr/speech_asr_aishell1_trainsets',
    None,
    _preprocess_aishell1_dataset,
    get_dataset_from_repo,
    train_split=['validation'],
    val_split=['test'],
    function_kwargs={'val_dataset_sample': 200},
    tags=['chat', 'multi-modal', 'audio', '🔥'],
    is_main=False)


def _repair_agent_conversations(conversations: str, use_mini: bool) -> List[Dict[str, str]]:
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


def _repair_ms_bench(conversations: str) -> List[Dict[str, str]]:
    if isinstance(conversations, str):
        conversations = ast.literal_eval(conversations)
    default_system = 'You are a helpful assistant.'
    if conversations[0]['from'] == 'system' and conversations[0]['value'] == default_system:
        conversations.pop(0)
    # skip MOSS
    for c in conversations:
        value = c['value'].lower()
        if 'moss' in value or 'human:' in value or 'assistant:' in value or 'user:' in value:
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
    'AI-ModelScope/LongAlpaca-12k',
    None,
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
    DatasetName.ruozhiba,
    'AI-ModelScope/ruozhiba', ['post-annual', 'title-good', 'title-norm'],
    _preprocess_ruozhiba,
    get_dataset_from_repo,
    tags=['pretrain', '🔥'])

register_dataset(
    DatasetName.ms_bench,
    'iic/ms_bench',
    None,
    ConversationsPreprocessor(
        repair_conversations=_repair_ms_bench, error_strategy='delete'),
    get_dataset_from_repo,
    tags=['chat', 'general', 'multi-round', '🔥'])

register_dataset(
    DatasetName.damo_agent_zh_mini,
    'damo/MSAgent-Bench',
    None,
    ConversationsPreprocessor(
        repair_conversations=partial(_repair_agent_conversations, use_mini=True), error_strategy='delete'),
    get_dataset_from_repo,
    val_split=['validation'],
    tags=['chat', 'agent', 'multi-round'],
    is_main=False)
register_dataset(
    DatasetName.damo_agent_zh,
    'damo/MSAgent-Bench',
    None,
    ConversationsPreprocessor(
        repair_conversations=partial(_repair_agent_conversations, use_mini=False, error_strategy='delete')),
    get_dataset_from_repo,
    val_split=['validation'],
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
    val_split=['validation'],
    tags=['text-generation', '🔥'],
    hf_dataset_id='shibing624/AdvertiseGen')

_firefly_kind_list = [
    'ProseGeneration', 'MRC', 'JinYongGeneration', 'TextCorrection', 'ClassicalChinese', 'BELLE', 'StoryGeneration',
    'Couplet', 'Cot', 'Dictionary', 'Translation', 'Program', 'SentimentAnalyze', 'OpenQA', 'AncientPoem',
    'TextMatching', 'NLI', 'Summary', 'KeywordRecognition', 'ProductDesc', 'LyricGeneration', 'Composition',
    'MusicComment', 'NER'
]


def _preprocess_firefly(dataset: List[Dict[str, str]], kind_list: List[str]) -> HfDataset:
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
    DatasetName.firefly_zh,
    'wyj123456/firefly',
    None,
    _preprocess_firefly,
    tags=['chat', 'general'],
    function_kwargs={'kind_list': _firefly_kind_list})
def get_firefly_zh_dataset(dataset_id: str, _, preprocess_func: PreprocessFunc,
                           *args, **kwargs) -> HfDataset:
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
    ClsPreprocessor(['neutral', 'entailment', 'contradiction'],
                    'Natural Language Inference', True),
    get_dataset_from_repo,
    val_split=['validation'],
    tags=['text-generation', 'classification'],
    hf_dataset_id='clue')

register_dataset(
    DatasetName.jd_sentiment_zh,
    'DAMO_NLP/jd',
    None,
    ClsPreprocessor(['negative', 'positive'], 'Sentiment Classification',
                    False),
    get_dataset_from_repo,
    val_split=['validation'],
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
    'modelscope/DuReader_robust-QG',
    None,
    _preprocess_dureader_robust,
    get_dataset_from_repo,
    train_split=['train', 'validation'],
    val_split=['test'],
    tags=['text-generation', '🔥'])


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

    return dataset.map(reorganize_row).filter(lambda row: row['query'] is not None)


register_dataset(
    DatasetName.hh_rlhf,
    'AI-ModelScope/hh-rlhf', [
        'harmless-base', 'helpful-base', 'helpful-online',
        'helpful-rejection-sampled', 'red-team-attempts'
    ],
    process_hh_rlhf,
    get_dataset_from_repo,
    val_split=['test'],
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

    return dataset.filter(row_can_be_parsed).map(reorganize_row).filter(lambda row: row['query'])


register_dataset(
    DatasetName.hh_rlhf_cn,
    'AI-ModelScope/hh_rlhf_cn', [
        'hh_rlhf', 'harmless_base_cn', 'harmless_base_en', 'helpful_base_cn',
        'helpful_base_en'
    ],
    process_hh_rlhf_cn,
    get_dataset_from_repo,
    val_split=['test'],
    tags=['rlhf', 'dpo', 'pairwise', '🔥'])


def _preprocess_sharegpt(dataset: HfDataset) -> HfDataset:
    query = []
    response = []
    history: List[History] = []
    for d in tqdm(dataset):
        if isinstance(d['conversation'], str):
            try:
                conversation = ast.literal_eval(d['conversation'])
            except SyntaxError:
                continue
        query.append(conversation[-1]['human'])
        response.append(conversation[-1]['assistant'])
        h = []
        for c in conversation[:-1]:
            h.append([c['human'], c['assistant']])
        history.append(h)
    return HfDataset.from_dict({'query': query, 'response': response, 'history': history})


register_dataset(
    DatasetName.sharegpt,
    'huangjintao/sharegpt',
    ['common-zh', 'computer-zh', 'unknow-zh', 'common-en', 'computer-en'],
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
    dataset = HfDataset.from_dict({'query': [query] * len(response), 'response': response, 'images': images})
    dataset._info.features._column_requires_decoding['images'] = True
    return dataset


def _repair_planner(conversations: list) -> list:
    if isinstance(conversations, str):
        conversations = ast.literal_eval(conversations)
    if len(conversations) == 2 and conversations[0]['from'] != 'user':
        conversations[0]['from'] = 'user'
    return conversations


register_dataset(
    DatasetName.capcha_images,
    'AI-ModelScope/captcha-images',
    None,
    _preprocess_capcha_images,
    get_dataset_from_repo,
    val_split=['validation'],
    tags=['chat', 'multi-modal', 'vision'])

register_dataset(
    DatasetName.alpha_umi_toolbench,
    'shenweizhou/alpha-umi-toolbench-processed-v2',
    ['backbone', 'caller', 'planner', 'summarizer'], {
        'backbone':
        ConversationsPreprocessor('system', system_role='-'),
        'caller':
        ConversationsPreprocessor('system', 'caller', '-'),
        'planner':
        ConversationsPreprocessor(
            repair_conversations=_repair_planner, error_strategy='delete'),
        'summarizer':
        ConversationsPreprocessor('system', 'conclusion', None),
    },
    get_dataset_from_repo,
    tags=['chat', 'agent', '🔥'])


def _preprocess_blossom_math(dataset: HfDataset) -> HfDataset:
    response = []
    for d in tqdm(dataset):
        output, answer = d['output'], d['answer']
        response.append(f'{output}\n\nAnswer: {answer}')
    return HfDataset.from_dict({'query': dataset['input'], 'response': response})


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
    'AI-ModelScope/tigerbot-law-plugin',
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
    'huangjintao/AgentInstruct_copy',
    ['alfworld', 'db', 'kg', 'mind2web', 'os', 'webshop'],
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
            system += conv_prompt.format(name=conv[i]['from'], content=conv[i]['value'])
        query.append(system)
    return HfDataset.from_dict({'query': query, 'response': response})


register_dataset(
    DatasetName.ms_agent_multirole,
    'iic/MSAgent-MultiRole',
    None,
    _preprocess_msagent_multirole_dataset,
    get_dataset_from_repo,
    tags=['chat', 'agent', 'multi-round', 'role-play', 'multi-agent'])


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
    'simpleai/HC3-Chinese', [
        'baike', 'open_qa', 'nlpcc_dbqa', 'finance', 'medicine', 'law',
        'psychology'
    ],
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


def _check_dataset(dataset: Optional[None], check_dataset_strategy: Literal['none', 'discard', 'error',
                                                                            'warning']) -> HfDataset:
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


def _safe_split(s: str, sep: str, use_0: bool) -> Tuple[str, str]:
    # use_0: When the length of the part is 1, is it considered as part0 or part1.
    if s is None or len(s) == 0:
        return None, None
    part = s.split(sep)
    if len(part) == 1:
        if use_0:
            part = part[0], None
        else:
            part = None, part[0]
    else:
        assert len(part) == 2
    return part


def parse_dataset_name(
        dataset_name: str) -> Tuple[Optional[str], str, List[str], int, int]:
    # HF::dataset_name:subset1/subset2/subset3#train_sample/val_sample
    use_hf, other = _safe_split(dataset_name, '::', False)
    part1, part2 = _safe_split(other, '#', True)
    dataset_name, subsets = _safe_split(part1, ':', True)
    if subsets is not None:
        subset_list = subsets.split('/')
        subset_list = [subset.strip() for subset in subset_list]
    else:
        subset_list = None
    train_sample, val_sample = _safe_split(part2, '/', True)
    train_sample, val_sample = [
        sample if sample is None else int(sample)
        for sample in [train_sample, val_sample]
    ]
    if train_sample is None:
        train_sample = -1
    if val_sample is None:
        val_sample = -1
    return tuple(
        t.strip() if isinstance(t, str) else t
        for t in [use_hf, dataset_name, subset_list, train_sample, val_sample])


def _dataset_name_exists(dataset_list: str, dataset_name: str) -> List[int]:
    dataset_name = parse_dataset_name(dataset_name)[1]
    cache_name_list = [
        parse_dataset_name(dataset)[1] for dataset in dataset_list
    ]
    res = []
    for i, cache_name in enumerate(cache_name_list):
        if cache_name == dataset_name:
            res.append(i)
    return res


def _preprocess_self_cognition_dataset(
    dataset_list: Tuple[HfDataset, Optional[HfDataset]],
    model_name: Tuple[str, Optional[str]],
    model_author: Tuple[str, Optional[str]],
) -> Tuple[HfDataset, HfDataset]:
    # model_name: Tuple[zh, en]
    assert model_name[0] is not None
    assert model_author[0] is not None
    if model_name[1] is None:
        model_name = (model_name[0], model_name[0])
    if model_author[1] is None:
        model_author = (model_author[0], model_author[0])
    res_d_list = []
    for dataset in dataset_list:
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
        res_d_list.append(dataset)
    return tuple(res_d_list)


def _dataset_id_to_name(dataset_name_list: List[str]) -> List[int]:
    # register dataset_id (ms/hf). Convert dataset_id to dataset_name.
    ms_dataset_mapping = {}
    hf_dataset_mapping = {}
    for k_name, container in zip(['dataset_id_or_path', 'hf_dataset_id'],
                                 [ms_dataset_mapping, hf_dataset_mapping]):
        for k, v in DATASET_MAPPING.items():
            if v.get(k_name) is None or not v.get('is_main', True):
                continue
            if v[k_name] not in container:
                container[v[k_name]] = []
            container[v[k_name]].append(k)

    dataset_list = []
    res_dataset = []
    for d in dataset_name_list:
        use_hf, d_name = parse_dataset_name(d)[:2]
        if use_hf is None:
            use_hf = strtobool(os.environ.get('USE_HF', 'False'))
        elif isinstance(use_hf, str):
            use_hf = {'hf': 1, 'ms': 0}[use_hf.lower()]
        if '/' in d:
            dataset_list.append((d, use_hf, d_name))
        else:
            res_dataset.append(d)

    extra_dataset = []
    for d, use_hf, d_name in dataset_list:
        dataset_mapping = hf_dataset_mapping if use_hf else ms_dataset_mapping
        if d_name in dataset_mapping:
            for d_name2 in dataset_mapping[d_name]:
                res_dataset.append(d.replace(d_name, d_name2))
        else:
            extra_dataset.append((d, use_hf, d_name))

    for i, (d, use_hf, d_name) in enumerate(extra_dataset):
        d_info = {}
        if use_hf:
            d_info['hf_dataset_id'] = d_name
        else:
            d_info['dataset_id'] = d_name
        d_name2 = f'_{i}'
        register_dataset_info(d_name2, d_info)
        res_dataset.append(d.replace(d_name, d_name2))
    return res_dataset


def get_dataset(
    dataset_name_list: Union[List[str], str],
    dataset_test_ratio: float = 0.,
    dataset_seed: Union[RandomState, int] = 42,
    check_dataset_strategy: Literal['none', 'discard', 'error',
                                    'warning'] = 'none',
    *,
    # for self-cognition
    model_name: Optional[Tuple[str, str]] = None,
    model_author: Optional[Tuple[str, str]] = None
) -> Tuple[HfDataset, Optional[HfDataset]]:
    """Returns train_dataset and val_dataset"""
    if isinstance(dataset_name_list, str):
        dataset_name_list = [dataset_name_list]
    train_dataset_list: List[HfDataset] = []
    val_dataset_list: List[HfDataset] = []
    if isinstance(dataset_seed, int):
        dataset_seed = RandomState(dataset_seed)

    dataset_name_list = _dataset_id_to_name(dataset_name_list)
    for dataset_name in dataset_name_list:
        use_hf, dataset_name, subsets, train_sample, val_sample = parse_dataset_name(
            dataset_name)
        dataset_info = DATASET_MAPPING[dataset_name]
        if subsets is None:
            subsets = dataset_info['subsets']
        if train_sample == -1:
            train_sample = dataset_info.get('train_sample', -1)
        if val_sample is None:
            val_sample = dataset_info.get('val_sample', -1)

        get_function: GetDatasetFunction = dataset_info['get_function']
        is_local = dataset_info.get('is_local', False)
        dataset_id_or_path = dataset_info['dataset_id_or_path']
        remove_useless_columns = dataset_info.get('remove_useless_columns',
                                                  True)
        if not is_local:
            if use_hf is None:
                use_hf = strtobool(os.environ.get('USE_HF', 'False'))
            elif isinstance(use_hf, str):
                use_hf = {'hf': 1, 'ms': 0}[use_hf.lower()]
            dataset_str_f = 'Downloading the dataset from {hub}, dataset_id: {dataset_id}'
            if use_hf:
                dataset_id_or_path = dataset_info['hf_dataset_id']
                dataset_str = dataset_str_f.format(
                    hub='HuggingFace', dataset_id=dataset_id_or_path)
            else:
                dataset_str = dataset_str_f.format(
                    hub='ModelScope', dataset_id=dataset_id_or_path)
            logger.info(dataset_str)
            assert dataset_id_or_path is not None, (
                f'dataset_name: {dataset_name}, use_hf: {use_hf}, '
                f'dataset_id_or_path: {dataset_id_or_path}.')
        dataset = get_function(
            dataset_id_or_path,
            subsets,
            dataset_info['preprocess_func'],
            dataset_info['train_split'],
            dataset_info['val_split'],
            train_sample,
            val_sample,
            random_state=dataset_seed,
            dataset_test_ratio=dataset_test_ratio,
            remove_useless_columns=remove_useless_columns,
            use_hf=use_hf)

        if dataset_name == 'self-cognition':
            assert model_name is not None and model_author is not None
            dataset = _preprocess_self_cognition_dataset(
                dataset, model_name, model_author)
        train_d: HfDataset
        if isinstance(dataset, (list, tuple)):
            train_d, val_d = dataset
        else:
            train_d, val_d = dataset, None
        assert train_d is not None or val_d is not None
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


def load_dataset_from_local(dataset_path_list: Optional[Union[str, List[str]]],
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
            raise ValueError('The custom dataset only supports CSV, JSONL or JSON format. You can refer to the link '
                             '`https://github.com/modelscope/swift/blob/main/docs/source/LLM/自定义与拓展.md#注册数据集的方式` '
                             'for more information.')
        dataset = HfDataset.from_dict(df.to_dict(orient='list'))
        dataset_list.append(preprocess_func(dataset))
    return concatenate_datasets(dataset_list)


def get_local_dataset(_1: str,
                      _2: Optional[List[str]],
                      preprocess_func: PreprocessFunc,
                      train_split: List[str],
                      val_split: List[str],
                      train_sample: int = -1,
                      val_sample: int = -1,
                      random_state: Optional[RandomState] = None,
                      dataset_test_ratio: float = 0.,
                      remove_useless_columns: bool = True,
                      **kwargs) -> Tuple[HfDataset, Optional[HfDataset]]:
    train_dataset = load_dataset_from_local(train_split, preprocess_func)
    val_dataset = load_dataset_from_local(val_split, preprocess_func)
    return _post_preprocess([train_dataset, val_dataset], train_sample,
                            val_sample, random_state, None, dataset_test_ratio,
                            remove_useless_columns)


def _register_dataset_info_file(
        dataset_info_path: Optional[str] = None) -> None:
    if dataset_info_path is None:
        dataset_info_path = os.path.abspath(
            os.path.join(__file__, '..', '..', 'data', 'dataset_info.json'))
    if not os.path.isfile(dataset_info_path):
        return

    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)

    for dataset_name, d_info in dataset_info.items():
        if 'dataset_id' in d_info or 'hf_dataset_id' in d_info:
            register_dataset_info(dataset_name, d_info)
        elif 'train_dataset_path' in d_info or 'val_dataset_path' in d_info:
            base_dir = os.path.abspath(
                os.path.join(__file__, '..', '..', 'data'))
            register_local_dataset(dataset_name,
                                   d_info.pop('train_dataset_path', None),
                                   d_info.pop('val_dataset_path', None),
                                   base_dir, **d_info)
    logger.info('Successfully registered `dataset_info.json`')


_register_dataset_info_file()
