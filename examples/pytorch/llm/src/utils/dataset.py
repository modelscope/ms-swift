# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import os
import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import json
import numpy as np
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets
from modelscope import MsDataset
from tqdm.auto import tqdm

from .preprocess import History
from .utils import download_dataset


def _preprocess_alpaca_dataset(
        dataset: HfDataset,
        preprocess_input: Optional[Callable[[str], str]] = None) -> HfDataset:
    instruction = dataset['instruction']
    input_ = dataset['input']
    new_instruction: List[str] = []
    for inst, inp in zip(instruction, input_):
        if inp is None:
            inp = ''
        if preprocess_input is not None:
            inp = preprocess_input(inp)
        inst = f'{inst}\n{inp}'
        new_instruction.append(inst)
    dataset = HfDataset.from_dict({
        'query': new_instruction,
        'response': dataset['output']
    })
    return dataset


def get_alpaca_gpt4_en_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'AI-ModelScope/alpaca-gpt4-data-en', split='train').to_hf_dataset()
    return _preprocess_alpaca_dataset(dataset)


def get_alpaca_gpt4_zh_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'AI-ModelScope/alpaca-gpt4-data-zh', split='train').to_hf_dataset()

    def _preprocess_input(inp: str) -> str:
        if inp.startswith('输入：'):
            inp = inp[3:]
        return inp

    return _preprocess_alpaca_dataset(dataset, _preprocess_input)


def get_finance_en_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'wyj123456/finance_en', split='train').to_hf_dataset()
    return _preprocess_alpaca_dataset(dataset)


_multi_alpaca_language_list = [
    'ar', 'de', 'es', 'fr', 'id', 'ja', 'ko', 'pt', 'ru', 'th', 'vi'
]


def _get_multi_alpaca(subset_name: str) -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'damo/nlp_polylm_multialpaca_sft',
        subset_name=subset_name,
        split='train').to_hf_dataset()
    return _preprocess_alpaca_dataset(dataset)


def get_multi_alpaca(language_list: List[str]) -> HfDataset:
    """language_list:
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
    """
    dataset_list: List[HfDataset] = []
    for subset_name in language_list:
        dataset = _get_multi_alpaca(subset_name)
        dataset_list.append(dataset)
    dataset = concatenate_datasets(dataset_list)
    return dataset


def get_multi_alpaca_all() -> HfDataset:
    return get_multi_alpaca(_multi_alpaca_language_list)


def get_code_alpaca_en_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'wyj123456/code_alpaca_en', split='train').to_hf_dataset()
    return _preprocess_alpaca_dataset(dataset)


def get_instinwild_zh_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'wyj123456/instinwild', subset_name='default',
        split='train').to_hf_dataset()
    return _preprocess_alpaca_dataset(dataset)


def get_instinwild_en_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'wyj123456/instinwild', subset_name='subset',
        split='train').to_hf_dataset()
    return _preprocess_alpaca_dataset(dataset)


def get_cot_en_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'YorickHe/CoT', split='train').to_hf_dataset()
    return _preprocess_alpaca_dataset(dataset)


def get_cot_zh_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'YorickHe/CoT_zh', split='train').to_hf_dataset()
    return _preprocess_alpaca_dataset(dataset)


def _preprocess_mutimodal_dataset(dataset: HfDataset, prompt: str,
                                  image_key: str,
                                  response_key: str) -> HfDataset:
    dataset._info.features._column_requires_decoding['image'] = False
    query_format = f'<img>{{image_path}}</img>{prompt}'
    query = []
    response = []
    for d in tqdm(dataset):
        query.append(query_format.format(image_path=d[image_key]['path']))
        if '&&' in d[response_key]:
            d[response_key] = d[response_key].split('&&')[0]
        response.append(d[response_key])
    dataset = HfDataset.from_dict({'query': query, 'response': response})
    return dataset


def get_coco_en_dataset() -> HfDataset:
    dataset_dict = MsDataset.load('modelscope/coco_2014_caption')
    dataset: HfDataset = concatenate_datasets([
        dataset_dict['train'].to_hf_dataset(),
        dataset_dict['validation'].to_hf_dataset()
    ])
    return _preprocess_mutimodal_dataset(dataset, 'please describe the image',
                                         'image', 'caption')


def _filter_agent_dataset(dataset: List[Dict[str, Any]],
                          use_mini: bool) -> List[Dict[str, Any]]:
    if use_mini:
        pattern = r'\d\. {"plugin_name": "(.+?)"'
    else:
        pattern = r'\d\. {"(?:plugin_)?name": "(.+?)"'
    res: List[Dict[str, Any]] = []
    for d in tqdm(dataset):
        idx = d['conversations'].find(r"'from': 'user")
        if idx == -1:
            continue
        find_list = re.findall(pattern, d['conversations'][:idx])
        # remove dirty data
        if len(set(find_list)) <= 1:
            continue
        d['conversations'] = ast.literal_eval(d['conversations'])
        if len(d['conversations']) == 1:
            continue
        res.append(d)
    return res


def _preprocess_agent_dataset(dataset: List[Dict[str, str]]) -> HfDataset:
    system: List[str] = []
    query: List[str] = []
    response: List[str] = []
    history: List[Optional[History]] = []
    for d in tqdm(dataset):
        conversations = d['conversations']
        assert len(conversations) >= 3
        assert conversations[0]['from'] == 'system'
        system.append(conversations[0]['value'])
        query.append(conversations[-2]['value'])
        response.append(conversations[-1]['value'])
        h: Optional[History] = None
        if len(conversations) > 3:
            assert len(conversations) % 2 == 1
            conversations_h = conversations[1:-2]
            h = [(q['value'], r['value'])
                 for q, r in zip(conversations_h[::2], conversations_h[1::2])]
        history.append(h)
    dataset = HfDataset.from_dict({
        'system': system,
        'query': query,
        'response': response,
        'history': history
    })
    return dataset


def get_damo_agent_zh_dataset(use_mini: bool = False) -> HfDataset:
    dataset_dict = MsDataset.load('damo/MSAgent-Bench')
    dataset: HfDataset = concatenate_datasets([
        dataset_dict['train'].to_hf_dataset(),
        dataset_dict['validation'].to_hf_dataset()
    ])
    dataset = _filter_agent_dataset(dataset, use_mini)
    return _preprocess_agent_dataset(dataset)


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
    for d in dataset:
        if d['kind'] not in kind_set:
            continue
        query.append(d['input'])
        response.append(d['target'])

    return HfDataset.from_dict({
        'query': query,
        'response': response,
    })


def get_firefly_zh_dataset(kind_list: List[str]) -> HfDataset:
    model_id = 'wyj123456/firefly'
    file = 'firefly-train-1.1M.jsonl'
    dataset_dir = download_dataset(model_id, [file])
    fpath = os.path.join(dataset_dir, file)
    with open(fpath, 'r') as f:
        text = f.read()
        text = text.replace('}{', '},{')
        text = f'[{text}]'
        dataset = json.loads(text)
    return _preprocess_firefly(dataset, kind_list)


def get_firefly_all_zh_dataset() -> HfDataset:
    return get_firefly_zh_dataset(_firefly_kind_list)


def get_poetry_zh_dataset() -> HfDataset:
    dataset_dict = MsDataset.load('modelscope/chinese-poetry-collection')
    dataset: HfDataset = concatenate_datasets([
        dataset_dict['train'].to_hf_dataset(),
        dataset_dict['test'].to_hf_dataset()
    ])
    return HfDataset.from_dict({
        'query': ['写诗'] * len(dataset),
        'response': dataset['text1']
    })


def get_instruct_en_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'wyj123456/instruct', split='train').to_hf_dataset()
    dataset = dataset.rename_column('prompt', 'query')
    dataset = dataset.rename_column('completion', 'response')
    return dataset


def get_gpt4all_en_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'wyj123456/GPT4all', split='train').to_hf_dataset()
    return _preprocess_alpaca_dataset(dataset)


def _preprocess_cls_dataset(dataset: HfDataset, cls_mapping: List[str],
                            task: str, pair_seq: bool) -> HfDataset:
    category = ', '.join(cls_mapping)
    if pair_seq:
        input_ = 'Sentence1: {sentence1}\nSentence2: {sentence2}'
    else:
        input_ = 'Sentence: {sentence}'
    prompt = f"""Task: {task}
{input_}
Category: {category}
Label: """
    query = []
    response = []
    for d in tqdm(dataset):
        if d['label'] is None:
            continue
        if pair_seq:
            q = prompt.format(
                sentence1=d['sentence1'], sentence2=d['sentence2'])
        else:
            q = prompt.format(sentence=d['sentence'])
        query.append(q)
        label = int(d['label'])
        response.append(cls_mapping[label])
    return HfDataset.from_dict({'query': query, 'response': response})


def get_cmnli_zh_dataset() -> HfDataset:
    """Natural Language Inference"""
    dataset_dict = MsDataset.load('clue', subset_name='cmnli')
    dataset: HfDataset = concatenate_datasets([
        dataset_dict['train'].to_hf_dataset(),
        dataset_dict['validation'].to_hf_dataset(),
        dataset_dict['test'].to_hf_dataset(),
    ])
    cls_mapping = ['neutral', 'entailment', 'contradiction']
    return _preprocess_cls_dataset(dataset, cls_mapping,
                                   'Natural Language Inference', True)


def get_jd_zh_dataset() -> HfDataset:
    """Sentiment classification"""
    dataset_dict = MsDataset.load('DAMO_NLP/jd')
    dataset: HfDataset = concatenate_datasets([
        dataset_dict['train'].to_hf_dataset(),
        dataset_dict['validation'].to_hf_dataset()
    ])

    cls_mapping = ['negative', 'positive']
    return _preprocess_cls_dataset(dataset, cls_mapping,
                                   'Sentiment Classification', False)


def _preprocess_dureader_robust(dataset: HfDataset) -> HfDataset:
    prompt = """Task: Question Generation
Context: {context}
Answer: {answer}
Question: """
    query = []
    response = []
    for d in dataset:
        answer, context = d['text1'].split('[SEP]')
        q = prompt.format(context=context, answer=answer)
        query.append(q)
        response.append(d['text2'])
    return HfDataset.from_dict({'query': query, 'response': response})


def get_dureader_robust_qg_zh_dataset() -> HfDataset:
    """Question Generation"""
    dataset_dict = MsDataset.load('modelscope/DuReader_robust-QG')
    dataset: HfDataset = concatenate_datasets([
        dataset_dict['train'].to_hf_dataset(),
        dataset_dict['validation'].to_hf_dataset(),
        dataset_dict['test'].to_hf_dataset()
    ])
    return _preprocess_dureader_robust(dataset)


def _preprocess_medical(dataset: HfDataset, subset_name: str) -> HfDataset:
    query = []
    for d in tqdm(dataset):
        if subset_name == 'zh':
            q = d['instruction']
        else:
            q = d['input']
        query.append(q)
    return HfDataset.from_dict({'query': query, 'response': dataset['output']})


def get_medical_dataset(subset_name: str,
                        dataset_sample: int = -1) -> HfDataset:
    """
    mode: Literal['en', zh]
    """
    dataset_dict = MsDataset.load(
        'huangjintao/medical_zh', subset_name=subset_name)
    dataset: HfDataset = concatenate_datasets([
        dataset_dict['train'].to_hf_dataset(),
        dataset_dict['val'].to_hf_dataset(),
        dataset_dict['test'].to_hf_dataset(),
    ])
    if dataset_sample != -1:
        idxs = np.random.permutation(dataset_sample)
        dataset = dataset.select(idxs)
    return _preprocess_medical(dataset, subset_name)


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
            h.append((c['human'], c['assistant']))
        history.append(h)
    return HfDataset.from_dict({
        'query': query,
        'response': response,
        'history': history
    })


def get_sharegpt_dataset(subset_name_list: List[str]) -> HfDataset:
    dataset_list = []
    for subset_name in subset_name_list:
        dataset = MsDataset.load(
            'huangjintao/sharegpt', subset_name=subset_name,
            split='train').to_hf_dataset()
        dataset_list.append(dataset)
    dataset = concatenate_datasets(dataset_list)
    return _preprocess_sharegpt(dataset)


_sharegpt_zh_subset_list = ['common-zh', 'computer-zh', 'unknow-zh']

_sharegpt_en_subset_list = ['common-en', 'computer-en']


def get_sharegpt_all_zh_dataset():
    """multi-round chat"""
    return get_sharegpt_dataset(_sharegpt_zh_subset_list)


def get_sharegpt_all_en_dataset():
    """multi-round chat"""
    return get_sharegpt_dataset(_sharegpt_en_subset_list)


def get_cls_fudan_news_zh() -> HfDataset:
    """Sequence Classification """
    dataset = MsDataset.load('damo/zh_cls_fudan-news').to_hf_dataset()
    return HfDataset.from_dict({
        'query': dataset['prompt'],
        'response': dataset['answer']
    })


def get_ner_jave_zh() -> HfDataset:
    """Named Entity Recognition"""
    dataset = MsDataset.load('damo/zh_ner-JAVE').to_hf_dataset()
    return HfDataset.from_dict({
        'query': dataset['prompt'],
        'response': dataset['answer']
    })


DATASET_MAPPING = {
    # nlp chat
    'alpaca-en':
    get_alpaca_gpt4_en_dataset,
    'alpaca-zh':
    get_alpaca_gpt4_zh_dataset,
    'finance-en':
    get_finance_en_dataset,
    'multi-alpaca-all':
    get_multi_alpaca_all,
    'code-en':
    get_code_alpaca_en_dataset,
    'instinwild-en':
    get_instinwild_en_dataset,
    'instinwild-zh':
    get_instinwild_zh_dataset,
    'cot-en':
    get_cot_en_dataset,
    'cot-zh':
    get_cot_zh_dataset,
    'firefly-all-zh':
    get_firefly_all_zh_dataset,
    'poetry-zh':
    get_poetry_zh_dataset,
    'instruct-en':
    get_instruct_en_dataset,
    'gpt4all-en':
    get_gpt4all_en_dataset,
    'medical-en':
    partial(get_medical_dataset, subset_name='en'),
    'medical-zh':
    partial(get_medical_dataset, subset_name='zh'),
    'medical-mini-zh':
    partial(get_medical_dataset, subset_name='zh', dataset_sample=100000),
    # multi-round chat
    'damo-agent-mini-zh':
    partial(get_damo_agent_zh_dataset, use_mini=True),
    'damo-agent-zh':
    get_damo_agent_zh_dataset,  # containing normal chat
    'sharegpt-en':
    get_sharegpt_all_en_dataset,
    'sharegpt-zh':
    get_sharegpt_all_zh_dataset,
    # nlp text-generation (please use model:base, template:default-generation)
    'cmnli-zh':
    get_cmnli_zh_dataset,
    'jd-zh':
    get_jd_zh_dataset,
    'dureader-robust-zh':
    get_dureader_robust_qg_zh_dataset,
    # multi-modal chat
    'coco-en':
    get_coco_en_dataset,

    # other (e.g. example dataset for specific model)
    'cls-fudan-news-zh':
    get_cls_fudan_news_zh,  # seqgpt-560m
    'ner-jave-zh':
    get_ner_jave_zh,  # seqgpt-560m
}


def get_dataset(dataset_name_list: List[str]) -> HfDataset:
    dataset_list: List[HfDataset] = []
    for dataset_name in dataset_name_list:
        get_function = DATASET_MAPPING[dataset_name]
        dataset_list.append(get_function())
    dataset = concatenate_datasets(dataset_list)
    return dataset
