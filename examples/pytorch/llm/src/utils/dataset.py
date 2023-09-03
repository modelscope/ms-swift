import ast
import os
import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import json
import numpy as np
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets
from modelscope import MsDataset
from tqdm.auto import tqdm

from swift.utils import get_seed
from .preprocess import History
from .utils import download_dataset


def _process_alpaca_dataset(
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
    return _process_alpaca_dataset(dataset)


def get_alpaca_gpt4_zh_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'AI-ModelScope/alpaca-gpt4-data-zh', split='train').to_hf_dataset()

    def _preprocess_input(inp: str) -> str:
        if inp.startswith('输入：'):
            inp = inp[3:]
        return inp

    return _process_alpaca_dataset(dataset, _preprocess_input)


def get_finance_en_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'wyj123456/finance_en', split='train').to_hf_dataset()
    return _process_alpaca_dataset(dataset)


_multi_alpaca_language_list = [
    'ar', 'de', 'es', 'fr', 'id', 'ja', 'ko', 'pt', 'ru', 'th', 'vi'
]


def _get_multi_alpaca(subset_name: str) -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'damo/nlp_polylm_multialpaca_sft',
        subset_name=subset_name,
        split='train').to_hf_dataset()
    return _process_alpaca_dataset(dataset)


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
    return _process_alpaca_dataset(dataset)


def get_instinwild_zh_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'wyj123456/instinwild', subset_name='default',
        split='train').to_hf_dataset()
    return _process_alpaca_dataset(dataset)


def get_instinwild_en_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'wyj123456/instinwild', subset_name='subset',
        split='train').to_hf_dataset()
    return _process_alpaca_dataset(dataset)


def get_cot_en_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'YorickHe/CoT', split='train').to_hf_dataset()
    return _process_alpaca_dataset(dataset)


def get_cot_zh_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'YorickHe/CoT_zh', split='train').to_hf_dataset()
    return _process_alpaca_dataset(dataset)


def _process_mutimodal_dataset(dataset: HfDataset, prompt: str, image_key: str,
                               response_key: str) -> HfDataset:
    dataset._info.features._column_requires_decoding['image'] = False
    query_format = f'<img>{{image_path}}</img>{prompt}'
    query = [
        query_format.format(image_path=d[image_key]['path']) for d in dataset
    ]
    dataset = HfDataset.from_dict({
        'query': query,
        'response': dataset[response_key]
    })
    return dataset


def get_coco_en_dataset() -> HfDataset:
    dataset_dict = MsDataset.load('modelscope/coco_2014_caption')
    dataset: HfDataset = concatenate_datasets([
        dataset_dict['train'].to_hf_dataset(),
        dataset_dict['validation'].to_hf_dataset()
    ])
    return _process_mutimodal_dataset(dataset, 'please describe the image',
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


def _process_agent_dataset(dataset: List[Dict[str, str]]) -> HfDataset:
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
    return _process_agent_dataset(dataset)


_firefly_kind_list = [
    'ProseGeneration', 'MRC', 'JinYongGeneration', 'TextCorrection',
    'ClassicalChinese', 'BELLE', 'StoryGeneration', 'Couplet', 'Cot',
    'Dictionary', 'Translation', 'Program', 'SentimentAnalyze', 'OpenQA',
    'AncientPoem', 'TextMatching', 'NLI', 'Summary', 'KeywordRecognition',
    'ProductDesc', 'LyricGeneration', 'Composition', 'MusicComment', 'NER'
]


def _process_firefly(dataset: List[Dict[str, str]],
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
    return _process_firefly(dataset, kind_list)


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
    return _process_alpaca_dataset(dataset)


DATASET_MAPPING = {
    # nlp
    'alpaca-en': get_alpaca_gpt4_en_dataset,
    'alpaca-zh': get_alpaca_gpt4_zh_dataset,
    'finance-en': get_finance_en_dataset,
    'multi-alpaca-all': get_multi_alpaca_all,
    'code-en': get_code_alpaca_en_dataset,
    'instinwild-en': get_instinwild_en_dataset,
    'instinwild-zh': get_instinwild_zh_dataset,
    'cot-en': get_cot_en_dataset,
    'cot-zh': get_cot_zh_dataset,
    'damo-agent-mini-zh': partial(get_damo_agent_zh_dataset, use_mini=True),
    'damo-agent-zh': get_damo_agent_zh_dataset,  # containing normal chat
    'firefly-all-zh': get_firefly_all_zh_dataset,
    'poetry-zh': get_poetry_zh_dataset,
    'instruct-en': get_instruct_en_dataset,
    'gpt4all-en': get_gpt4all_en_dataset,
    # multi-modal
    'coco-en': get_coco_en_dataset,
}


def get_dataset(dataset_name_list: List[str]) -> HfDataset:
    dataset_list: List[HfDataset] = []
    for dataset_name in dataset_name_list:
        get_function = DATASET_MAPPING[dataset_name]
        dataset_list.append(get_function())
    dataset = concatenate_datasets(dataset_list)
    return dataset


def process_dataset(dataset: HfDataset, dataset_test_size: float,
                    dataset_sample: int,
                    dataset_seed: int) -> Tuple[HfDataset, HfDataset]:
    random_state = np.random.RandomState(dataset_seed)
    if dataset_sample >= 0:
        index = random_state.permutation(len(dataset))[:dataset_sample]
        dataset = dataset.select(index)
    dataset = dataset.train_test_split(
        dataset_test_size, seed=get_seed(random_state))
    return dataset['train'], dataset['test']
