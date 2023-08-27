import os
import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import json
import numpy as np
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets
from modelscope import MsDataset

from swift.utils import get_seed
from .utils import download_dataset


def _processing_alpaca(
        dataset: HfDataset,
        preprocess_input: Optional[Callable[[str], str]] = None) -> HfDataset:
    instruction = dataset['instruction']
    input_ = dataset['input']
    new_instruction = []
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
    return _processing_alpaca(dataset)


def get_alpaca_gpt4_zh_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'AI-ModelScope/alpaca-gpt4-data-zh', split='train').to_hf_dataset()

    def _preprocess_input(inp: str) -> str:
        if inp.startswith('输入：'):
            inp = inp[3:]
        return inp

    return _processing_alpaca(dataset, _preprocess_input)


def get_finance_en_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'wyj123456/finance_en', split='train').to_hf_dataset()
    return _processing_alpaca(dataset)


_multi_alpaca_language_list = [
    'ar', 'de', 'es', 'fr', 'id', 'ja', 'ko', 'pt', 'ru', 'th', 'vi'
]


def get_multi_alpaca(subset_name: str) -> HfDataset:
    """
    subset_name:
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
    dataset: HfDataset = MsDataset.load(
        'damo/nlp_polylm_multialpaca_sft',
        subset_name=subset_name,
        split='train').to_hf_dataset()
    return _processing_alpaca(dataset)


def get_multi_alpaca_all() -> HfDataset:
    dataset_list = []
    for subset_name in _multi_alpaca_language_list:
        dataset = get_multi_alpaca(subset_name)
        dataset_list.append(dataset)
    dataset = concatenate_datasets(dataset_list)
    return dataset


def get_code_alpaca_en_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'wyj123456/code_alpaca_en', split='train').to_hf_dataset()
    return _processing_alpaca(dataset)


def get_instinwild_zh_dataset():
    dataset: HfDataset = MsDataset.load(
        'wyj123456/instinwild', subset_name='default',
        split='train').to_hf_dataset()
    return _processing_alpaca(dataset)


def get_instinwild_en_dataset():
    dataset: HfDataset = MsDataset.load(
        'wyj123456/instinwild', subset_name='subset',
        split='train').to_hf_dataset()
    return _processing_alpaca(dataset)


def get_cot_en_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'YorickHe/CoT', split='train').to_hf_dataset()
    return _processing_alpaca(dataset)


def get_cot_zh_dataset() -> HfDataset:
    dataset: HfDataset = MsDataset.load(
        'YorickHe/CoT_zh', split='train').to_hf_dataset()
    return _processing_alpaca(dataset)


def _processing_captions(dataset: HfDataset,
                         get_image_path: Callable[[Dict[str, Any]], str],
                         response_key: str) -> HfDataset:
    query_format = '<img>{image_path}</img>Please describe the image.'
    query = [
        query_format.format(image_path=get_image_path(d)) for d in dataset
    ]
    dataset = HfDataset.from_dict({
        'query': query,
        'response': dataset[response_key]
    })
    return dataset


def get_coco_en_dataset() -> HfDataset:
    dataset_id = 'modelscope/coco_2014_caption'
    dataset_dict = MsDataset.load(dataset_id)
    dataset: HfDataset = concatenate_datasets([
        dataset_dict['train'].to_hf_dataset(),
        dataset_dict['validation'].to_hf_dataset()
    ])
    dataset._info.features._column_requires_decoding['image'] = False
    return _processing_captions(dataset, lambda d: d['image']['path'],
                                'caption')


def _filter_agent_dataset(
        dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    res = []
    for d in dataset:
        system = d['conversations'][0]['value']
        find_list = re.findall(r'{"plugin_name": "(.+?)"', system)
        if len(set(find_list)) <= 1:
            continue
        res.append(d)
    return res


def _process_agent_dataset(dataset: List[Dict[str, Any]]) -> HfDataset:
    system = []
    query = []
    response = []
    history = []
    for d in dataset:
        conversations = d['conversations']
        assert len(conversations) >= 3
        assert conversations[0]['from'] == 'system'
        system.append(conversations[0]['value'])
        query.append(conversations[-2]['value'])
        response.append(conversations[-1]['value'])
        h = None
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


def get_agent_zh_dataset():
    model_id = 'damo/MSAgent-Bench'
    files = ['train.jsonl', 'dev.jsonl']
    dataset_dir = download_dataset(model_id, files)
    dataset = []
    for file in files:
        fpath = os.path.join(dataset_dir, file)
        with open(fpath, 'r') as f:
            text = f.read()
        text = text.replace('}{', '},{')
        text = f'[{text}]'
        dataset += json.loads(text)
    dataset = _filter_agent_dataset(dataset)
    return _process_agent_dataset(dataset)


DATASET_MAPPING = {
    # nlp
    'alpaca-en': get_alpaca_gpt4_en_dataset,
    'alpaca-zh': get_alpaca_gpt4_zh_dataset,
    'finance-en': get_finance_en_dataset,
    'multi-alpaca-all': get_multi_alpaca_all,
    **{
        f'multi-alpaca-{k}': partial(get_multi_alpaca, k)
        for k in _multi_alpaca_language_list
    },
    'code-en': get_code_alpaca_en_dataset,
    'instinwild-en': get_instinwild_en_dataset,
    'instinwild-zh': get_instinwild_zh_dataset,
    'cot-en': get_cot_en_dataset,
    'cot-zh': get_cot_zh_dataset,
    'agent-zh': get_agent_zh_dataset,
    # multi-modal
    'coco-en': get_coco_en_dataset,
}


def get_dataset(dataset_name_list: List[str]) -> HfDataset:
    dataset_list = []
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
