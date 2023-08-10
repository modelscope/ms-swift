from functools import partial
from typing import Callable, List, Optional, Tuple

import numpy as np
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets
from modelscope import MsDataset

from swift.utils import get_seed


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
        'instruction': new_instruction,
        'output': dataset['output']
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
        'wyj123456/instinwild', subset_name='default', split='train').to_hf_dataset()
    return _processing_alpaca(dataset)


def get_instinwild_en_dataset():
    dataset: HfDataset = MsDataset.load(
        'wyj123456/instinwild', subset_name='subset', split='train').to_hf_dataset()
    return _processing_alpaca(dataset)


DATASET_MAPPING = {
    'alpaca-en': get_alpaca_gpt4_en_dataset,
    'alpaca-zh': get_alpaca_gpt4_zh_dataset,
    'finance-en': get_finance_en_dataset,
    'multi-alpaca-all': get_multi_alpaca_all,
    **{
        f'multi-alpaca-{k}': partial(get_multi_alpaca, k)
        for k in _multi_alpaca_language_list
    },
    'code-en': get_code_alpaca_en_dataset,
    'instinwild-zh': get_instinwild_zh_dataset,
    'instinwild-en': get_instinwild_en_dataset,
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
