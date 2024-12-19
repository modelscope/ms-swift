import os
import re

import numpy as np

from swift.llm import DATASET_MAPPING, EncodePreprocessor, get_model_tokenizer, get_template, load_dataset
from swift.utils import stat_array

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def get_cache_mapping(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()
    idx = text.find('| Dataset ID |')
    text = text[idx:]
    text_list = text.split('\n')[2:]
    cache_mapping = {}  # dataset_id -> (dataset_size, stat)
    for text in text_list:
        if not text:
            continue
        items = text.split('|')
        key = items[1] if items[1] != '-' else items[6]
        key = re.search(r'\[(.+?)\]', key).group(1)
        stat = items[3:5]
        if stat[0] == '-':
            stat = ('huge dataset', '-')
        cache_mapping[key] = stat
    return cache_mapping


def get_dataset_id(key):
    for dataset_id in key:
        if dataset_id is not None:
            break
    return dataset_id


def run_dataset(key, template, cache_mapping):
    ms_id, hf_id, _ = key
    dataset_meta = DATASET_MAPPING[key]
    tags = ', '.join(tag for tag in dataset_meta.tags) or '-'
    dataset_id = ms_id or hf_id
    use_hf = ms_id is None
    if ms_id is not None:
        ms_id = f'[{ms_id}](https://modelscope.cn/datasets/{ms_id})'
    else:
        ms_id = '-'
    if hf_id is not None:
        hf_id = f'[{hf_id}](https://huggingface.co/datasets/{hf_id})'
    else:
        hf_id = '-'
    subsets = '<br>'.join(subset.name for subset in dataset_meta.subsets)

    if dataset_meta.huge_dataset:
        dataset_size = 'huge dataset'
        stat_str = '-'
    elif dataset_id in cache_mapping:
        dataset_size, stat_str = cache_mapping[dataset_id]
    else:
        num_proc = 4
        dataset, _ = load_dataset(
            f'{dataset_id}:all', strict=False, num_proc=num_proc, use_hf=use_hf, download_mode='force_redownload')
        dataset_size = len(dataset)
        random_state = np.random.RandomState(42)
        idx_list = random_state.choice(dataset_size, size=min(dataset_size, 100000), replace=False)
        encoded_dataset = EncodePreprocessor(template)(dataset.select(idx_list), num_proc=num_proc)

        input_ids = encoded_dataset['input_ids']
        token_len = [len(tokens) for tokens in input_ids]
        stat = stat_array(token_len)[0]
        stat_str = f"{stat['mean']:.1f}±{stat['std']:.1f}, min={stat['min']}, max={stat['max']}"

    return f'|{ms_id}|{subsets}|{dataset_size}|{stat_str}|{tags}|{hf_id}|'


def write_dataset_info() -> None:
    fpaths = ['docs/source/Instruction/支持的模型和数据集.md', 'docs/source_en/Instruction/Supported-models-and-datasets.md']
    cache_mapping = get_cache_mapping(fpaths[0])
    res_text_list = []
    res_text_list.append('| Dataset ID | Subset Name | Dataset Size | Statistic (token) | Tags | HF Dataset ID |')
    res_text_list.append('| ---------- | ----------- | -------------| ------------------| ---- | ------------- |')

    all_keys = list(DATASET_MAPPING.keys())
    all_keys = sorted(all_keys, key=lambda x: get_dataset_id(x))
    _, tokenizer = get_model_tokenizer('Qwen/Qwen2.5-7B-Instruct', load_model=False)
    template = get_template(tokenizer.model_meta.template, tokenizer)
    try:
        for i, key in enumerate(all_keys):
            res = run_dataset(key, template, cache_mapping)
            res_text_list.append(res)
            print(res)
    finally:
        for fpath in fpaths:
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read()
            idx = text.find('| Dataset ID |')

            new_text = '\n'.join(res_text_list)
            text = text[:idx] + new_text + '\n'
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(text)
    print(f'数据集总数: {len(all_keys)}')


if __name__ == '__main__':
    write_dataset_info()
