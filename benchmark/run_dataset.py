import os
from typing import List

from datasets import concatenate_datasets

from swift.llm import (DATASET_MAPPING, MODEL_MAPPING, DatasetName, ModelType,
                       dataset_map, get_dataset, get_model_tokenizer,
                       get_template)
from swift.utils import stat_array


def get_dataset_name_list() -> List[str]:
    res = []
    for k in DatasetName.__dict__.keys():
        if k.startswith('__'):
            continue
        res.append(getattr(DatasetName, k))
    return res


def write_dataset_info(fpath: str, model_type: str) -> None:
    """Statistical analysis was performed on the dataset"""
    if not os.path.exists(fpath):
        with open(fpath, 'w') as f:
            f.write("""| Dataset Name | Dataset ID | Size | Statistic (token) |
| ------------ | ---------- | ---- | ----------------- |
""")
    with open(fpath, 'r') as f:
        text_list = f.readlines()[2:]
    ignore_dataset = set([text.split('|', 2)[1] for text in text_list])
    dataset_name_list = get_dataset_name_list()
    _, tokenizer = get_model_tokenizer(model_type, load_model=False)
    template_type = MODEL_MAPPING[model_type]['template']
    template = get_template(template_type, tokenizer)
    for dataset_name in dataset_name_list:
        if dataset_name in ignore_dataset:
            continue
        dataset_info = DATASET_MAPPING[dataset_name]
        train_dataset, val_dataset = get_dataset([dataset_name])
        raw_dataset = train_dataset
        if val_dataset is not None:
            raw_dataset = concatenate_datasets([raw_dataset, val_dataset])
        dataset = dataset_map(raw_dataset, template.encode)

        _token_len = []
        input_ids = dataset['input_ids']
        for i in range(len(dataset)):
            _token_len.append(len(input_ids[i]))
        stat = stat_array(_token_len)[0]
        stat_str = f"{stat['mean']:.1f}±{stat['std']:.1f}, min={stat['min']}, max={stat['max']}"
        url = f"https://modelscope.cn/datasets/{dataset_info['dataset_id_or_path']}/summary"
        with open(fpath, 'a') as f:
            f.write(
                f"""|{dataset_name}|[{dataset_info['dataset_id_or_path']}]({url})|{stat['size']}|{stat_str}|
""")


if __name__ == '__main__':
    write_dataset_info('dataset_info.txt', ModelType.qwen_7b_chat)
