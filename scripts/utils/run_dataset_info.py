import os

from datasets import concatenate_datasets

from swift.llm import (DATASET_MAPPING, DatasetName, ModelType, dataset_map,
                       get_dataset, get_default_template_type,
                       get_model_tokenizer, get_template)
from swift.utils import stat_array


def write_dataset_info() -> None:
    fpaths = [
        'docs/source/LLM/支持的模型和数据集.md',
        'docs/source_en/LLM/Supported-models-datasets.md'
    ]
    pre_texts = []
    for fpath in fpaths:
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read()
            idx = text.find('| Dataset Name |')
            pre_texts.append(text[:idx])

            text = text[idx:]
            text_list = [t for t in text.split('\n') if len(t.strip()) > 0]
        else:
            text_list = []
            pre_texts.append('')

    res_text_list = []

    res_text_list.append(
        '| Dataset Name | Dataset ID | Train Size | Val Size | Statistic (token) | Tags |'
    )
    res_text_list.append(
        '| ------------ | ---------- | ---------- | -------- | ----------------- | ---- |'
    )
    if len(text_list) >= 2:
        text_list = text_list[2:]
    else:
        text_list = []

    ignore_dataset = {
        text.split('|', 2)[1].lstrip('🔥 '): text
        for text in text_list
    }
    dataset_name_list = DatasetName.get_dataset_name_list()
    mapping = {}
    _iter = zip(
        ['llm', 'vision', 'audio'],
        [
            ModelType.qwen_7b_chat, ModelType.qwen_vl_chat,
            ModelType.qwen_audio_chat
        ],
    )
    for task_type, model_type in _iter:
        _, tokenizer = get_model_tokenizer(model_type, load_model=False)
        template_type = get_default_template_type(model_type)
        template = get_template(template_type, tokenizer)
        mapping[task_type] = template
    for dataset_name in dataset_name_list:
        dataset_info = DATASET_MAPPING[dataset_name]
        tags = dataset_info.get('tags', [])
        if 'audio' in tags:
            template = mapping['audio']
        elif 'vision' in tags:
            template = mapping['vision']
        else:
            template = mapping['llm']
        if dataset_name in ignore_dataset:
            train_size, val_size, stat_str = ignore_dataset[
                dataset_name].split('|')[3:6]
        else:
            train_dataset, val_dataset = get_dataset([dataset_name])
            train_size = len(train_dataset)
            val_size = 0
            if val_dataset is not None:
                val_size = len(val_dataset)

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

        if '🔥' in tags:
            tags.remove('🔥')
            dataset_name = '🔥' + dataset_name
        tags_str = ', '.join(tags)
        if len(tags_str) == 0:
            tags_str = '-'
        res_text_list.append(
            f"|{dataset_name}|[{dataset_info['dataset_id_or_path']}]({url})|{train_size}|"
            f'{val_size}|{stat_str}|{tags_str}|')
    print(f'数据集总数: {len(dataset_name_list)}')

    for idx in range(len(fpaths)):
        text = '\n'.join(res_text_list)
        text = pre_texts[idx] + text + '\n'
        with open(fpaths[idx], 'w', encoding='utf-8') as f:
            f.write(text)


if __name__ == '__main__':
    write_dataset_info()
