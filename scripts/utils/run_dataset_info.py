import os
from typing import Dict, List

from swift.llm import DATASET_MAPPING, EncodePreprocessor, TemplateType, get_model_tokenizer, get_template
from swift.llm.dataset.loader import load_dataset
from swift.utils import stat_array


def find_dataset_row(dataset: str, ignore_datasets: Dict[str, str]):
    if not dataset:
        return None
    for ignore_dataset in ignore_datasets:
        if dataset in ignore_dataset:
            return ignore_datasets[ignore_dataset]
    return None


def write_dataset_info() -> None:
    fpaths = ['docs/source/Instruction/支持的模型和数据集.md', 'docs/source_en/Instruction/Supported-models-datasets.md']
    pre_texts = []
    for fpath in fpaths:
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read()
            idx = text.find('| MS Dataset ID |')
            pre_texts.append(text[:idx])

            text = text[idx:]
            text_list = [t for t in text.split('\n') if len(t.strip()) > 0]
        else:
            text_list = []
            pre_texts.append('')

    res_text_list = []

    res_text_list.append('| MS Dataset ID | HF Dataset ID | Subset name | Real Subset  |'
                         ' Subset split | Dataset Size | Statistic (token) | Tags |')
    res_text_list.append('| ------------ | ------------- | ----------- |------------- |'
                         ' -------------| -------------| ----------------- | ---- |')
    if len(text_list) >= 2:
        text_list = text_list[2:]
    else:
        text_list = []

    hf_ignore_datasets = {text.split('|', 3)[2].lstrip('🔥 '): text for text in text_list}
    ms_ignore_datasets = {text.split('|', 3)[1].lstrip('🔥 '): text for text in text_list}
    all_keys = set(DATASET_MAPPING.keys())
    mapping = {}
    _iter = zip(
        ['llm', 'vision', 'audio'],
        ['qwen/Qwen2-7B-Instruct', 'Qwen/Qwen2-VL-7B-Instruct', 'qwen/Qwen2-Audio-7B-Instruct'],
        [TemplateType.qwen, TemplateType.qwen2_vl, TemplateType.qwen2_audio],
    )
    for task_type, model_id, template_type in _iter:
        _, tokenizer = get_model_tokenizer(model_id, load_model=False)
        template = get_template(template_type, tokenizer)
        mapping[task_type] = template
    all_keys = list(all_keys)
    all_keys.sort(key=lambda k: k[0] or '')
    for key in all_keys:
        ms_id, hf_id, _ = key
        print(f'Processing {ms_id or hf_id}')
        ms_dataset = find_dataset_row(ms_id, ms_ignore_datasets)
        hf_dataset = find_dataset_row(hf_id, hf_ignore_datasets)
        dataset_info = DATASET_MAPPING[key]
        tags = dataset_info.tags
        tags_str = ', '.join(tags)
        if len(tags_str) == 0:
            tags_str = '-'
        try:
            if ms_id is not None:
                ms_id = f'[{ms_id}](https://modelscope.cn/datasets/{ms_id}/summary)'
            else:
                ms_id = '-'
            if hf_id is not None:
                hf_id = f'[{hf_id}](https://huggingface.co/datasets/{hf_id})'
            else:
                hf_id = '-'
            r = (f'|{ms_id}|{hf_id}|{",".join([s.name for s in dataset_info.subsets])}|'
                 f'{",".join([s.subset for s in dataset_info.subsets])}|'
                 f'{",".join(dataset_info.split) or "train"}')
            if ms_dataset or hf_dataset:
                dataset_size, stat_str = (ms_dataset or hf_dataset).split('|')[6:8]
            else:
                if 'audio' in tags:
                    template = mapping['audio']
                elif 'vision' in tags:
                    template = mapping['vision']
                else:
                    template = mapping['llm']

                if dataset_info.huge_dataset:
                    dataset_size = '-'
                    stat_str = 'Dataset is too huge, please click the original link to view the dataset stat.'
                else:
                    train_dataset, val_dataset = load_dataset(
                        key[0] + ':all',
                        split_dataset_ratio=0.0,
                        strict=False,
                        num_proc=12,
                        model_name=['小黄', 'Xiao Huang'],
                        model_author=['魔搭', 'ModelScope'])
                    dataset_size = len(train_dataset)
                    assert val_dataset is None
                    raw_dataset = train_dataset
                    if len(raw_dataset) < 5000:
                        num_proc = 1
                    else:
                        num_proc = 4

                    dataset = EncodePreprocessor(template)(
                        raw_dataset.select(range(min(5000, len(raw_dataset)))), num_proc=num_proc)

                    _token_len = []
                    input_ids = dataset['input_ids']
                    for i in range(len(dataset)):
                        _token_len.append(len(input_ids[i]))
                    stat = stat_array(_token_len)[0]
                    stat_str = f"{stat['mean']:.1f}±{stat['std']:.1f}, min={stat['min']}, max={stat['max']}"

            res_text_list.append(f'{r}|{dataset_size}|{stat_str}|{tags_str}|')
        except Exception:
            import traceback
            print(traceback.format_exc())
            break
        finally:
            for idx in range(len(fpaths)):
                text = '\n'.join(res_text_list)
                text = pre_texts[idx] + text + '\n'
                with open(fpaths[idx], 'w', encoding='utf-8') as f:
                    f.write(text)
    print(f'数据集总数: {len(all_keys)}, 子数据集总数: {len(res_text_list)}')


if __name__ == '__main__':
    write_dataset_info()
