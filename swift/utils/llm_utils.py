# Copyright (c) Alibaba, Inc. and its affiliates.

import heapq
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset as HfDataset
from torch.nn import Linear, Module
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import TextStreamer

from .logger import get_logger

logger = get_logger()


def stat_dataset(dataset: HfDataset) -> None:
    """Statistical analysis was performed on the dataset"""
    _token_len = []
    for d in dataset:
        _token_len.append(len(d['input_ids']))
    _token_len = np.array(_token_len)
    mean = _token_len.mean().item()
    std = _token_len.std().item()
    min_ = _token_len.min().item()
    max_ = _token_len.max().item()
    logger.info(
        f'Dataset Token Length: {mean:.6f}±{std:.6f}, min={min_:.6f}, max={max_:.6f}, size={_token_len.shape[0]}'
    )


def data_collate_fn(batch: List[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
    assert tokenizer.pad_token_id is not None
    input_ids = [torch.tensor(b['input_ids']) for b in batch]
    labels = [torch.tensor(b['labels']) for b in batch]
    attention_mask = [
        torch.ones(len(input_ids[i]), dtype=torch.int64)
        for i in range(len(input_ids))
    ]

    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(
        attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def lower_bound(lo: int, hi: int, cond: Callable[[int], bool]) -> int:
    # The lower bound satisfying the condition "cond".
    while lo < hi:
        mid = (lo + hi) >> 1
        if cond(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def print_example(example: Dict[str, Any], tokenizer) -> None:
    input_ids, labels = example['input_ids'], example['labels']
    logger.info(f'[INPUT_IDS] {input_ids}')
    logger.info(f'[INPUT] {tokenizer.decode(input_ids)}')
    n_mask = lower_bound(0, len(labels), lambda i: labels[i] != -100)
    logger.info(f'[LABLES_IDS] {labels}')
    logger.info(
        f'[LABLES] [-100 * {n_mask}]{tokenizer.decode(labels[n_mask:])}')


def find_all_linear_for_lora(model: Module,
                             quantization_bit: int,
                             model_type: Optional[str] = None) -> List[str]:
    """ref: https://github.com/artidoro/qlora"""
    head_module_name = 'lm_head'
    if model_type.startswith('chatglm2-6b'):
        head_module_name = 'output_layer'
    if quantization_bit == 4:
        from bitsandbytes.nn import Linear4bit
        linear_cls = Linear4bit
    elif quantization_bit == 8:
        from bitsandbytes.nn import Linear8bitLt
        linear_cls = Linear8bitLt
    else:
        linear_cls = Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls):
            module_name = name.split('.')[-1]
            if head_module_name not in module_name:
                lora_module_names.add(module_name)
    return list(lora_module_names)


def sort_by_max_length(dataset: HfDataset, num_dataset: int) -> HfDataset:
    dataset_len = [len(d['input_ids']) for d in tqdm(dataset)]
    idx = heapq.nlargest(
        num_dataset, range(len(dataset_len)), key=lambda i: dataset_len[i])
    input_ids = []
    labels = []
    for i in tqdm(idx):
        input_ids.append(dataset[i]['input_ids'])
        labels.append(dataset[i]['labels'])
    return HfDataset.from_dict({'input_ids': input_ids, 'labels': labels})


def inference(input_ids: List[int],
              model,
              tokenizer,
              streamer: Optional[TextStreamer] = None) -> str:
    generation_config = getattr(model, 'generation_config', None)
    streamer.skip_prompt = True
    print(f'[PROMPT]{tokenizer.decode(input_ids)}[OUTPUT]', end='')
    input_ids = torch.tensor(input_ids)[None].cuda()
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    generate_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        generation_config=generation_config)
    output_text = tokenizer.decode(generate_ids[0, len(input_ids[0]):])
    if streamer is None:
        print(output_text)
    return output_text
