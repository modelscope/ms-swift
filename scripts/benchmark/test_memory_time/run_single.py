import time
from dataclasses import dataclass, field
from typing import *

import numpy as np
import torch

from swift.llm import sft_main
from swift.llm.utils import *
from swift.utils import *


@dataclass
class TrainArguments(SftArguments):
    run_time: int = 1
    global_seed: int = 42

    def __post_init__(self):
        if self.model_type is None:
            self.model_type = 'qwen-7b-chat'
        if self.use_flash_attn is None:
            self.use_flash_attn = True
        return


def get_non_default_args(train_args) -> Dict[str, Any]:
    train_args_default = train_args.__class__()
    res = {}
    for k, v in train_args.__dict__.items():
        v_default = getattr(train_args_default, k)
        if v != v_default or k in {'use_flash_attn', 'model_type'}:
            res[k] = v
    return res


def test_memory_time(train_args: TrainArguments) -> Dict[str, Dict[str, Any]]:
    random_state = np.random.RandomState(train_args.global_seed)
    args_kwargs = get_non_default_args(train_args)
    print(f'args_kwargs: {args_kwargs}')
    train_dataset_sample = 1000  # save time
    if args_kwargs.get('max_length', 2048) <= 2048:
        train_dataset_sample = -1
    for i in range(train_args.run_time):
        sft_args = SftArguments(
            dataset_test_ratio=0,
            dataset=DatasetName.cls_fudan_news_zh,
            train_dataset_sample=train_dataset_sample,
            save_strategy='no',
            check_dataset_strategy='warning',
            seed=get_seed(random_state),
            **args_kwargs)
        output = sft_main(sft_args)
        torch.cuda.empty_cache()
    res = {
        'samples/s': f"{output['train_time']['train_samples_per_second']:.2f}",
        'memory': output['memory'],
        'train_args': check_json_format(args_kwargs),
        'model_info': output['model_info'],
        'dataset_info': output['dataset_info']
    }
    append_to_jsonl('scripts/benchmark/test_memory_time/result.jsonl', res)
    print(res)
    return res


test_memory_time_main = get_main(TrainArguments, test_memory_time)

if __name__ == '__main__':
    test_memory_time_main()
