import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from swift.llm import (MODEL_MAPPING, DatasetName, InferArguments,
                       SftArguments, infer_main, sft_main)
from swift.utils import append_to_jsonl, get_main, get_seed, stat_array

DEBUG = False


def test_eval_acc(result: List[Dict[str, str]]) -> float:
    result_list = []
    for line in result:
        result_list.append(line['response'] == line['label'])

    return np.array(result_list).mean()


@dataclass
class TrainArguments(SftArguments):
    run_time: int = 3
    global_seed: int = 42

    def __post_init__(self):
        return


def test_method(train_args: TrainArguments) -> Dict[str, Dict[str, Any]]:
    start_t = time.time()
    if DEBUG:
        eval_steps = 50
        train_dataset_sample = 50 * 16
        val_dataset_sample = 100
    else:
        eval_steps = 100
        train_dataset_sample = 20000
        val_dataset_sample = -1
    t_list = []
    m_list = []
    acc_list = []
    random_state = np.random.RandomState(train_args.global_seed)
    for i in range(train_args.run_time):
        sft_args = SftArguments(
            model_type=train_args.model_type,
            eval_steps=eval_steps,
            check_dataset_strategy='warning',
            train_dataset_sample=train_dataset_sample,
            seed=get_seed(random_state),
            val_dataset_sample=val_dataset_sample,
            dataset=[DatasetName.jd_sentiment_zh],
            weight_decay=0.1,
            output_dir='output',
            acc_strategy='sentence',
            use_flash_attn=True,
            sft_type=train_args.sft_type)
        output = sft_main(sft_args)
        best_model_checkpoint = output['best_model_checkpoint']
        print(f'best_model_checkpoint: {best_model_checkpoint}')
        t = time.time() - start_t
        max_memory = torch.cuda.max_memory_reserved()
        torch.cuda.empty_cache()
        infer_args = InferArguments(
            ckpt_dir=best_model_checkpoint,
            val_dataset_sample=val_dataset_sample,
            verbose=False)
        result = infer_main(infer_args)
        torch.cuda.empty_cache()
        acc = test_eval_acc(result['result'])
        print({'time': t, 'acc': acc, 'memory': max_memory})
        t_list.append(t)
        m_list.append(max_memory)
        acc_list.append(acc)
    t = stat_array(t_list)
    m = stat_array(m_list)
    acc = stat_array(acc_list)
    output = {
        'time': f"{t['mean']:.6f}±{t['std']:.6f}",
        'acc': f"{acc['mean']:.6f}±{acc['std']:.6f}",
        'memory': m / 1e9,
        'train_args': train_args.__dict__
    }
    append_to_jsonl('scripts/benchmark/test_method/result.jsonl', output)
    print(output)
    return result


test_method_main = get_main(TrainArguments, test_method)

if __name__ == '__main__':
    test_method_main()
