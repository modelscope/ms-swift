import os
import time
from typing import Any, Dict, List

import torch

from swift.llm import (DatasetName, InferArguments, ModelType, SftArguments,
                       infer_main, sft_main)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_eval_acc(result: List[Dict[str, str]]) -> float:
    n_correct = 0
    for line in result:
        if line['response'] == line['label']:
            n_correct += 1
    return n_correct / len(result)


def test_method(method_kwargs: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    model_type = ModelType.qwen_7b_chat
    start_t = time.time()
    sft_args = SftArguments(
        model_type=model_type,
        eval_steps=100,
        check_dataset_strategy='warning',
        train_dataset_sample=20000,
        dataset_test_ratio=0.05,
        dataset=[DatasetName.hc3_zh, DatasetName.hc3_en],
        output_dir='output',
        acc_strategy='sentence')
    output = sft_main(sft_args)
    best_model_checkpoint = output['best_model_checkpoint']
    print(f'best_model_checkpoint: {best_model_checkpoint}')
    t = time.time() - start_t
    torch.cuda.empty_cache()
    infer_args = InferArguments(
        ckpt_dir=best_model_checkpoint, val_dataset_sample=10, verbose=False)
    result = infer_main(infer_args)
    torch.cuda.empty_cache()
    acc = test_eval_acc(result['result'])
    return {'time': t, 'acc': acc}


result = test_method({'sft_type': 'lora'})
