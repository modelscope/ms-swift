import os
from pprint import pprint

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
kwargs = {
    'per_device_train_batch_size': 4,
    'per_device_eval_batch_size': 4,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
    'save_steps': 100,
    'max_length': 512,
    'task_type': 'seq_cls',
    'num_labels': 2,
}


def calc_acc(infer_result):
    n_correct = 0
    for res in infer_result:
        if res['response'] == res['labels']:
            n_correct += 1
    return f'acc: {n_correct/len(infer_result)}, n_correct: {n_correct}, len(res): {len(infer_result)}'


def test_llm():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments, Template
    res = []
    for model in ['Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-0.5B', 'AI-ModelScope/bert-base-chinese']:
        dataset = ['DAMO_NLP/jd:cls#2000']
        result = sft_main(TrainArguments(model=model, dataset=dataset, split_dataset_ratio=0.1, **kwargs))
        last_model_checkpoint = result['last_model_checkpoint']
        infer_result = infer_main(
            InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True, truncation_strategy='right'))
        res.append(calc_acc(infer_result))
        infer_result2 = infer_main(
            InferArguments(
                ckpt_dir=last_model_checkpoint, load_data_args=True, max_batch_size=16, truncation_strategy='right'))
        res.append(calc_acc(infer_result2))

    model = 'Qwen/Qwen2.5-0.5B-Instruct'
    dataset = ['DAMO_NLP/jd#2000']
    train_kwargs = kwargs.copy()
    train_kwargs.pop('task_type')
    train_kwargs.pop('num_labels')
    result = sft_main(TrainArguments(model=model, dataset=dataset, split_dataset_ratio=0.1, **train_kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_result = infer_main(
        InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True, truncation_strategy='right'))
    res.append(calc_acc(infer_result))
    infer_result2 = infer_main(
        InferArguments(
            ckpt_dir=last_model_checkpoint, load_data_args=True, max_batch_size=16, truncation_strategy='right'))
    res.append(calc_acc(infer_result2))
    pprint(res)


if __name__ == '__main__':
    test_llm()
