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
    'max_length': 8192,
}


def calc_acc(infer_result):
    n_correct = 0
    for res in infer_result:
        if res['response'] == res['labels']:
            n_correct += 1
    return f'acc: {n_correct/len(infer_result)}, n_correct: {n_correct}, len(res): {len(infer_result)}'


def calc_diff(infer_result, infer_result2):
    n_correct = 0
    for x1, x2 in zip(infer_result, infer_result2):
        if x1['response'] == x2['response']:
            n_correct += 1
    return f'acc: {n_correct/len(infer_result)}, n_correct: {n_correct}, len(res): {len(infer_result)}'


def test_llm():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments, Template
    res = []
    for padding_side in ['left', 'right']:
        model = 'Qwen/Qwen2.5-0.5B-Instruct'
        dataset = ['damo/zh_cls_fudan-news#2000']
        result = sft_main(
            TrainArguments(model=model, dataset=dataset, split_dataset_ratio=0.1, padding_side=padding_side, **kwargs))
        last_model_checkpoint = result['last_model_checkpoint']
        infer_result = infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True))
        res.append(calc_acc(infer_result))
        infer_result2 = infer_main(
            InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True, max_batch_size=16))
        res.append(calc_acc(infer_result2))
    pprint(res)


def test_mllm():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments, Template
    res = []
    for padding_side in ['left', 'right']:
        model = 'Qwen/Qwen2-VL-2B-Instruct'
        dataset = ['AI-ModelScope/LaTeX_OCR#2000']
        result = sft_main(TrainArguments(model=model, dataset=dataset, padding_side=padding_side, **kwargs))
        last_model_checkpoint = result['last_model_checkpoint']
        infer_result = infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True))
        res.append(infer_result)
        infer_result2 = infer_main(
            InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True, max_batch_size=16))
        res.append(infer_result2)
    print(calc_diff(res[0], res[1]))
    print(calc_diff(res[2], res[3]))
    print(calc_diff(res[0], res[2]))
    print(calc_diff(res[0], res[3]))
    print(calc_diff(res[2], res[1]))


if __name__ == '__main__':
    test_llm()
    test_mllm()
