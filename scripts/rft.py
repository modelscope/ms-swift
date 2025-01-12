import contextlib
import os
import shutil
import subprocess
import time
from typing import List

import torch.cuda

conda_prefix = 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate py311 && '


def do_sample(model: str, dataset: List[str], iter: int):
    gpu_prefix = ''
    if torch.cuda.device_count() > 1:
        gpu_prefix = f'NPROC_PER_NODE={torch.cuda.device_count()} '
    sample_cmd = (f'{conda_prefix} {gpu_prefix} swift sample '
                  f'--model {model} '
                  f'--dataset {" ".join(dataset)} '
                  f'--max_length 2048 '
                  f'--system "You are a math model, you should **think step by step** carefully, '
                  f'and always consider the basic math principles to avoid making calculating mistakes.'
                  f'Give the final answer wrapped with \\boxed{{}}" '
                  f'--num_train_epochs 2 '
                  f'--load_args false '
                  f'--sampler_engine lmdeploy '
                  f'--orm_model math '
                  f'--max_new_tokens 768 '
                  f'--override_exist_file false '
                  f'--eval_strategy no '
                  f'--num_sampling_per_gpu_batch_size 2 '
                  f'--num_return_sequences 64 '
                  f'--file_prefix iter_{iter} '
                  f'--temperature 0.6 ')
    print(f'Sampling iter {iter}.')
    handler = subprocess.Popen(f'{sample_cmd}' + f' > logs/sample_iter_{iter}.log 2>&1', env=os.environ.copy(),
                               shell=True, executable='/bin/bash')
    handler.wait()
    datasets = []
    for proc in range(torch.cuda.device_count()):
        datasets.append(os.path.join('sample_output', f'iter_{iter}_proc_{proc}_sampling.jsonl'))
    print(f'Sampling done, files:{datasets}')
    return datasets


def do_train(model: str, datasets: List[str], iter, cmd='sft'):
    gpu_prefix = ''
    if torch.cuda.device_count() > 1:
        gpu_prefix = f'NPROC_PER_NODE={torch.cuda.device_count()} '
    extra_args = ''
    if cmd == 'rlhf':
        extra_args = (f'--rlhf_type dpo '
                      f'--beta 2.0 '
                      )
    ga = 128 // torch.cuda.device_count() // 2
    train_cmd = (f'{conda_prefix} {gpu_prefix} swift {cmd} '
                 f'--model {model} '
                 f'--dataset {" ".join(datasets)} '
                 f'--max_length 2048 '
                 f'--num_train_epochs 1 '
                 f'--load_args false '
                 f'--train_type full '
                 f'{extra_args} ' 
                 f'--eval_strategy no '
                 f'--split_dataset_ratio 0 '
                 f'--per_device_train_batch_size 2 '
                 f'--gradient_accumulation_steps {ga} '
                 f'--save_steps 1 '
                 f'--save_strategy epoch '
                 f'--deepspeed zero3 '
                 f'--learning_rate 4e-6 ')

    print(f'Training {iter}.')
    handler = subprocess.Popen(f'{train_cmd}' + f' > logs/train_iter_{iter}.log 2>&1', shell=True, env=os.environ.copy(),
                               executable='/bin/bash')
    handler.wait()
    ckpt = None
    with open(f'logs/train_iter_{iter}.log', 'r') as f:
        for line in f.readlines():
            if 'last_model_checkpoint: ' in line:
                ckpt = line.split('last_model_checkpoint: ')[1]
                break
    assert ckpt is not None
    print(f'Training done, ckpt: {ckpt.strip()}.')
    return ckpt.strip()


def do_eval(model, iter):
    eval_cmd = (f'{conda_prefix} swift eval '
                '--eval_dataset math '
                '--infer_backend lmdeploy '
                f'--model {model} '
                '--model_type llama3_1 --system "You are a math model, you should **think step by step** carefully, '
                'and always consider the basic math principles to avoid making calculating mistakes. '
                'Give the final answer wrapped with \\boxed{}"')
    print(f'Evaluating.', flush=True)
    replace_math_dataset()

    if iter is None:
        iter = 'origin'
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    handler = subprocess.Popen(f'{eval_cmd}' + f' > logs/eval_iter_{iter}.log 2>&1',
                               shell=True, env=env, executable='/bin/bash')
    handler.wait()

    acc = None
    # | math | 393424 | accuracy | gen | 39.00 |
    with open(f'logs/eval_iter_{iter}.log', 'r') as f:
        for line in f.readlines():
            if '| math |' in line:
                parts = [l for l in line.split('|') if l.strip()]
                acc = float(parts[-1])
                break

    print(f'Iter {iter} eval done with acc: {acc}.', flush=True)
    return acc


def replace_math_dataset():
    user_dir = os.path.expanduser('~')
    os.remove(os.path.join(user_dir, '.cache', 'opencompass', 'data', 'math', 'math.json'))
    shutil.move(os.path.join('scripts', 'math.json'),
                os.path.join(user_dir, '.cache', 'opencompass', 'data', 'math', 'math.json'))


def main():
    os.makedirs('logs', exist_ok=True)
    max_acc = 0.
    # model = 'LLM-Research/Meta-Llama-3.1-8B-Instruct'
    first_model = '/mnt/nas3/yzhao/tastelikefeet/swift/output/checkpoint-2000/v0-20250111-163224/checkpoint-600'

    if False:
        do_eval(first_model, None)

    model = first_model
    for i in range(5):
        ts = time.time()
        datasets = do_sample(model, ['modelscope/competition_math:rl'], i)
        print(f'do sample cost: {(time.time()-ts)/60} seconds.')
        ts = time.time()
        ckpt = do_train(model, datasets, i)
        print(f'do train cost: {(time.time() - ts) / 60} seconds.')
        ts = time.time()
        acc = do_eval(ckpt, i)
        print(f'do eval cost: {(time.time() - ts) / 60} seconds.')
        if acc > max_acc:
            max_acc = acc
        model = ckpt
        print(f'acc: {acc}, upgrade model to : {model}', flush=True)


if __name__ == '__main__':
    main()
