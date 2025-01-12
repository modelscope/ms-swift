import os
import shutil
import subprocess
import time
from typing import List

import torch.cuda

conda_prefix = 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate py311 && '


def do_sample(model: str, model_type:str, dataset: List[str], iter: int):
    device_count = torch.cuda.device_count()
    handlers = []
    datasets = []
    for device in range(device_count):
        sample_cmd = (f'{conda_prefix} CUDA_VISIBLE_DEVICES={device} swift sample '
                      f'--model {model} --model_type {model_type} '
                      f'--dataset {" ".join(dataset)} '
                      f'--data_range {device} {device_count} '
                      f'--max_length 2048 '
                      f'--system "You are a math model, you should **think step by step** carefully, '
                      f'and always consider the basic math principles to avoid making calculating mistakes.'
                      f'Give the final answer wrapped with \\boxed{{}}" '
                      f'--load_args false '
                      f'--sampler_engine lmdeploy '
                      f'--orm_model math '
                      f'--max_new_tokens 768 '
                      f'--override_exist_file false '
                      f'--num_sampling_per_gpu_batch_size 2 '
                      f'--num_sampling_per_gpu_batches 5 '
                      f'--num_return_sequences 64 '
                      f'--file_prefix iter_{iter}_proc_{device} '
                      '--engine_kwargs {\\"cache_max_entry_count\\":0.6} '
                      f'--temperature 1.0 ')
        print(f'Sampling iter {iter}, part {device}.', flush=True)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(device)
        handler = subprocess.Popen(f'{sample_cmd}' + f' > logs/sample_iter_{iter}_proc_{device}.log 2>&1', env=os.environ.copy(),
                                   shell=True, executable='/bin/bash')
        handlers.append(handler)

    for proc, handler in enumerate(handlers):
        handler.wait()
        assert os.path.exists(os.path.join('sample_output', f'iter_{iter}_proc_{proc}_sampling.jsonl'))
        datasets.append(os.path.join('sample_output', f'iter_{iter}_proc_{proc}_sampling.jsonl'))
    print(f'Sampling done, files:{datasets}', flush=True)
    return datasets


def do_train(model: str, model_type:str, datasets: List[str], iter, cmd='sft'):
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
                 f'--model {model} --model_type {model_type} '
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

    print(f'Training iter {iter}.', flush=True)
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
    print(f'Training done, ckpt: {ckpt.strip()}.', flush=True)
    return ckpt.strip()


def do_eval(model, model_type:str, iter):
    eval_cmd = (f'{conda_prefix} swift eval '
                '--eval_dataset math '
                '--infer_backend lmdeploy '
                f'--model {model} --model_type {model_type} '
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
    if os.path.exists(os.path.join(user_dir, '.cache', 'opencompass', 'data', 'math', 'math.json')):
        os.remove(os.path.join(user_dir, '.cache', 'opencompass', 'data', 'math', 'math.json'))
    shutil.copy(os.path.join('scripts', 'math.json'),
                os.path.join(user_dir, '.cache', 'opencompass', 'data', 'math', 'math.json'))


def main():
    os.makedirs('logs', exist_ok=True)
    max_acc = 0.
    first_model = 'LLM-Research/Meta-Llama-3.1-8B-Instruct'
    model_type = 'llama3_1'
    # first_model = '/mnt/nas3/yzhao/tastelikefeet/swift/output/checkpoint-2000/v0-20250111-163224/checkpoint-600'

    if False:
        do_eval(first_model, None)

    model = first_model
    for i in range(5):
        ts = time.time()
        datasets = do_sample(model, model_type, ['tastelikefeet/competition_math:rl'], i)
        print(f'do sample cost: {(time.time()-ts)/60} seconds.', flush=True)
        ts = time.time()
        # ckpt = do_train(model, model_type, datasets, i)
        ckpt = '/mnt/nas3/yzhao/tastelikefeet/swift/output/Meta-Llama-3.1-8B-Instruct/v70-20250112-214821/checkpoint-1'
        print(f'do train cost: {(time.time() - ts) / 60} seconds.', flush=True)
        ts = time.time()
        acc = do_eval(ckpt, model_type, i)
        print(f'do eval cost: {(time.time() - ts) / 60} seconds.', flush=True)
        if acc > max_acc:
            max_acc = acc
        model = ckpt
        print(f'acc: {acc}, upgrade model to : {model}', flush=True)


if __name__ == '__main__':
    main()
