import os
import subprocess
import time

os.makedirs('logs', exist_ok=True)
GPUs = 4
max_acc = 0.

#model = 'LLM-Research/Meta-Llama-3.1-8B-Instruct'
model='/mnt/nas3/yzhao/tastelikefeet/swift/output/checkpoint-2000/v0-20250111-163224/checkpoint-600'

cmd = (f'source /root/miniconda3/etc/profile.d/conda.sh && conda activate py311 && swift rlft '
# cmd = (f'swift rlft '
       f'--model #model# '
       f'--model_type llama3_1 '
       f'--dataset #dataset# '
       f'--max_length 2048 '
       f'--system "You are a math model, you should **think step by step** carefully, '
       f'and always consider the basic math principles to avoid making calculating mistakes.'
       f'Give the final answer wrapped with \\boxed{{}}" '
       f'--num_train_epochs 2 '
       f'--load_args false '
       f'--sampler_output rollout_output '
       f'--orm_model math '
       f'--max_new_tokens 1024 '
       f'--train_type full '
       f'--num_rollout_batches 50 '
       f'--use_cache_dataset true '
       f'--rlft_type dpo '
       f'--beta 2.0 '
       f'--eval_strategy no '
       f'--split_dataset_ratio 0 '
       f'--per_device_train_batch_size 2 '
       f'--num_return_sequences 10 '
       f'--gradient_accumulation_steps 8 '
       f'--temperature 0.6 '
       f'--save_steps 1 '
       f'--gpu #gpu# '
       f'--save_strategy epoch '
       f'--learning_rate 4e-6 '
       f'--iter #iter# '
       f'--task #task# ')

if True:
    print(f'Evaluating original model...', flush=True)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    eval_cmd = ('source /root/miniconda3/etc/profile.d/conda.sh && conda activate py311 && swift eval '
                '--eval_dataset math '
                '--eval_limit 500 '
                '--infer_backend lmdeploy '
                f'--model {model} '
                '--model_type llama3_1 --system "You are a math model, you should **think step by step** carefully, '
                'and always consider the basic math principles to avoid making calculating mistakes. '
                'Give the final answer wrapped with \\boxed{}"')
    handler = subprocess.Popen(f'{eval_cmd}' + f' > logs/eval_origin_model.log 2>&1',
                            shell=True, env=env, executable='/bin/bash')
    handler.wait()

    acc = None
    # | math | 393424 | accuracy | gen | 39.00 |
    with open(f'logs/eval_origin_model.log', 'r') as f:
        for line in f.readlines():
            if '| math |' in line:
                parts = [l for l in line.split('|') if l.strip()]
                acc = float(parts[-1])
                break

    print(f'Original model eval done with acc: {acc}.', flush=True)
# max_acc = acc

for i in range(15):
    handlers = []
    time1 = time.time()
    for gpu in range(GPUs):
        rollout_cmd = (cmd.replace('#model#', model)
                       .replace('#dataset#', 'modelscope/competition_math:rl')
                       .replace('#iter#', str(i))
                       .replace('#gpu#', str(gpu))
                       .replace('#task#', 'rollout'))
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        rollout_cmd = f'{rollout_cmd}' + f' > logs/rollout_iter_{i}_gpu_{gpu}.log 2>&1'
        print(rollout_cmd, flush=True)
        handler = subprocess.Popen(rollout_cmd, env=env,
                                   shell=True, executable='/bin/bash')
        handlers.append(handler)

    print(f'Iter {i} rollout to log: logs/rollout_iter_{i}_gpu_x.log', flush=True)
    for handler in handlers:
        handler.wait()

    print(f'Iter {i} rollout done, costing: {time.time() - time1} seconds, begin training.', flush=True)

    all_datasets = []
    for gpu in range(GPUs):
        assert os.path.exists(f'rollout_output/rollout_iter_{i}_gpu_{gpu}.jsonl')
        all_datasets.append(f'rollout_output/rollout_iter_{i}_gpu_{gpu}.jsonl')

    env = os.environ.copy()
    train_cmd = (cmd.replace('#model#', model)
                 .replace('#dataset#', ' '.join(all_datasets))
                 .replace('#iter#', str(i))
                 .replace('#gpu#', '0')
                 .replace('#task#', 'train'))
    train_cmd += '--deepspeed zero3 '
    train_cmd = train_cmd.replace('swift rlft', f'NPROC_PER_NODE={GPUs} swift rlft')
    handler = subprocess.Popen(f'{train_cmd}' + f' > logs/train_iter_{i}.log 2>&1', shell=True, env=env,
                               executable='/bin/bash')
    handler.wait()
    ckpt = None
    with open(f'logs/train_iter_{i}.log', 'r') as f:
        for line in f.readlines():
            if 'last_model_checkpoint: ' in line:
                ckpt = line.split('last_model_checkpoint: ')[1]
                break
    assert ckpt is not None
    temp_model = ckpt.strip()

    print(f'Iter {i} training done with ckpt: {temp_model}, evaluating...', flush=True)

    env['CUDA_VISIBLE_DEVICES'] = '0'
    eval_cmd = ('source /root/miniconda3/etc/profile.d/conda.sh && conda activate py311 && swift eval '
                '--eval_dataset math '
                '--eval_limit 500 '
                '--infer_backend lmdeploy '
                f'--model {temp_model} '
                '--model_type llama3_1 --system "You are a math model, you should **think step by step** carefully, '
                'and always consider the basic math principles to avoid making calculating mistakes. '
                'Give the final answer wrapped with \\boxed{}"')
    handler = subprocess.Popen(f'{eval_cmd}' + f' > logs/eval_iter_{i}.log 2>&1',
                               shell=True, env=env, executable='/bin/bash')
    handler.wait()

    acc = None
    # | math | 393424 | accuracy | gen | 39.00 |
    with open(f'logs/eval_iter_{i}.log', 'r') as f:
        for line in f.readlines():
            if '| math |' in line:
                parts = [l for l in line.split('|') if l.strip()]
                acc = float(parts[-1])
                break

    print(f'Iter {i} eval done with acc: {acc}.', flush=True)
    if acc > max_acc:
        max_acc = acc

        model = temp_model
        print(f'acc: {max_acc}, upgrade model to : {model}', flush=True)
