import os
import subprocess
import time
from typing import List

import json
from modelscope.msdatasets import MsDataset

conda_prefix = ''


def client_sample(model: str, orm: str, dataset_path: str, iter: int, device_count: int, output_dir: str):
    handlers = []
    # Sampling cache
    api_key = os.getenv('DASHSCOPE_API_KEY')

    for device in range(device_count):

        output_file = f'iter_{iter}_proc_{device}.jsonl'
        cache_file = f'iter_{iter}_proc_{device}_cache.jsonl'
        dataset = f'train_{device:02}.jsonl'

        # output_file_path = os.path.join(output_dir, output_file)
        cache_file_path = os.path.join(output_dir, cache_file)
        single_dataset_path = os.path.join(dataset_path, dataset)

        if not os.path.exists(cache_file_path):
            open(cache_file_path, 'w').close()
        sample_cmd = (f'USE_OPENCOMPASS_EVALUATOR=True '
                      f'swift sample '
                      f'--model {model} '
                      f'--orm_model {orm} '
                      f'--sampler_type mcts '
                      f'--process_reward_rate 0 '
                      f'--stop_words ки '
                      f'--seed 42 '
                      f'--api_key {api_key} '
                      f'--dataset {single_dataset_path} '
                      f'--max_length 2048 '
                      f'--system ./scripts/sampler/system_prompt.txt '
                      f'--load_args false '
                      f'--sampler_engine client '
                      f'--max_new_tokens 768 '
                      f'--override_exist_file true '
                      f'--num_sampling_per_gpu_batch_size 1 '
                      f'--num_return_sequences 8 '
                      f'--exploration_rate 0.2 '
                      f'--max_iterations 200 '
                      f'--output_dir {output_dir} '
                      f'--cache_files {cache_file} '
                      f'--output_file {output_file} '
                      f'--temperature 1.0 ')
        print(f'Sampling caches of iter {iter}, part {device}.', flush=True)
        # env['CUDA_VISIBLE_DEVICES'] = str(device)
        handler = subprocess.Popen(
            f'{sample_cmd}' + f' > mcts_logs/sample_iter_{iter}_proc_{device}_cache.log 2>&1',
            env=os.environ.copy(),
            shell=True,
            executable='/bin/bash')
        handlers.append(handler)

    datasets = []
    for proc, handler in enumerate(handlers):
        handler.wait()
        assert os.path.exists(os.path.join(output_dir, f'iter_{iter}_proc_{proc}.jsonl'))
        datasets.append(os.path.join('sample_output', f'iter_{iter}_proc_{proc}.jsonl'))
    print(f'Sampling done, files:{datasets}', flush=True)


def split_dataset(ds, split_size, out_path):
    data_size = int(len(ds) / split_size) + 1

    for i in range(split_size):
        file_name = f'train_{i:02}.jsonl'
        file_path = os.path.join(out_path, file_name)
        print(file_path)
        ds_split = ds[data_size * i:min(data_size * (i + 1), len(ds))]
        print(f"split_size: {len(ds_split['problem'])}")
        with open(file_path, 'w', encoding='utf-8') as file:
            for problem, solution in zip(ds_split['problem'], ds_split['solution']):
                message = {
                    'messages': [
                        {
                            'role': 'user',
                            'content': problem,
                        },
                        {
                            'role': 'assistant',
                            'content': solution,
                        },
                    ]
                }
                file.write(json.dumps(message, ensure_ascii=False) + '\n')


def main():
    server_model = 'qwen-max'
    orm = 'math'
    device_count = 20
    output_dir = 'output/sampler/client_mcts/'
    dataset_dir = 'datasets/competition_math/'
    log_dir = 'mcts_logs/'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    ds = MsDataset.load('tastelikefeet/competition_math', subset_name='default', split='train')
    split_dataset(ds, device_count, dataset_dir)

    ts = time.time()
    client_sample(server_model, orm, dataset_dir, 0, device_count, output_dir)
    print(f'do sample cost: {(time.time() - ts) / 60:.1f} minutes.', flush=True)


if __name__ == '__main__':
    main()
