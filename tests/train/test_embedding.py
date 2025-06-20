import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 4,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_llm():
    from swift.llm import sft_main, TrainArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen3-Embedding-0.6B',
            task_type='embedding',
            dataset=['sentence-transformers/stsb:positive'],
            load_from_cache_file=False,
            loss_type='infonce',
            **kwargs,
        ))
    last_model_checkpoint = result['last_model_checkpoint']
    print(f'last_model_checkpoint: {last_model_checkpoint}')


if __name__ == '__main__':
    test_llm()
