import os

kwargs = {
    'per_device_train_batch_size': 2,
    'per_device_eval_batch_size': 2,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_train_eval_loop():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
    from swift.llm import sft_main, TrainArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2.5-0.5B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100'],
            target_modules=['all-linear', 'all-embedding'],
            modules_to_save=['all-embedding', 'all-norm'],
            eval_strategy='steps',
            eval_steps=1,
            eval_use_evalscope=True,
            eval_datasets=['gsm8k'],
            eval_limit=10,
            **kwargs))

if __name__ == '__main__':
    test_train_eval_loop()