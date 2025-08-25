import os

kwargs = {
    'per_device_train_batch_size': 5,
    'save_steps': 5,
    'gradient_accumulation_steps': 1,
    'num_train_epochs': 1,
}


def test_train_eval_loop():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from swift.llm import sft_main, TrainArguments
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2.5-0.5B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#100'],
            target_modules=['all-linear', 'all-embedding'],
            modules_to_save=['all-embedding', 'all-norm'],
            eval_strategy='steps',
            eval_steps=5,
            per_device_eval_batch_size=5,
            eval_use_evalscope=True,
            eval_dataset=['gsm8k'],
            eval_dataset_args={'gsm8k': {
                'few_shot_num': 0
            }},
            eval_limit=10,
            **kwargs))


if __name__ == '__main__':
    test_train_eval_loop()
