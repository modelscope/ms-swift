import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
kwargs = {
    'per_device_train_batch_size': 2,
    'save_steps': 30,
    'gradient_accumulation_steps': 2,
    'num_train_epochs': 1,
}


def test_sft():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2.5-0.5B-Instruct',
            dataset=['swift/self-cognition#200'],
            split_dataset_ratio=0.01,
            use_cce=True,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


if __name__ == '__main__':
    test_sft()