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
            model='Qwen/Qwen2.5-7B-Instruct',
            dataset=['swift/self-cognition#200'],
            split_dataset_ratio=0.01,
            use_liger_kernel=True,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True))


def test_mllm_dpo():
    os.environ['MAX_PIXLES'] = f'{1280 * 28 * 28}'
    from swift.llm import rlhf_main, RLHFArguments, infer_main, InferArguments
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='dpo',
            model='Qwen/Qwen2.5-VL-3B-Instruct',
            train_type='full',
            dataset=['swift/RLAIF-V-Dataset#1000'],
            split_dataset_ratio=0.01,
            dataset_num_proc=8,
            deepspeed='zero3',
            use_liger_kernel=True,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_data_args=True))


if __name__ == '__main__':
    test_sft()
    # test_mllm_dpo()
