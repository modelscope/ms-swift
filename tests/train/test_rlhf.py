import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 2,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_llm():
    from swift.llm import rlhf_main, RLHFArguments, infer_main, InferArguments
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='dpo',
            model='qwen/Qwen2-7B-Instruct',
            dataset=['hjh0119/shareAI-Llama3-DPO-zh-en-emoji:zh#100'],
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_dataset_config=True, merge_lora=True))


def test_mllm():
    from swift.llm import rlhf_main, RLHFArguments, infer_main, InferArguments
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='dpo', model='qwen/Qwen2-VL-7B-Instruct', dataset=['swift/RLAIF-V-Dataset#100'], **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(ckpt_dir=last_model_checkpoint, load_dataset_config=True, merge_lora=True))


def test_mllm_zero3():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    from swift.llm import rlhf_main, RLHFArguments, infer_main, InferArguments
    rlhf_main(
        RLHFArguments(
            rlhf_type='dpo',
            model='qwen/Qwen2-VL-7B-Instruct',
            dataset=['swift/RLAIF-V-Dataset#100'],
            deepspeed='zero3',
            **kwargs))


if __name__ == '__main__':
    # test_llm()
    # test_mllm()
    test_mllm_zero3()
