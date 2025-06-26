import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 4,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_llm():
    from swift.llm import rlhf_main, RLHFArguments, infer_main, InferArguments
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='gkd',
            model='Qwen/Qwen2.5-0.5B',
            teacher_model='Qwen/Qwen2.5-1.5B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-en#2000'],
            load_from_cache_file=False,
            seq_kd=True,
            **kwargs,
        ))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_mllm():
    from swift.llm import rlhf_main, RLHFArguments, infer_main, InferArguments
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='gkd',
            model='OpenGVLab/InternVL3-2B-Pretrained',
            teacher_model='OpenGVLab/InternVL3-8B',
            dataset=['AI-ModelScope/LaTeX_OCR#2000', 'AI-ModelScope/alpaca-gpt4-data-en#2000'],
            load_from_cache_file=False,
            **kwargs,
        ))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


if __name__ == '__main__':
    # test_llm()
    test_mllm()
