import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

kwargs = {
    'per_device_train_batch_size': 2,
    'save_steps': 50,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 3,
}


def test_llm():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#1000', 'swift/self-cognition#1000'],
            packing=True,
            max_length=4096,
            attn_impl='flash_attn',
            logging_steps=1,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_streaming():
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#10000'],
            packing=True,
            max_length=4096,
            streaming=True,
            attn_impl='flash_attn',
            max_steps=100,
            dataset_num_proc=1,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_mllm_streaming():
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from swift.llm import sft_main, TrainArguments, infer_main, InferArguments
    result = sft_main(
        TrainArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['HF::linxy/LaTeX_OCR#20000'],
            packing=True,
            max_length=8192,
            streaming=True,
            attn_impl='flash_attn',
            max_steps=100,
            dataset_num_proc=4,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


if __name__ == '__main__':
    # test_llm()
    test_streaming()
    # test_mllm_streaming()
