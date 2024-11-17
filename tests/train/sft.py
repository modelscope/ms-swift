kwargs = {
    'per_device_train_batch_size': 2,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'logging_first_step': True
}


def test_llm():
    from swift.llm import sft_main, SftArguments
    sft_main(
        SftArguments(
            model='qwen/Qwen2-7B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#1000', 'AI-ModelScope/alpaca-gpt4-data-en#200'],
            **kwargs))


def test_mllm():
    from swift.llm import sft_main, SftArguments
    result = sft_main(
        SftArguments(
            model='qwen/Qwen2-VL-7B-Instruct',
            dataset=['modelscope/coco_2014_caption:validation#20', 'AI-ModelScope/alpaca-gpt4-data-en#20'],
            **kwargs))
    print()


if __name__ == '__main__':
    # test_llm()
    test_mllm()
