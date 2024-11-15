def test_sft():
    from swift.llm import sft_main, SftArguments
    sft_main(
        SftArguments(
            model='qwen/Qwen2-7B-Instruct',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#1000', 'AI-ModelScope/alpaca-gpt4-data-en#200']))


if __name__ == '__main__':
    test_sft()
