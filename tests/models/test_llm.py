import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_llama3():
    from swift.llm import infer_main, InferArguments
    infer_main(
        InferArguments(
            model='LLM-Research/Meta-Llama-3.1-8B-Instruct',
            max_batch_size=2,
            val_dataset='AI-ModelScope/alpaca-gpt4-data-en#2'))


if __name__ == '__main__':
    test_llama3()
