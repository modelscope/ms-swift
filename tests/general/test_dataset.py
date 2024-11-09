from typing import List

from swift.llm import load_dataset


def _test_dataset(datasets: List[str]):
    dataset = load_dataset(datasets, num_proc=1)
    print(f'dataset[0]: {dataset[0]}')
    print(f'dataset[1]: {dataset[1]}')


def test_alpaca():
    _test_dataset(['AI-ModelScope/alpaca-gpt4-data-zh#1000', 'AI-ModelScope/alpaca-gpt4-data-en#200'])


def test_coco():
    _test_dataset(['modelscope/coco_2014_caption:val'])


def test_llava_instruct():
    _test_dataset(['AI-ModelScope/LLaVA-Instruct-150K'])


def test_ms_bench():
    _test_dataset(['iic/ms_bench'])


if __name__ == '__main__':
    # test_alpaca()
    # test_coco()
    test_llava_instruct()
    # test_ms_bench()
