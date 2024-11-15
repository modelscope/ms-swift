from typing import List

from swift.llm import load_dataset


def _test_dataset(datasets: List[str], num_proc: int = 1, strict: bool = True):
    dataset = load_dataset(datasets, num_proc=num_proc, strict=strict)
    print(f'dataset[0]: {dataset[0]}')
    print(f'dataset[1]: {dataset[1]}')


def test_alpaca():
    _test_dataset(['AI-ModelScope/alpaca-gpt4-data-zh#1000', 'AI-ModelScope/alpaca-gpt4-data-en#200'])


def test_coco():
    _test_dataset(['modelscope/coco_2014_caption:validation'])


def test_llava_instruct():
    _test_dataset(['AI-ModelScope/LLaVA-Instruct-150K'], num_proc=16)


def test_ms_bench():
    _test_dataset(['iic/ms_bench'], strict=False)


def test_ms_agent():
    _test_dataset(['AI-ModelScope/ms_agent_for_agentfabric:all'])


def test_dpo():
    _test_dataset(['AI-ModelScope/orpo-dpo-mix-40k'])


def test_pretrain():
    _test_dataset(['AI-ModelScope/ruozhiba:all'])


def test_dataset_info():
    _test_dataset(['codefuse-ai/CodeExercise-Python-27k'])


def test_aishell1():
    _test_dataset(['speech_asr/speech_asr_aishell1_trainsets:validation'])


def test_messages():
    _test_dataset(['swift/tagengo-gpt4'], strict=False)


if __name__ == '__main__':
    # test_alpaca()
    # test_coco()
    # test_llava_instruct()
    # test_ms_bench()
    # test_ms_agent()
    # test_dpo()
    # test_pretrain()
    # test_dataset_info()
    # test_aishell1()
    test_messages()
