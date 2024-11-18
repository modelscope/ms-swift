from typing import List

from swift.llm import load_dataset


def _test_dataset(datasets: List[str], num_proc: int = 1, strict: bool = True, **kwargs):
    dataset = load_dataset(datasets, num_proc=num_proc, strict=strict, **kwargs)
    print(f'dataset[0]: {dataset[0]}')
    print(f'dataset[1]: {dataset[1]}')


def test_sft():
    _test_dataset(['AI-ModelScope/alpaca-gpt4-data-zh#1000', 'AI-ModelScope/alpaca-gpt4-data-en#200'])
    _test_dataset(['iic/ms_bench'], strict=False)
    _test_dataset(['swift/tagengo-gpt4'], strict=False)


def test_mllm():
    _test_dataset(['AI-ModelScope/LaTeX_OCR:all'])
    _test_dataset(['swift/VideoChatGPT:all'])
    _test_dataset(['speech_asr/speech_asr_aishell1_trainsets:validation'])
    _test_dataset(['AI-ModelScope/captcha-images'])
    _test_dataset(['swift/RLAIF-V-Dataset:all'])
    _test_dataset(['swift/gpt4v-dataset:all'])

    # _test_dataset(['modelscope/coco_2014_caption:validation'])
    # _test_dataset(['AI-ModelScope/LLaVA-Instruct-150K'], num_proc=16)


def test_ms_agent():
    _test_dataset(['AI-ModelScope/ms_agent_for_agentfabric:all'])


def test_dpo():
    _test_dataset(['AI-ModelScope/orpo-dpo-mix-40k'], strict=False)
    _test_dataset(['AI-ModelScope/hh-rlhf:all'], strict=False)
    _test_dataset(['AI-ModelScope/hh_rlhf_cn:all'], strict=False)
    _test_dataset(['hjh0119/shareAI-Llama3-DPO-zh-en-emoji:all'])


def test_pretrain():
    _test_dataset(['AI-ModelScope/ruozhiba:all'])


def test_dataset_info():
    _test_dataset(['codefuse-ai/CodeExercise-Python-27k'])


if __name__ == '__main__':
    # test_sft()
    test_mllm()
    # test_ms_agent()
    # test_dpo()
    # test_pretrain()
    # test_dataset_info()
