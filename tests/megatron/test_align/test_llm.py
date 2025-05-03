import os

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _test_model(model_id):
    from swift.llm import export_main, ExportArguments
    if model_id.endswith('mcore'):
        export_main(
            ExportArguments(
                mcore_model=model_id,
                to_hf=True,
                exist_ok=True,
                test_convert_precision=True,
                torch_dtype=torch.bfloat16))
    else:
        export_main(
            ExportArguments(
                model=model_id,
                to_mcore=True,
                exist_ok=True,
                test_convert_precision=True,
                torch_dtype=torch.bfloat16,
            ))


def test_qwen2():
    _test_model('Qwen/Qwen2-0.5B-Instruct')


def test_llama2():
    _test_model('modelscope/Llama-2-7b-chat-ms')


def test_llama3():
    _test_model('LLM-Research/Meta-Llama-3-8B-Instruct')


def test_marco_o1():
    _test_model('AIDC-AI/Marco-o1')


def test_deepseek_r1_llama():
    _test_model('deepseek-ai/DeepSeek-R1-Distill-Llama-8B')


def test_deepseek_r1_qwen():
    _test_model('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')


def test_yi():
    _test_model('01ai/Yi-1.5-6B-Chat')


def test_megrez():
    _test_model('InfiniAI/Megrez-3b-Instruct')


def test_llama3_1():
    _test_model('LLM-Research/Meta-Llama-3.1-8B-Instruct')


def test_llama3_2():
    _test_model('LLM-Research/Llama-3.2-1B-Instruct')


def test_qwen3():
    _test_model('Qwen/Qwen3-0.6B-Base')


def test_qwen2_moe():
    _test_model('Qwen/Qwen1.5-MoE-A2.7B-Chat')


def test_qwen3_moe():
    _test_model('Qwen/Qwen3-15B-A2B-Base')


if __name__ == '__main__':
    # test_qwen2()
    # test_llama2()
    # test_llama3()
    # test_marco_o1()
    # test_deepseek_r1_llama()
    # test_deepseek_r1_qwen()
    # test_yi()
    # test_megrez()
    # test_llama3_1()
    # test_llama3_2()
    # test_qwen3()
    # test_qwen2_moe()
    test_qwen3_moe()
