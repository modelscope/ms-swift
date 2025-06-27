import os

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


def _test_model(model_id, **kwargs):
    from swift.llm import export_main, ExportArguments
    if model_id.endswith('mcore'):
        export_main(
            ExportArguments(
                mcore_model=model_id,
                to_hf=True,
                exist_ok=True,
                test_convert_precision=True,
                torch_dtype='bfloat16',
                **kwargs,
            ))
    else:
        export_main(
            ExportArguments(
                model=model_id,
                to_mcore=True,
                exist_ok=True,
                test_convert_precision=True,
                torch_dtype='bfloat16',
                **kwargs,
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


def test_deepseek_r1_qwen_0528():
    _test_model('deepseek-ai/DeepSeek-R1-0528-Qwen3-8B')


def test_yi():
    _test_model('01ai/Yi-1.5-6B-Chat')


def test_megrez():
    _test_model('InfiniAI/Megrez-3b-Instruct')


def test_llama3_1():
    _test_model('LLM-Research/Meta-Llama-3.1-8B-Instruct')


def test_llama3_2():
    _test_model('LLM-Research/Llama-3.2-1B-Instruct')


def test_qwen3():
    # _test_model('Qwen/Qwen3-0.6B-Base')
    _test_model('deepseek-ai/DeepSeek-Prover-V2-7B')


def test_internlm3():
    _test_model('Shanghai_AI_Laboratory/internlm3-8b-instruct')


def test_qwen2_moe():
    _test_model('Qwen/Qwen1.5-MoE-A2.7B-Chat')


def test_qwen3_moe():
    _test_model('Qwen/Qwen3-30B-A3B')


def test_mimo():
    # _test_model('XiaomiMiMo/MiMo-7B-RL')
    _test_model('XiaomiMiMo/MiMo-7B-RL-0530')


def test_moonlight():
    _test_model('moonshotai/Moonlight-16B-A3B-Instruct')


def test_deepseek_v2():
    # _test_model('deepseek-ai/DeepSeek-V2-Lite')
    _test_model('deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct')


def test_deepseek_moe():
    _test_model('deepseek-ai/deepseek-moe-16b-chat')


def test_dots():
    _test_model('rednote-hilab/dots.llm1.inst')


def test_kimi_dev():
    _test_model('moonshotai/Kimi-Dev-72B')


def test_hunyuan():
    _test_model('Tencent-Hunyuan/Hunyuan-A13B-Instruct')


if __name__ == '__main__':
    # test_qwen2()
    # test_llama2()
    # test_llama3()
    # test_marco_o1()
    # test_deepseek_r1_llama()
    # test_deepseek_r1_qwen()
    # test_deepseek_r1_qwen_0528()
    # test_yi()
    # test_megrez()
    # test_llama3_1()
    # test_llama3_2()
    # test_qwen3()
    # test_qwen2_moe()
    # test_qwen3_moe()
    # test_internlm3()
    # test_mimo()
    # test_moonlight()
    # test_deepseek_v2()
    # test_deepseek_moe()
    # test_dots()
    # test_kimi_dev()
    test_hunyuan()
