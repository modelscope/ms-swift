import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def _test_model(model_id, **kwargs):
    from swift.llm import export_main, ExportArguments
    if model_id.endswith('mcore') or 'megatron_output' in model_id and 'hf' not in model_id:
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


def test_qwen2_5_vl():
    os.environ['MAX_PIXELS'] = str(1280 * 28 * 28)
    _test_model('Qwen/Qwen2.5-VL-7B-Instruct')


def test_qwen2_vl():
    os.environ['MAX_PIXELS'] = str(1280 * 28 * 28)
    _test_model('Qwen/Qwen2-VL-7B-Instruct')


def test_qwen2_5_omni():
    os.environ['MAX_PIXELS'] = str(1280 * 28 * 28)
    _test_model('Qwen/Qwen2.5-Omni-7B')


def test_internvl3():
    _test_model('OpenGVLab/InternVL3-8B')
    # _test_model('OpenGVLab/InternVL3-1B')


def test_internvl3_5():
    _test_model('OpenGVLab/InternVL3_5-1B')


def test_internvl3_5_moe():
    _test_model('OpenGVLab/InternVL3_5-30B-A3B')


def test_internvl3_hf():
    _test_model('OpenGVLab/InternVL3-1B-hf')


def test_internvl3_5_hf():
    _test_model('OpenGVLab/InternVL3_5-1B-HF')


def test_internvl3_5_moe_hf():
    _test_model('OpenGVLab/InternVL3_5-30B-A3B-HF')


def test_glm4_5v():
    _test_model('ZhipuAI/GLM-4.5V')


def test_ovis2_5():
    _test_model('AIDC-AI/Ovis2.5-2B')


def test_kimi_vl():
    _test_model('moonshotai/Kimi-VL-A3B-Thinking-2506')


def test_qwen3_vl():
    _test_model('Qwen/Qwen3-VL')


def test_qwen3_vl_moe():
    _test_model('Qwen/Qwen3-VL-Moe')


def test_qwen3_omni():
    _test_model('Qwen/Qwen3-Omni-30B-A3B-Instruct')


if __name__ == '__main__':
    # test_qwen2_5_vl()
    # test_qwen2_vl()
    # test_qwen2_5_omni()
    # test_internvl3()
    # test_internvl3_5()
    # test_internvl3_5_moe()
    # test_internvl3_hf()
    # test_internvl3_5_hf()
    # test_internvl3_5_moe_hf()
    # test_glm4_5v()
    # test_ovis2_5()
    # test_kimi_vl()
    test_qwen3_vl()
    # test_qwen3_vl_moe()
    # test_qwen3_omni()
