import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['SWIFT_DEBUG'] = '1'


def _infer_model(pt_engine, system=None, messages=None, images=None):
    seed_everything(42)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    if messages is None:
        messages = []
        if system is not None:
            messages += [{'role': 'system', 'content': system}]
        messages += [{'role': 'user', 'content': '你好'}]
        resp = pt_engine.infer([{'messages': messages}], request_config=request_config)
        response = resp[0].choices[0].message.content
        messages += [{'role': 'assistant', 'content': response}, {'role': 'user', 'content': '<image>这是什么'}]
    if images is None:
        images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
    resp = pt_engine.infer([{'messages': messages, 'images': images}], request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    logger.info(f'model: {pt_engine.model_info.model_name}, messages: {messages}')
    return messages


def test_qwen2_vl():
    pt_engine = PtEngine('qwen/Qwen2-VL-2B-Instruct')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_internvl2():
    pt_engine = PtEngine('OpenGVLab/InternVL2-2B')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_internvl2_phi3():
    pt_engine = PtEngine('OpenGVLab/Mini-InternVL-Chat-4B-V1-5')
    _infer_model(pt_engine, system='')
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine, system='')


def test_llava():
    pt_engine = PtEngine('AI-ModelScope/llava-v1.6-mistral-7b')
    _infer_model(pt_engine)


def test_yi_vl():
    pt_engine = PtEngine('01ai/Yi-VL-6B')
    _infer_model(pt_engine)


def test_glm4v():
    # There will be differences in '\n'. This is normal.
    pt_engine = PtEngine('ZhipuAI/glm-4v-9b')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_minicpmv():
    pt_engine = PtEngine('OpenBMB/MiniCPM-V-2_6')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_got_ocr():
    # https://github.com/modelscope/ms-swift/issues/2122
    pt_engine = PtEngine('stepfun-ai/GOT-OCR2_0')
    _infer_model(
        pt_engine,
        messages=[{
            'role': 'user',
            'content': 'OCR: '
        }],
        images=['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/ocr.png'])


def test_llama_vision():
    pt_engine = PtEngine('LLM-Research/Llama-3.2-11B-Vision-Instruct')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_llava_hf():
    pt_engine = PtEngine('swift/llava-v1.6-mistral-7b-hf')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_florence():
    pt_engine = PtEngine('AI-ModelScope/Florence-2-base-ft')
    _infer_model(
        pt_engine,
        messages=[{
            'role': 'user',
            'content': '<OD>'
        }],
        images=['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png'])


def test_phi3_vision():
    pt_engine = PtEngine('LLM-Research/Phi-3-vision-128k-instruct')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_qwen_vl():
    pt_engine = PtEngine('qwen/Qwen-VL-Chat')
    _infer_model(pt_engine)


def test_llava_onevision_hf():
    pass


def test_xcomposer2_5():
    pt_engine = PtEngine('Shanghai_AI_Laboratory/internlm-xcomposer2d5-7b')
    _infer_model(pt_engine, system='')
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine, system='')


def test_deepseek_vl():
    pt_engine = PtEngine('deepseek-ai/deepseek-vl-1.3b-chat')
    _infer_model(pt_engine)


def test_deepseek_janus():
    pt_engine = PtEngine('deepseek-ai/Janus-1.3B')
    _infer_model(pt_engine)


def test_mplug_owl2():
    pass


def test_mplug_owl3():
    # pt_engine = PtEngine('iic/mPLUG-Owl3-7B-240728')
    pt_engine = PtEngine('iic/mPLUG-Owl3-7B-241101')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_ovis1_6():
    pt_engine = PtEngine('AIDC-AI/Ovis1.6-Gemma2-9B')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_paligemma():
    pt_engine = PtEngine('AI-ModelScope/paligemma-3b-pt-224')
    _infer_model(pt_engine, messages=[{'role': 'user', 'content': 'caption en'}])


def test_pixtral():
    pt_engine = PtEngine('AI-ModelScope/pixtral-12b')
    _infer_model(pt_engine, messages=[{'role': 'user', 'content': '<image>这是什么'}])


def test_glm_edge_v():
    pt_engine = PtEngine('ZhipuAI/glm-edge-v-2b')
    _infer_model(pt_engine, messages=[{'role': 'user', 'content': '<image>这是什么'}])


def test_internvl2_5():
    pt_engine = PtEngine('OpenGVLab/InternVL2_5-26B')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine, system='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。')


if __name__ == '__main__':
    from swift.llm import PtEngine, RequestConfig, get_template
    from swift.utils import get_logger, seed_everything

    logger = get_logger()
    # test_qwen2_vl()
    # test_internvl2()
    # test_internvl2_phi3()
    # test_llava()
    # test_ovis1_6()
    # test_yi_vl()
    # test_deepseek_vl()
    # test_deepseek_janus()
    # test_qwen_vl()
    # test_glm4v()
    # test_minicpmv()
    # test_got_ocr()
    # test_paligemma()
    # test_pixtral()
    # test_llama_vision()
    # test_llava_hf()
    # test_xcomposer2_5()
    # test_florence()
    # test_glm_edge_v()
    #
    # test_mplug_owl3()
    # test_phi3_vision()
    test_internvl2_5()
