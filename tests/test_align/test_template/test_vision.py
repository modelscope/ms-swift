from swift.llm import PtEngine, RequestConfig, get_template
from swift.utils import get_logger, seed_everything

logger = get_logger()


def _infer_model(pt_engine, system=None):
    seed_everything(42)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    messages = []
    if system is not None:
        messages += [{'role': 'system', 'content': system}]
    messages += [{'role': 'user', 'content': '你好'}]
    resp = pt_engine.infer([{'messages': messages}], request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}, {'role': 'user', 'content': '<image>这是什么'}]
    resp = pt_engine.infer([{
        'messages': messages,
        'images': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
    }],
                           request_config=request_config)
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
    pass


def test_minicpmv():
    pass


def test_got_ocr():
    pass


def test_llama_vision():
    pass


def test_llava_hf():
    pass


def test_florence():
    pass


def test_phi3_vision():
    pass


def test_qwen_vl():
    pass


def test_ovis1_6():
    pass


def test_llava_onevision_hf():
    pass


def test_xcomposer2_5():
    pass


def test_cogvlm2_video():
    pass


def test_deepseek_vl_janus():
    pass


def test_mplug_owl2():
    pass


def test_mplug_owl3():
    pass


def test_qwq():
    'Qwen/QwQ-32B-Preview'


if __name__ == '__main__':
    # test_qwen2_vl()
    # test_internvl2()
    # test_internvl2_phi3()
    # test_llava()
    #
    test_yi_vl()
