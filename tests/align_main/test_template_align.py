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


def test_internvl2_phi3():
    pt_engine = PtEngine('OpenGVLab/Mini-InternVL-Chat-4B-V1-5')
    _infer_model(pt_engine, system='')
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine, system='')


def test_qwen2_audio():
    pt_engine = PtEngine('qwen/Qwen2-Audio-7B-Instruct')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model_jinja(pt_engine)


def test_qwen2_5():
    pt_engine = PtEngine('Qwen/Qwen2.5-3B')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_qwen1half():
    pt_engine = PtEngine('Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_llava():
    pt_engine = PtEngine('AI-ModelScope/llava-v1.6-mistral-7b')
    _infer_model(pt_engine)


if __name__ == '__main__':
    # test_qwen2_vl()
    # test_internvl2()
    # test_internvl2_phi3()
    # test_qwen2_audio()
    # test_qwen2_5()
    # test_qwen1half()
    test_llava()
