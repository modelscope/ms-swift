import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def _infer_model(pt_engine, system=None):
    seed_everything(42)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    messages = []
    if system is not None:
        messages += [{'role': 'system', 'content': system}]
    messages += [{'role': 'user', 'content': '你好'}]
    resp = pt_engine.infer([{'messages': messages}], request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}, {'role': 'user', 'content': '<audio>这段语音说了什么'}]
    resp = pt_engine.infer([{
        'messages': messages,
        'audios': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav']
    }],
                           request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    logger.info(f'model: {pt_engine.model_info.model_name}, messages: {messages}')
    return response


def test_qwen_audio():
    pt_engine = PtEngine('Qwen/Qwen-Audio-Chat')
    _infer_model(pt_engine)


def test_qwen2_audio():
    pt_engine = PtEngine('Qwen/Qwen2-Audio-7B-Instruct')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_xcomposer2d5_ol():
    pt_engine = PtEngine('Shanghai_AI_Laboratory/internlm-xcomposer2d5-ol-7b:audio')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


if __name__ == '__main__':
    from swift.llm import PtEngine, RequestConfig, get_template
    from swift.utils import get_logger, seed_everything
    logger = get_logger()
    # test_qwen_audio()
    # test_qwen2_audio()
    test_xcomposer2d5_ol()
