import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def _infer_model(pt_engine, system=None, messages=None, audios=None):
    seed_everything(42)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    if messages is None:
        messages = []
        if system is not None:
            messages += [{'role': 'system', 'content': system}]
        messages += [{'role': 'user', 'content': '你好'}]
        resp = pt_engine.infer([{'messages': messages}], request_config=request_config)
        response = resp[0].choices[0].message.content
        messages += [{'role': 'assistant', 'content': response}, {'role': 'user', 'content': '<audio>这段语音说了什么'}]
    else:
        messages = messages.copy()
    if audios is None:
        audios = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav']
    resp = pt_engine.infer([{'messages': messages, 'audios': audios}], request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    logger.info(f'model: {pt_engine.model_info.model_name}, messages: {messages}')
    return response


def test_qwen_audio():
    pt_engine = PtEngine('Qwen/Qwen-Audio-Chat')
    _infer_model(pt_engine)


def test_qwen2_audio():
    # transformers==4.48.3
    pt_engine = PtEngine('Qwen/Qwen2-Audio-7B-Instruct')
    messages = [{'role': 'user', 'content': '<audio>'}]
    audios = ['https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav']
    response = _infer_model(pt_engine, messages=messages, audios=audios)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages, audios=audios)
    assert response == response2 == 'Yes, the speaker is female and in her twenties.'


def test_xcomposer2d5_ol():
    pt_engine = PtEngine('Shanghai_AI_Laboratory/internlm-xcomposer2d5-ol-7b:audio')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_step_audio_chat():
    pt_engine = PtEngine('stepfun-ai/Step-Audio-Chat')
    response = _infer_model(pt_engine, messages=[{'role': 'user', 'content': '<audio>'}])
    assert response == ('是的呢，今天天气晴朗，阳光明媚，微风和煦，非常适合外出活动。天空湛蓝，白云朵朵，让人心情愉悦。希望你能好好享受这美好的一天！')


if __name__ == '__main__':
    from swift.llm import PtEngine, RequestConfig, get_template
    from swift.utils import get_logger, seed_everything
    logger = get_logger()
    # test_qwen_audio()
    test_qwen2_audio()
    # test_xcomposer2d5_ol()
    # test_step_audio_chat()
