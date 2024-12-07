import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def _infer_model(pt_engine, system=None, messages=None, videos=None):
    seed_everything(42)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    if messages is None:
        messages = []
        if system is not None:
            messages += [{'role': 'system', 'content': system}]
        messages += [{'role': 'user', 'content': '你好'}]
        resp = pt_engine.infer([{'messages': messages}], request_config=request_config)
        response = resp[0].choices[0].message.content
        messages += [{'role': 'assistant', 'content': response}, {'role': 'user', 'content': '<video>描述视频'}]
        videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
    resp = pt_engine.infer([{'messages': messages, 'videos': videos}], request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    logger.info(f'model: {pt_engine.model_info.model_name}, messages: {messages}')
    return messages


def test_qwen2_vl():
    os.environ['NFRAMES'] = '24'
    os.environ['MAX_PIXELS'] = '100352'
    os.environ['VIDEO_MAX_PIXELS'] = str(100352 // 4)
    os.environ['SIZE_FACTOR'] = '12'
    pt_engine = PtEngine('qwen/Qwen2-VL-2B-Instruct')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_internvl2_5():
    pt_engine = PtEngine('OpenGVLab/InternVL2_5-2B')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine, system='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。')


if __name__ == '__main__':
    from swift.llm import PtEngine, RequestConfig, get_template
    from swift.utils import get_logger, seed_everything
    logger = get_logger()
    test_qwen2_vl()
    test_internvl2_5()
