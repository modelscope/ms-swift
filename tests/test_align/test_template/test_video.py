import os

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['SWIFT_DEBUG'] = '1'


def _infer_model(pt_engine, system=None, messages=None, videos=None, max_tokens=128):
    seed_everything(42)
    request_config = RequestConfig(max_tokens=max_tokens, temperature=0)
    if messages is None:
        messages = []
    if not messages:
        if system is not None:
            messages += [{'role': 'system', 'content': system}]
        messages += [{'role': 'user', 'content': '你好'}]
        resp = pt_engine.infer([{'messages': messages}], request_config=request_config)
        response = resp[0].choices[0].message.content
        messages += [{'role': 'assistant', 'content': response}, {'role': 'user', 'content': '<video>描述视频'}]
    if videos is None:
        videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
    resp = pt_engine.infer([{'messages': messages, 'videos': videos}], request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    logger.info(f'model: {pt_engine.model_info.model_name}, messages: {messages}')
    return response


def test_qwen2_vl():
    os.environ['NFRAMES'] = '24'
    os.environ['MAX_PIXELS'] = '100352'
    os.environ['VIDEO_MAX_PIXELS'] = str(100352 // 4)
    os.environ['SIZE_FACTOR'] = '12'
    pt_engine = PtEngine('Qwen/Qwen2-VL-2B-Instruct')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_internvl2_5():
    pt_engine = PtEngine('OpenGVLab/InternVL2_5-2B')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine, system='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。')


def test_internvl2_5_mpo():
    pt_engine = PtEngine('OpenGVLab/InternVL2_5-1B-MPO', model_type='internvl2_5')
    response = _infer_model(pt_engine, messages=[{'role': 'user', 'content': '<video>这是什么'}])
    assert response == ('这是一段婴儿在阅读的视频。婴儿穿着浅绿色的上衣和粉色的裤子，戴着黑框眼镜，坐在床上，正在翻阅一本打开的书。'
                        '背景中可以看到婴儿床、衣物和一些家具。视频中可以看到“clipo.com”的水印。婴儿看起来非常专注，似乎在认真地阅读。')


def test_xcomposer2_5():
    pt_engine = PtEngine('Shanghai_AI_Laboratory/internlm-xcomposer2d5-ol-7b:base', torch.float16)
    messages = [{'role': 'user', 'content': '<video>Describe the video'}]
    messages_with_system = messages.copy()
    messages_with_system.insert(0, {'role': 'system', 'content': ''})
    response = _infer_model(pt_engine, messages=messages_with_system)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages, system='')
    assert response == response2

    response = _infer_model(pt_engine, messages=messages)
    std_response = (
        'The video features a young child sitting on a bed, deeply engaged in reading a book. '
        'The child is dressed in a light blue sleeveless top and pink pants, and is wearing glasses. '
        'The bed is covered with a textured white blanket, and there are various items scattered on it, '
        'including a white cloth and a striped piece of clothing. In the background, '
        'a wooden crib and a dresser with a mirror can be seen. The child flips through the pages of the book, '
        'occasionally pausing to look at the illustrations. The child appears to be enjoying the book, '
        'and the overall atmosphere is one of quiet concentration and enjoyment.')

    assert response == std_response[:len(response)]


def test_mplug3():
    pt_engine = PtEngine('iic/mPLUG-Owl3-7B-240728')
    # pt_engine = PtEngine('iic/mPLUG-Owl3-7B-241101')
    _infer_model(pt_engine, system='')
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine, system='')


def test_minicpmv():
    pt_engine = PtEngine('OpenBMB/MiniCPM-V-2_6')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


if __name__ == '__main__':
    from swift.llm import PtEngine, RequestConfig, get_template
    from swift.utils import get_logger, seed_everything
    logger = get_logger()
    # test_qwen2_vl()
    # test_internvl2_5()
    # test_xcomposer2_5()
    # test_internvl2_5_mpo()
    test_mplug3()
    test_minicpmv()
