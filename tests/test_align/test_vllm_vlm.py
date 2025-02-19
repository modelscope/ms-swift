import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _infer_audio(model, use_chat_template: bool = True, max_model_len=8192, system=None):
    engine = VllmEngine(model, max_model_len=max_model_len, limit_mm_per_prompt={'audio': 2})
    if not use_chat_template:
        engine.default_template.use_chat_template = False
    audios = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav']
    messages = []
    if system is not None:
        messages += [{'role': 'system', 'content': system}]
    messages.append({'role': 'user', 'content': 'describe the audio.'})
    resp_list = engine.infer([InferRequest(messages=messages, audios=audios)],
                             RequestConfig(temperature=0, max_tokens=64, repetition_penalty=1.))
    return resp_list[0].choices[0].message.content


def _infer_image(model, use_chat_template: bool = True, max_model_len=8192, system=None):
    engine = VllmEngine(model, max_model_len=max_model_len, limit_mm_per_prompt={'image': 5, 'video': 2})
    if not use_chat_template:
        engine.default_template.use_chat_template = False
    images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
    messages = []
    if system is not None:
        messages += [{'role': 'system', 'content': system}]
    messages.append({'role': 'user', 'content': 'describe the image.'})
    resp_list = engine.infer([InferRequest(messages=messages, images=images)],
                             RequestConfig(temperature=0, max_tokens=64, repetition_penalty=1.))
    return resp_list[0].choices[0].message.content


def test_qwen2_audio():
    response = _infer_audio('Qwen/Qwen2-Audio-7B-Instruct')
    assert response == "The audio is a man speaking in Mandarin saying '今天天气真好呀'."


def test_qwen2_vl():
    response = _infer_image('Qwen/Qwen2-VL-2B-Instruct')
    assert response == (
        'The image depicts a cute kitten with a fluffy, white and gray striped coat. The kitten has large, '
        'expressive blue eyes and is looking directly at the camera. Its ears are perked up, and it has a '
        'small red mark on its left ear. The background is blurred, focusing attention on the kitten. The overall')


def test_qwen2_5_vl():
    response = _infer_image('Qwen/Qwen2.5-VL-3B-Instruct')
    assert response == (
        'The image depicts a cute, fluffy kitten with striking blue eyes and a white and gray fur pattern. '
        'The kitten has a small, pink nose and is looking directly at the camera with a curious expression. '
        "The background is blurred, drawing attention to the kitten's face. "
        'The overall appearance is very endearing and charming.')


def test_deepseek_vl_v2():
    response = _infer_image('deepseek-ai/deepseek-vl2-tiny', max_model_len=4096)
    assert response == ('The image depicts a close-up of a adorable kitten with large, expressive eyes. The kitten has '
                        'a mix of white and gray fur with distinct black stripes, giving it a tabby-like appearance. '
                        'Its ears are perked up, and its whiskers are prominently visible. The background is blurred, '
                        'focusing attention on the kitten')


def test_internvl2():
    response = _infer_image('OpenGVLab/InternVL2-2B', max_model_len=4096, system='')
    assert response == ('The image features a kitten with striking blue eyes and a mix of white and black fur. '
                        'The kitten has large, expressive eyes and a small, pink nose. Its ears are perked up, '
                        'and it appears to be looking directly at the camera. The fur is soft and fluffy, with a mix')


if __name__ == '__main__':
    from swift.llm import VllmEngine, InferRequest, RequestConfig
    # test_qwen2_vl()
    # test_qwen2_5_vl()
    # test_deepseek_vl_v2()
    # test_internvl2()
    test_qwen2_audio()
