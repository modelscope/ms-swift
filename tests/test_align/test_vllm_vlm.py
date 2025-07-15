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


def _infer_video(model, use_chat_template: bool = True, max_model_len=8192, system=None, limit_mm_per_prompt=None):
    limit_mm_per_prompt = limit_mm_per_prompt or {'image': 16, 'video': 2}
    engine = VllmEngine(model, max_model_len=max_model_len, limit_mm_per_prompt=limit_mm_per_prompt)
    if not use_chat_template:
        engine.default_template.use_chat_template = False
    videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
    messages = []
    if system is not None:
        messages += [{'role': 'system', 'content': system}]
    messages.append({'role': 'user', 'content': 'describe the video.'})
    resp_list = engine.infer([InferRequest(messages=messages, videos=videos)],
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


def test_minicpmv_2_5():
    response = _infer_image('OpenBMB/MiniCPM-Llama3-V-2_5', max_model_len=4096)
    assert response == (
        "The image is a digital painting of a kitten that captures the essence of a young feline's innocence "
        "and curiosity. The kitten's fur is rendered with a mix of gray, white, and black stripes, "
        'giving it a realistic and adorable appearance. Its large, expressive eyes are a striking blue, '
        "which draws the viewer's")


def test_minicpmv_2_6():
    response = _infer_image('OpenBMB/MiniCPM-V-2_6', max_model_len=4096)
    assert response == (
        'The image features a close-up of a kitten with striking blue eyes and a mix of '
        "white and dark fur, possibly gray or black. The kitten's gaze is directed forward, giving it an "
        "expressive and captivating look. The background is blurred, drawing focus to the kitten's face. "
        "The overall composition emphasizes the kitten's features")


def test_minicpmo_2_6_video():
    response = _infer_video('OpenBMB/MiniCPM-o-2_6')
    assert response == ('The video features a young child sitting on a bed, deeply engaged in reading a book. '
                        'The child, dressed in a light blue sleeveless top and pink pants, is surrounded by a '
                        'cozy and homely environment. The bed is adorned with a patterned blanket, and a white cloth '
                        'is casually draped over the side.')


def test_qwen2_5_vl_video():
    response = _infer_video('Qwen/Qwen2.5-VL-3B-Instruct')
    assert response == ('A baby wearing sunglasses is sitting on a bed and reading a book. '
                        'The baby is holding the book with both hands and is looking at the pages. '
                        'The baby is wearing a light blue shirt and pink pants. The baby is sitting '
                        'on a white blanket. The baby is looking at the book and is smiling. The baby')


def test_qwen2_5_omni():
    limit_mm_per_prompt = {'image': 1, 'video': 1, 'audio': 1}
    response = _infer_video('Qwen/Qwen2.5-Omni-7B', limit_mm_per_prompt=limit_mm_per_prompt)
    # response = _infer_audio('Qwen/Qwen2.5-Omni-7B')
    assert response


def test_ovis2():
    response = _infer_image('AIDC-AI/Ovis2-1B', max_model_len=4096)
    assert response[:200] == ('The image showcases a charming digital painting of a kitten, capturing its '
                              'adorable features in a unique style. The kitten has a predominantly white face '
                              'with black stripes and spots, giving it a stri')


def test_keye_vl():
    response = _infer_image('Kwai-Keye/Keye-VL-8B-Preview', max_model_len=4096)
    assert response[:200] == ('<analysis>This question asks for a description of the image, which is '
                              'straightforward and involves observing the visual content. Therefore, '
                              '/no_think is more appropriate.</analysis>The image features ')


def test_kimi_vl():
    response = _infer_image('moonshotai/Kimi-VL-A3B-Instruct', max_model_len=4096)
    print(f'response: {response}')


def test_glm4v():
    response = _infer_image('ZhipuAI/glm-4v-9b', max_model_len=4096)
    print(f'response: {response}')


def test_glm4_1v():
    response = _infer_image('ZhipuAI/GLM-4.1V-9B-Thinking', max_model_len=4096)
    print(f'response: {response}')


if __name__ == '__main__':
    from swift.llm import VllmEngine, InferRequest, RequestConfig
    # test_qwen2_vl()
    # test_qwen2_5_vl()
    # test_deepseek_vl_v2()
    # test_internvl2()
    # test_qwen2_audio()
    # test_minicpmv_2_5()
    # test_minicpmv_2_6()
    # test_minicpmo_2_6_video()
    # test_qwen2_5_vl_video()
    # test_qwen2_5_omni()
    # test_ovis2()
    # test_keye_vl()
    # test_kimi_vl()
    # test_glm4v()
    test_glm4_1v()
