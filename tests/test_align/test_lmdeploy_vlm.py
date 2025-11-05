import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _infer_image(model, system=None, images=None):
    engine = LmdeployEngine(model)
    if images is None:
        images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
    messages = []
    if system is not None:
        messages += [{'role': 'system', 'content': system}]
    messages.append({'role': 'user', 'content': 'describe the image.'})
    resp_list = engine.infer([InferRequest(messages=messages, images=images)],
                             RequestConfig(temperature=0, max_tokens=64, repetition_penalty=1.))
    return resp_list[0].choices[0].message.content


def _infer_image_pipeline(model, images=None, prefix='<IMAGE_TOKEN>\n'):
    from lmdeploy import pipeline, GenerationConfig
    from lmdeploy.vl import load_image
    from swift.llm import safe_snapshot_download
    gen_config = GenerationConfig(temperature=0., repetition_penalty=1., max_new_tokens=64)
    pipe = pipeline(safe_snapshot_download(model))

    image = load_image('http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png')
    response = pipe((f'{prefix}describe the image.', image), gen_config=gen_config)
    return response.text


def test_internvl2_5():
    model = 'OpenGVLab/InternVL2_5-4B'
    response = _infer_image(model)
    response2 = _infer_image_pipeline(model)
    assert response == response2


def test_internvl2():
    model = 'OpenGVLab/InternVL2-2B'
    response = _infer_image(model)
    response2 = _infer_image_pipeline(model)  # Missing '\n' after '<|im_end|>'
    assert response == response2


def test_deepseek_vl():
    model = 'deepseek-ai/deepseek-vl-1.3b-chat'
    response = _infer_image(model)
    response2 = _infer_image_pipeline(model, prefix='<IMAGE_TOKEN>')
    assert response == response2


def test_qwen_vl():
    model = 'Qwen/Qwen-VL-Chat'
    response = _infer_image_pipeline(model)  # Missing: 'Picture 1: '
    response2 = _infer_image(model)
    assert response == response2


def test_qwen2_vl():
    model = 'Qwen/Qwen2-VL-2B-Instruct'
    response = _infer_image_pipeline(model, prefix='<IMAGE_TOKEN>')
    response2 = _infer_image(model)
    assert response == response2


def test_qwen2_5_vl():
    model = 'Qwen/Qwen2.5-VL-3B-Instruct'
    response = _infer_image(model)
    response2 = _infer_image_pipeline(model, prefix='<IMAGE_TOKEN>')
    assert response == response2


if __name__ == '__main__':
    from swift.llm import LmdeployEngine, InferRequest, RequestConfig
    # test_internvl2()
    # test_internvl2_5()
    # test_deepseek_vl()
    # test_qwen_vl()
    # test_qwen2_vl()
    test_qwen2_5_vl()
