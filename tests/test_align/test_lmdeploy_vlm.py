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


def _infer_image_pipeline(model, images=None):
    from lmdeploy import pipeline, GenerationConfig
    from lmdeploy.vl import load_image
    from swift.llm import safe_snapshot_download
    gen_config = GenerationConfig(temperature=0., repetition_penalty=1., max_new_tokens=64)
    pipe = pipeline(safe_snapshot_download(model))

    image = load_image('http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png')
    response = pipe(('<IMAGE_TOKEN>\ndescribe the image.', image), gen_config=gen_config)
    return response.text


def test_internvl2_5():
    model = 'OpenGVLab/InternVL2_5-2B'
    response = _infer_image(model)
    response2 = _infer_image_pipeline(model)
    assert response == response2 == (
        'The image is a close-up of a kitten. The kitten has a mix of white and dark fur, with large, expressive '
        "eyes and a curious expression. The background is blurred, highlighting the kitten's features, "
        'such as its whiskers and fur texture. The overall tone of the image is soft and')


def test_internvl2():
    model = 'OpenGVLab/InternVL2-4B'
    response = _infer_image(model)
    response2 = _infer_image_pipeline(model)
    assert response == response2 == (
        'The image is a close-up of a kitten. The kitten has a mix of white and dark fur, with large, expressive '
        "eyes and a curious expression. The background is blurred, highlighting the kitten's features, "
        'such as its whiskers and fur texture. The overall tone of the image is soft and')


if __name__ == '__main__':
    from swift.llm import LmdeployEngine, InferRequest, RequestConfig
    test_internvl2()
    test_internvl2_5()
