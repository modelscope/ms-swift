import os

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['SWIFT_DEBUG'] = '1'


def _infer_model(pt_engine, system=None, messages=None, images=None):
    seed_everything(42)
    request_config = RequestConfig(max_tokens=128, temperature=0)
    if messages is None:
        messages = []
        if system is not None:
            messages += [{'role': 'system', 'content': system}]
        messages += [{'role': 'user', 'content': '你好'}]
        resp = pt_engine.infer([{'messages': messages}], request_config=request_config)
        response = resp[0].choices[0].message.content
        messages += [{'role': 'assistant', 'content': response}, {'role': 'user', 'content': '<image>这是什么'}]
    if images is None:
        images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
    resp = pt_engine.infer([{'messages': messages, 'images': images}], request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    logger.info(f'model: {pt_engine.model_info.model_name}, messages: {messages}')
    return response


def test_qwen2_vl():
    pt_engine = PtEngine('Qwen/Qwen2-VL-2B-Instruct')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_qvq():
    pt_engine = PtEngine('Qwen/QVQ-72B-Preview')
    response = _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine)
    assert response == response2


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
    # There will be differences in '\n'. This is normal.
    pt_engine = PtEngine('ZhipuAI/glm-4v-9b')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_minicpmv():
    pt_engine = PtEngine('OpenBMB/MiniCPM-V-2_6')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_got_ocr():
    # https://github.com/modelscope/ms-swift/issues/2122
    pt_engine = PtEngine('stepfun-ai/GOT-OCR2_0')
    _infer_model(
        pt_engine,
        messages=[{
            'role': 'user',
            'content': 'OCR: '
        }],
        images=['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/ocr.png'])


def test_llama_vision():
    pt_engine = PtEngine('LLM-Research/Llama-3.2-11B-Vision-Instruct')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_llava_hf():
    pt_engine = PtEngine('swift/llava-v1.6-mistral-7b-hf')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_florence():
    pt_engine = PtEngine('AI-ModelScope/Florence-2-base-ft')
    _infer_model(pt_engine, messages=[{'role': 'user', 'content': 'who are you?'}], images=[])

    _infer_model(
        pt_engine,
        messages=[{
            'role': 'user',
            'content': '<OD>'
        }],
        images=['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png'])


def test_phi3_vision():
    # pt_engine = PtEngine('LLM-Research/Phi-3-vision-128k-instruct')
    pt_engine = PtEngine('LLM-Research/Phi-3.5-vision-instruct')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_qwen_vl():
    pt_engine = PtEngine('Qwen/Qwen-VL-Chat')
    _infer_model(pt_engine)


def test_llava_onevision_hf():
    pass


def test_xcomposer2_5():
    pt_engine = PtEngine('Shanghai_AI_Laboratory/internlm-xcomposer2d5-ol-7b:base', torch.float16)
    # pt_engine = PtEngine('Shanghai_AI_Laboratory/internlm-xcomposer2d5-7b')
    response = _infer_model(pt_engine, system='')
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine)
    assert response == response2


def test_deepseek_vl():
    # pt_engine = PtEngine('deepseek-ai/deepseek-vl-1.3b-chat')
    pt_engine = PtEngine('deepseek-ai/Janus-1.3B')
    _infer_model(pt_engine)


def test_deepseek_vl2():
    pt_engine = PtEngine('deepseek-ai/deepseek-vl2-small')
    response = _infer_model(pt_engine)
    assert response == ('这是一只可爱的小猫。它有着大大的蓝色眼睛和柔软的毛发，看起来非常天真无邪。小猫的耳朵竖立着，显得非常警觉和好奇。'
                        '它的鼻子小巧而粉红，嘴巴微微张开，似乎在探索周围的环境。整体来看，这只小猫非常可爱，充满了活力和好奇心。')


def test_mplug_owl2():
    # pt_engine = PtEngine('iic/mPLUG-Owl2')
    pt_engine = PtEngine('iic/mPLUG-Owl2.1')
    _infer_model(pt_engine, messages=[{'role': 'user', 'content': '<image>这是什么'}])


def test_mplug_owl3():
    # pt_engine = PtEngine('iic/mPLUG-Owl3-7B-240728')
    pt_engine = PtEngine('iic/mPLUG-Owl3-7B-241101')
    _infer_model(pt_engine, system='')
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine, system='')


def test_ovis1_6():
    pt_engine = PtEngine('AIDC-AI/Ovis1.6-Gemma2-9B')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_paligemma():
    pt_engine = PtEngine('AI-ModelScope/paligemma-3b-pt-224')
    _infer_model(pt_engine, messages=[{'role': 'user', 'content': 'caption en'}])


def test_pixtral():
    pt_engine = PtEngine('AI-ModelScope/pixtral-12b')
    _infer_model(pt_engine, messages=[{'role': 'user', 'content': '<image>这是什么'}])


def test_glm_edge_v():
    pt_engine = PtEngine('ZhipuAI/glm-edge-v-2b')
    _infer_model(pt_engine, messages=[{'role': 'user', 'content': '<image>这是什么'}])


def test_internvl2_5():
    pt_engine = PtEngine('OpenGVLab/InternVL2_5-26B')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine, system='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。')


def test_internvl2_5_mpo():
    pt_engine = PtEngine('OpenGVLab/InternVL2_5-1B-MPO', model_type='internvl2_5')
    response = _infer_model(pt_engine, messages=[{'role': 'user', 'content': 'Hello, who are you?'}], images=[])
    assert response == ("Hello! I'm an AI assistant whose name is InternVL, developed jointly by Shanghai AI Lab, "
                        'Tsinghua University and other partners.')
    response2 = _infer_model(pt_engine, messages=[{'role': 'user', 'content': '<image>这是什么'}])
    assert response2 == ('这是一只小猫的特写照片。照片中的小猫有大大的蓝色眼睛和毛发，看起来非常可爱。这种照片通常用于展示宠物的可爱瞬间。')


def test_megrez_omni():
    pt_engine = PtEngine('InfiniAI/Megrez-3B-Omni')
    _infer_model(pt_engine)
    response = _infer_model(
        pt_engine,
        messages=[{
            'role': 'user',
            'content': [
                {
                    'type': 'image'
                },
                {
                    'type': 'audio',
                    'audio': 'weather.wav'
                },
            ]
        }])
    assert response == ('根据图片，无法确定确切的天气状况。然而，猫咪放松的表情和柔和的光线可能暗示着是一个晴朗或温和的日子。'
                        '没有阴影或明亮的阳光表明这不是正午时分，也没有雨滴或雪花的迹象，这可能意味着不是下雨或下雪的日子。')


def test_molmo():
    # pt_engine = PtEngine('LLM-Research/Molmo-7B-O-0924')
    pt_engine = PtEngine('LLM-Research/Molmo-7B-D-0924')
    _infer_model(pt_engine)
    response = _infer_model(pt_engine, messages=[{'role': 'user', 'content': '<image>这是什么'}])
    assert response == (
        ' This is a close-up photograph of a young kitten. '
        'The kitten has striking blue eyes and a mix of white and black fur, '
        'with distinctive black stripes on its head and face. '
        "It's looking directly at the camera with an alert and curious expression. "
        "The kitten's fur appears soft and fluffy, and its pink nose and white whiskers are clearly visible. "
        'The background is blurred, which emphasizes the kitten as the main subject of the image.')


def test_molmoe():
    pt_engine = PtEngine('LLM-Research/MolmoE-1B-0924')
    response = _infer_model(pt_engine, messages=[{'role': 'user', 'content': '<image>这是什么'}])
    assert response == (" This is a close-up photograph of a kitten's face. The kitten has striking blue eyes and "
                        "a mix of white, black, and brown fur. It's looking directly at the camera with an adorable "
                        "expression, its ears perked up and whiskers visible. The image captures the kitten's cute "
                        'features in sharp detail, while the background is blurred, creating a soft, out-of-focus '
                        "effect that emphasizes the young feline's charm.")


def test_doc_owl2():
    pt_engine = PtEngine('iic/DocOwl2', torch_dtype=torch.float16)
    response = _infer_model(pt_engine, messages=[{'role': 'user', 'content': '你是谁'}], images=[])
    images = [
        'https://modelscope.cn/models/iic/DocOwl2/resolve/master/examples/docowl2_page0.png',
        'https://modelscope.cn/models/iic/DocOwl2/resolve/master/examples/docowl2_page1.png',
        'https://modelscope.cn/models/iic/DocOwl2/resolve/master/examples/docowl2_page2.png',
        'https://modelscope.cn/models/iic/DocOwl2/resolve/master/examples/docowl2_page3.png',
        'https://modelscope.cn/models/iic/DocOwl2/resolve/master/examples/docowl2_page4.png',
        'https://modelscope.cn/models/iic/DocOwl2/resolve/master/examples/docowl2_page5.png',
    ]
    response = _infer_model(
        pt_engine,
        messages=[{
            'role': 'user',
            'content': '<image>' * len(images) + 'what is this paper about? provide detailed information.'
        }],
        images=images)
    assert response == (
        'This paper is about multimodal Language Models(MLMs) achieving promising OCR-free '
        'Document Understanding by performing understanding by the cost of generating thorough sands of visual '
        'tokens for a single document image, leading to excessive GPU computation time. The paper also discusses '
        'the challenges and limitations of existing multimodal OCR approaches and proposes a new framework for '
        'more efficient and accurate OCR-free document understanding.')


if __name__ == '__main__':
    from swift.llm import PtEngine, RequestConfig, get_template
    from swift.utils import get_logger, seed_everything

    logger = get_logger()
    # test_qwen2_vl()
    # test_internvl2()
    # test_internvl2_phi3()
    # test_llava()
    # test_ovis1_6()
    # test_yi_vl()
    # test_deepseek_vl()
    # test_deepseek_vl2()
    # test_qwen_vl()
    # test_glm4v()
    # test_minicpmv()
    # test_got_ocr()
    # test_paligemma()
    # test_pixtral()
    # test_llama_vision()
    # test_llava_hf()
    # test_florence()
    # test_glm_edge_v()
    # test_phi3_vision()
    # test_internvl2_5()
    # test_internvl2_5_mpo()
    # test_mplug_owl3()
    # test_xcomposer2_5()
    # test_megrez_omni()
    # test_qvq()
    # test_mplug_owl2()
    # test_molmo()
    # test_molmoe()
    test_doc_owl2()
