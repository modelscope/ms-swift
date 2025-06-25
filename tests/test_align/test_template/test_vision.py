import os

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['SWIFT_DEBUG'] = '1'


def _infer_model(pt_engine, system=None, messages=None, images=None, **kwargs):
    seed_everything(42)
    request_config = RequestConfig(max_tokens=128, temperature=0, repetition_penalty=1)
    if messages is None:
        messages = []
        if system is not None:
            messages += [{'role': 'system', 'content': system}]
        messages += [{'role': 'user', 'content': '你好'}]
        resp = pt_engine.infer([{'messages': messages}], request_config=request_config)
        response = resp[0].choices[0].message.content
        messages += [{'role': 'assistant', 'content': response}, {'role': 'user', 'content': '<image>这是什么'}]
    else:
        messages = messages.copy()
    if images is None:
        images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
    resp = pt_engine.infer([{'messages': messages, 'images': images, **kwargs}], request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    logger.info(f'model: {pt_engine.model_info.model_name}, messages: {messages}')
    return response


def test_qwen2_vl():
    pt_engine = PtEngine('Qwen/Qwen2-VL-2B-Instruct')
    response = _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine)
    assert response == response2 == '这是一只小猫的图片。它有黑白相间的毛发，眼睛大而圆，显得非常可爱。'


def test_qwen2_5_vl():
    pt_engine = PtEngine('Qwen/Qwen2.5-VL-7B-Instruct')
    messages = [{'role': 'user', 'content': '<image>What kind of dog is this?'}]
    images = ['https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/demo_small.jpg']
    response = _infer_model(pt_engine, messages=messages, images=images)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages, images=images)
    assert response == response2 == (
        'The dog in the picture appears to be a Labrador Retriever. Labradors are known for their '
        'friendly and energetic nature, which is evident in the image where the dog seems to be '
        "interacting playfully with the person. The dog's size, coat color, and build are "
        'characteristic of the Labrador Retriever breed.')


def test_qwen2_5_omni():
    pt_engine = PtEngine('Qwen/Qwen2.5-Omni-7B')
    response = _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine)
    assert response == response2


def test_qvq():
    pt_engine = PtEngine('Qwen/QVQ-72B-Preview')
    response = _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine)
    assert response == response2


def test_internvl2():
    pt_engine = PtEngine('OpenGVLab/InternVL2-2B')
    response = _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine)
    assert response == response2


def test_internvl2_phi3():
    pt_engine = PtEngine('OpenGVLab/Mini-InternVL-Chat-4B-V1-5')
    _infer_model(pt_engine, system='')
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine, system='')


def test_internvl3_8b():
    pt_engine = PtEngine('OpenGVLab/InternVL3-8B')
    response = _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, system='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。')
    assert response == response2


def test_internvl3_9b():
    pt_engine = PtEngine('OpenGVLab/InternVL3-9B')
    response = _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, system='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。')
    assert response == response2


def test_llava():
    pt_engine = PtEngine('AI-ModelScope/llava-v1.6-mistral-7b')
    _infer_model(pt_engine)


def test_yi_vl():
    pt_engine = PtEngine('01ai/Yi-VL-6B')
    _infer_model(pt_engine)


def test_glm4v():
    # There will be differences in '\n'. This is normal.
    pt_engine = PtEngine('ZhipuAI/glm-4v-9b')
    messages = [{'role': 'user', 'content': '描述这张图片'}]
    response = _infer_model(pt_engine, messages=messages)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages)
    assert response == ('这张图片是一只小猫的特写，它有着非常醒目的蓝色眼睛和混合了灰色、白色和棕色毛发的皮毛。小猫的耳朵竖立着，胡须清晰可见。它的眼神看起来既好奇又警觉，整体上显得非常可爱。')
    assert response2 == ('这是一张特写照片，展示了一只毛茸茸的小猫。小猫的眼睛大而圆，呈深蓝色，眼珠呈金黄色，非常明亮。它的鼻子短而小巧，'
                         '是粉色的。小猫的嘴巴紧闭，胡须细长。它的耳朵竖立着，耳朵内侧是白色的，外侧是棕色的。小猫的毛发看起来柔软而浓密，'
                         '主要是白色和棕色相间的花纹。背景模糊不清，但似乎是一个室内环境。')


def test_cogagent():
    pt_engine = PtEngine('ZhipuAI/cogagent-9b-20241220')
    messages = [{
        'role':
        'user',
        'content':
        """<image>Task: I'm looking for a software to \"edit my photo with grounding\"
History steps:
(Platform: Mac)
(Answer in Action-Operation-Sensitive format.)"""
    }]
    images = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/agent.png']
    response = _infer_model(pt_engine, messages=messages, images=images)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages, images=images)
    assert response == response2 == (
        """Action: Click on the 'Adobe Photoshop 2023' icon located in the middle of the screen to open the application.
Grounded Operation: CLICK(box=[[346,574,424,710]], element_type='卡片', element_info='Adobe Photoshop 2023')
<<一般操作>>""")


def test_minicpmv():
    pt_engine = PtEngine('OpenBMB/MiniCPM-V-2_6')
    _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    _infer_model(pt_engine)


def test_minicpmo():
    pt_engine = PtEngine('OpenBMB/MiniCPM-o-2_6')
    messages = [{
        'role':
        'user',
        'content':
        '<image><image>Compare image 1 and image 2, tell me about the differences between image 1 and image 2.'
    }]
    images = [
        'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png',
        'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png'
    ]
    response = _infer_model(pt_engine, messages=messages, images=images)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages, images=images)
    assert response == response2 == (
        'The main difference between image 1 and image 2 is the subject matter. '
        'Image 1 features a close-up of a kitten, while image 2 depicts a cartoon illustration of four sheep '
        'standing in a grassy field. The setting, the number of subjects, and the overall style of the images '
        'are distinct from each other.')


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


def test_got_ocr_hf():
    pt_engine = PtEngine('stepfun-ai/GOT-OCR-2.0-hf')
    response = _infer_model(
        pt_engine,
        messages=[{
            'role': 'user',
            'content': 'OCR: '
        }],
        images=['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/ocr.png'])
    assert response[:200] == ('简介 SWIFT支持250+LLM和35+MLLM（多模态大模型）的训练、推理、 评测和部署。开发者可以直接将'
                              '我们的框架应用到自己的Research和 生产环境中，实现模型训练评测到应用的完整链路。我们除支持了 PEFT提供的轻量训练方案外'
                              '，也提供了一个完整的Adapters库以支持 最新的训练技术，如NEFTune、LoRA+、LLaMA-PRO等，这个适配器 库可以脱离训练脚本'
                              '直接使用在自己的')


def test_llama_vision():
    pt_engine = PtEngine('LLM-Research/Llama-3.2-11B-Vision-Instruct')
    response = _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine)
    assert response == response2


def test_llava_hf():
    pt_engine = PtEngine('llava-hf/llava-v1.6-mistral-7b-hf')
    response = _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine)
    assert response == response2


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
    pt_engine = PtEngine('llava-hf/llava-onevision-qwen2-0.5b-ov-hf')
    response = _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine)
    assert response == response2


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


def test_deepseek_janus():
    pt_engine = PtEngine('deepseek-ai/Janus-Pro-7B')
    messages = [{'role': 'user', 'content': '描述图片'}]
    response = _infer_model(pt_engine, messages=messages)
    assert response == ('这是一张非常可爱的猫咪图片。猫咪的毛色主要是白色，并带有灰色的条纹。它的眼睛非常大，呈现出明亮的蓝色，'
                        '显得非常可爱和无辜。猫咪的耳朵竖立着，显得非常警觉和好奇。背景模糊，使得猫咪成为图片的焦点。'
                        '整体画面给人一种温暖和愉悦的感觉。')


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
    response = _infer_model(pt_engine, system='')
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, system='')
    assert response == response2


def test_ovis1_6():
    pt_engine = PtEngine('AIDC-AI/Ovis1.6-Gemma2-9B')
    # pt_engine = PtEngine('AIDC-AI/Ovis1.6-Gemma2-27B')
    response = _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine)
    assert response == response2


def test_ovis1_6_llama3():
    pt_engine = PtEngine('AIDC-AI/Ovis1.6-Llama3.2-3B')
    messages = [{'role': 'user', 'content': '这是什么'}]
    # llama3
    response = _infer_model(pt_engine, messages=messages)
    pt_engine.default_template.template_backend = 'jinja'
    # llama3_2
    _infer_model(pt_engine, messages=messages, system='You are a helpful and honest multimodal assistant.')
    assert response == '这是一只小猫。从图中可见的特征如大眼睛、细长的白色鼻毛和毛发的图案，表明它可能属于常见的猫种。猫的表情和毛发的质感显示出它年轻，可能是幼猫。'


def test_ovis2():
    pt_engine = PtEngine('AIDC-AI/Ovis2-2B')  # with flash_attn
    response = _infer_model(pt_engine, messages=[{'role': 'user', 'content': 'Describe the image.'}])
    assert response[:200] == ('The image features a close-up portrait of a young kitten with striking blue eyes. '
                              'The kitten has a distinctive coat pattern with a mix of gray, black, and white fur, '
                              'typical of a tabby pattern. Its ea')


def test_paligemma():
    pt_engine = PtEngine('AI-ModelScope/paligemma-3b-mix-224')
    response = _infer_model(pt_engine, messages=[{'role': 'user', 'content': 'detect cat'}])
    assert response == '<loc0000><loc0000><loc1022><loc1022> cat'


def test_paligemma2():
    pt_engine = PtEngine('AI-ModelScope/paligemma2-3b-ft-docci-448', torch_dtype=torch.bfloat16)
    response = _infer_model(pt_engine, messages=[{'role': 'user', 'content': 'caption en'}])
    assert response == (
        'A close up view of a white and gray kitten with black stripes on its head and face staring forward with '
        'its light blue eyes. The kitten is sitting on a white surface with a blurry background. '
        "There is a light shining on the top of the kitten's head and the front of its body.")


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


def test_valley():
    pt_engine = PtEngine('bytedance-research/Valley-Eagle-7B')
    _infer_model(pt_engine)


def test_ui_tars():
    os.environ['MAX_PIXELS'] = str(1280 * 28 * 28)
    pt_engine = PtEngine('bytedance-research/UI-TARS-2B-SFT')
    prompt = ('You are a GUI agent. You are given a task and your action history, with screenshots. '
              'You need to perform the next action to complete the task.' + r"""

## Output Format
```\nThought: ...
Action: ...\n```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use \"\
\" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


## Note
- Use Chinese in `Thought` part.
- Summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
""")
    instruction = "I'm looking for a software to \"edit my photo with grounding\""
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': prompt + instruction
                },
            ],
        },
    ]
    images = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/agent.png']
    response = _infer_model(pt_engine, messages=messages, images=images)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages, images=images)
    assert response == response2


def test_phi4_vision():
    pt_engine = PtEngine('LLM-Research/Phi-4-multimodal-instruct')
    response = _infer_model(pt_engine, messages=[{'role': 'user', 'content': 'describe the image.'}])
    assert response == (
        "The image features a close-up of a kitten's face. The kitten has large, "
        'round eyes with a bright gaze, and its fur is predominantly white with black stripes. '
        "The kitten's ears are pointed and alert, and its whiskers are visible. The background is blurred, "
        "drawing focus to the kitten's face.")
    response = _infer_model(
        pt_engine,
        messages=[{
            'role': 'user',
            'content': 'describe the audio.'
        }],
        images=[],
        audios=['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav'])
    assert response == '今天天气真好呀'


def test_gemma3_vision():
    pt_engine = PtEngine('LLM-Research/gemma-3-4b-it')
    response = _infer_model(pt_engine, messages=[{'role': 'user', 'content': '<image>Describe this image in detail.'}])
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=[{'role': 'user', 'content': '<image>Describe this image in detail.'}])
    assert response[:80] == response2[:80] == (
        "Here's a detailed description of the image:\n\n**Overall Impression:**\n\nThe image ")


def test_mistral_2503():
    pt_engine = PtEngine('mistralai/Mistral-Small-3.1-24B-Instruct-2503')
    response = _infer_model(pt_engine, messages=[{'role': 'user', 'content': 'What is shown in this image?'}])
    assert response == (
        'The image shows a close-up of a Siamese kitten. The kitten has distinctive blue almond-shaped eyes, '
        'a pink nose, and a light-colored coat with darker points on the ears, paws, tail, and face, '
        'which are characteristic features of the Siamese breed. '
        'The kitten appears to be looking directly at the viewer with a curious and endearing expression.')


def test_llama4():
    pt_engine = PtEngine('LLM-Research/Llama-4-Scout-17B-16E-Instruct')
    messages = [{'role': 'user', 'content': '<image><image>What is the difference between the two images?'}]
    images = [
        'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png',
        'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png'
    ]
    response = _infer_model(pt_engine, messages=messages, images=images)
    assert response[:128] == ('The two images are distinct in their subject matter and style. The first image features '
                              'a realistic depiction of a kitten, while') and len(response) == 654


def test_kimi_vl():
    pt_engine = PtEngine('moonshotai/Kimi-VL-A3B-Instruct')
    messages = [{'role': 'user', 'content': '<image><image>What is the difference between the two images?'}]
    images = [
        'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png',
        'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png'
    ]
    response = _infer_model(pt_engine, messages=messages, images=images)
    assert response == ('The first image is a close-up of a kitten with a blurred background, '
                        'while the second image is a cartoon of four sheep standing in a field.')


def test_kimi_vl_thinking():
    pt_engine = PtEngine('moonshotai/Kimi-VL-A3B-Thinking-2506')
    messages = [{'role': 'user', 'content': '<image><image>What is the difference between the two images?'}]
    images = [
        'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png',
        'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png'
    ]
    response = _infer_model(pt_engine, messages=messages, images=images)
    assert response[:200] == ("◁think▷So, let's analyze the two images. The first image is a close - "
                              'up of a real kitten with detailed fur, whiskers, and a realistic style. '
                              'The second image is an illustration of four sheep in a car')


if __name__ == '__main__':
    from swift.llm import PtEngine, RequestConfig
    from swift.utils import get_logger, seed_everything

    logger = get_logger()
    # test_qwen2_vl()
    # test_qwen2_5_vl()
    # test_qwen2_5_omni()
    # test_internvl2()
    # test_internvl2_phi3()
    # test_llava()
    # test_ovis1_6()
    # test_ovis1_6_llama3()
    # test_ovis2()
    # test_yi_vl()
    # test_deepseek_vl()
    # test_deepseek_janus()
    # test_deepseek_vl2()
    # test_qwen_vl()
    # test_glm4v()
    # test_cogagent()
    # test_llava_onevision_hf()
    # test_minicpmv()
    # test_got_ocr()
    # test_got_ocr_hf()
    # test_paligemma()
    # test_paligemma2()
    # test_pixtral()
    # test_llama_vision()
    # test_llava_hf()
    # test_florence()
    # test_glm_edge_v()
    # test_phi3_vision()
    # test_phi4_vision()
    # test_internvl2_5()
    # test_internvl2_5_mpo()
    # test_mplug_owl3()
    # test_xcomposer2_5()
    # test_megrez_omni()
    # test_qvq()
    # test_mplug_owl2()
    # test_molmo()
    # test_molmoe()
    # test_doc_owl2()
    # test_minicpmo()
    # test_valley()
    # test_ui_tars()
    # test_gemma3_vision()
    # test_mistral_2503()
    # test_llama4()
    # test_internvl3_8b()
    # test_internvl3_9b()
    # test_kimi_vl()
    test_kimi_vl_thinking()
