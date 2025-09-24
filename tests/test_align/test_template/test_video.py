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
    else:
        messages = messages.copy()
    if videos is None:
        videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
    resp = pt_engine.infer([{'messages': messages, 'videos': videos}], request_config=request_config)
    response = resp[0].choices[0].message.content
    messages += [{'role': 'assistant', 'content': response}]
    logger.info(f'model: {pt_engine.model_info.model_name}, messages: {messages}')
    return response


def test_qwen2_vl():
    os.environ['FPS_MAX_FRAMES'] = '24'
    os.environ['MAX_PIXELS'] = '100352'
    os.environ['VIDEO_MAX_PIXELS'] = str(100352 // 4)
    pt_engine = PtEngine('Qwen/Qwen2-VL-2B-Instruct')
    response = _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine)
    assert response == response2


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


def test_minicpmo():
    os.environ['VIDEO_MAX_SLICE_NUMS'] = '2'
    pt_engine = PtEngine('OpenBMB/MiniCPM-o-2_6')
    messages = [{'role': 'user', 'content': '<video>Describe the video'}]
    response = _infer_model(pt_engine, messages=messages)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages)
    assert response == response2 == (
        'The video features a young child sitting on a bed, deeply engrossed in reading a large book. The child, '
        'dressed in a light blue sleeveless top and pink pants, is surrounded by a cozy and homely environment. '
        'The bed is adorned with a patterned blanket, and a white cloth is casually draped over the side. '
        'In the background, a crib and a television are visible, adding to the domestic setting. '
        'The child is seen flipping through the pages of the book, occasionally pausing to look at the pages, '
        'and then continuing to turn them. The video captures the child\'s focused and curious demeanor as they '
        'explore the contents of the book, creating a heartwarming '
        'scene of a young reader immersed in their world of stories.')[:len(response)]


def test_valley():
    pt_engine = PtEngine('bytedance-research/Valley-Eagle-7B')
    _infer_model(pt_engine)


def _run_qwen2_5_vl_hf(messages, model, template):
    from qwen_vl_utils import process_vision_info
    processor = template.processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=text, images=images, videos=videos, do_resize=False, return_tensors='pt', **video_kwargs)
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]


def test_qwen2_5_vl():
    os.environ['FPS'] = '1'
    os.environ['VIDEO_MAX_PIXELS'] = str(360 * 420)
    pt_engine = PtEngine('Qwen/Qwen2.5-VL-7B-Instruct')
    query = 'What happened in the video?'
    messages = [{'role': 'user', 'content': query}]
    videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
    response = _infer_model(pt_engine, messages=messages, videos=videos)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages, videos=videos)
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'video',
                    'video': videos[0]
                },
                {
                    'type': 'text',
                    'text': query
                },
            ],
        },
    ]
    response3 = _run_qwen2_5_vl_hf(messages, pt_engine.model, pt_engine.default_template)
    assert response == response2 == response3


def test_qwen2_5_omni():
    USE_AUDIO_IN_VIDEO = True
    os.environ['USE_AUDIO_IN_VIDEO'] = str(USE_AUDIO_IN_VIDEO)
    pt_engine = PtEngine('Qwen/Qwen2.5-Omni-7B', attn_impl='flash_attn')
    system = ('You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, '
              'capable of perceiving auditory and visual inputs, as well as generating text and speech.')
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': '<video>'}]
    videos = ['https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4']
    response = _infer_model(pt_engine, messages=messages, videos=videos)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages, videos=videos)
    if USE_AUDIO_IN_VIDEO:
        ground_truth = ("Oh, that's a really cool drawing! It looks like a guitar. You've got the body "
                        'and the neck drawn in a simple yet effective way. The lines are clean and the '
                        'shape is well-defined. What made you choose to draw a guitar?')
    else:
        ground_truth = ('嗯，你是在用平板画画呢。你画的这把吉他，看起来很简洁明了。你用的笔触也很流畅，线条很清晰。你对颜色的运用也很不错，整体看起来很协调。你要是还有啥想法或者问题，随时跟我说哈。')
    assert response == response2 == ground_truth


def _run_qwen3_omni_hf(model, processor, messages):
    from qwen_omni_utils import process_mm_info
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors='pt',
        padding=True,
        use_audio_in_video=True)
    inputs = inputs.to(device=model.device, dtype=model.dtype)
    text_ids = model.generate(**inputs, use_audio_in_video=True, do_sample=False, max_new_tokens=128)
    text = processor.decode(
        text_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text


def test_qwen3_omni():
    USE_AUDIO_IN_VIDEO = True
    os.environ['USE_AUDIO_IN_VIDEO'] = str(USE_AUDIO_IN_VIDEO)
    pt_engine = PtEngine('Qwen/Qwen3-Omni-30B-A3B-Thinking')
    query = 'describe the video.'
    messages = [{'role': 'user', 'content': query}]
    videos = ['https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4']
    response = _infer_model(pt_engine, messages=messages, videos=videos)

    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'video',
                    'video': videos[0]
                },
                {
                    'type': 'text',
                    'text': query
                },
            ],
        },
    ]
    response2 = _run_qwen3_omni_hf(pt_engine.model, pt_engine.processor, messages)
    assert response == response2


def test_glm4_1v():
    messages = [{'role': 'user', 'content': '<video>What happened in the video?'}]
    videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
    pt_engine = PtEngine('ZhipuAI/GLM-4.1V-9B-Thinking')
    response = _infer_model(pt_engine, messages=messages, videos=videos)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages, videos=videos)
    assert response == response2


def test_glm4_5v():
    messages = [{'role': 'user', 'content': '<video>What happened in the video?'}]
    videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
    pt_engine = PtEngine('ZhipuAI/GLM-4.5V')
    response = _infer_model(pt_engine, messages=messages, videos=videos)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages, videos=videos)
    assert response == response2


def test_keye_vl():
    pt_engine = PtEngine('Kwai-Keye/Keye-VL-8B-Preview')
    messages = [{'role': 'user', 'content': '<video>Describe this video.'}]
    videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
    response = _infer_model(pt_engine, messages=messages, videos=videos)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages, videos=videos)
    assert response == response2


def test_keye_vl_1_5():
    pt_engine = PtEngine('Kwai-Keye/Keye-VL-1_5-8B')
    messages = [{'role': 'user', 'content': '<video>Describe this video.'}]
    videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
    response = _infer_model(pt_engine, messages=messages, videos=videos)
    assert response[:200] == ('The video features a young child sitting on a bed, engrossed in '
                              'reading a book. The child is wearing a light blue sleeveless top and pink '
                              'pants. The book appears to be a hardcover with illustrations, ')


def test_ovis2_5():
    pt_engine = PtEngine('AIDC-AI/Ovis2.5-2B')
    messages = [{'role': 'user', 'content': '<video>Describe this video in detail.'}]
    videos = ['baby.mp4']
    response = _infer_model(pt_engine, messages=messages, videos=videos)
    print(f'response: {response}')


def run_hf(model, processor, messages):
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(
            model.device, dtype=torch.bfloat16)
    generate_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    decoded_output = processor.decode(generate_ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return decoded_output


def test_interns1():
    pt_engine = PtEngine('Shanghai_AI_Laboratory/Intern-S1-mini')
    query = 'Describe this video in detail.'
    messages = [{'role': 'user', 'content': f'<video>{query}'}]
    videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
    response = _infer_model(pt_engine, messages=messages, videos=videos)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages, videos=videos)
    messages = [{
        'role': 'user',
        'content': [
            {
                'type': 'video',
                'url': videos[0]
            },
            {
                'type': 'text',
                'text': query
            },
        ],
    }]
    response2 = run_hf(pt_engine.model, pt_engine.processor, messages)
    assert response == ('<think>' + response2)[:len(response)]


def test_internvl3_5():
    models = [
        'OpenGVLab/InternVL3_5-1B', 'OpenGVLab/InternVL3_5-2B', 'OpenGVLab/InternVL3_5-4B', 'OpenGVLab/InternVL3_5-8B',
        'OpenGVLab/InternVL3_5-14B', 'OpenGVLab/InternVL3_5-38B', 'OpenGVLab/InternVL3_5-30B-A3B',
        'OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview'
    ]
    for model in models:
        pt_engine = PtEngine(model)
        messages = [{'role': 'user', 'content': '<video>Describe this video in detail.'}]
        videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
        response = _infer_model(pt_engine, messages=messages, videos=videos)
        pt_engine.default_template.template_backend = 'jinja'
        response2 = _infer_model(pt_engine, messages=messages, videos=videos)
        assert response == response2


def test_minicpmv4_5():
    pt_engine = PtEngine('OpenBMB/MiniCPM-V-4_5')
    messages = [{'role': 'user', 'content': '<video>Describe this video in detail.'}]
    videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
    response = _infer_model(pt_engine, messages=messages, videos=videos)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages, videos=videos)
    assert response == response2


def _run_qwen3_vl_hf(messages, model, template):
    from qwen_vl_utils import process_vision_info
    processor = template.processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        do_resize=False,
        return_tensors='pt',
        **video_kwargs)
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]


def test_qwen3_vl():
    pt_engine = PtEngine('Qwen/Qwen3-VL')
    videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
    query = 'describe this video.'
    messages = [{'role': 'user', 'content': query}]
    response = _infer_model(pt_engine, messages=messages, videos=videos)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine, messages=messages, videos=videos)
    messages = [{
        'role': 'user',
        'content': [
            {
                'type': 'video',
                'video': videos[0],
            },
            {
                'type': 'text',
                'text': query
            },
        ],
    }]
    response3 = _run_qwen3_vl_hf(messages, pt_engine.model, pt_engine.default_template)
    assert response == response2 == response3


def test_qwen3_moe_vl():
    pt_engine = PtEngine('Qwen/Qwen3-VL-Moe')
    response = _infer_model(pt_engine)
    pt_engine.default_template.template_backend = 'jinja'
    response2 = _infer_model(pt_engine)
    assert response == response2


if __name__ == '__main__':
    from swift.llm import PtEngine, RequestConfig
    from swift.utils import get_logger, seed_everything
    logger = get_logger()
    # test_qwen2_vl()
    # test_internvl2_5()
    # test_xcomposer2_5()
    # test_internvl2_5_mpo()
    # test_mplug3()
    # test_minicpmv()
    # test_minicpmo()
    # test_valley()
    # test_qwen2_5_vl()
    # test_qwen2_5_omni()
    # test_qwen3_omni()
    # test_glm4_1v()  # bug now, wait model fix
    # test_keye_vl()
    # test_keye_vl_1_5()
    # test_glm4_5v()
    # test_ovis2_5()
    # test_interns1()
    # test_internvl3_5()
    # test_minicpmv4_5()
    test_qwen3_vl()
    # test_qwen3_moe_vl()
