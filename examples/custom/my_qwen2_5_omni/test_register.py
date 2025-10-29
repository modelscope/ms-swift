import os
import sys

import requests
from modelscope import snapshot_download
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from swift.llm import InferRequest, PtEngine, RequestConfig

sys.path.append('examples/custom/my_qwen2_5_omni')


def infer_hf():
    model_dir = snapshot_download('Qwen/Qwen2.5-Omni-7B')
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2')
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir)
    # Use decord to read video (url not yet supported)
    resp = requests.get('https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4')
    with open('_baby.mp4', 'wb') as f:
        f.write(resp.content)

    conversation = [
        {
            'role':
            'user',
            'content': [
                {
                    'type': 'video',
                    'video': '_baby.mp4'
                },
                {
                    'type': 'image',
                    'image': 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png'
                },
                {
                    'type': 'text',
                    'text': 'Describe the video and image.'
                },
            ],
        },
    ]

    USE_AUDIO_IN_VIDEO = False
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors='pt',
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)
    text_ids = model.generate(
        **inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, thinker_do_sample=False, return_audio=False)
    text = processor.batch_decode(
        text_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return inputs['input_ids'][0].tolist(), text[0]


def test_my_qwen2_5_omni():
    engine = PtEngine('Qwen/Qwen2.5-Omni-7B', model_type='my_qwen2_5_omni', attn_impl='flash_attention_2')
    infer_request = InferRequest(
        messages=[{
            'role': 'user',
            'content': '<video><image>Describe the video and image.',
        }],
        videos=['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'],
        images=['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png'],
    )
    request_config = RequestConfig(temperature=0, max_tokens=512)
    input_ids = engine.default_template.encode(infer_request)['input_ids']
    resp_list = engine.infer([infer_request], request_config)
    resp = resp_list[0].choices[0].message.content
    return input_ids, resp


if __name__ == '__main__':
    import my_register
    # Enable debug mode, will print input_ids and generate_ids from `PtEngine.infer`
    os.environ['SWIFT_DEBUG'] = '1'
    input_ids_hf, response_hf = infer_hf()
    input_ids_swift, response_swift = test_my_qwen2_5_omni()
    # Test input_ids and response alignment
    assert input_ids_hf == input_ids_swift
    assert response_hf == response_swift
