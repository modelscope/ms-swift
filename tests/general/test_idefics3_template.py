# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import re
import torch
from PIL import Image
from transformers import AutoProcessor

from swift.model import get_model_processor
from swift.template import get_template

MODELS = ['HuggingFaceTB/SmolVLM-256M-Instruct', 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct']
GREEN = Image.new('RGB', (640, 480), (10, 200, 30))
RED = Image.new('RGB', (200, 900), (250, 20, 30))


def _get_template(model_id):
    _, processor = get_model_processor(model_id, load_model=False)
    template = get_template(processor)
    template.set_mode('train')
    return template, AutoProcessor.from_pretrained(processor.model_info.model_dir)


def _to_hf_messages(messages):
    """ms-swift messages -> typed chat-template messages (`<image>`/`<video>` become media items)."""
    hf_messages = []
    for message in messages:
        content = []
        for part in re.split(r'(<image>|<video>)', message['content']):
            if part in {'<image>', '<video>'}:
                content.append({'type': part[1:-1]})
            elif part:
                content.append({'type': 'text', 'text': part})
        hf_messages.append({'role': message['role'], 'content': content})
    return hf_messages


def _hf_encode(hf_processor, text, images=(), videos=None, video_metadata=None):
    """Encode chat-template text with the processor, as `apply_chat_template(tokenize=True)` does it."""
    kwargs = {}
    if videos is not None:
        video_processor = hf_processor.video_processor
        kwargs = {
            'videos': [videos],
            'video_metadata': [video_metadata],
            'do_sample_frames': True,
            'num_frames': video_processor.num_frames,
            'fps': video_processor.fps,
        }
    images = list(images)[:text.count('<image>')]
    return hf_processor(
        text=text, images=[images] if images else None, return_tensors='pt', add_special_tokens=False, **kwargs)


def _hf_expected(hf_processor, messages, **kwargs):
    """input_ids, labels and pixels from the processor's chat template; every assistant turn is trained."""
    messages = _to_hf_messages(messages)
    encoded = _hf_encode(hf_processor, hf_processor.apply_chat_template(messages), **kwargs)
    input_ids = encoded['input_ids'][0].tolist()
    assert hf_processor.tokenizer.decode(input_ids[-1:]) == '\n'
    input_ids = input_ids[:-1]  # ms-swift ends with `<end_of_utterance>`, the chat template adds '\n'
    labels = [-100] * len(input_ids)
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        prompt = hf_processor.apply_chat_template(messages[:i], add_generation_prompt=True)
        start = _hf_encode(hf_processor, prompt, **kwargs)['input_ids'].shape[1]
        conversation = hf_processor.apply_chat_template(messages[:i + 1])
        end = _hf_encode(hf_processor, conversation, **kwargs)['input_ids'].shape[1] - 1
        labels[start:end] = input_ids[start:end]
    return input_ids, labels, encoded


CASES = {
    'system_multi_turn': ([('system', 'You are helpful.'), ('user', '<image>Describe.'), ('assistant', 'A green box.'),
                           ('user', 'What color?'), ('assistant', 'Green.')], [GREEN]),
    'images_mid_text': ([('user', 'Compare <image> and <image> please'),
                         ('assistant', 'Green, then red.')], [GREEN, RED]),
    'text_only': ([('user', 'Hello'), ('assistant', 'Hi there.')], []),
}


@pytest.mark.parametrize('model_id', MODELS)
@pytest.mark.parametrize('case', list(CASES))
def test_encode_matches_hf_processor(model_id, case):
    template, hf_processor = _get_template(model_id)
    turns, images = CASES[case]
    messages = [{'role': role, 'content': content} for role, content in turns]
    encoded = template.encode({'messages': messages, 'images': images})

    input_ids, labels, expected = _hf_expected(hf_processor, messages, images=images)
    assert encoded['input_ids'] == input_ids
    assert encoded['labels'] == labels
    if images:
        assert torch.equal(encoded['pixel_values'], expected['pixel_values'])
        assert torch.equal(encoded['pixel_attention_mask'], expected['pixel_attention_mask'])
    else:
        assert 'pixel_values' not in encoded


def _write_video(path):
    av = pytest.importorskip('av')
    import numpy as np
    container = av.open(path, 'w')
    stream = container.add_stream('mpeg4', rate=8)
    stream.width, stream.height, stream.pix_fmt = 320, 240, 'yuv420p'
    for i in range(40):  # 5 seconds
        frame = np.zeros((240, 320, 3), np.uint8)
        frame[:, i * 8:i * 8 + 16] = 255
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def test_smolvlm2_video_matches_hf_processor(tmp_path):
    from swift.template.vision_utils import load_video_hf
    video_path = str(tmp_path / 'clip.mp4')
    _write_video(video_path)
    template, hf_processor = _get_template(MODELS[1])
    messages = [{'role': 'user', 'content': '<video>What happens?'}, {'role': 'assistant', 'content': 'A bar moves.'}]
    encoded = template.encode({'messages': messages, 'videos': [video_path]})

    videos, video_metadata = load_video_hf([video_path])
    input_ids, labels, expected = _hf_expected(hf_processor, messages, videos=videos, video_metadata=video_metadata)
    assert expected['pixel_values'].shape[1] == 5  # sampled at 1 fps, as in the processor's chat template
    assert encoded['input_ids'] == input_ids
    assert encoded['labels'] == labels
    assert torch.equal(encoded['pixel_values'], expected['pixel_values'])
    assert torch.equal(encoded['pixel_attention_mask'], expected['pixel_attention_mask'])


@pytest.mark.parametrize('model_id', MODELS)
def test_data_collator_pads_patches(model_id):
    template, _ = _get_template(model_id)
    reply = {'role': 'assistant', 'content': 'Hello.'}
    batch = [
        template.encode({
            'messages': [{
                'role': 'user',
                'content': '<image>Hi'
            }, reply],
            'images': [image]
        }) for image in [Image.new('RGB', (224, 224)), RED]
    ]
    batch.append(template.encode({'messages': [{'role': 'user', 'content': 'Hi'}, reply]}))
    num_patches = [b['pixel_values'].shape[1] for b in batch[:2]]
    assert num_patches[0] != num_patches[1]

    res = template.data_collator(batch)
    max_num = max(num_patches)
    assert res['pixel_values'].shape[:2] == (2, max_num)
    assert res['pixel_attention_mask'].shape[:2] == (2, max_num)
    for i, n in enumerate(num_patches):
        assert torch.equal(res['pixel_values'][i, :n], batch[i]['pixel_values'][0])
        assert not res['pixel_values'][i, n:].any()
        assert not res['pixel_attention_mask'][i, n:].any()
