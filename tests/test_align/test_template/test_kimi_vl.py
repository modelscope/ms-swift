# Copyright (c) ModelScope Contributors. All rights reserved.
import base64
import numpy as np
import os
import pytest
import torch
from PIL import Image

from swift.model import get_processor
from swift.template import get_template

# Only the official tokenizer and processor are loaded; no model weights are required.
MODEL_ID = os.environ.get('KIMI_VL_MODEL_ID', 'moonshotai/Kimi-VL-A3B-Thinking-2506')


@pytest.fixture(scope='module')
def processor():
    return get_processor(MODEL_ID, model_type='kimi_vl')


@pytest.fixture(scope='module')
def video(tmp_path_factory):
    av = pytest.importorskip('av')
    decord = pytest.importorskip('decord')
    path = tmp_path_factory.mktemp('kimi_vl') / 'video.mp4'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=2)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = 'yuv420p'
        for i in range(4):
            pixels = np.zeros((48, 64, 3), dtype=np.uint8)
            pixels[:, :, i % 3] = 60 + i * 40
            frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    reader = decord.VideoReader(str(path), num_threads=1)
    # Official cookbook: a two-second clip at 1 sample/s selects indices 0, 2, 3.
    images = [Image.fromarray(reader[i].asnumpy()).resize((448, 336)) for i in [0, 2, 3]]
    return str(path), images, ['00:00:00', '00:00:01', '00:00:01']


def _messages(case, video):
    path, frames, timestamps = video
    with open(path, 'rb') as f:
        video_url = 'data:video/mp4;base64,' + base64.b64encode(f.read()).decode()
    image1 = Image.new('RGB', (84, 56), (12, 34, 56))
    image2 = Image.new('RGB', (56, 84), (56, 34, 12))
    parts = {
        'video': [path, 'Describe the video.'],
        # Digits adjacent to a timestamp must be tokenized together, including with the Jinja backend.
        'mixed': [image1, 'Before:12', path, image2, 'After:', video_url, image1, 'Compare.'],
    }[case]
    swift_parts, official_parts, images = [], [], []
    for part in parts:
        if isinstance(part, Image.Image):
            swift_parts.append({'type': 'image', 'image': part})
            official_parts.append({'type': 'image', 'image': part})
            images.append(part)
        elif part in (path, video_url):
            swift_parts.append({'type': 'video_url', 'video_url': {'url': part}})
            for frame, timestamp in zip(frames, timestamps):
                official_parts.extend([{'type': 'text', 'text': timestamp}, {'type': 'image', 'image': frame}])
                images.append(frame)
        else:
            swift_parts.append({'type': 'text', 'text': part})
            official_parts.append({'type': 'text', 'text': part})
    swift_messages = [{'role': 'user', 'content': swift_parts}]
    official_messages = [{'role': 'user', 'content': official_parts}]
    return swift_messages, official_messages, images


@pytest.mark.parametrize('backend', ['swift', 'jinja'])
@pytest.mark.parametrize('mode', ['transformers', 'train', 'vllm'])
@pytest.mark.parametrize('case', ['video', 'mixed'])
def test_official_processor_alignment(processor, video, backend, mode, case):
    template = get_template(processor, template_backend=backend)
    template.set_mode(mode)
    messages, official_messages, images = _messages(case, video)
    if case == 'mixed':
        system = {'role': 'system', 'content': 'Follow the media in order.'}
        messages.insert(0, system)
        official_messages.insert(0, system)
    if mode == 'train':
        answer = {'role': 'assistant', 'content': '◁think▷Inspect the scene.◁/think▷Colored frames.'}
        messages.append(answer)
        official_messages.append(answer)
    encoded = template.encode({'messages': messages})
    text = processor.apply_chat_template(official_messages, tokenize=False, add_generation_prompt=mode != 'train')
    if mode == 'vllm':
        assert encoded['input_ids'] == processor.tokenizer(text)['input_ids']
        assert 'videos' not in encoded
        assert len(encoded['images']) == len(images)
        for actual, expected in zip(encoded['images'], images):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    else:
        official = processor(text=[text], images=images, return_tensors='pt')
        assert encoded['input_ids'] == official['input_ids'][0].tolist()
        torch.testing.assert_close(encoded['pixel_values'], official['pixel_values'], rtol=0, atol=0)
        assert torch.equal(encoded['image_grid_hws'], official['image_grid_hws'])
        if mode == 'train':
            assert len(encoded['labels']) == len(encoded['input_ids'])
            if backend == 'swift':
                supervised = [token for token in encoded['labels'] if token != -100]
                assert processor.tokenizer.decode(supervised) == answer['content'] + '<|im_end|>'
