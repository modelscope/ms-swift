# Copyright (c) ModelScope Contributors. All rights reserved.
import base64
import numpy as np
import os
import pytest
import torch
from PIL import Image

from swift.model import get_processor
from swift.template import get_template
from swift.template.vision_utils import load_video_kimi_vl

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
    image = Image.new('RGB', (84, 56), (12, 34, 56))
    parts = {
        'text': ['Describe the scene.'],
        'image': [image, 'Describe the image.'],
        'video': [path, 'Describe the video.'],
        # Digits adjacent to a timestamp must be tokenized together, including with the Jinja backend.
        'mixed': [image, 'Before:12', path, image, 'After:', path, image, 'Compare.'],
        'multi_turn': [path, 'Describe the video.'],
    }[case]
    swift_parts, official_parts, images = [], [], []
    for part in parts:
        if isinstance(part, Image.Image):
            swift_parts.append({'type': 'image', 'image': part})
            official_parts.append({'type': 'image', 'image': part})
            images.append(part)
        elif part == path:
            swift_parts.append({'type': 'video_url', 'video_url': {'url': path}})
            for frame, timestamp in zip(frames, timestamps):
                official_parts.extend([{'type': 'text', 'text': timestamp}, {'type': 'image', 'image': frame}])
                images.append(frame)
        else:
            swift_parts.append({'type': 'text', 'text': part})
            official_parts.append({'type': 'text', 'text': part})
    swift_messages = [{'role': 'user', 'content': swift_parts}]
    official_messages = [{'role': 'user', 'content': official_parts}]
    if case == 'multi_turn':
        history = [
            {
                'role': 'assistant',
                'content': '◁think▷Inspect the frames.◁/think▷Colored frames.'
            },
            {
                'role': 'user',
                'content': 'Summarize in Chinese.'
            },
        ]
        swift_messages.extend(history)
        official_messages.extend(history)
    return swift_messages, official_messages, images


@pytest.mark.parametrize('backend', ['swift', 'jinja'])
@pytest.mark.parametrize('mode', ['transformers', 'train', 'vllm'])
@pytest.mark.parametrize('case', ['text', 'image', 'video', 'mixed', 'multi_turn'])
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
        assert len(encoded.get('images', [])) == len(images)
        for actual, expected in zip(encoded.get('images', []), images):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    else:
        official = processor(text=[text], images=images or None, return_tensors='pt')
        assert encoded['input_ids'] == official['input_ids'][0].tolist()
        if images:
            torch.testing.assert_close(encoded['pixel_values'], official['pixel_values'], rtol=0, atol=0)
            assert torch.equal(encoded['image_grid_hws'], official['image_grid_hws'])
        if mode == 'train':
            assert len(encoded['labels']) == len(encoded['input_ids'])
            if backend == 'swift':
                supervised = [token for token in encoded['labels'] if token != -100]
                answers = ''.join(message['content'] + '<|im_end|>' for message in official_messages
                                  if message['role'] == 'assistant')
                assert processor.tokenizer.decode(supervised) == answers


@pytest.mark.parametrize('source_type', ['path', 'bytes', 'base64'])
def test_video_decoding(video, source_type):
    path, expected_images, _ = video
    source = path
    if source_type != 'path':
        with open(path, 'rb') as f:
            source = f.read()
        if source_type == 'base64':
            source = 'data:video/mp4;base64,' + base64.b64encode(source).decode()
    images, timestamps = load_video_kimi_vl(source)
    assert timestamps == [0, 1, 1]
    assert len(images) == len(expected_images)
    for actual, expected in zip(images, expected_images):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize('mode', ['transformers', 'vllm'])
def test_base64_video_request(processor, video, mode):
    template = get_template(processor)
    template.set_mode(mode)
    with open(video[0], 'rb') as f:
        url = 'data:video/mp4;base64,' + base64.b64encode(f.read()).decode()
    encoded = template.encode({
        'messages': [{
            'role':
            'user',
            'content': [{
                'type': 'text',
                'text': 'describe'
            }, {
                'type': 'video_url',
                'video_url': {
                    'url': url
                }
            }],
        }],
    })
    expected = template.encode({'messages': [{'role': 'user', 'content': 'describe<video>'}], 'videos': [video[0]]})
    assert encoded['input_ids'] == expected['input_ids']
    if mode == 'vllm':
        assert 'videos' not in encoded
        assert len(encoded['images']) == 3
    else:
        assert torch.equal(encoded['pixel_values'], expected['pixel_values'])


def test_implicit_video_tag_and_batch(processor, video):
    template = get_template(processor)
    template.set_mode('train')
    messages = [{'role': 'user', 'content': 'Describe.'}, {'role': 'assistant', 'content': 'Colors.'}]
    implicit = template.encode({'messages': messages, 'videos': [video[0]]})
    explicit = template.encode({
        'messages': [{
            'role': 'user',
            'content': '<video>Describe.'
        }, messages[1]],
        'videos': [video[0]],
    })
    assert implicit['input_ids'] == explicit['input_ids']
    expected_grids = torch.cat([implicit['image_grid_hws']] * 2)
    expected_pixels = torch.cat([implicit['pixel_values']] * 2)
    batch = template.data_collator([implicit, explicit])
    assert torch.equal(batch['image_grid_hws'], expected_grids)
    assert torch.equal(batch['pixel_values'], expected_pixels)
