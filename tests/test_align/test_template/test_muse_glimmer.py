import os

os.environ['SWIFT_DEBUG'] = '1'

MODEL = 'meta-models/Muse-Glimmer-30B'


def _get_processor_template():
    from swift.model import get_model_processor
    from swift.template import get_template
    _, processor = get_model_processor(MODEL, load_model=False)
    return processor, get_template(processor)


def _hf_text(processor, messages):
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def test_text():
    """The rendered text must match chat_template.jinja byte for byte."""
    from swift import InferRequest
    processor, template = _get_processor_template()
    cases = [
        [{
            'role': 'user',
            'content': 'Hello!'
        }],
        [{
            'role': 'system',
            'content': 'You are a cat.'
        }, {
            'role': 'user',
            'content': 'Hi'
        }],
        [{
            'role': 'user',
            'content': 'q1'
        }, {
            'role': 'assistant',
            'content': 'a1'
        }, {
            'role': 'user',
            'content': 'q2'
        }],
    ]
    for messages in cases:
        encoded = template.encode(InferRequest(messages=messages))
        assert processor.tokenizer.decode(encoded['input_ids']) == _hf_text(processor, messages)


def test_image():
    """`<|patch|>` expansion, pixel_values and image_grid_thw must match the HF processor."""
    from PIL import Image

    from swift import InferRequest
    processor, template = _get_processor_template()
    images = [Image.new('RGB', (112, 112), (255, 0, 0)), Image.new('RGB', (224, 168), (0, 128, 255))]
    for n in (1, 2):
        imgs = images[:n]
        content = [{'type': 'image', 'image': img} for img in imgs] + [{'type': 'text', 'text': 'Compare'}]
        text = _hf_text(processor, [{'role': 'user', 'content': content}])
        # `add_special_tokens=False`: the rendered text already carries `<|begin_of_text|>`.
        ref = processor(text=[text], images=imgs, return_tensors='pt', add_special_tokens=False)
        encoded = template.encode(
            InferRequest(messages=[{
                'role': 'user',
                'content': '<image>' * n + 'Compare'
            }], images=imgs))
        assert ref['input_ids'][0].tolist() == encoded['input_ids']
        assert ref['pixel_values'].shape == encoded['pixel_values'].shape
        assert (ref['image_grid_thw'] == encoded['image_grid_thw']).all()


def test_video():
    """Videos interleave per-frame timestamps and separators; metadata must not leak downstream."""
    import numpy as np

    from swift import InferRequest
    processor, template = _get_processor_template()
    video = np.stack([np.full((112, 112, 3), t * 30, dtype=np.uint8) for t in range(8)])
    content = [{'type': 'video', 'video': video}, {'type': 'text', 'text': 'Describe'}]
    text = _hf_text(processor, [{'role': 'user', 'content': content}])
    ref = processor(text=[text], videos=[video], return_tensors='pt', add_special_tokens=False)
    encoded = template.encode(InferRequest(messages=[{'role': 'user', 'content': '<video>Describe'}], videos=[video]))
    assert ref['input_ids'][0].tolist() == encoded['input_ids']
    assert 'video_metadata' not in encoded


if __name__ == '__main__':
    test_text()
    test_image()
    test_video()
