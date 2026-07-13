from swift.dataset import EncodePreprocessor, load_dataset
from swift.model import get_processor
from swift.template import TemplateInputs, get_template


def test_template():
    tokenizer = get_processor('Qwen/Qwen2-7B-Instruct')
    template = get_template(tokenizer)
    template_inputs = TemplateInputs.from_dict({
        'messages': [{
            'role': 'system',
            'content': 'AAA'
        }, {
            'role': 'user',
            'content': 'BBB'
        }, {
            'role': 'assistant',
            'content': 'CCC'
        }, {
            'role': 'user',
            'content': 'DDD'
        }]
    })
    inputs = template.encode(template_inputs)
    print(f'inputs.keys(): {inputs.keys()}')
    print(tokenizer.decode(inputs['input_ids']))


def test_mllm():
    processor = get_processor('Qwen/Qwen2-VL-7B-Instruct')
    template = get_template(processor)
    template_inputs = TemplateInputs(
        chosen={
            'messages': [{
                'role': 'system',
                'content': 'AAA'
            }, {
                'role': 'user',
                'content': '<image>BBB'
            }, {
                'role': 'assistant',
                'content': 'CCC'
            }, {
                'role': 'user',
                'content': 'DDD'
            }],
            'images': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
        })
    inputs = template.encode(template_inputs)
    print(f'inputs.keys(): {inputs.keys()}')
    print(template.safe_decode(inputs['input_ids']))


def _test_dataset_map(model_id: str, dataset_id: str):
    tokenizer = get_processor(model_id)
    template = get_template(tokenizer)
    dataset = load_dataset([dataset_id], num_proc=2)[0]

    # 1: 1500
    # 16: 10766.36 examples/s
    new_dataset = EncodePreprocessor(template)(dataset, num_proc=4)
    print(f'new_dataset: {new_dataset}')
    print(template.safe_decode(new_dataset[0]['input_ids']))
    print(template.safe_decode(new_dataset[1]['input_ids']))


def test_llm_dataset_map():
    _test_dataset_map('Qwen/Qwen2-7B-Instruct', 'AI-ModelScope/alpaca-gpt4-data-zh')


def test_mllm_dataset_map():
    _test_dataset_map('Qwen/Qwen2-VL-7B-Instruct', 'modelscope/coco_2014_caption:validation#100')


def test_save_pil_image_dimension_collision():
    from PIL import Image

    from swift.template.base import Template

    # Two images that share the same flattened pixel bytes but differ in shape.
    width_a, height_a = 120, 80
    width_b, height_b = 80, 120
    assert width_a * height_a == width_b * height_b
    pixels = bytearray()
    for i in range(width_a * height_a):
        row = i // width_a
        pixels.extend((255, 60, 60) if row % 10 < 5 else (60, 60, 255))
    img_bytes = bytes(pixels)
    image_a = Image.frombytes('RGB', (width_a, height_a), img_bytes)
    image_b = Image.frombytes('RGB', (width_b, height_b), img_bytes)
    assert image_a.tobytes() == image_b.tobytes()

    path_a = Template._save_pil_image(image_a)
    path_b = Template._save_pil_image(image_b)

    # Different dimensions must not collide onto the same cache file.
    assert path_a != path_b
    with Image.open(path_a) as saved_a:
        assert saved_a.size == (width_a, height_a)
    with Image.open(path_b) as saved_b:
        assert saved_b.size == (width_b, height_b)


if __name__ == '__main__':
    test_template()
    test_mllm()
    test_llm_dataset_map()
    test_mllm_dataset_map()
    test_save_pil_image_dimension_collision()
