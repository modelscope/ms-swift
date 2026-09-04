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


def test_load_video_minicpmv_handles_zero_fps():
    import numpy as np
    import sys
    import types
    from unittest import mock

    from swift.template import vision_utils

    n_frames = 8

    class _FakeVideoReader:

        def __len__(self):
            return n_frames

        def get_avg_fps(self):
            # decord returns 0.0 when the container carries no / broken fps metadata.
            return 0.0

        def get_batch(self, idx):
            arr = np.zeros((len(list(idx)), 4, 4, 3), dtype='uint8')
            return types.SimpleNamespace(asnumpy=lambda: arr)

    fake_decord = types.SimpleNamespace(
        VideoReader=lambda *args, **kwargs: _FakeVideoReader(),
        cpu=lambda *args, **kwargs: None,
    )

    with mock.patch.dict(sys.modules, {'decord': fake_decord}), \
            mock.patch.object(vision_utils, 'load_file', lambda video: video):
        # A 0-fps video used to raise "range() arg 3 must not be zero" here.
        frames = vision_utils.load_video_minicpmv_mplug_owl3(b'video-bytes', max_num_frames=4)

    assert len(frames) == 4


def test_load_audio_bytes_input_does_not_crash_on_fallback(monkeypatch):
    import sys
    import types

    from swift.template import vision_utils

    calls = []

    fake_librosa = types.ModuleType('librosa')

    def fake_load(audio_io, sr):
        calls.append(audio_io)
        if len(calls) == 1:
            # First attempt fails (e.g. a format soundfile can't read), forcing
            # the except branch that used to call bytes.startswith and crash.
            raise RuntimeError('first load fails')
        return ([0.1, 0.2], sr)

    fake_librosa.load = fake_load
    monkeypatch.setitem(sys.modules, 'librosa', fake_librosa)

    # bytes audio (allowed by the Union[str, bytes] signature) must not raise a
    # TypeError from `audio.startswith(...)` or from `_check_path(bytes)` when
    # the first decode fails and the except branch runs.
    result = vision_utils.load_audio(b'\x00\x01raw-audio-bytes', sampling_rate=16000)

    assert result == [0.1, 0.2]


def test_check_path_with_bytes_returns_none():
    from swift.template.vision_utils import _check_path

    # bytes input is not a path; it must return None instead of raising a
    # TypeError from the str-only checks (len/os.path/startswith) below.
    assert _check_path(b'\x00\x01raw-bytes') is None


if __name__ == '__main__':
    test_template()
    test_mllm()
    test_llm_dataset_map()
    test_mllm_dataset_map()
    test_save_pil_image_dimension_collision()
    test_load_video_minicpmv_handles_zero_fps()
