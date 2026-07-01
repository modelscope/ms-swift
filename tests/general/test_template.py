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


if __name__ == '__main__':
    test_template()
    test_mllm()
    test_llm_dataset_map()
    test_mllm_dataset_map()


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
