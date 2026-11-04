import os
import torch
import unittest
from functools import lru_cache
from transformers.utils import strtobool

# NOTE: All tests here only load the processor (tokenizer + vision processor) and the
# remote-code python files via `get_model_processor` (i.e. `load_model=False`), so
# *.safetensors / *.bin weight shards are NOT downloaded (see `safe_snapshot_download`,
# which appends ['*.bin', '*.safetensors'] to `ignore_patterns` when `download_model=False`).
# The template `encode`/official processor `__call__` paths are pure tokenization +
# image/video preprocessing and never run a model forward, so no weights are required.

# The ModelScope and Hugging Face hubs host this model under different orgs; pick the
# default id by download channel so both CI (ModelScope) and local USE_HF=1 runs work.
_USE_HF = strtobool(os.environ.get('USE_HF', '0'))
MODEL = os.getenv('MOSS_VL_TEST_MODEL',
                  None) or ('OpenMOSS-Team/MOSS-VL-Instruct-0708' if _USE_HF else 'openmoss/MOSS-VL-Instruct-0708')
VIDEO = 'https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'


@lru_cache(maxsize=1)
def _get_processor():
    from swift.model import get_model_processor
    try:
        _, processor = get_model_processor(MODEL, load_model=False)
    except ImportError as e:
        # The remote video processor imports torchcodec, whose wheels must match the
        # local torch version; environments without a compatible torchcodec (e.g. CI
        # images on older torch) skip these tests instead of erroring out.
        raise unittest.SkipTest(f'MOSS-VL processor dependency unavailable: {e}')
    return processor


def _get_template(mode='transformers'):
    from swift.template import get_template
    template = get_template(_get_processor())
    template.set_mode(mode)
    return template


@lru_cache(maxsize=1)
def _get_video():
    video = os.getenv('MOSS_VL_TEST_VIDEO', VIDEO)
    if video.startswith(('http://', 'https://')):
        from swift.utils import download_file
        video = download_file(video)
    return video


def _assert_processor_outputs_equal(reference, encoded):
    assert reference['input_ids'][0].tolist() == encoded['input_ids']
    for key in ('pixel_values', 'grid_thw', 'cross_attention_mask'):
        assert torch.equal(reference[key], encoded[key]), key
    assert reference['media_nums_per_sample'] == encoded['media_nums_per_sample']


class TestMossVLProcessorAlignment(unittest.TestCase):

    def test_text(self):
        from swift import InferRequest
        processor = _get_processor()
        template = _get_template()
        messages = [{'role': 'user', 'content': 'Hello!'}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        reference = processor(text=[text], padding=False, return_tensors='pt')
        encoded = template.encode(InferRequest(messages=messages))
        _assert_processor_outputs_equal(reference, encoded)

    def test_image(self):
        from PIL import Image

        from swift import InferRequest
        processor = _get_processor()
        template = _get_template()
        image = Image.new('RGB', (128, 96), (255, 0, 0))
        content = [{'type': 'image', 'image': image}, {'type': 'text', 'text': 'Describe the image.'}]
        native_messages = [{'role': 'user', 'content': content}]
        text = processor.apply_chat_template(native_messages, tokenize=False, add_generation_prompt=True)
        reference = processor(text=[text], images=[image], padding=False, return_tensors='pt')
        encoded = template.encode(
            InferRequest(messages=[{
                'role': 'user',
                'content': '<image>Describe the image.'
            }], images=[image]))
        _assert_processor_outputs_equal(reference, encoded)

    def test_video(self):
        from swift import InferRequest
        processor = _get_processor()
        template = _get_template()
        video = _get_video()
        content = [{'type': 'video', 'video': video}, {'type': 'text', 'text': 'Describe the video.'}]
        native_messages = [{'role': 'user', 'content': content}]
        text = processor.apply_chat_template(native_messages, tokenize=False, add_generation_prompt=True)
        reference = processor(text=[text], videos=[video], padding=False, return_tensors='pt')
        encoded = template.encode(
            InferRequest(messages=[{
                'role': 'user',
                'content': '<video>Describe the video.'
            }], videos=[video]))
        _assert_processor_outputs_equal(reference, encoded)

    def test_training_labels(self):
        from PIL import Image

        from swift import InferRequest
        processor = _get_processor()
        template = _get_template('train')
        image = Image.new('RGB', (128, 96), (0, 128, 255))
        content = [{'type': 'image', 'image': image}, {'type': 'text', 'text': 'What color is it?'}]
        native_messages = [
            {
                'role': 'user',
                'content': content
            },
            {
                'role': 'assistant',
                'content': 'It is blue.'
            },
        ]
        text = processor.apply_chat_template(native_messages, tokenize=False, add_generation_prompt=False)
        response_start = text.index('It is blue.')
        response_end = response_start + len('It is blue.<|im_end|>')
        reference = processor(
            text=[text],
            images=[image],
            labels_spans=[[[response_start, response_end]]],
            padding=False,
            return_tensors='pt')
        encoded = template.encode(
            InferRequest(
                messages=[{
                    'role': 'user',
                    'content': '<image>What color is it?'
                }, {
                    'role': 'assistant',
                    'content': 'It is blue.'
                }],
                images=[image]))
        _assert_processor_outputs_equal(reference, encoded)
        self.assertEqual(reference['labels'][0].tolist(), encoded['labels'])
        supervised = processor.tokenizer.decode([token for token in encoded['labels'] if token != -100])
        self.assertEqual(supervised, 'It is blue.<|im_end|>')


if __name__ == '__main__':
    unittest.main()
