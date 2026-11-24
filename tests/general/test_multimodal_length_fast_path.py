"""Hermetic unit tests for the header-only multimodal length pre-pass fast path.

These tests run without downloading any model weights or requiring a GPU. They verify:
- ``load_image_size`` reads dimensions from headers without decoding pixels (path, bytes,
  BytesIO, base64, PIL.Image, and Arrow dict payloads).
- ``rescale_image`` produces identical dimensions for ``ImageSize`` and ``PIL.Image``.
- ``Template.encode(..., return_length=True, length_only=True)`` and ``AddLengthPreprocessor``
  produce exact ``lengths`` matching the full pixel-decoding path across Qwen2-VL, Qwen2.5-VL,
  Qwen3-VL, and Qwen3.5 templates, multi-image inputs, custom resize kwargs, grounding bboxes,
  and truncation strategies, while skipping ``fetch_image`` and ``image_processor`` entirely.
- Subclasses that override ``replace_tag`` or ``_encode`` (e.g. ``Qwen2_5OmniTemplate``) opt out
  safely so they continue receiving decoded ``PIL.Image`` instances.
"""
import base64
import math
import os
import sys
import tempfile
import torch
import types
import unittest
from io import BytesIO
from PIL import Image
from unittest import mock

from swift.dataset.utils import AddLengthPreprocessor
from swift.template.base import MaxLengthError
from swift.template.constant import MLLMTemplateType
from swift.template.register import TEMPLATE_MAPPING
from swift.template.templates.qwen import (Qwen2_5OmniTemplate, Qwen2_5VLTemplate, Qwen2VLTemplate, Qwen3_5Template,
                                           Qwen3VLTemplate)
from swift.template.vision_utils import ImageSize, load_image_size, rescale_image


def _make_fake_qwen_vl_utils(*, modern: bool = True, max_ratio: int = 200):
    """Build an in-memory ``qwen_vl_utils`` module mirroring upstream ``vision_process``."""
    vp = types.ModuleType('qwen_vl_utils.vision_process')
    vp.MAX_RATIO = max_ratio
    if modern:
        vp.SPATIAL_MERGE_SIZE = 2
        vp.IMAGE_MIN_TOKEN_NUM = 4
        vp.IMAGE_MAX_TOKEN_NUM = 16384
    else:
        vp.IMAGE_FACTOR = 28
        vp.MIN_PIXELS = 4 * 28 * 28
        vp.MAX_PIXELS = 16384 * 28 * 28

    def round_by_factor(number: int, factor: int) -> int:
        return round(number / factor) * factor

    def ceil_by_factor(number: int, factor: int) -> int:
        return math.ceil(number / factor) * factor

    def floor_by_factor(number: int, factor: int) -> int:
        return math.floor(number / factor) * factor

    def smart_resize(height: int, width: int, factor: int = 28, min_pixels=None, max_pixels=None):
        if modern:
            max_pixels = max_pixels if max_pixels is not None else (vp.IMAGE_MAX_TOKEN_NUM * factor**2)
            min_pixels = min_pixels if min_pixels is not None else (vp.IMAGE_MIN_TOKEN_NUM * factor**2)
        else:
            max_pixels = max_pixels if max_pixels is not None else vp.MAX_PIXELS
            min_pixels = min_pixels if min_pixels is not None else vp.MIN_PIXELS
        if max(height, width) / min(height, width) > vp.MAX_RATIO:
            raise ValueError(f'absolute aspect ratio must be smaller than {vp.MAX_RATIO}')
        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    vp.smart_resize = smart_resize
    fetch_calls = []

    def fetch_image(ele, image_patch_size: int = 14):
        fetch_calls.append(ele)
        img = ele['image']
        if not isinstance(img, Image.Image):
            raise TypeError(f'expected PIL.Image, got {type(img)!r}')
        if modern:
            patch_factor = int(image_patch_size * vp.SPATIAL_MERGE_SIZE)
            default_min = vp.IMAGE_MIN_TOKEN_NUM * patch_factor**2
            default_max = vp.IMAGE_MAX_TOKEN_NUM * patch_factor**2
        else:
            patch_factor = vp.IMAGE_FACTOR
            default_min = vp.MIN_PIXELS
            default_max = vp.MAX_PIXELS
        if 'resized_height' in ele and 'resized_width' in ele:
            rh, rw = smart_resize(ele['resized_height'], ele['resized_width'], factor=patch_factor)
        else:
            min_p = ele.get('min_pixels', default_min)
            max_p = ele.get('max_pixels', default_max)
            rh, rw = smart_resize(img.height, img.width, factor=patch_factor, min_pixels=min_p, max_pixels=max_p)
        return img.resize((rw, rh))

    def fetch_video(*args, **kwargs):
        raise NotImplementedError

    pkg = types.ModuleType('qwen_vl_utils')
    pkg.vision_process = vp
    pkg.fetch_image = fetch_image
    pkg.fetch_video = fetch_video
    pkg._fetch_calls = fetch_calls
    return pkg, vp


class _DummyImageProcessor:

    def __init__(self, patch_size: int, merge_size: int = 2):
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.calls = 0

    def __call__(self, images=None, return_tensors='pt', do_resize=False, **kwargs):
        self.calls += 1
        grids = []
        for img in images:
            if not isinstance(img, Image.Image):
                raise TypeError(f'image_processor expects PIL.Image, got {type(img)!r}')
            grids.append([1, img.height // self.patch_size, img.width // self.patch_size])
        return {
            'pixel_values': torch.zeros((sum(g[1] * g[2] for g in grids), 3), dtype=torch.float32),
            'image_grid_thw': torch.tensor(grids, dtype=torch.long),
        }


class _DummyTokenizer:
    """Minimal deterministic tokenizer supporting special vision tokens."""

    def __init__(self, image_token_id: int, video_token_id: int):
        self.special = {
            '<|im_start|>': 151644,
            '<|im_end|>': 151645,
            '<|vision_start|>': 151652,
            '<|vision_end|>': 151653,
            '<|image_pad|>': image_token_id,
            '<|video_pad|>': video_token_id,
            '<|object_ref_start|>': 151646,
            '<|object_ref_end|>': 151647,
            '<|box_start|>': 151648,
            '<|box_end|>': 151649,
            '<|endoftext|>': 151643,
        }
        self._vocab = dict(self.special)
        self._next_id = 100
        self.pad_token = '<|endoftext|>'
        self.eos_token = '<|im_end|>'
        self.bos_token = None
        self.pad_token_id = 151643
        self.eos_token_id = 151645
        self.bos_token_id = None
        self.model_info = types.SimpleNamespace(
            max_model_len=32768,
            config=types.SimpleNamespace(vision_config=object()),
            task_type='causal_lm',
        )
        self.model_meta = types.SimpleNamespace(is_multimodal=True)
        self.chat_template = None

    def convert_tokens_to_ids(self, token):
        if isinstance(token, list):
            return [self.convert_tokens_to_ids(t) for t in token]
        return self.special.get(token, 1)

    def __call__(self, text: str, **kwargs):
        return {'input_ids': self.encode(text)}

    def encode(self, text: str, add_special_tokens: bool = False):
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for tok, tid in self.special.items():
                if text.startswith(tok, i):
                    ids.append(tid)
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                ch = text[i]
                if ch not in self._vocab:
                    self._vocab[ch] = self._next_id
                    self._next_id += 1
                ids.append(self._vocab[ch])
                i += 1
        return ids


class _DummyProcessor:

    def __init__(self, patch_size: int, image_token_id: int, video_token_id: int):
        self.tokenizer = _DummyTokenizer(image_token_id=image_token_id, video_token_id=video_token_id)
        self.image_processor = _DummyImageProcessor(patch_size=patch_size, merge_size=2)
        self.model_info = self.tokenizer.model_info
        self.model_meta = self.tokenizer.model_meta


def _png_bytes(width: int, height: int, mode: str = 'RGB') -> bytes:
    buf = BytesIO()
    color = (10, 20, 30, 128) if mode == 'RGBA' else (10, 20, 30) if mode == 'RGB' else 128
    Image.new(mode, (width, height), color).save(buf, format='PNG')
    return buf.getvalue()


def _jpeg_bytes(width: int, height: int) -> bytes:
    buf = BytesIO()
    Image.new('RGB', (width, height), (40, 80, 120)).save(buf, format='JPEG')
    return buf.getvalue()


def _build_template(template_type: str, patch_size: int, **kwargs):
    template_meta = TEMPLATE_MAPPING[template_type]
    cls = template_meta.template_cls
    processor = _DummyProcessor(
        patch_size=patch_size, image_token_id=cls.image_token_id, video_token_id=cls.video_token_id)
    template = cls(processor=processor, template_meta=template_meta, **kwargs)
    template.set_mode('train')
    return template, processor


class TestLoadImageSize(unittest.TestCase):

    def test_header_only_formats_without_pixel_decode(self):
        png_data = _png_bytes(640, 360, mode='RGBA')
        jpg_data = _jpeg_bytes(1024, 512)
        b64_data = 'data:image/png;base64,' + base64.b64encode(png_data).decode('ascii')

        with tempfile.TemporaryDirectory() as tmpdir:
            png_path = os.path.join(tmpdir, 'sample.png')
            with open(png_path, 'wb') as f:
                f.write(png_data)

            with mock.patch.object(Image.Image, 'convert', side_effect=AssertionError('must not decode pixels')):
                self.assertEqual(load_image_size(png_path), ImageSize(640, 360))
                self.assertEqual(load_image_size(png_data), ImageSize(640, 360))
                self.assertEqual(load_image_size(BytesIO(jpg_data)), ImageSize(1024, 512))
                self.assertEqual(load_image_size(b64_data), ImageSize(640, 360))
                self.assertEqual(load_image_size(Image.new('L', (123, 45))), ImageSize(123, 45))
                self.assertEqual(load_image_size(ImageSize(77, 88)), ImageSize(77, 88))

    def test_rescale_image_matches_pil_dimensions(self):
        for w, h, max_pixels in [(3840, 2176, 1000000), (777, 1000, 200000), (300, 200, 500000)]:
            proxy = rescale_image(ImageSize(w, h), max_pixels)
            ratio = w / h
            if w * h <= max_pixels:
                expected = (w, h)
            else:
                hs = math.sqrt(max_pixels / ratio)
                ws = hs * ratio
                expected = (int(ws), int(hs))
            self.assertEqual(proxy.size, expected)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_image_size('/nonexistent/path/to/image.png')


class TestQwenVLLengthFastPath(unittest.TestCase):

    def _assert_fast_matches_full(self, template, processor, fake_pkg, row):
        fake_pkg._fetch_calls.clear()
        processor.image_processor.calls = 0

        full_encoded = template.encode(row, return_length=True, length_only=False)
        self.assertGreater(len(fake_pkg._fetch_calls), 0)
        self.assertGreater(processor.image_processor.calls, 0)

        fake_pkg._fetch_calls.clear()
        processor.image_processor.calls = 0

        fast_encoded = template.encode(row, return_length=True, length_only=True)
        self.assertEqual(len(fake_pkg._fetch_calls), 0, 'fast path must skip fetch_image')
        self.assertEqual(processor.image_processor.calls, 0, 'fast path must skip image_processor')
        self.assertEqual(fast_encoded['lengths'], full_encoded['lengths'])
        self.assertNotIn('pixel_values', fast_encoded)
        self.assertFalse(template._length_only)

    def test_exact_length_parity_across_qwen_vl_family(self):
        fake_pkg, fake_vp = _make_fake_qwen_vl_utils(modern=True)
        cases = [
            (MLLMTemplateType.qwen2_vl, 14),
            (MLLMTemplateType.qwen2_5_vl, 14),
            (MLLMTemplateType.qwen3_vl, 16),
            (MLLMTemplateType.qwen3_5, 16),
        ]
        # Include 4K, odd non-multiples of 32, RGBA, JPEG, and multi-image rows.
        img_4k = _png_bytes(3840, 2176, mode='RGBA')
        img_odd = _jpeg_bytes(1000, 777)
        img_small = _png_bytes(53, 41, mode='RGB')

        with mock.patch.dict(sys.modules, {'qwen_vl_utils': fake_pkg, 'qwen_vl_utils.vision_process': fake_vp}):
            for template_type, patch_size in cases:
                template, processor = _build_template(template_type, patch_size)
                for row in [
                    {
                        'messages': [
                            {
                                'role': 'user',
                                'content': '<image>Describe this 4K screenshot.'
                            },
                            {
                                'role': 'assistant',
                                'content': 'It shows a dashboard.'
                            },
                        ],
                        'images': [{
                            'bytes': img_4k,
                            'path': None
                        }],
                    },
                    {
                        'messages': [
                            {
                                'role': 'user',
                                'content': '<image><image>Compare both images.'
                            },
                            {
                                'role': 'assistant',
                                'content': 'First is larger than second.'
                            },
                        ],
                        'images': [{
                            'bytes': img_odd,
                            'path': None
                        }, {
                            'bytes': img_small,
                            'path': None
                        }],
                    },
                    {
                        'messages': [
                            {
                                'role': 'user',
                                'content': '<image>Custom resize bounds.'
                            },
                            {
                                'role': 'assistant',
                                'content': 'Done.'
                            },
                        ],
                        'images': [{
                            'bytes': img_4k,
                            'path': None
                        }],
                        'chat_template_kwargs': {
                            'min_pixels': 4 * 32 * 32,
                            'max_pixels': 256 * 32 * 32
                        },
                    },
                    {
                        'messages': [
                            {
                                'role': 'user',
                                'content': '<image>Explicit resized_height/width.'
                            },
                            {
                                'role': 'assistant',
                                'content': 'OK.'
                            },
                        ],
                        'images': [{
                            'bytes': img_odd,
                            'path': None
                        }],
                        'chat_template_kwargs': {
                            'resized_height': 448,
                            'resized_width': 672
                        },
                    },
                ]:
                    with self.subTest(template=template_type, msg=row['messages'][0]['content']):
                        self._assert_fast_matches_full(template, processor, fake_pkg, row)

    def test_legacy_qwen_vl_utils_parity(self):
        fake_pkg, fake_vp = _make_fake_qwen_vl_utils(modern=False)
        img_bytes = _png_bytes(1920, 1080)
        row = {
            'messages': [
                {
                    'role': 'user',
                    'content': '<image>Legacy qwen_vl_utils check.'
                },
                {
                    'role': 'assistant',
                    'content': 'Looks good.'
                },
            ],
            'images': [{
                'bytes': img_bytes,
                'path': None
            }],
        }
        with mock.patch.dict(sys.modules, {'qwen_vl_utils': fake_pkg, 'qwen_vl_utils.vision_process': fake_vp}):
            template, processor = _build_template(MLLMTemplateType.qwen2_5_vl, patch_size=14)
            self._assert_fast_matches_full(template, processor, fake_pkg, row)

    def test_grounding_bbox_and_max_pixels_parity(self):
        fake_pkg, fake_vp = _make_fake_qwen_vl_utils(modern=True)
        img_bytes = _png_bytes(800, 600)
        row = {
            'messages': [
                {
                    'role': 'user',
                    'content': '<image>Locate <ref-object>.'
                },
                {
                    'role': 'assistant',
                    'content': '<bbox>'
                },
            ],
            'images': [{
                'bytes': img_bytes,
                'path': None
            }],
            'objects': {
                'ref': ['cat'],
                'bbox': [[100, 150, 400, 450]],
                'bbox_type': 'real',
                'image_id': [0]
            },
        }
        with mock.patch.dict(sys.modules, {'qwen_vl_utils': fake_pkg, 'qwen_vl_utils.vision_process': fake_vp}):
            for template_type in (MLLMTemplateType.qwen2_vl, MLLMTemplateType.qwen2_5_vl):
                template, processor = _build_template(template_type, patch_size=14, max_pixels=200000)
                self._assert_fast_matches_full(template, processor, fake_pkg, row)

    def test_truncation_and_max_length_error_restores_state(self):
        fake_pkg, fake_vp = _make_fake_qwen_vl_utils(modern=True)
        img_bytes = _png_bytes(1024, 1024)
        row = {
            'messages': [
                {
                    'role': 'user',
                    'content': '<image>' + 'hello ' * 20
                },
                {
                    'role': 'assistant',
                    'content': 'world ' * 20
                },
            ],
            'images': [{
                'bytes': img_bytes,
                'path': None
            }],
        }
        with mock.patch.dict(sys.modules, {'qwen_vl_utils': fake_pkg, 'qwen_vl_utils.vision_process': fake_vp}):
            template, processor = _build_template(
                MLLMTemplateType.qwen3_vl, patch_size=16, max_length=64, truncation_strategy='raise')
            with self.assertRaises(MaxLengthError):
                template.encode(row, return_length=True, length_only=True)
            self.assertFalse(template._length_only, '_length_only must reset on exception')
            self.assertEqual(processor.image_processor.calls, 0)

            for strategy in ('right', 'left'):
                t, p = _build_template(
                    MLLMTemplateType.qwen3_vl, patch_size=16, max_length=64, truncation_strategy=strategy)
                self._assert_fast_matches_full(t, p, fake_pkg, row)

    def test_add_length_preprocessor_preserves_row_images(self):
        fake_pkg, fake_vp = _make_fake_qwen_vl_utils(modern=True)
        img_bytes = _png_bytes(640, 480)
        row = {
            'messages': [
                {
                    'role': 'user',
                    'content': '<image>What is this?'
                },
                {
                    'role': 'assistant',
                    'content': 'A test image.'
                },
            ],
            'images': [{
                'bytes': img_bytes,
                'path': None
            }],
        }
        with mock.patch.dict(sys.modules, {'qwen_vl_utils': fake_pkg, 'qwen_vl_utils.vision_process': fake_vp}):
            template, processor = _build_template(MLLMTemplateType.qwen3_5, patch_size=16)
            expected = template.encode(row, return_length=True, length_only=False)['lengths']

            fake_pkg._fetch_calls.clear()
            processor.image_processor.calls = 0

            preprocessor = AddLengthPreprocessor(template)
            out = preprocessor.preprocess(row)
            self.assertEqual(out['lengths'], expected)
            self.assertEqual(out['images'], [{'bytes': img_bytes, 'path': None}])
            self.assertEqual(len(fake_pkg._fetch_calls), 0)
            self.assertEqual(processor.image_processor.calls, 0)

    def test_real_qwen_vl_utils_parity(self):
        try:
            import qwen_vl_utils  # noqa: F401
        except ImportError:
            self.skipTest('qwen_vl_utils not installed')

        img_4k = _png_bytes(3840, 2176, mode='RGBA')
        img_odd = _jpeg_bytes(1000, 777)
        row = {
            'messages': [
                {
                    'role': 'user',
                    'content': '<image><image>Compare real qwen_vl_utils resize.'
                },
                {
                    'role': 'assistant',
                    'content': 'Matched.'
                },
            ],
            'images': [{
                'bytes': img_4k,
                'path': None
            }, {
                'bytes': img_odd,
                'path': None
            }],
        }
        for template_type, patch_size in ((MLLMTemplateType.qwen2_5_vl, 14), (MLLMTemplateType.qwen3_5, 16)):
            template, processor = _build_template(template_type, patch_size=patch_size)
            full_encoded = template.encode(row, return_length=True, length_only=False)
            processor.image_processor.calls = 0
            fast_encoded = template.encode(row, return_length=True, length_only=True)
            self.assertEqual(fast_encoded['lengths'], full_encoded['lengths'])
            self.assertEqual(processor.image_processor.calls, 0)

    def test_overriding_subclasses_opt_out_safely(self):

        class CustomReplaceTagTemplate(Qwen3VLTemplate):

            def replace_tag(self, media_type, index, inputs):
                return super().replace_tag(media_type, index, inputs)

        class CustomEncodeTemplate(Qwen3_5Template):

            def _encode(self, inputs):
                return super()._encode(inputs)

        for cls in (Qwen2_5OmniTemplate, CustomReplaceTagTemplate, CustomEncodeTemplate):
            inst = cls.__new__(cls)
            inst.mode = 'train'
            self.assertFalse(inst._can_use_length_only_images(), f'{cls.__name__} must opt out')

        for cls in (Qwen2VLTemplate, Qwen2_5VLTemplate, Qwen3VLTemplate, Qwen3_5Template):
            inst = cls.__new__(cls)
            inst.mode = 'train'
            self.assertTrue(inst._can_use_length_only_images(), f'{cls.__name__} should opt in')
            inst.mode = 'vllm'
            self.assertFalse(inst._can_use_length_only_images(), f'{cls.__name__} must opt out in vllm mode')


if __name__ == '__main__':
    unittest.main()
