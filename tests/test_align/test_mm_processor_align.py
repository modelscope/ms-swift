"""Swift train encode vs vLLM rollout multimodal alignment (GPU + vLLM).

Run:
  cd workspace/swift
  CUDA_VISIBLE_DEVICES=0 python tests/test_align/test_mm_processor_align.py
  CUDA_VISIBLE_DEVICES=0 pytest tests/test_align/test_mm_processor_align.py -v
"""
import copy
import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swift.model import get_processor
from swift.template import get_template

try:
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.inputs import nested_tensors_equal
except ImportError:
    ModelConfig = None
    MULTIMODAL_REGISTRY = None
    nested_tensors_equal = None

pytestmark = pytest.mark.skipif(ModelConfig is None, reason='vLLM not available')

WEATHER_AUDIO = 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav'
BABY_VIDEO = 'https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'
DRAW_VIDEO = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4'
CAT_IMAGE = 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png'
_SKIP_TRAIN_KEYS = frozenset({'input_ids', 'labels', 'loss_scale', 'mm_token_type_ids'})

os.environ['SWIFT_AUDIO_LOAD_BACKEND'] = 'soundfile_pyav'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def as_list_ids(x):
    if isinstance(x, torch.Tensor):
        return x.reshape(-1).tolist()
    return list(x)


def tensors_aligned(a, b) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        a, b = a.detach().cpu(), b.detach().cpu()
        if a.ndim == b.ndim + 1 and a.shape[0] == 1 and a.shape[1:] == b.shape:
            a = a.squeeze(0)
        elif b.ndim == a.ndim + 1 and b.shape[0] == 1 and b.shape[1:] == a.shape:
            b = b.squeeze(0)
        if a.shape != b.shape:
            return False
        if a.dtype.is_floating_point or b.dtype.is_floating_point:
            a = a.to(torch.bfloat16).float()
            b = b.to(torch.bfloat16).float()
            return torch.allclose(a, b, rtol=0, atol=0)
        return torch.equal(a, b)
    if nested_tensors_equal is None:
        return a == b
    return nested_tensors_equal(a, b)


def build_vllm_mm_data(vllm_encoded: Dict[str, Any]) -> Dict[str, Any]:
    mm_data = {}
    for plural, singular in [('images', 'image'), ('videos', 'video'), ('audios', 'audio')]:
        data = vllm_encoded.get(plural)
        if not data:
            continue
        if len(data) == 1 and not isinstance(data[0], tuple):
            mm_data[singular] = data[0]
        else:
            mm_data[singular] = data
    return mm_data


def swift_train_encode(template, sample: dict) -> Dict[str, Any]:
    train_template = copy.deepcopy(template)
    train_template.set_mode('train')
    return train_template.encode(sample)


def vllm_forward_kwargs(model_id: str, template, sample: dict) -> Dict[str, Any]:
    if ModelConfig is None:
        raise RuntimeError('vLLM is not available')
    vllm_template = copy.deepcopy(template)
    vllm_template.set_mode('vllm')
    encoded = vllm_template.encode(sample)
    mm_data = build_vllm_mm_data(encoded)
    if not mm_data:
        return {'input_ids': encoded['input_ids'], 'mm_tensors': {}}
    model_config = ModelConfig(model_id, trust_remote_code=True, dtype='auto', seed=0)
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    mm_items = processor.info.parse_mm_data(mm_data)
    result = processor(
        encoded['input_ids'],
        mm_items=mm_items,
        hf_processor_mm_kwargs=encoded.get('mm_processor_kwargs') or {},
    )
    return {
        'input_ids': result['prompt_token_ids'],
        'mm_tensors': result['mm_kwargs'].get_data(),
    }


@contextmanager
def audio_backend(backend: str):
    prev = os.environ.get('SWIFT_AUDIO_LOAD_BACKEND')
    os.environ['SWIFT_AUDIO_LOAD_BACKEND'] = backend
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop('SWIFT_AUDIO_LOAD_BACKEND', None)
        else:
            os.environ['SWIFT_AUDIO_LOAD_BACKEND'] = prev


def _vllm_audio_feature_lengths(train: dict, vllm_tensors: dict) -> None:
    """vLLM sets audio_feature_lengths; Swift derives the same value from mask.sum()."""
    vllm_afl = vllm_tensors.get('audio_feature_lengths')
    mask = train.get('feature_attention_mask')
    if vllm_afl is None or mask is None:
        return
    derived = mask.sum(-1)
    if derived.ndim == 0:
        derived = derived.unsqueeze(0)
    assert tensors_aligned(derived, vllm_afl), 'mask.sum() != vLLM audio_feature_lengths'


@contextmanager
def use_audio_in_video(enabled: bool = True):
    prev = os.environ.get('USE_AUDIO_IN_VIDEO')
    if enabled:
        os.environ['USE_AUDIO_IN_VIDEO'] = 'true'
    else:
        os.environ.pop('USE_AUDIO_IN_VIDEO', None)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop('USE_AUDIO_IN_VIDEO', None)
        else:
            os.environ['USE_AUDIO_IN_VIDEO'] = prev


def _assert_mm_align(
    model_id,
    sample,
    *,
    tensor_key_aliases=None,
    check_input_ids=True,
    check_vllm_audio_feature_lengths=False,
    use_audio_in_video_flag=False,
):
    tensor_key_aliases = tensor_key_aliases or {}
    ctx = use_audio_in_video() if use_audio_in_video_flag else nullcontext()
    with ctx:
        processor = get_processor(model_id)
        template = get_template(processor)
        train = swift_train_encode(template, sample)
        vllm = vllm_forward_kwargs(model_id, template, sample)

    if check_input_ids:
        assert as_list_ids(train['input_ids']) == as_list_ids(vllm['input_ids'])

    vllm_tensors = dict(vllm['mm_tensors'])
    compared = [
        (tk, tensor_key_aliases.get(tk, tk))
        for tk in sorted(k for k, v in train.items() if v is not None and k not in _SKIP_TRAIN_KEYS)
        if tensor_key_aliases.get(tk, tk) in vllm_tensors
    ]
    for train_key, vllm_key in compared:
        assert tensors_aligned(train[train_key], vllm_tensors[vllm_key]), f'{train_key}!={vllm_key}'
    if check_vllm_audio_feature_lengths:
        _vllm_audio_feature_lengths(train, vllm_tensors)


# ---------------------------------------------------------------------------
# Qwen2.5-Omni
# ---------------------------------------------------------------------------

def test_qwen2_5_omni_image():
    _assert_mm_align(
        'Qwen/Qwen2.5-Omni-7B',
        {'messages': [{'role': 'user', 'content': 'describe the image.'}], 'images': [CAT_IMAGE]},
    )


def test_qwen2_5_omni_video():
    _assert_mm_align(
        'Qwen/Qwen2.5-Omni-7B',
        {'messages': [{'role': 'user', 'content': 'describe the video.'}], 'videos': [BABY_VIDEO]},
        tensor_key_aliases={'video_second_per_grid': 'second_per_grid_ts'},
    )


def test_qwen2_5_omni_audio():
    # Standalone audio: sample has `audios` field (not extracted from video).
    # vLLM path loads as (wav, sr) in _preprocess_inputs; train path uses ndarray.
    _assert_mm_align(
        'Qwen/Qwen2.5-Omni-7B',
        {'messages': [{'role': 'user', 'content': 'describe the audio.'}], 'audios': [WEATHER_AUDIO]},
        check_vllm_audio_feature_lengths=True,
    )


def test_qwen2_5_omni_video_use_audio_in_video():
    # Video track extracted in replace_tag; vLLM uses different audio/video token layout.
    _assert_mm_align(
        'Qwen/Qwen2.5-Omni-7B',
        {'messages': [{'role': 'user', 'content': 'describe the video.'}], 'videos': [DRAW_VIDEO]},
        tensor_key_aliases={'video_second_per_grid': 'second_per_grid_ts'},
        check_input_ids=False,
        check_vllm_audio_feature_lengths=True,
        use_audio_in_video_flag=True,
    )


# ---------------------------------------------------------------------------
# Qwen3-Omni
# ---------------------------------------------------------------------------

def test_qwen3_omni_audio():
    _assert_mm_align(
        'Qwen/Qwen3-Omni-30B-A3B-Instruct',
        {'messages': [{'role': 'user', 'content': 'describe the audio.'}], 'audios': [WEATHER_AUDIO]},
        tensor_key_aliases={
            'input_features': 'input_audio_features',
            'feature_attention_mask': 'feature_attention_mask',
        },
        check_vllm_audio_feature_lengths=True,
    )


def test_qwen3_omni_video_use_audio_in_video():
    _assert_mm_align(
        'Qwen/Qwen3-Omni-30B-A3B-Instruct',
        {'messages': [{'role': 'user', 'content': 'describe the video.'}], 'videos': [DRAW_VIDEO]},
        tensor_key_aliases={
            'input_features': 'input_audio_features',
            'feature_attention_mask': 'feature_attention_mask',
            'video_second_per_grid': 'second_per_grid_ts',
        },
        check_input_ids=False,
        check_vllm_audio_feature_lengths=True,
        use_audio_in_video_flag=True,
    )


# ---------------------------------------------------------------------------
# Gemma4
# ---------------------------------------------------------------------------

def test_gemma4_audio():
    _assert_mm_align(
        'google/gemma-4-E2B-it',
        {'messages': [{'role': 'user', 'content': 'describe the audio.'}], 'audios': [WEATHER_AUDIO]},
        tensor_key_aliases={
            'input_features': 'input_features_padded',
            'input_features_mask': 'input_features_mask',
        },
    )


@pytest.mark.xfail(reason='vLLM gemma4 video timestamp/soft-token path differs from HF Gemma4VideoProcessor')
def test_gemma4_video():
    _assert_mm_align(
        'google/gemma-4-E2B-it',
        {'messages': [{'role': 'user', 'content': '<video>describe the video.'}], 'videos': [BABY_VIDEO]},
        tensor_key_aliases={
            'pixel_values_videos': 'pixel_values_videos',
            'video_position_ids': 'video_position_ids',
        },
    )


if __name__ == '__main__':
    tests = [
        test_qwen2_5_omni_image,
        test_qwen2_5_omni_video,
        test_qwen2_5_omni_audio,
        test_qwen2_5_omni_video_use_audio_in_video,
        test_qwen3_omni_audio,
        test_qwen3_omni_video_use_audio_in_video,
        test_gemma4_audio,
        test_gemma4_video,
    ]
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f'{name}: PASS')
        except Exception:
            if fn is test_gemma4_video:
                print(f'{name}: XFAIL (expected vLLM upstream mismatch)')
            else:
                raise
    print('all mm processor align tests finished')
