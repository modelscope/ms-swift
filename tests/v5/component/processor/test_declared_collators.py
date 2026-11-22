"""Dev family media hooks preserve legacy tensor batching without importing legacy at runtime."""

import copy
from types import SimpleNamespace

import pytest
import torch


def _tensor(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _items(n, seed):
    return [_tensor(2, 3, seed=seed + i) for i in range(n)]


CASES = {
    "chatglm4v": lambda s: {"images": _tensor(1, 3, 4, 4, seed=s)},
    "mplug_owl2": lambda s: {"images": _tensor(1, 3, 4, 4, seed=s)},
    "yi_vl": lambda s: {"images": _tensor(1, 3, 4, 4, seed=s)},
    "ernie_vl": lambda s: {
        "images": _tensor(4, 3, seed=s),
        "grid_thw": torch.tensor([[1, 2, 2]]),
        "image_type_ids": torch.tensor([0, 0, 0, 0]),
    },
    "doc_owl2": lambda s: {"images": _tensor(2, 3, 4, 4, seed=s), "patch_positions": torch.tensor([[0, 0], [0, 1]])},
    "phi4_multimodal": lambda s: {
        "input_image_embeds": _tensor(1, 2, 3, 4, 4, seed=s),
        "image_sizes": torch.tensor([[4, 4]]),
        "image_attention_mask": torch.ones(1, 2, 2, 2),
        "input_mode": torch.tensor([1]),
    },
    "ovis1_6": lambda s: {"pixel_values": _items(2, s)},
    "minimax_vl": lambda s: {"pixel_values": _items(2, s), "image_sizes": [torch.tensor([4, 4]), torch.tensor([8, 4])]},
    "pixtral": lambda s: {"pixel_values": _items(1, s), "image_sizes": [torch.tensor([4, 4])]},
    "got_ocr2": lambda s: {"images": _items(1, s)},
    "llava1_6_mistral_hf": lambda s: {
        "pixel_values": _tensor(1, 3 + s % 2, 3, 4, 4, seed=s),
        "image_sizes": torch.tensor([[4, 4]]),
    },
    "valley": lambda s: {"images": _items(2, s), "image_sizes": torch.tensor([[4, 4], [8, 8]])},
    "minicpmv4_6": lambda s: {"pixel_values": _tensor(3, 4 + s, seed=s), "target_sizes": torch.tensor([[2, 2]])},
}


def _kernel(template_type, padding_free=False, *, legacy=False):
    if legacy:
        from swift.template import TEMPLATE_MAPPING
    else:
        from swift.dev.template.register import TEMPLATE_MAPPING, _ensure_registered

        _ensure_registered()
    cls = TEMPLATE_MAPPING[template_type].template_cls
    kernel = object.__new__(cls)
    kernel.__dict__.update(
        processor=SimpleNamespace(pad_token_id=0),
        padding_side="right",
        padding_free=padding_free,
        use_megatron=False,
        sequence_parallel_size=1,
        downsample_mode="16x",
    )
    if legacy:
        kernel.mode = "train"
    return kernel


def _rows(build, with_text=False):
    text = {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "length": 3}
    rows = [{**text, **build(1)}, {**text, **build(2)}]
    return rows + [dict(text)] if with_text else rows


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.mark.parametrize("template_type", sorted(CASES))
def test_family_media_hook_matches_full_legacy_collator(template_type):
    dev_kernel = _kernel(template_type)
    legacy_kernel = _kernel(template_type, legacy=True)
    rows = _rows(CASES[template_type])
    full = legacy_kernel._data_collator(copy.deepcopy(rows))
    media_rows = [
        {key: value for key, value in row.items() if key not in {"input_ids", "labels", "length"}} for row in rows
    ]
    direct = dev_kernel.collate_media(copy.deepcopy(media_rows))

    assert direct
    for key, value in direct.items():
        assert key in full and _same(value, full[key]), key


def test_unhandled_family_payload_fails_instead_of_being_dropped():
    from swift.dev.template.base import Template as DevFamilyBase

    with pytest.raises(NotImplementedError, match="_data_collator_mm_data"):
        object.__new__(DevFamilyBase).collate_media([{"unknown_payload": [1]}])
