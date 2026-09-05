"""Multimodal (VL) encode + collate regression tests.

Scope: multimodal is a sample->batch data-format concern. The dev QwenVLTemplate
(train mode) must produce input_ids/pixel_values/image_grid_thw bit-identical to legacy swift
(the ground truth the model was trained on), and the dev InputProcessor must collate a ragged
multi-image batch by CONCATENATING VLM_CONCAT_FIELDS (pixel_values/grid_thw along dim 0), not
padding. CP/SP splitting is explicitly the Model's job (out of scope here).

Marked slow: needs a real Qwen2.5-VL model + image processing. Run with -m slow.
"""
import numpy as np
import pytest
import torch

VL_MODEL = 'Qwen/Qwen2.5-VL-3B-Instruct'


def _rand_image(sz=56):
    from PIL import Image
    return Image.fromarray((np.random.rand(sz, sz, 3) * 255).astype('uint8'))


def _dev_and_legacy_templates():
    """(dev-derived, plain legacy) for the SAME VL family, as two independent instances.

    dev's template is now the legacy class + DevMixin, so the only expected difference is the
    next-token label shift; input_ids and the media tensors must stay identical. Two instances are
    built because the derivation rewrites __class__ in place.
    """
    from swift.dev.template import shifted_template_class
    from swift.model import get_model_processor
    from swift.template import get_template
    _, proc = get_model_processor(VL_MODEL, load_model=False)
    dev = get_template(proc, template_type='qwen2_5_vl', max_length=1024)
    dev.__class__ = shifted_template_class(type(dev))
    legacy = get_template(proc, template_type='qwen2_5_vl', max_length=1024)
    return dev, legacy


@pytest.mark.slow
def test_vl_train_encode_matches_legacy_bit_exact():
    """dev QwenVLTemplate train-mode encode == legacy: input_ids (token-strict), pixel_values &
    image_grid_thw (maxdiff==0). Labels differ by design: dev next-token shifts, legacy aligned.
    """
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.template import DevMixin
    dev, legacy = _dev_and_legacy_templates()
    legacy.set_mode('train')
    dev.set_mode('train')
    sample = {
        'messages': [{
            'role': 'user',
            'content': '<image>What is this?'
        }, {
            'role': 'assistant',
            'content': 'A cat.'
        }],
        'images': [_rand_image()]
    }
    le = legacy.encode(dict(sample))
    de = dev.encode(dict(sample))

    # input_ids: token-strict equality (ground truth = legacy = what the model trained on)
    assert list(le['input_ids']) == list(de['input_ids']), 'VL input_ids diverged from legacy'

    # mm tensors: bit-exact (same processor path). Compare the intersection of VLM keys.
    for k in ('pixel_values', 'image_grid_thw'):
        assert k in le and k in de, f'{k} missing (legacy={k in le} dev={k in de})'
        a = torch.as_tensor(le[k]).float()
        b = torch.as_tensor(de[k]).float()
        assert a.shape == b.shape, f'{k} shape {tuple(a.shape)} != {tuple(b.shape)}'
        assert (a - b).abs().max().item() == 0.0, f'{k} not bit-exact vs legacy'

    # labels: dev is next-token shifted; legacy is aligned. Verify the shift marker.
    assert de.get(DevMixin.SHIFTED_KEY) is True
    # dev labels == legacy labels shifted-by-one (aligned -> next-token)
    legacy_shifted = list(le['labels'][1:]) + [-100]
    assert list(de['labels']) == legacy_shifted, 'dev VL labels are not legacy labels next-token shifted'


@pytest.mark.slow
def test_vl_batch_collate_concats_ragged_images():
    """dev InputProcessor collates a ragged multi-image batch by CONCATENATING VLM_CONCAT_FIELDS
    (pixel_values / image_grid_thw along dim 0), while text keys (input_ids/labels) are padded."""
    if not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    from swift.dev.processor import InputProcessor
    dev, _ = _dev_and_legacy_templates()
    dev.set_mode('train')

    def enc(n_img):
        imgs = [_rand_image() for _ in range(n_img)]
        content = '<image>' * n_img + 'describe'
        return dev.encode({
            'messages': [{
                'role': 'user',
                'content': content
            }, {
                'role': 'assistant',
                'content': 'ok'
            }],
            'images': imgs
        })

    s1, s2 = enc(1), enc(2)  # ragged: 1 image vs 2 images
    ip = InputProcessor()
    batch = ip.prepare_inputs([dict(s1), dict(s2)])
    collated = ip.collate_fn(batch)[0]

    # image_grid_thw: one row per image -> 1 + 2 == 3 rows (CONCAT, not pad)
    gt = torch.as_tensor(collated['image_grid_thw'])
    assert gt.shape[0] == 3, f'image_grid_thw should concat to 3 rows, got {gt.shape[0]}'

    # pixel_values: dim0 == sum of per-sample patch rows (CONCAT)
    pv = torch.as_tensor(collated['pixel_values'])
    exp = torch.as_tensor(s1['pixel_values']).shape[0] + torch.as_tensor(s2['pixel_values']).shape[0]
    assert pv.shape[0] == exp, f'pixel_values should concat to {exp} rows, got {pv.shape[0]}'

    # text keys are batched/padded to [B, T]
    ids = torch.as_tensor(collated['input_ids'])
    assert ids.dim() == 2 and ids.shape[0] == 2, f'input_ids should be [2, T], got {tuple(ids.shape)}'


def test_mixed_image_text_batch_fills_missing_mm_token_type_ids():
    """A mixed image+text batch must collate without KeyError, matching legacy.

    Only multimodal samples carry sequence-level mm fields like mm_token_type_ids (dev Template
    emits it under `requires_mm_token_type_ids and any(mm_mask)`, exactly like legacy qwen.py). But
    twinkle's _collate_macro_batch unions all sample keys and then indexes every sample with
    `item[key]`, so a text row with no mm_token_type_ids raised KeyError. legacy pads such fields up
    to the full batch with the padding value (a 2-image/2-text batch -> shape (4, L), text rows
    all-zero); dev's prepare_inputs reproduces that fill. Pure-CPU: builds the ragged dicts directly,
    no model needed.
    """
    from swift.dev.processor import InputProcessor
    ip = InputProcessor()
    # image rows carry mm_token_type_ids (non-zero where the image token sits); text rows do not.
    img_a = {'input_ids': [10, 151655, 12], 'labels': [-100, -100, 12], 'mm_token_type_ids': torch.tensor([0, 1, 0])}
    txt = {'input_ids': [10, 11], 'labels': [-100, 11]}
    img_b = {
        'input_ids': [10, 151655, 151655, 12],
        'labels': [-100, -100, -100, 12],
        'mm_token_type_ids': torch.tensor([0, 1, 1, 0])
    }
    prepared = ip.prepare_inputs([dict(img_a), dict(txt), dict(img_b), dict(txt)])

    # every row now carries mm_token_type_ids, length == its own input_ids (legacy fill semantics)
    for row in prepared:
        assert row.get('mm_token_type_ids') is not None
        assert len(row['mm_token_type_ids']) == len(row['input_ids'])
    # the filled text rows are all-zero (padding value 0 == "not a multimodal token")
    assert int(prepared[1]['mm_token_type_ids'].sum()) == 0
    assert int(prepared[3]['mm_token_type_ids'].sum()) == 0
    # the real image rows are untouched
    assert int(prepared[0]['mm_token_type_ids'].sum()) == 1

    # and the collate stage no longer KeyErrors on the union index
    collated = ip.collate_fn(prepared)
    mm = collated[0]['mm_token_type_ids'] if isinstance(collated, list) else collated['mm_token_type_ids']
    assert torch.as_tensor(mm).shape[0] == 4, 'mm_token_type_ids should pad to the full batch of 4'


def test_all_text_batch_leaves_no_mm_token_type_ids():
    """A pure-text batch must NOT gain a phantom mm_token_type_ids: the fill only fires when the
    field is present on some-but-not-all rows, never when it is absent everywhere (legacy emits it
    for no text row either)."""
    from swift.dev.processor import InputProcessor
    ip = InputProcessor()
    txt = {'input_ids': [10, 11], 'labels': [-100, 11]}
    prepared = ip.prepare_inputs([dict(txt), dict(txt)])
    assert all(row.get('mm_token_type_ids') is None for row in prepared)
