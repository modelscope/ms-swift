# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise the actual truncation method of every registered template.

These are encoded-field contract tests, not tokenizer, processor or model tests.
Model-specific encoding and collation are covered separately.
"""
import pytest
import torch
from types import SimpleNamespace

from swift.template import TEMPLATE_MAPPING
from swift.template.templates.moss import MossVLTemplate


@pytest.mark.parametrize('template_type', sorted(TEMPLATE_MAPPING))
@pytest.mark.parametrize('strategy', ['left', 'right'])
def test_every_registered_template_truncates_aligned_fields(template_type, strategy):
    meta = TEMPLATE_MAPPING[template_type]
    template = meta.template_cls.__new__(meta.template_cls)
    # SailVL reads these processor attributes in its constructor even when the
    # processor initialization is deferred for an encoded-field test.
    template.processor = SimpleNamespace(num_image_token=2, convert_tokens_to_ids=lambda token: 90)
    meta.template_cls.__init__(template, None, meta, max_length=5, truncation_strategy=strategy)

    if isinstance(template, MossVLTemplate):
        # MOSS-VL uses contiguous truncation and a cross-attention mask instead
        # of token-type fields. Keep its complete vision span in both directions.
        tokens = {'<|vision_start|>': 98, '<|image_pad|>': 99, '<|vision_end|>': 97}
        template.processor = SimpleNamespace(unk_token_id=-1, convert_tokens_to_ids=tokens.get)
        template.max_length = 6
        input_ids = [10, 11, 98, 99, 97, 12, 13, 14]
        kept = list(range(2, 8)) if strategy == 'left' else list(range(6))
        mask = torch.arange(16).reshape(1, 1, 8, 2)
        encoded = {'cross_attention_mask': mask.clone(), 'loss_scale': None}
        actual_ids, actual_labels = template._truncate(input_ids, input_ids.copy(), encoded, strategy)
        assert actual_ids == [input_ids[i] for i in kept]
        assert actual_labels == [-100] + actual_ids[1:]
        torch.testing.assert_close(encoded['cross_attention_mask'], mask[:, :, kept, :])
        return

    template.placeholder_tokens = [99, 90]
    input_ids = [10, 99, 11, 90, 90, 12, 98, 13]
    kept = [1, 3, 4, 6, 7] if strategy == 'left' else [0, 1, 2, 3, 4]
    types = [0, 0, 0, 1, 1, 0, 0, 0]
    for token_types in (None, types, torch.tensor(types, dtype=torch.int32), torch.tensor([types])):
        encoded = {
            'loss_scale': [0., 0., 0.25, 0., 0., 0.5, 0.75, 1.],
            'token_type_ids': token_types,
            'mm_token_type_ids': torch.arange(8),
            'image_token_types': torch.tensor([-1, -1, -1, 0, 0, -1, -1, -1]),
        }
        expected_loss = [encoded['loss_scale'][i] for i in kept]
        expected_loss[0] = 0
        expected_image_types = encoded['image_token_types'][kept]
        actual_ids, actual_labels = template._truncate(input_ids, input_ids.copy(), encoded, strategy)
        assert actual_ids == [input_ids[i] for i in kept]
        assert actual_labels == [-100] + actual_ids[1:]
        assert encoded['loss_scale'] == expected_loss
        torch.testing.assert_close(encoded['mm_token_type_ids'], torch.tensor(kept))
        torch.testing.assert_close(encoded['image_token_types'], expected_image_types)
        if token_types is None:
            assert encoded['token_type_ids'] is None
        elif isinstance(token_types, torch.Tensor):
            expected = torch.tensor([types[i] for i in kept], dtype=token_types.dtype)
            if token_types.ndim == 2:
                expected = expected[None]
            torch.testing.assert_close(encoded['token_type_ids'], expected)
        else:
            assert encoded['token_type_ids'] == [types[i] for i in kept]
