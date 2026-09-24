# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from swift.template import TEMPLATE_MAPPING
from swift.template.base import Template
from swift.template.templates.seed import SailVLTemplate


def _template():
    template = SailVLTemplate.__new__(SailVLTemplate)
    template.processor = SimpleNamespace(num_image_token=2, convert_tokens_to_ids=lambda token: 90)
    template.__init__(None, TEMPLATE_MAPPING['sail_vl2'])
    return template


@pytest.mark.parametrize('has_image', [False, True])
@pytest.mark.parametrize('mode', ['train', 'train_with_loss_scale', 'transformers'])
def test_sailvl_text_and_image_encoding(has_image, mode):
    template = _template()
    base_ids = [10, -100, 11, 12] if has_image else [10, 11, 12]
    base = {
        'input_ids': base_ids,
        'labels': [-100] * (len(base_ids) - 1) + [12] if mode != 'transformers' else None,
        'loss_scale': [0.] * (len(base_ids) - 1) + [0.5] if mode == 'train_with_loss_scale' else None,
    }
    pixels = torch.ones(2, 3, 2, 2)
    template.processor.image_processor = lambda images: {'num_patches_list': [2], 'pixel_values': pixels}
    template.processor.encode = lambda *args, **kwargs: [90]
    with patch.object(Template, '_encode', return_value=deepcopy(base)):
        encoded = template._encode(SimpleNamespace(images=['image'] if has_image else []))

    expected = [10, 90, 90, 90, 90, 11, 12] if has_image else base_ids
    assert encoded['input_ids'] == expected
    assert encoded['labels'] == ([-100] * (len(expected) - 1) + [12] if mode != 'transformers' else None)
    assert encoded['loss_scale'] == ([0.] * (len(expected) - 1) + [0.5] if mode == 'train_with_loss_scale' else None)
    assert encoded['pixel_values'] is (pixels if has_image else None)


@pytest.mark.parametrize('has_image', [False, True])
@pytest.mark.parametrize('deepspeed_enabled', [False, True])
def test_sailvl_post_encode_preserves_text_and_connects_gradients(has_image, deepspeed_enabled):
    template = _template()
    embedding = torch.nn.Embedding(100, 4)
    vision = torch.nn.Linear(1, 4)
    model = SimpleNamespace(
        language_model=SimpleNamespace(get_input_embeddings=lambda: embedding),
        extract_feature=lambda pixels: vision(pixels.mean().reshape(1, 1)),
    )
    input_ids = torch.tensor([[10, 90, 12] if has_image else [10, 11, 12]])
    pixels = torch.ones(1, 3, 2, 2) if has_image else None
    expected = embedding(input_ids).detach().clone()
    if has_image:
        expected[:, 1] = model.extract_feature(pixels).detach()
    with patch('swift.template.templates.seed.is_deepspeed_enabled', return_value=deepspeed_enabled):
        encoded = template._post_encode(model, {'input_ids': input_ids, 'pixel_values': pixels})
    torch.testing.assert_close(encoded['inputs_embeds'], expected)
    encoded['inputs_embeds'].sum().backward()
    assert embedding.weight.grad is not None
    if has_image or deepspeed_enabled:
        assert vision.weight.grad is not None
    else:
        assert vision.weight.grad is None
