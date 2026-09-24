# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from swift.template import TEMPLATE_MAPPING, TemplateType
from swift.template.base import Template
from swift.template.templates.baidu import ERNIE_VLTemplate
from swift.template.templates.gemma import Gemma3VisionTemplate


def _make_template(kind, **kwargs):
    template_type = TemplateType.ernie_vl if 'tensor' in kind else TemplateType.gemma3_vision
    template_cls = ERNIE_VLTemplate if 'tensor' in kind else Gemma3VisionTemplate
    template = template_cls(None, TEMPLATE_MAPPING[template_type], **kwargs)
    template.placeholder_tokens = [99]
    template.processor = SimpleNamespace(pad_token_id=0)
    template.mode = 'train'
    return template


def _make_encoded(input_ids, kind):
    token_types = [int(token == 90) for token in input_ids]
    if kind == 'tuple':
        token_types = tuple(token_types)
    elif 'tensor' in kind:
        token_types = torch.tensor(token_types, dtype=torch.int32)
        if kind == 'batched_tensor':
            token_types = token_types[None]
    return {
        'input_ids': input_ids.copy(),
        'labels': input_ids.copy(),
        'loss_scale': list(range(1,
                                 len(input_ids) + 1)),
        'token_type_ids': token_types,
        'mm_token_type_ids': torch.arange(len(input_ids)),
        'image_token_types': torch.tensor([0 if token == 90 else -1 for token in input_ids]),
    }


def _encode_truncated(template, encoded):
    with patch.object(template, '_preprocess_inputs'), patch.object(template, '_encode', return_value=encoded):
        return template._encode_truncated(None)


@pytest.mark.parametrize('kind', ['list', 'tuple', 'tensor', 'batched_tensor'])
@pytest.mark.parametrize('padding_side', ['left', 'right'])
@pytest.mark.parametrize(
    'strategy,max_length,kept',
    [
        ('left', 8, [2, 7, 8, 9, 10, 11, 12, 13]),
        ('right', 8, [0, 1, 2, 3, 4, 5, 6, 8]),
        ('left', 5, [2, 8, 11, 12, 13]),
        ('right', 5, [0, 1, 2, 3, 8]),
        # All protected positions survive even when they exhaust the token budget.
        ('left', 2, [2, 8]),
        ('right', 1, [2, 8]),
    ])
def test_truncated_token_types_and_mixed_length_padding(kind, padding_side, strategy, max_length, kept):
    template = _make_template(kind, max_length=max_length, truncation_strategy=strategy, padding_side=padding_side)
    encoded = _make_encoded([10, 11, 99, 90, 90, 98, 12, 13, 99, 90, 90, 98, 14, 15], kind)
    original = deepcopy(encoded)
    result = _encode_truncated(template, encoded)
    expected = {key: torch.as_tensor(value).reshape(-1)[kept].tolist() for key, value in original.items()}
    expected['labels'][0] = -100
    expected['loss_scale'][0] = 0
    assert result['length'] == len(kept)
    for key, value in expected.items():
        assert torch.as_tensor(result[key]).reshape(-1).tolist() == value
    if 'tensor' in kind:
        token_types = result['token_type_ids']
        assert isinstance(token_types, torch.Tensor)
        assert token_types.dtype == original['token_type_ids'].dtype
        assert token_types.device == original['token_type_ids'].device
        assert token_types.shape == ((1, len(kept)) if kind == 'batched_tensor' else (len(kept), ))

    short = _make_encoded([42], kind)
    short['labels'] = [-100]
    short['loss_scale'] = [0]
    short = _encode_truncated(template, short)
    pad_values = dict(
        input_ids=0, labels=-100, loss_scale=0, token_type_ids=0, mm_token_type_ids=0, image_token_types=-1)
    for padding_to in (None, len(kept) + 2):
        batch = template._data_collator(deepcopy([short, result]), padding_to=padding_to)
        target_length = padding_to or len(kept)
        for key, value in expected.items():
            rows = [torch.as_tensor(short[key]).reshape(-1).tolist(), value]
            for i, row in enumerate(rows):
                padding = [pad_values[key]] * (target_length - len(row))
                rows[i] = padding + row if padding_side == 'left' else row + padding
            assert batch[key].shape == (2, target_length)
            assert batch[key].tolist() == rows


@pytest.mark.parametrize('kind', ['list', 'tuple', 'tensor', 'batched_tensor'])
@pytest.mark.parametrize('strategy', ['left', 'right'])
def test_text_only_inference_truncation(kind, strategy):
    template = _make_template(kind, max_length=1, truncation_strategy=strategy)
    template.mode = 'transformers'
    encoded = _make_encoded([10, 11, 12], kind)
    encoded['labels'] = encoded['loss_scale'] = None
    result = _encode_truncated(template, encoded)
    assert result['input_ids'] == ([12] if strategy == 'left' else [10])
    assert result['labels'] is None
    batch = template._data_collator([result])
    assert batch['token_type_ids'].shape == batch['input_ids'].shape == (1, 1)
    assert batch['token_type_ids'].tolist() == [[0]]


@pytest.mark.parametrize('kind', ['list', 'tuple', 'tensor', 'batched_tensor'])
@pytest.mark.parametrize('max_length', [None, 3, 4])
def test_token_types_are_unchanged_without_truncation(kind, max_length):
    template = _make_template(kind, max_length=max_length, truncation_strategy='left')
    encoded = _make_encoded([10, 90, 11], kind)
    token_types = encoded['token_type_ids']
    result = _encode_truncated(template, encoded)
    assert result['input_ids'] == [10, 90, 11]
    assert result['token_type_ids'] is token_types
    assert result['length'] == 3


@pytest.mark.parametrize('missing', [True, False])
@pytest.mark.parametrize('strategy', ['left', 'right'])
def test_truncation_without_token_types(missing, strategy):
    template = _make_template('list', max_length=2, truncation_strategy=strategy)
    encoded = _make_encoded([10, 11, 12], 'list')
    if missing:
        encoded.pop('token_type_ids')
    else:
        encoded['token_type_ids'] = None
    result = _encode_truncated(template, encoded)
    assert result['input_ids'] == ([11, 12] if strategy == 'left' else [10, 11])
    assert ('token_type_ids' not in result) if missing else (result['token_type_ids'] is None)
    assert 'token_type_ids' not in template._data_collator([result])


@pytest.mark.parametrize('strategy,expected_ids,expected_types', [
    ('left', [99, 99, 12, 13], [1, 1, 0, 0]),
    ('right', [10, 11, 99, 99], [0, 0, 1, 1]),
])
def test_ernie_encoder_produces_truncatable_batched_token_types(strategy, expected_ids, expected_types):
    template = _make_template('batched_tensor', max_length=4, truncation_strategy=strategy)

    class Processor:
        pad_token_id = 0

        def __call__(self, **kwargs):
            return {
                'input_ids': torch.tensor([[99, 99]]),
                'token_type_ids': torch.tensor([[1, 1]]),
                'position_ids': torch.tensor([[[0, 0, 0], [0, 0, 1]]]),
            }

    template.processor = Processor()
    base_encoded = {'input_ids': [10, 11, 99, 12, 13], 'labels': [10, 11, -100, 12, 13], 'loss_scale': None}
    inputs = SimpleNamespace(images=['image'], videos=[])
    with patch.object(template, '_preprocess_inputs'), patch.object(
            Template, '_encode', return_value=base_encoded), patch.object(
                template, '_tokenize', side_effect=lambda text: [99 if text == '<|IMAGE_PLACEHOLDER|>' else 100]):
        result = template._encode_truncated(inputs)
    assert result['input_ids'] == expected_ids
    assert isinstance(result['token_type_ids'], torch.Tensor)
    assert result['token_type_ids'].shape == (1, 4)
    assert result['token_type_ids'].tolist() == [expected_types]
