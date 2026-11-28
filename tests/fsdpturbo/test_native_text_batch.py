# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import pytest
import torch
from types import SimpleNamespace

from swift.fsdpturbo.data import _native_text_collator, _normalize_native_text_batch


def text_batch():
    return {
        'input_ids': torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]]),
        'labels': torch.tensor([[-100, 2, 3, 4], [-100, 6, -100, -100]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
    }


def test_native_text_types_use_the_generic_backend_input_without_changing_tensors():
    batch = text_batch()
    original = dict(batch)
    types = torch.full_like(batch['input_ids'], -1)
    batch['image_token_types'] = types

    result = _normalize_native_text_batch(batch)

    assert result['token_types'] is types
    assert 'image_token_types' not in result
    for name, tensor in original.items():
        assert result[name] is tensor


def test_text_batch_without_private_fields_keeps_its_field_set():
    batch = text_batch()
    assert _normalize_native_text_batch(batch) is batch
    assert 'token_types' not in batch


def test_right_padded_contiguous_positions_are_checked_then_removed():
    batch = text_batch()
    batch['position_ids'] = torch.tensor([[0, 1, 2, 3], [0, 1, 99, 99]])

    assert 'position_ids' not in _normalize_native_text_batch(batch)


@pytest.mark.parametrize('field', ['image_token_types', 'token_types'])
@pytest.mark.parametrize('value', [0, -2])
def test_non_text_token_types_are_rejected(field, value):
    batch = text_batch()
    batch[field] = torch.full_like(batch['input_ids'], -1)
    batch[field][0, 1] = value

    with pytest.raises(ValueError, match='TEXT=-1'):
        _normalize_native_text_batch(batch)


@pytest.mark.parametrize('field', ['image_token_types', 'token_types', 'position_ids', 'attention_mask'])
def test_sequence_metadata_must_have_the_input_shape(field):
    batch = text_batch()
    batch[field] = torch.ones(1, 4, dtype=torch.int64)

    with pytest.raises(ValueError, match='shape'):
        _normalize_native_text_batch(batch)


@pytest.mark.parametrize('field', ['image_token_types', 'token_types', 'position_ids', 'attention_mask'])
def test_sequence_metadata_is_validated_on_cpu(field):
    batch = text_batch()
    batch[field] = torch.empty(2, 4, device='meta', dtype=torch.int64)

    with pytest.raises(ValueError, match='CPU'):
        _normalize_native_text_batch(batch)


@pytest.mark.parametrize('mask', [[[0, 1, 1, 1], [1, 1, 0, 0]], [[1, 0, 1, 0], [1, 1, 0, 0]]])
def test_left_padding_and_mask_gaps_are_not_silently_reinterpreted(mask):
    batch = text_batch()
    batch['attention_mask'] = torch.tensor(mask)

    with pytest.raises(ValueError, match='right-padded'):
        _normalize_native_text_batch(batch)


def test_nonbinary_attention_mask_is_rejected():
    batch = text_batch()
    batch['attention_mask'][0, 1] = 2

    with pytest.raises(ValueError, match='binary'):
        _normalize_native_text_batch(batch)


def test_noncontiguous_active_positions_are_rejected():
    batch = text_batch()
    batch['position_ids'] = torch.tensor([[0, 1, 3, 4], [0, 1, 0, 0]])

    with pytest.raises(ValueError, match='contiguous'):
        _normalize_native_text_batch(batch)


@pytest.mark.parametrize('field', ['images', 'pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw'])
def test_actual_image_inputs_are_explicitly_unsupported(field):
    batch = text_batch()
    batch[field] = torch.ones(1)

    with pytest.raises(ValueError, match='text-only'):
        _normalize_native_text_batch(batch)


def test_native_collator_normalizes_before_device_transfer():

    def collate(rows, *, padding_to):
        assert rows == ['sample']
        assert padding_to is None
        batch = text_batch()
        batch['image_token_types'] = torch.full_like(batch['input_ids'], -1)
        return batch

    result = _native_text_collator(collate, ['sample'])

    assert result['token_types'].device.type == 'cpu'
    assert 'image_token_types' not in result


@pytest.mark.skipif(importlib.util.find_spec('fsdp_turbo') is None, reason='requires the optional FSDPTurbo package')
@pytest.mark.parametrize('model_type', ['deepseek_v41', 'qwen3_5_moe'])
def test_dataloader_selects_normalization_only_for_native_models(model_type):
    from swift.fsdpturbo.model_specs import get_model_spec
    from swift.fsdpturbo.trainer import FSDPTurboTrainer

    def collate(rows, *, padding_to):
        assert rows == ['first', 'second']
        assert padding_to is None
        batch = text_batch()
        batch['image_token_types'] = torch.full_like(batch['input_ids'], -1)
        return batch

    trainer = object.__new__(FSDPTurboTrainer)
    trainer.spec = get_model_spec(model_type)
    trainer.args = SimpleNamespace(data_seed=42)
    trainer.train_dataset = ['first', 'second']
    trainer.template = SimpleNamespace(data_collator=collate, tokenizer=SimpleNamespace(pad_token_id=0))
    trainer.train_args = SimpleNamespace(
        train_dataloader_shuffle=False,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        per_device_train_batch_size=2,
        dataloader_pin_memory=False,
    )
    trainer.model = SimpleNamespace(
        model=SimpleNamespace(model_args=SimpleNamespace(compress_ratios=(0, ), n_layers=1)),
        parallel_state=SimpleNamespace(
            get_data_group_size=lambda: 1, get_data_rank=lambda: 0, get_cp_group_size=lambda: 1))

    result = next(iter(trainer.build_dataloader()))

    assert ('token_types' in result) == (model_type == 'deepseek_v41')
    assert ('image_token_types' in result) == (model_type == 'qwen3_5_moe')
