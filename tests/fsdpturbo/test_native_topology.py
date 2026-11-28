# Copyright (c) ModelScope Contributors. All rights reserved.
"""Native topology must reuse upstream TP/CP without corrupting text batches."""
import importlib.util
import pytest
import torch
from types import SimpleNamespace
from unittest import mock

from swift.fsdpturbo.data import _native_text_collator
from swift.fsdpturbo.model_specs import get_model_spec
from .test_cp_config import make_args

requires_fsdpturbo = pytest.mark.skipif(
    importlib.util.find_spec('fsdp_turbo') is None, reason='requires the optional FSDPTurbo package')


@requires_fsdpturbo
def test_v41_tp_shards_shared_experts_and_keeps_attention_replicated():
    from swift.fsdpturbo.trainer import build_fsdpturbo_config

    args = make_args(1)
    args.tp_size = 2
    args.gradient_checkpointing = False
    config = build_fsdpturbo_config(args, get_model_spec('deepseek_v41'))

    assert config.distributed.tp_plan.colwise_parallel == [
        'model.layers.{*}.ffn.shared_experts.w1', 'model.layers.{*}.ffn.shared_experts.w3'
    ]
    assert config.distributed.tp_plan.rowwise_parallel == ['model.layers.{*}.ffn.shared_experts.w2']


@requires_fsdpturbo
def test_v41_cp_uses_native_kv_allgather_and_model_owned_loss():
    from swift.fsdpturbo.trainer import build_fsdpturbo_config

    config = build_fsdpturbo_config(make_args(2), get_model_spec('deepseek_v41'))

    assert config.distributed.kv_allgather_parallel_size == 2
    assert config.distributed.ulysses_parallel_size == 1
    assert config.distributed.kcp_parallel_size == 1
    assert not config.distributed.cp_plan.kv_allgather_function_patches
    assert not config.distributed.cp_plan.loss_function_patches
    assert not config.module_patches
    assert config.memory.recompute
    assert config.memory.recompute_plan == ['model.layers.{*}']


@pytest.mark.parametrize('cp_size', [1, 2])
@requires_fsdpturbo
def test_native_cp_recompute_reaches_existing_backend_configuration(cp_size):
    from swift.fsdpturbo.trainer import FSDPTurboTrainer

    args = make_args(cp_size)
    args.model_type = 'deepseek_v41'
    args.router_aux_loss_coef = 0
    with mock.patch('fsdp_turbo.training.trainer.BaseTrainer.__init__', autospec=True) as initialize:
        FSDPTurboTrainer(args, None, None, None)
    assert initialize.call_args.kwargs['config'].distributed.kv_allgather_parallel_size == cp_size


@requires_fsdpturbo
def test_activation_offload_reuses_backend_plan():
    from swift.fsdpturbo.trainer import build_fsdpturbo_config

    args = make_args(1)
    args.offload_activations = True
    config = build_fsdpturbo_config(args, get_model_spec('deepseek_v41'))
    assert config.memory.swap_activation
    assert config.memory.swap_activation_plan.apply_modules == ['model.layers.{*}']


def test_cp_padding_uses_compression_alignment_and_keeps_labels_complete():
    original = {
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'labels': torch.tensor([[-100, 2, 3, 4, 5]]),
        'image_token_types': torch.full((1, 5), -1),
    }
    result = _native_text_collator(lambda rows, padding_to: dict(original), [], sequence_alignment=4, pad_token_id=2)

    torch.testing.assert_close(result['input_ids'], torch.tensor([[1, 2, 3, 4, 5, 2, 2, 2]]))
    torch.testing.assert_close(result['attention_mask'], torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]))
    torch.testing.assert_close(result['labels'], torch.tensor([[-100, 2, 3, 4, 5, -100, -100, -100]]))
    assert 'token_types' not in result
    assert 'image_token_types' not in result


@pytest.mark.parametrize('rank', [0, 1])
@requires_fsdpturbo
def test_native_text_is_sharded_even_with_hf_vision_metadata(rank):
    from swift.fsdpturbo.trainer import FSDPTurboTrainer

    trainer = object.__new__(FSDPTurboTrainer)
    trainer.spec = get_model_spec('deepseek_v41')
    trainer.model = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace())),
        parallel_state=SimpleNamespace(get_cp_group_size=lambda: 2, get_cp_group=lambda: object()))
    labels = torch.arange(8).reshape(1, 8)
    batch = {
        'input_ids': torch.arange(8).reshape(1, 8),
        'labels': labels,
        'attention_mask': torch.ones(1, 8, dtype=torch.int64)
    }

    with mock.patch('torch.distributed.get_world_size', return_value=2), \
            mock.patch('torch.distributed.get_rank', return_value=rank):
        local = trainer._split_batch_for_context_parallel(batch)

    torch.testing.assert_close(local['input_ids'], batch['input_ids'][:, rank * 4:(rank + 1) * 4])
    torch.testing.assert_close(local['position_ids'], batch['input_ids'][:, rank * 4:(rank + 1) * 4])
    assert local['labels'] is labels


@requires_fsdpturbo
def test_native_tp_and_kv_cp_preserve_combination_constraint():
    from swift.fsdpturbo.trainer import build_fsdpturbo_config

    args = make_args(2)
    args.tp_size = 2
    with pytest.raises(ValueError, match='KV-AllGather requires Ulysses=KCP=TP=1'):
        build_fsdpturbo_config(args, get_model_spec('deepseek_v41'))
