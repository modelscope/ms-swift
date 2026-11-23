import importlib.util
import pytest
import torch
from torch import nn
from types import SimpleNamespace
from unittest.mock import patch

from swift.arguments import SftArguments
from tests.fsdpturbo import test_arguments as argument_tests


@pytest.mark.parametrize('weight_decay', [0.0, 0.2])
@pytest.mark.parametrize('host_engram', [False, True])
@pytest.mark.skipif(importlib.util.find_spec('fsdp_turbo') is None, reason='requires the optional FSDPTurbo package')
def test_bias_and_norm_keep_their_values_under_weight_decay(weight_decay, host_engram):
    from swift.fsdpturbo.trainer import FSDPTurboTrainer

    model = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))
    model.register_parameter('frozen', nn.Parameter(torch.ones(2), requires_grad=False))
    if host_engram:
        from fsdp_turbo.models.deepseek.deepseek_v41.engram.host_offload_engram import TorchHostOffloadEngram

        table = TorchHostOffloadEngram(8, 4, layer_id=0, num_max_tokens_per_rank=4, pin_memory=False)
        table.to_empty(device='cpu')
        nn.init.ones_(table.weight)
        model.add_module('engram', table)
    trainer = object.__new__(FSDPTurboTrainer)
    trainer.model = model
    trainer.spec = SimpleNamespace(model_type='deepseek_v41' if host_engram else 'qwen3_5_moe')
    trainer.config = SimpleNamespace(
        optimizer=SimpleNamespace(
            optimizer_type='AdamW',
            lr=0.1,
            weight_decay=weight_decay,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8))
    optimizer = trainer.build_optimizer()
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and name != 'engram.weight':
            parameter.grad = torch.zeros_like(parameter)
    if host_engram:
        table.accumulate_sparse_gradient(torch.tensor([0]), torch.ones(1, 4))
    optimizer.step()
    torch.testing.assert_close(model[0].weight, original['0.weight'] * (1 - 0.1 * weight_decay))
    for name in ('0.bias', '1.weight', '1.bias', 'frozen'):
        torch.testing.assert_close(dict(model.named_parameters())[name], original[name], rtol=0, atol=0)
    assert all(parameter is not model.frozen for group in optimizer.param_groups for parameter in group['params'])
    if host_engram:
        assert all(parameter is not table.weight for group in optimizer.dense_optimizer.param_groups
                   for parameter in group['params'])
        assert optimizer.checkpoint_optimizers == (optimizer.dense_optimizer, optimizer.sparse_optimizer)
        assert torch.all(table.weight[0] < original['engram.weight'][0])
        torch.testing.assert_close(table.weight[1:], original['engram.weight'][1:], rtol=0, atol=0)
        state = optimizer.sparse_optimizer.state[table.weight]
        assert state['step'] == 1
        assert state['exp_avg'].dtype == state['exp_avg_sq'].dtype == torch.float32
        assert table.pending_sparse_grad() is None


@pytest.mark.parametrize('optim', ['adamw_torch_fused', 'adamw_bnb_8bit'])
def test_optimizer_variants_are_rejected_instead_of_silently_replaced(optim):
    args = argument_tests.TestFSDPTurboArguments._bare_sft_args(
        training_args=SimpleNamespace(save_strategy='no', optim=optim), vit_gradient_checkpointing=True)
    with patch.object(SftArguments, '__post_init__'), pytest.raises(ValueError, match='AdamW'):
        args.__post_init__()
