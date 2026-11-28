# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
from torch import nn

from swift.model.models.qwen import (_HostEmbeddingProxy, _pin_qwen4_exp_ple_ngram_on_host,
                                     _restore_qwen4_exp_ple_state_dict)


def test_host_embedding_stays_on_cpu():
    embedding = nn.Embedding(13, 4, padding_idx=0)
    embedding.requires_grad_(False)
    proxy = _HostEmbeddingProxy(embedding)
    ids = torch.tensor([[0, 2, 12, 2]])
    expected = embedding(ids)

    proxy.to(device='meta', dtype=torch.float64)
    torch.testing.assert_close(proxy(ids), expected)
    assert proxy.weight.device.type == 'cpu'
    assert proxy.weight.dtype == torch.float32
    assert not list(proxy.parameters())
    assert not proxy.state_dict()


@pytest.mark.parametrize('frozen', [False, True])
def test_pin_ple_preserves_trainability(frozen):
    model = nn.Module()
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    layer = nn.Module()
    layer.ple = nn.Module()
    layer.ple.ple_embedding = nn.Module()
    embedding = nn.Embedding(8, 3)
    embedding.requires_grad_(not frozen)
    layer.ple.ple_embedding.ngram_embedding = embedding
    model.model.language_model.layers = nn.ModuleList([layer])
    ids = torch.tensor([[1, 3, 1]])
    expected = embedding(ids).detach().clone()

    _pin_qwen4_exp_ple_ngram_on_host(model)

    actual = layer.ple.ple_embedding.ngram_embedding
    torch.testing.assert_close(actual(ids), expected)
    assert actual.weight.device.type == 'cpu'
    assert isinstance(actual, _HostEmbeddingProxy) == frozen
    assert actual.weight.requires_grad == (not frozen)
    assert any(p is actual.weight for p in model.parameters()) == (not frozen)
    assert any('ngram_embedding' in key for key in model.state_dict()) == (not frozen)
    if not frozen:
        actual(ids).sum().backward()
        assert actual.weight.grad is not None
        assert actual.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('checkpointing', [False, True])
def test_full_checkpoint_restores_host_table(tmp_path, checkpointing):
    from safetensors.torch import load_file
    from types import SimpleNamespace

    from swift.trainers.mixin import SwiftMixin

    model = nn.Sequential(nn.Embedding(8, 3), nn.Linear(3, 2))
    expected = model[0].weight.detach().clone()
    model[0].requires_grad_(False)
    model[0] = _HostEmbeddingProxy(model[0])
    if checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
        model = checkpoint_wrapper(model)
    gathered = model.state_dict()
    state = _restore_qwen4_exp_ple_state_dict(model, gathered)
    assert '0.weight' not in gathered
    assert state['0.weight'].data_ptr() == model[0].weight.data_ptr()
    assert state['0.weight'].device.type == 'cpu'
    model.model_meta = SimpleNamespace(model_type='qwen4_exp')
    trainer = SimpleNamespace(
        model=model, args=SimpleNamespace(tuner_type='full', safe_serialization=True, use_flash_ckpt=False))
    SwiftMixin._save_model(trainer, str(tmp_path), gathered)
    restored = load_file(tmp_path / 'model.safetensors')
    ordinary = nn.Sequential(nn.Embedding(8, 3), nn.Linear(3, 2))
    ordinary.load_state_dict(restored, strict=True)
    model[0].weight.zero_()
    model.load_state_dict(restored, strict=True)
    torch.testing.assert_close(ordinary[0].weight, expected)
    torch.testing.assert_close(model[0].weight, expected)


def test_lora_targets_exclude_indexer():
    import re
    from types import SimpleNamespace

    from swift.pipelines.train.tuner import get_target_modules

    model = nn.Module()
    model.config = SimpleNamespace(model_type='qwen4_exp')
    model.model_meta = SimpleNamespace(
        is_multimodal=True,
        model_arch=SimpleNamespace(language_model=['text'], vision_tower=[], aligner=[], lm_head='lm_head'))
    model.text = nn.Module()
    model.text.indexer = nn.ModuleDict({'key_proj': nn.Linear(3, 3)})
    model.text.ple = nn.ModuleDict({'key_proj': nn.Linear(3, 3), 'value_proj': nn.Linear(3, 3)})
    args = SimpleNamespace(
        model_type='qwen4_exp', target_modules=['all-linear'], freeze_llm=False, freeze_vit=True, freeze_aligner=True)
    for multimodal in (True, False):
        model.model_meta.is_multimodal = multimodal
        pattern = get_target_modules(args, model)
        assert not re.fullmatch(pattern, 'text.indexer.key_proj')
        assert re.fullmatch(pattern, 'text.ple.key_proj')
        assert re.fullmatch(pattern, 'text.ple.value_proj')
