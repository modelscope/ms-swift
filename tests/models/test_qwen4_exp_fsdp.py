# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
from torch import nn

from swift.model.models.qwen import _HostEmbeddingProxy, _pin_qwen4_exp_ple_ngram_on_host


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
