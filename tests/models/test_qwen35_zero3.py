# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
import torch
import types

from swift.model.models import qwen


def _install_fake_deepspeed(monkeypatch, events):

    class GatheredParameters:

        def __init__(self, params):
            self.params = list(params)
            events.append(('init', self.params))

        def __enter__(self):
            events.append('enter')
            return self

        def __exit__(self, exc_type, exc, tb):
            events.append('exit')

    deepspeed = types.SimpleNamespace(zero=types.SimpleNamespace(GatheredParameters=GatheredParameters))
    monkeypatch.setitem(sys.modules, 'deepspeed', deepspeed)


def test_qwen35_conv1d_gather_context_collects_zero3_params(monkeypatch):
    events = []
    _install_fake_deepspeed(monkeypatch, events)
    monkeypatch.setattr(qwen, 'is_deepspeed_enabled', lambda: True)

    mod = types.SimpleNamespace(conv1d=torch.nn.Conv1d(4, 4, kernel_size=3, groups=4, bias=False))
    mod.conv1d.weight.ds_tensor = object()

    with qwen._gather_qwen3_5_conv1d_params_if_zero3(mod):
        events.append('body')

    assert events == [('init', [mod.conv1d.weight]), 'enter', 'body', 'exit']


def test_qwen35_conv1d_gather_context_ignores_non_zero3_params(monkeypatch):
    events = []
    _install_fake_deepspeed(monkeypatch, events)
    monkeypatch.setattr(qwen, 'is_deepspeed_enabled', lambda: True)

    mod = types.SimpleNamespace(conv1d=torch.nn.Conv1d(4, 4, kernel_size=3, groups=4, bias=False))

    with qwen._gather_qwen3_5_conv1d_params_if_zero3(mod):
        events.append('body')

    assert events == ['body']


def test_qwen35_patched_forward_gathers_conv1d_params_for_origin_forward(monkeypatch):
    events = []
    _install_fake_deepspeed(monkeypatch, events)
    monkeypatch.setattr(qwen, 'is_deepspeed_enabled', lambda: True)

    qwen35_package = types.ModuleType('transformers.models.qwen3_5')
    qwen35_package.__path__ = []
    qwen35_modeling = types.ModuleType('transformers.models.qwen3_5.modeling_qwen3_5')

    class Qwen3_5GatedDeltaNet(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1d = torch.nn.Conv1d(4, 4, kernel_size=3, groups=4, bias=False)
            self.conv1d.weight.ds_tensor = object()

        def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
            events.append('origin')
            return hidden_states + 1

    qwen35_modeling.Qwen3_5GatedDeltaNet = Qwen3_5GatedDeltaNet
    qwen35_package.modeling_qwen3_5 = qwen35_modeling
    monkeypatch.setitem(sys.modules, 'transformers.models.qwen3_5', qwen35_package)
    monkeypatch.setitem(sys.modules, 'transformers.models.qwen3_5.modeling_qwen3_5', qwen35_modeling)

    qwen._patch_qwen3_5_linear_attention_sequence_parallel()

    hidden_states = torch.zeros(1, 2, 4)
    mod = Qwen3_5GatedDeltaNet()
    output = mod(hidden_states)

    assert torch.equal(output, hidden_states + 1)
    assert events[0] == ('init', [mod.conv1d.weight])
    assert events[1:] == ['enter', 'origin', 'exit']
