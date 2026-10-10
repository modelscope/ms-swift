# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU regressions for the opt-in decision template and causal supervision."""
import importlib.util
import pytest
import torch
import torch.nn.functional as F
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from swift.template import TEMPLATE_MAPPING
from swift.trainers.mixin import SwiftMixin


@pytest.fixture
def decision_template():
    path = Path(__file__).resolve().parents[2] / 'examples/train/intern_decision_npu/decision_plugin.py'
    spec = importlib.util.spec_from_file_location('decision_plugin_test', path)
    module = importlib.util.module_from_spec(spec)
    # Registration must not escape the test or replace the built-in Qwen template.
    with patch.dict(TEMPLATE_MAPPING):
        spec.loader.exec_module(module)
        yield module.DecisionTemplate


class Tokenizer:
    """Synthetic token boundaries; real tokenizer parity is a separate audit."""

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2, 9, 3, 9, 4]

    def encode(self, text, **kwargs):
        return {'<decision>': [9], 'A': [5], 'B': [6], 'invalid': [7, 8]}[text]


def test_answer_labels_do_not_change_prompt(decision_template):
    stub = SimpleNamespace(tokenizer=Tokenizer())
    inputs = SimpleNamespace(
        images=[],
        videos=[],
        system=None,
        messages=[{
            'role': 'assistant',
            'content': 'skeleton'
        }],
        extra_kwargs={'decision_targets': ['A', 'B']})
    original = deepcopy(inputs)
    encoded = decision_template._encode(stub, inputs)
    assert encoded['labels'] == [-100, -100, 5, -100, 6, -100]
    inputs.extra_kwargs['decision_targets'] = ['B', 'A']
    changed = decision_template._encode(stub, inputs)
    assert changed['input_ids'] == encoded['input_ids']
    assert inputs.messages == original.messages
    assert changed['labels'] != encoded['labels']


@pytest.mark.parametrize('targets', [[], ['A'], ['A', 'invalid']])
def test_invalid_supervision_is_rejected(decision_template, targets):
    inputs = SimpleNamespace(images=[], videos=[], system=None, messages=[], extra_kwargs={'decision_targets': targets})
    with pytest.raises(ValueError):
        decision_template._encode(SimpleNamespace(tokenizer=Tokenizer()), inputs)


@pytest.mark.parametrize('distributed_single_batch', [False, True])
@pytest.mark.parametrize('positions', [[4], [3, 7]])
def test_selected_causal_loss_and_gradient(distributed_single_batch, positions):
    generator = torch.Generator().manual_seed(42)
    stub = SimpleNamespace(
        template=SimpleNamespace(sequence_parallel_size=1), args=SimpleNamespace(tuner_backend='peft'))
    logits = torch.randn(1, 11, 13, generator=generator, dtype=torch.float64, requires_grad=True)
    labels = torch.full((1, 11), -100)
    for index, position in enumerate(positions):
        labels[0, position] = index + 2
    full = F.cross_entropy(logits[:, :-1].reshape(-1, 13), labels[:, 1:].reshape(-1))
    expected_grad, = torch.autograd.grad(full, logits, retain_graph=True)
    inputs = {'labels': labels.clone()}
    with patch.dict(SwiftMixin.prepare_logits_to_keep.__globals__, {'is_mp': lambda: not distributed_single_batch}):
        SwiftMixin.prepare_logits_to_keep(stub, inputs)
    keep = inputs['logits_to_keep']
    selected = logits[:, -keep:] if isinstance(keep, int) else logits[:, keep]
    actual = F.cross_entropy(selected[:, :-1].reshape(-1, 13), inputs['labels'][:, 1:].reshape(-1))
    actual_grad, = torch.autograd.grad(actual, logits)
    torch.testing.assert_close(full, actual, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(expected_grad, actual_grad, rtol=1e-12, atol=1e-12)
    assert set(actual_grad.abs().sum(-1)[0].nonzero().flatten().tolist()) == {p - 1 for p in positions}
