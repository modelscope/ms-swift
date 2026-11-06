"""Numerical check of SWIFT's selected-logit causal supervision."""
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from unittest.mock import patch

from swift.trainers.mixin import SwiftMixin

torch.manual_seed(42)
stub = SimpleNamespace(template=SimpleNamespace(sequence_parallel_size=1), args=SimpleNamespace(tuner_backend='peft'))
for distributed_single_batch in (False, True):
    with patch.dict(SwiftMixin.prepare_logits_to_keep.__globals__, {'is_mp': lambda: not distributed_single_batch}):
        for positions in ([4], [3, 7]):
            logits = torch.randn(1, 11, 13, requires_grad=True)
            labels = torch.full((1, 11), -100)
            for index, position in enumerate(positions):
                labels[0, position] = index + 2
            full = F.cross_entropy(logits[:, :-1].reshape(-1, 13), labels[:, 1:].reshape(-1))
            expected_grad, = torch.autograd.grad(full, logits, retain_graph=True)
            inputs = {'labels': labels.clone()}
            SwiftMixin.prepare_logits_to_keep(stub, inputs)
            keep = inputs['logits_to_keep']
            selected = logits[:, -keep:] if isinstance(keep, int) else logits[:, keep]
            actual = F.cross_entropy(selected[:, :-1].reshape(-1, 13), inputs['labels'][:, 1:].reshape(-1))
            actual_grad, = torch.autograd.grad(actual, logits)
            torch.testing.assert_close(full, actual)
            torch.testing.assert_close(expected_grad, actual_grad)
            assert set(expected_grad.abs().sum(-1)[0].nonzero().flatten().tolist()) == {p - 1 for p in positions}
            print({'positions': positions, 'loss': full.item(), 'gradient_equal': True}, flush=True)
