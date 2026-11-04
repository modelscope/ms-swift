# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
import torch.nn.functional as F

from swift.sequence_parallel.utils import ChunkedCrossEntropyLoss


@pytest.mark.parametrize('chunk_size', [1, 3, 16])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('leaf', [False, True])
def test_chunked_ce_preserves_logits_and_matches_gradients(chunk_size, dtype, leaf):
    generator = torch.Generator().manual_seed(7)
    raw = torch.randn(7, 11, dtype=dtype, generator=generator, requires_grad=True)
    logits = raw if leaf else raw.tanh()
    original = logits.detach().clone()
    labels = torch.tensor([1, -100, 3, 5, 0, -100, 8])
    weights = torch.tensor([0.5, 1., 2., 0., 1., 3., 0.25], dtype=dtype)
    expected_input = original.clone().requires_grad_()
    expected = F.cross_entropy(expected_input, labels, reduction='none')
    actual = ChunkedCrossEntropyLoss.apply(logits, labels, chunk_size)
    torch.testing.assert_close(actual, expected)
    actual_grad, = torch.autograd.grad(actual, raw, weights, retain_graph=True)
    reference_grad, = torch.autograd.grad(expected, expected_input, weights)
    if not leaf:
        reference_grad *= 1 - original.square()
    torch.testing.assert_close(actual_grad, reference_grad)
    torch.testing.assert_close(logits, original, rtol=0, atol=0)
    # Reusing the retained graph must not see gradient data in place of logits.
    repeated_grad, = torch.autograd.grad(actual, raw, weights)
    torch.testing.assert_close(repeated_grad, reference_grad)


def test_chunked_ce_shared_storage_with_auxiliary_loss():
    torch.manual_seed(12)
    model = torch.nn.Linear(5, 11, dtype=torch.float64)
    x = torch.randn(7, 5, dtype=torch.float64)
    labels = torch.arange(7)
    logits = model(x)
    baseline = F.cross_entropy(logits, labels) + 0.1 * logits.square().mean()
    expected = torch.autograd.grad(baseline, tuple(model.parameters()))
    logits = model(x)
    actual = 0.1 * logits.square().mean() + ChunkedCrossEntropyLoss.apply(logits, labels, 3).mean()
    grads = torch.autograd.grad(actual, tuple(model.parameters()))
    for grad, reference in zip(grads, expected):
        torch.testing.assert_close(grad, reference)
