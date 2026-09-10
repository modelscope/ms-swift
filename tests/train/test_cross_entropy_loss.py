# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn.functional as F
import unittest
from transformers.modeling_outputs import CausalLMOutputWithPast

from swift.loss.causal_lm import CustomCrossEntropyLoss


class TestCustomCrossEntropyLoss(unittest.TestCase):

    def test_token_weights_and_normalization(self):
        for scale_mode in ['none', 'ones', 'weighted', 'zero']:
            for num_items_in_batch in [None, 20]:
                with self.subTest(scale_mode=scale_mode, num_items_in_batch=num_items_in_batch):
                    torch.manual_seed(42)
                    logits = torch.randn(2, 5, 7, requires_grad=True)
                    reference_logits = logits.detach().clone().requires_grad_()
                    labels = torch.tensor([[-100, 1, 2, 3, 4], [-100, 2, -100, 4, 5]])
                    weights = torch.ones(2, 5)
                    if scale_mode == 'weighted':
                        weights = torch.tensor([[0., 0.25, 2., 0., 1.], [0., 3., 0., 0.5, 2.]])
                    elif scale_mode == 'zero':
                        weights.zero_()
                    # Seq2SeqTrainer has already shifted and flattened loss_scale at the loss callback boundary.
                    loss_scale = None if scale_mode == 'none' else weights.roll(-1, dims=-1).reshape(-1)
                    actual = CustomCrossEntropyLoss(None, None)(
                        CausalLMOutputWithPast(logits=logits),
                        labels,
                        num_items_in_batch=num_items_in_batch,
                        loss_scale=loss_scale)
                    token_loss = F.cross_entropy(
                        reference_logits[:, :-1].reshape(-1, 7), labels[:, 1:].reshape(-1),
                        reduction='none').reshape(2, 4)
                    denominator = (labels[:, 1:] != -100).sum() if num_items_in_batch is None else num_items_in_batch
                    expected = (token_loss * weights[:, 1:]).sum() / denominator
                    torch.testing.assert_close(actual, expected)
                    actual.backward()
                    expected.backward()
                    torch.testing.assert_close(logits.grad, reference_logits.grad)
