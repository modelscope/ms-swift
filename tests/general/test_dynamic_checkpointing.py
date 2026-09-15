# Copyright (c) ModelScope Contributors. All rights reserved.
import copy
import torch
import unittest
from functools import partial
from torch import nn
from torch.utils.checkpoint import checkpoint

from swift.trainers.utils import dynamic_gradient_checkpointing


class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, hidden_states):
        return hidden_states + self.proj(hidden_states).tanh()


class TinyTower(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(10)])
        self.head = nn.Linear(4, 1)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.head(hidden_states)


class TestDynamicCheckpointing(unittest.TestCase):

    def check_gradients(self, model, reference, inputs):
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        actual = model(inputs.clone())
        expected = reference(inputs.clone())
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for (name, parameter), (_, reference_parameter) in zip(model.named_parameters(), reference.named_parameters()):
            with self.subTest(parameter=name):
                if reference_parameter.grad is None:
                    self.assertIsNone(parameter.grad)
                else:
                    self.assertIsNotNone(parameter.grad)
                    torch.testing.assert_close(parameter.grad, reference_parameter.grad)

    def prepare(self, model, use_reentrant):
        dynamic_gradient_checkpointing(model)
        for layer in model.layers:
            layer.gradient_checkpointing = True
            layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=use_reentrant)

    def test_partially_frozen_tower_matches_eager_gradients(self):
        for frozen_prefix, use_reentrant in ((5, True), (0, True), (5, False)):
            with self.subTest(frozen_prefix=frozen_prefix, use_reentrant=use_reentrant):
                torch.manual_seed(42)
                model = TinyTower()
                for layer in model.layers[:frozen_prefix]:
                    layer.requires_grad_(False)
                reference = copy.deepcopy(model)
                self.prepare(model, use_reentrant)
                self.check_gradients(model, reference, torch.randn(2, 4))

    def test_unfreezing_layers_after_first_forward(self):
        torch.manual_seed(42)
        model = TinyTower()
        model.layers.requires_grad_(False)
        reference = copy.deepcopy(model)
        self.prepare(model, True)
        inputs = torch.randn(2, 4)
        self.check_gradients(model, reference, inputs)
        model.layers[4].requires_grad_(True)
        reference.layers[4].requires_grad_(True)
        self.check_gradients(model, reference, inputs)


if __name__ == '__main__':
    unittest.main()
