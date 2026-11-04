# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import unittest

from swift.model.npu_patch.megatron_checkpoint import (_canonicalize_optimizer_steps_for_checkpoint,
                                                       _iter_optimizer_param_groups)


class InnerOptimizer:
    """A wrapper holding its optimizer as a plain attribute, the way `MegatronOptimizer` does."""

    def __init__(self, step, optimizer=None):
        self.param_groups = [{'params': [torch.zeros(1)], 'step': step}]
        if optimizer is not None:
            self.optimizer = optimizer


class ChainedOptimizer:
    """`optimizer` is a backwards-compatible property that refuses to answer for a chain of several.

    Mirrors megatron-core, where reading it raises `AssertionError` rather than `AttributeError`.
    """

    def __init__(self, *chained_optimizers):
        self.chained_optimizers = list(chained_optimizers)

    @property
    def optimizer(self):
        assert len(self.chained_optimizers) == 1, \
            'ChainedOptimizer has more than one optimizer when accessing self.optimizer'
        return self.chained_optimizers[0]


class TestOptimizerTraversal(unittest.TestCase):
    """Walking the optimizers must survive the optimizer types that hold more than one of them."""

    def test_a_chain_of_several_optimizers_is_walked(self):
        """Muon pairs the matrices with a scalar optimizer, so the chain holds two."""
        chained = ChainedOptimizer(InnerOptimizer(3), InnerOptimizer(3))
        self.assertEqual(len(list(_iter_optimizer_param_groups(chained))), 2)

    def test_a_chain_of_one_optimizer_is_walked(self):
        chained = ChainedOptimizer(InnerOptimizer(3))
        self.assertEqual(len(list(_iter_optimizer_param_groups(chained))), 1)

    def test_a_wrapped_optimizer_is_still_reached(self):
        """Wrappers keep theirs under `optimizer`, which is the only case worth reading that name for."""
        wrapper = InnerOptimizer(3, optimizer=InnerOptimizer(3))
        self.assertEqual(len(list(_iter_optimizer_param_groups(wrapper))), 2)

    def test_the_steps_of_a_chain_are_canonicalized_and_restored(self):
        """The npu path rewrites every param group's step, so it has to reach all of them."""
        chained = ChainedOptimizer(InnerOptimizer(torch.tensor(3)), InnerOptimizer(3))
        with _canonicalize_optimizer_steps_for_checkpoint(chained):
            steps = [group['step'] for opt in chained.chained_optimizers for group in opt.param_groups]
            self.assertEqual(steps, [3, 3])
        restored = [group['step'] for opt in chained.chained_optimizers for group in opt.param_groups]
        self.assertIsInstance(restored[0], torch.Tensor)
        self.assertEqual(restored[1], 3)

    def test_inconsistent_steps_across_a_chain_are_reported(self):
        chained = ChainedOptimizer(InnerOptimizer(3), InnerOptimizer(4))
        with self.assertRaisesRegex(RuntimeError, r'Inconsistent optimizer steps.*\[3, 4\]'):
            with _canonicalize_optimizer_steps_for_checkpoint(chained):
                pass


if __name__ == '__main__':
    unittest.main()
