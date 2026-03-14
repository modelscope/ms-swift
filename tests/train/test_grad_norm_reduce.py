# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for grad_norm all-reduce under ZeRO-0/DDP (fix #6815)."""
import unittest
from unittest.mock import MagicMock, patch

import torch

from swift.trainers.mixin import SwiftMixin


def _make_trainer():
    trainer = MagicMock()
    trainer.accelerator = MagicMock()
    trainer.accelerator.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return trainer


class TestGradNormReduce(unittest.TestCase):
    """Test _get_reduced_grad_norm_for_logging for consistent grad_norm logging under ZeRO-0."""

    def test_grad_norm_none(self):
        trainer = _make_trainer()
        self.assertIsNone(SwiftMixin._get_reduced_grad_norm_for_logging(trainer, None))

    def test_grad_norm_float(self):
        trainer = _make_trainer()
        self.assertEqual(SwiftMixin._get_reduced_grad_norm_for_logging(trainer, 1.5), 1.5)

    def test_grad_norm_tensor_single_process(self):
        trainer = _make_trainer()
        with patch('swift.trainers.mixin.is_dist', return_value=False):
            gn = torch.tensor(2.0)
            self.assertEqual(SwiftMixin._get_reduced_grad_norm_for_logging(trainer, gn), 2.0)

    def test_grad_norm_tensor_dist_zero3_no_reduce(self):
        trainer = _make_trainer()
        with patch('swift.trainers.mixin.is_dist', return_value=True), \
             patch('swift.trainers.mixin.get_dist_setting', return_value=(0, 0, 2, 2)), \
             patch('swift.trainers.mixin.is_deepspeed_zero3_enabled', return_value=True):
            gn = torch.tensor(0.025)
            out = SwiftMixin._get_reduced_grad_norm_for_logging(trainer, gn)
            self.assertAlmostEqual(out, 0.025)

    def test_grad_norm_tensor_dist_zero0_reduce(self):
        trainer = _make_trainer()
        with patch('swift.trainers.mixin.is_dist', return_value=True), \
             patch('swift.trainers.mixin.get_dist_setting', return_value=(0, 0, 2, 2)), \
             patch('swift.trainers.mixin.is_deepspeed_zero3_enabled', return_value=False), \
             patch('torch.distributed.all_reduce') as mock_all_reduce:
            gn = torch.tensor(1656.0)
            def _side_effect(tensor, *args, **kwargs):
                tensor.fill_(tensor.item() / 2)
            mock_all_reduce.side_effect = _side_effect
            out = SwiftMixin._get_reduced_grad_norm_for_logging(trainer, gn)
            self.assertEqual(mock_all_reduce.call_count, 1)
            self.assertAlmostEqual(out, 828.0)
