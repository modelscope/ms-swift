#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for EarlyStopCallback (gh#8330)."""
import unittest
from unittest.mock import MagicMock

from swift.callbacks.early_stop import EarlyStopCallback


class TestEarlyStopCallback(unittest.TestCase):
    """Test both (args, trainer) and (trainer,) init signatures."""

    def test_init_args_and_trainer(self):
        """Standard call: EarlyStopCallback(args, trainer)."""
        args = MagicMock()
        args.early_stop_interval = 3
        trainer = MagicMock()
        trainer.args = args
        cb = EarlyStopCallback(args, trainer)
        self.assertIs(cb.args, args)
        self.assertIs(cb.trainer, trainer)
        self.assertEqual(cb.total_interval, 3)

    def test_init_trainer_only(self):
        """Backward compat: EarlyStopCallback(trainer) when trainer.args is set (gh#8330)."""
        args = MagicMock()
        args.early_stop_interval = 5
        trainer = MagicMock()
        trainer.args = args
        cb = EarlyStopCallback(trainer)
        self.assertIs(cb.args, args)
        self.assertIs(cb.trainer, trainer)
        self.assertEqual(cb.total_interval, 5)

    def test_init_trainer_only_no_args_raises(self):
        """Single-arg call with no trainer.args must raise."""
        trainer = MagicMock(spec=[])  # no .args
        with self.assertRaises(TypeError) as ctx:
            EarlyStopCallback(trainer)
        self.assertIn('EarlyStopCallback requires', str(ctx.exception))
