# Copyright (c) ModelScope Contributors. All rights reserved.
import tempfile
import unittest
from unittest.mock import patch

from swift.rlhf_trainers import DPOConfig, DPOTrainer
from swift.rlhf_trainers.rlhf_mixin import RLHFTrainerMixin


class TestDPOPrecompute(unittest.TestCase):

    def setUp(self):
        output_dir = tempfile.TemporaryDirectory()
        self.addCleanup(output_dir.cleanup)
        self.config_kwargs = dict(output_dir=output_dir.name, use_cpu=True, bf16=False, fp16=False, report_to=[])

    def test_precompute_rejected_before_model_setup(self):
        args = DPOConfig(precompute_ref_log_probs=True, **self.config_kwargs)
        with patch.object(
                RLHFTrainerMixin, '__init__', side_effect=AssertionError('Model setup reached')) as initialize:
            with self.assertRaisesRegex(ValueError, r'precompute_ref_log_probs=True.*precompute_ref_log_probs=False'):
                DPOTrainer(args=args)
            initialize.assert_not_called()

    def test_precompute_enabled_after_config_creation(self):
        args = DPOConfig(**self.config_kwargs)
        args.precompute_ref_log_probs = True
        with patch.object(RLHFTrainerMixin, '__init__', side_effect=AssertionError('Model setup reached')):
            with self.assertRaisesRegex(ValueError, 'precompute_ref_log_probs=True'):
                DPOTrainer(args=args)

    def test_default_and_explicit_false_reach_model_setup(self):

        class ModelSetupReached(Exception):
            pass

        for options in ({}, {'precompute_ref_log_probs': False}):
            with self.subTest(options=options):
                args = DPOConfig(**options, **self.config_kwargs)
                with patch.object(RLHFTrainerMixin, '__init__', side_effect=ModelSetupReached) as initialize:
                    with self.assertRaises(ModelSetupReached):
                        DPOTrainer(args=args)
                    initialize.assert_called_once()


if __name__ == '__main__':
    unittest.main()
