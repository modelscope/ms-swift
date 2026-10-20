# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

_IMPORT_ERROR = ''
try:
    import torch
    from mcore_bridge.model import get_mcore_model_type, get_model_meta
    from transformers import Qwen3Config

    from swift.megatron.model import utils
except ImportError as e:
    utils = None
    _IMPORT_ERROR = str(e)

_HAS_CHUNKING = utils is not None and hasattr(utils.ModelConfig, 'mlp_chunks_for_training')


@dataclass
class _LegacyConfig:
    """Model configurations before MCore 0.18 have no training chunk field."""
    num_moe_experts: int = None


@unittest.skipUnless(utils is not None, f'Megatron dependencies not available: {_IMPORT_ERROR}')
class TestMLPChunking(unittest.TestCase):

    def setUp(self):
        self.hf_config = Qwen3Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=128)
        self.args = SimpleNamespace(
            megatron_model_meta=get_model_meta(get_mcore_model_type(self.hf_config.model_type)),
            task_type='causal_lm',
            torch_dtype=torch.bfloat16,
            decoder_first_pipeline_num_layers=None,
            decoder_last_pipeline_num_layers=None,
            fp4_param_gather=False,
            fp8_param_gather=False,
            router_replay_mode='disabled',
            padding_free=True,
            mlp_chunks_for_training=1,
            recompute_granularity='selective',
            recompute_modules=['core_attn'],
            megatron_extra_kwargs=None)
        # Weight conversion needs distributed groups; configuration validation does not.
        bridge_patch = patch.object(self.args.megatron_model_meta, 'bridge_cls', return_value=None)
        bridge_patch.start()
        self.addCleanup(bridge_patch.stop)

    def test_disabled_chunking_preserves_default_config(self):
        for extra_kwargs in (None, {'mlp_chunks_for_training': 1}):
            with self.subTest(extra_kwargs=extra_kwargs):
                self.args.megatron_extra_kwargs = extra_kwargs
                config = utils.get_mcore_model_config(self.args, self.hf_config)
                self.assertEqual(getattr(config, 'mlp_chunks_for_training', 1), 1)

    @unittest.skipIf(_HAS_CHUNKING, 'Requires a configuration predating megatron-core 0.18')
    def test_older_mcore_rejects_enabled_chunking(self):
        for chunks, extra_kwargs in ((2, None), (1, {'mlp_chunks_for_training': 4})):
            with self.subTest(chunks=chunks, extra_kwargs=extra_kwargs):
                self.args.mlp_chunks_for_training = chunks
                self.args.megatron_extra_kwargs = extra_kwargs
                with self.assertRaisesRegex(ValueError, 'megatron-core>=0.18'):
                    utils.get_mcore_model_config(self.args, self.hf_config)

    @unittest.skipUnless(_HAS_CHUNKING, 'Requires megatron-core>=0.18')
    def test_chunk_count_reaches_model_config(self):
        for chunks in (2, 4):
            with self.subTest(chunks=chunks):
                self.args.mlp_chunks_for_training = chunks
                config = utils.get_mcore_model_config(self.args, self.hf_config)
                self.assertEqual(config.mlp_chunks_for_training, chunks)
                self.assertEqual(config.recompute_modules, ['core_attn'])

    @unittest.skipUnless(_HAS_CHUNKING, 'Requires megatron-core>=0.18')
    def test_extra_kwargs_override_cli_chunk_count(self):
        self.args.mlp_chunks_for_training = 4
        for chunks in (1, 2):
            with self.subTest(chunks=chunks):
                self.args.megatron_extra_kwargs = {'mlp_chunks_for_training': chunks}
                config = utils.get_mcore_model_config(self.args, self.hf_config)
                self.assertEqual(config.mlp_chunks_for_training, chunks)

    def test_invalid_counts_are_rejected_in_both_inputs(self):
        for chunks in (0, -1, 1.5, True, None):
            for use_extra_kwargs in (False, True):
                with self.subTest(chunks=chunks, use_extra_kwargs=use_extra_kwargs):
                    self.args.mlp_chunks_for_training = 1 if use_extra_kwargs else chunks
                    self.args.megatron_extra_kwargs = {'mlp_chunks_for_training': chunks} if use_extra_kwargs else None
                    with self.assertRaisesRegex(ValueError, 'positive integer'):
                        utils.get_mcore_model_config(self.args, self.hf_config)

    @unittest.skipUnless(_HAS_CHUNKING, 'Requires megatron-core>=0.18')
    def test_selective_mlp_recomputation_checks_effective_overrides(self):
        self.args.mlp_chunks_for_training = 4
        self.args.megatron_extra_kwargs = {'recompute_modules': ['core_attn', 'mlp']}
        with self.assertRaisesRegex(ValueError, 'bypasses chunking'):
            utils.get_mcore_model_config(self.args, self.hf_config)

        self.args.megatron_extra_kwargs['mlp_chunks_for_training'] = 1
        config = utils.get_mcore_model_config(self.args, self.hf_config)
        self.assertIn('mlp', config.recompute_modules)
        self.assertEqual(config.mlp_chunks_for_training, 1)

    @unittest.skipUnless(_HAS_CHUNKING, 'Requires megatron-core>=0.18')
    def test_full_recomputation_is_allowed(self):
        self.args.mlp_chunks_for_training = 4
        self.args.megatron_extra_kwargs = {
            'recompute_granularity': 'full',
            'recompute_method': 'uniform',
            'recompute_num_layers': 1,
        }
        config = utils.get_mcore_model_config(self.args, self.hf_config)
        self.assertEqual(config.recompute_granularity, 'full')
        self.assertEqual(config.mlp_chunks_for_training, 4)

    @unittest.skipUnless(_HAS_CHUNKING, 'Requires megatron-core>=0.18')
    def test_moe_is_rejected_before_model_construction(self):
        self.args.mlp_chunks_for_training = 4
        self.args.megatron_extra_kwargs = {'num_moe_experts': 4}
        with self.assertRaisesRegex(ValueError, 'dense models'):
            utils.get_mcore_model_config(self.args, self.hf_config)

    def test_legacy_configs_keep_default_and_reject_enabled_chunking(self):
        for config in (_LegacyConfig, _LegacyConfig()):
            for overrides in ({}, {'mlp_chunks_for_training': 1}):
                with self.subTest(config=config, overrides=overrides):
                    self.args.mlp_chunks_for_training = 1
                    utils._check_mlp_chunks_for_training(self.args, config, overrides)
                    self.assertNotIn('mlp_chunks_for_training', overrides)
                    _LegacyConfig(**overrides)
            self.args.mlp_chunks_for_training = 4
            with self.assertRaisesRegex(ValueError, 'megatron-core>=0.18'):
                utils._check_mlp_chunks_for_training(self.args, config, {})
            self.args.mlp_chunks_for_training = 1
            with self.assertRaisesRegex(ValueError, 'megatron-core>=0.18'):
                utils._check_mlp_chunks_for_training(self.args, config, {'mlp_chunks_for_training': 4})


if __name__ == '__main__':
    unittest.main()
