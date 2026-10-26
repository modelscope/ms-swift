import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from swift.rlhf_trainers.rollout_mixin import RolloutTrainerMixin


class TestRolloutVllmLora(unittest.TestCase):

    def test_lora_uses_model_multimodal_metadata(self):
        for is_multimodal in (False, True):
            with self.subTest(is_multimodal=is_multimodal):
                # GKD does not initialize the GRPO-specific is_multimodal attribute.
                trainer = object.__new__(RolloutTrainerMixin)
                trainer.args = SimpleNamespace(
                    per_device_train_batch_size=1,
                    vllm_max_num_seqs=None,
                    tuner_type='lora',
                    vllm_enable_lora=True,
                    lora_rank=8,
                    vllm_engine_kwargs=None,
                    vllm_enable_prefix_caching=False,
                    vllm_enforce_eager=True,
                    vllm_limit_mm_per_prompt=None,
                    sleep_level=1,
                    vllm_max_model_len=128,
                    vllm_disable_cascade_attn=False,
                    vllm_mm_processor_cache_gb=0,
                )
                trainer.model = SimpleNamespace(
                    model_dir='unused-model',
                    model_info=SimpleNamespace(is_moe_model=False, quant_method=None, torch_dtype='bfloat16'),
                    model_meta=SimpleNamespace(is_multimodal=is_multimodal, model_type='test-model'),
                )
                trainer.template = SimpleNamespace(processor=None)
                trainer.accelerator = SimpleNamespace(process_index=0)
                trainer.vllm_tensor_parallel_size = 1
                trainer.vllm_gpu_memory_utilization = 0.3
                trainer.vllm_version_ge_0_10_2 = True
                logger = Mock()
                self.assertFalse(hasattr(trainer, 'is_multimodal'))

                with patch('swift.infer_engine.GRPOVllmEngine') as engine_cls, \
                        patch('swift.rlhf_trainers.rollout_mixin.Swift.grpo_context', return_value=nullcontext()), \
                        patch('swift.rlhf_trainers.rollout_mixin.set_expandable_segments'), \
                        patch('swift.rlhf_trainers.rollout_mixin.patch_vllm_load_adapter') as patch_adapter, \
                        patch('swift.rlhf_trainers.rollout_mixin.logger', logger):
                    engine = trainer._prepare_vllm_engine()

                self.assertIs(engine, engine_cls.return_value)
                engine_cls.assert_called_once()
                self.assertTrue(engine_cls.call_args.kwargs['enable_lora'])
                self.assertEqual(engine_cls.call_args.kwargs['max_lora_rank'], 8)
                self.assertTrue(trainer.rollout_enable_lora)
                patch_adapter.assert_called_once_with()
                if is_multimodal:
                    logger.warning.assert_called_once()
                    self.assertIn('multimodal model', logger.warning.call_args.args[0])
                else:
                    logger.warning.assert_not_called()


if __name__ == '__main__':
    unittest.main()
