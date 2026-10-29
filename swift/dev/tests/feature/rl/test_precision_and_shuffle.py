"""CPU regressions for online RL precision and prompt ordering (no model download)."""
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch


class PrecisionAndShuffleTests(unittest.TestCase):
    def test_fp32_storage_preserves_configured_compute_precision(self):
        from swift.dev.builders import model as builder
        from swift.dev.config import DistributedConfig, ModelConfig
        from twinkle.model.transformers.strategy.native_fsdp import _build_mp_policy

        for dtype, mode, compute in (('bfloat16', 'bf16', torch.bfloat16),
                                     ('float16', 'fp16', torch.float16),
                                     ('float32', 'no', None)):
            for override in (None, 'float32'):
                with self.subTest(dtype=dtype, override=override), ExitStack() as stack:
                    config = ModelConfig(model='/unused', torch_dtype=dtype)
                    stack.enter_context(patch('swift.dev.utils.hub.safe_snapshot_download', side_effect=lambda x, **kw: x))
                    stack.enter_context(patch.object(builder, '_resolve_model_loader', return_value=None))
                    stack.enter_context(patch.object(builder, '_build_hf_config', return_value=SimpleNamespace()))
                    stack.enter_context(patch.object(builder, '_apply_ray_placement'))
                    stack.enter_context(patch('swift.dev.model.TransformersModel', side_effect=lambda **kw: kw))
                    result = builder.build_model(config, DistributedConfig(fsdp='fsdp2', mode='ray'),
                                                 torch_dtype=override)
                    self.assertEqual(result['dtype'], getattr(torch, override or dtype))
                    self.assertEqual(result['mixed_precision'], mode)
                    policy = _build_mp_policy(result['mixed_precision'])
                    self.assertEqual((policy.param_dtype, policy.reduce_dtype, policy.output_dtype),
                                     (compute, compute, compute))
                    self.assertEqual(config.torch_dtype, dtype)

    def test_prompt_order_and_tail_batch(self):
        from swift.dev.builders.dataset import build_prompt_dataloader

        prompts = [[{'role': 'user', 'content': str(i)}] for i in range(17)]
        extras = [{'id': i} for i in range(17)]

        def loader(shuffle):
            return build_prompt_dataloader(prompts, extras, seed=42, batch_size=4, shuffle=shuffle)

        def read(dl):
            batches = list(dl)
            self.assertEqual([len(batch) for batch in batches], [4, 4, 4, 4, 1])
            rows = [row for batch in batches for row in batch]
            self.assertTrue(all(row['messages'][0]['content'] == str(row['id']) for row in rows))
            return [row['id'] for row in rows]

        ordered = loader(False)
        self.assertEqual(read(ordered), list(range(17)))
        self.assertEqual(read(ordered), list(range(17)))
        shuffled, replay = loader(True), loader(True)
        first = read(shuffled)
        shuffled.set_epoch(1)
        second = read(shuffled)
        self.assertCountEqual(first, range(17))
        self.assertNotEqual(first, list(range(17)))
        self.assertNotEqual(first, second)
        self.assertEqual(first, read(replay))
        replay.set_epoch(1)
        self.assertEqual(second, read(replay))


if __name__ == '__main__':
    unittest.main()
