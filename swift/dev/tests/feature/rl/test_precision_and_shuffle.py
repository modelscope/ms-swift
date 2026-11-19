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
                    # A real HF config always carries model_type; build_model's no-loader fallback
                    # re-resolves from it, so the mock must supply it (the value is irrelevant here --
                    # _resolve_model_loader is patched to None regardless -- it only has to exist).
                    stack.enter_context(
                        patch.object(builder, '_build_hf_config', return_value=SimpleNamespace(model_type='qwen2')))
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


    def test_prompt_shuffle_seed_controls_order(self):
        from swift.dev.builders.dataset import build_prompt_dataloader

        prompts = [[{'role': 'user', 'content': str(i)}] for i in range(17)]
        extras = [{'id': i} for i in range(17)]

        def order(seed):
            dl = build_prompt_dataloader(prompts, extras, seed=seed, batch_size=4, shuffle=True)
            return [row['id'] for batch in dl for row in batch]

        self.assertEqual(order(42), order(42))
        self.assertNotEqual(order(42), order(43))

    def test_recipe_prompt_dataloader_seed_source_is_data_seed(self):
        """defect98: the prompt shuffle must ride data_seed, not the run seed.

        PLAN
        | case | input | expected | failure it kills |
        | ast_guard×4 | run_{grpo,ppo,rft,distill}.py build_prompt_dataloader call | seed kwarg is dataset_config.data_seed | run seed leaking into data order (seed!=data_seed reshuffles prompts, confounding seed-vs-data reproducibility) |
        | behavior_seed_controls_order | builder seed 42 vs 43, shuffle=True | different row order | dead shuffle seed |
        Reverse-verify: revert any call site to train_config.seed -> the guard turns RED.
        """
        import ast
        from pathlib import Path

        recipes = {
            'grpo': 'swift/dev/recipe/grpo/run_grpo.py',
            'ppo': 'swift/dev/recipe/ppo/run_ppo.py',
            'rft': 'swift/dev/recipe/rft/run_rft.py',
            'distill': 'swift/dev/recipe/distill/run_distill.py',
        }
        root = Path(__file__).resolve().parents[5]
        for name, rel in recipes.items():
            with self.subTest(recipe=name):
                tree = ast.parse((root / rel).read_text())
                calls = [node for node in ast.walk(tree)
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                         and node.func.id == 'build_prompt_dataloader']
                self.assertEqual(len(calls), 1, f'{rel}: expected exactly one build_prompt_dataloader call')
                seed_kw = next((kw for kw in calls[0].keywords if kw.arg == 'seed'), None)
                self.assertIsNotNone(seed_kw, f'{rel}: build_prompt_dataloader call has no seed kwarg')
                self.assertIsInstance(seed_kw.value, ast.Attribute, f'{rel}: seed must be a config attribute')
                self.assertIsInstance(seed_kw.value.value, ast.Name)
                self.assertEqual((seed_kw.value.value.id, seed_kw.value.attr), ('dataset_config', 'data_seed'),
                                 f'{rel}: prompt shuffle seed must be dataset_config.data_seed, '
                                 f'got {ast.dump(seed_kw.value)}')


if __name__ == '__main__':
    unittest.main()
