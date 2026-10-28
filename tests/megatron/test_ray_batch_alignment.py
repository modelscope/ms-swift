# Copyright (c) ModelScope Contributors. All rights reserved.
import ast
import importlib.util
import sys
import torch
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple
from unittest.mock import patch

from swift.ray.megatron.driver_utils import estimate_dp_size

ROOT = Path(__file__).resolve().parents[2]


def load_methods(path, class_name, names, namespace):
    # Exercise the real pure-Python methods without importing Megatron's GPU stack.
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    cls.decorator_list = []
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[class_name]


BatchArgs = load_methods(
    ROOT / 'swift/megatron/arguments/megatron_args.py', 'RLHFMegatronArgumentsMixin',
    {'resolve_generation_batch_size', 'validate_batch_dp_alignment', '_init_generation_batch_params'}, {
        'Optional': Optional,
        'Tuple': Tuple,
        'torch': torch
    })
Pipeline = load_methods(ROOT / 'swift/ray/megatron/pipeline.py', 'MegatronRayPipeline',
                        {'_validate_grpo_train_batch_params', 'group_gpus'}, {
                            'Dict': Dict,
                            'estimate_dp_size': estimate_dp_size
                        })


class TestRayBatchAlignment(unittest.TestCase):

    def test_ray_accepts_complete_completion_microbatches(self):
        for values in [(512, 16, 32, 2, 32), (480, 16, 6, 2, 6), (96, 8, 6, 4, 12)]:
            with self.subTest(values=values):
                BatchArgs.validate_batch_dp_alignment(*values, use_ray=True)

    def test_ray_preserves_alignment_rejections(self):
        for values, message in [
            ((512, 16, 32, 3, 32), 'completion count'),
            ((480, 16, 32, 2, 32), 'divisible by dp_size'),
            ((0, 16, 4, 2, 4), 'per_device_num_rollout_prompt'),
            ((16, 16, 1, 1, 32), 'per_device_generation_batch_size'),
        ]:
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    BatchArgs.validate_batch_dp_alignment(*values, use_ray=True)

    def test_non_ray_keeps_prompt_microbatch_restriction(self):
        for kwargs in [{}, {'use_ray': False}]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, 'rollout prompt count'):
                    BatchArgs.validate_batch_dp_alignment(512, 16, 32, 2, 32, **kwargs)

    def test_prompt_aligned_batches_remain_valid(self):
        for use_ray in [False, True]:
            with self.subTest(use_ray=use_ray):
                BatchArgs.validate_batch_dp_alignment(512, 16, 16, 2, 32, use_ray=use_ray)
                BatchArgs.validate_batch_dp_alignment(512, 16, 32, 1, 32, use_ray=use_ray)

    def test_single_generation_keeps_same_divisibility(self):
        for use_ray in [False, True]:
            with self.subTest(use_ray=use_ray):
                BatchArgs.validate_batch_dp_alignment(24, 1, 6, 2, 12, use_ray=use_ray)
                with self.assertRaises(ValueError):
                    BatchArgs.validate_batch_dp_alignment(24, 1, 6, 3, 12, use_ray=use_ray)

    def worker_args(self, use_ray):
        args = BatchArgs.__new__(BatchArgs)
        vars(args).update(
            generation_batch_size=None,
            steps_per_generation=None,
            global_batch_size=512,
            num_generations=16,
            micro_batch_size=2,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            use_ray=use_ray)
        return args

    def test_worker_ray_uses_completion_units(self):
        for tp, pp, cp in [(1, 1, 1), (2, 2, 2)]:
            with self.subTest(tp=tp, pp=pp, cp=cp):
                args = self.worker_args(True)
                args.tensor_model_parallel_size = tp
                args.pipeline_model_parallel_size = pp
                args.context_parallel_size = cp
                world_size = 32 * tp * pp * cp
                with patch('torch.distributed.is_initialized', return_value=True), \
                        patch('torch.distributed.get_world_size', return_value=world_size):
                    args._init_generation_batch_params()
                self.assertEqual(args.generation_batch_size, 512)
                self.assertEqual(args.steps_per_generation, 1)
                self.assertEqual(args.per_device_generation_batch_size, 512 // world_size)

    def test_worker_non_ray_keeps_prompt_units(self):
        args = self.worker_args(False)
        with patch('torch.distributed.is_initialized', return_value=True), \
                patch('torch.distributed.get_world_size', return_value=32):
            with self.assertRaisesRegex(ValueError, 'rollout prompt count'):
                args._init_generation_batch_params()

    def pipeline(self, **overrides):
        pipeline = Pipeline.__new__(Pipeline)
        cfg = {'global_batch_size': 512, 'num_generations': 16, 'micro_batch_size': 2}
        cfg.update(overrides)
        pipeline.rlhf_type = 'grpo'
        pipeline.group_cfgs = {'train': cfg}
        pipeline.ray_config = SimpleNamespace(group_gpus={'train': 32})
        return pipeline

    def validate_pipeline(self, pipeline):
        name = 'swift.megatron.arguments.megatron_args'
        with patch.dict(sys.modules, {name: SimpleNamespace(RLHFMegatronArgumentsMixin=BatchArgs)}):
            pipeline._validate_grpo_train_batch_params()

    def test_pipeline_accepts_completion_aligned_batches(self):
        for tp, pp, cp in [(1, 1, 1), (2, 2, 2)]:
            with self.subTest(tp=tp, pp=pp, cp=cp):
                pipeline = self.pipeline(
                    tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp, context_parallel_size=cp)
                pipeline.group_gpus['train'] *= tp * pp * cp
                self.validate_pipeline(pipeline)

    def test_pipeline_resolves_generation_steps(self):
        self.validate_pipeline(self.pipeline(global_batch_size=256, steps_per_generation=2))

    def test_pipeline_preserves_completion_and_prompt_checks(self):
        for overrides, message in [
            ({
                'micro_batch_size': 3
            }, 'completion count'),
            ({
                'global_batch_size': 480
            }, 'divisible by dp_size'),
            ({
                'global_batch_size': 33
            }, 'divisible by num_generations'),
        ]:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    self.validate_pipeline(self.pipeline(**overrides))

    def test_pipeline_keeps_gpu_count_guard(self):
        pipeline = self.pipeline(global_batch_size=16, tensor_model_parallel_size=32)
        with self.assertRaisesRegex(ValueError, 'per_device_generation_batch_size'):
            self.validate_pipeline(pipeline)

    @unittest.skipUnless(importlib.util.find_spec('ray'), 'Ray is required for native collation')
    def test_native_collation_forms_full_microbatches(self):
        from swift.ray.megatron.base_trainer import BaseRayTrainer

        def collate(samples, template, **kwargs):
            return {'sample_ids': list(samples)}, SimpleNamespace(sample_ids=list(samples))

        trainer = SimpleNamespace(
            args=SimpleNamespace(micro_batch_size=2, loss_type='dr_grpo'),
            template=object(),
            device='cpu',
            _padding_to=None)
        with patch('swift.rlhf_trainers.utils.collate_to_grpo_micro_batch', side_effect=collate):
            dispatch, batches = BaseRayTrainer._collate_for_workers(trainer, SimpleNamespace(dp_size=32),
                                                                    list(range(512)))
        self.assertEqual(len(dispatch), 32)
        self.assertTrue(all(len(microbatches) == 8 for microbatches in dispatch.values()))
        self.assertTrue(all(len(batch.sample_ids) == 2 for batch in batches))
        self.assertEqual([sample for batch in batches for sample in batch.sample_ids], list(range(512)))
        dispatched = [inputs['grpo_batch'] for microbatches in dispatch.values() for inputs in microbatches]
        self.assertEqual(len(dispatched), len(batches))
        self.assertTrue(all(left is right for left, right in zip(dispatched, batches)))


if __name__ == '__main__':
    unittest.main()
