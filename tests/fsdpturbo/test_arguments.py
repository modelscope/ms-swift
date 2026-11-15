import unittest
from unittest import mock

from swift.arguments import SftArguments
from swift.fsdpturbo.arguments import FSDPTurboArguments, FSDPTurboSftArguments


class TestFSDPTurboArguments(unittest.TestCase):

    @staticmethod
    def _bare_sft_args(**overrides):
        values = {
            'fsdp': None,
            'deepspeed': None,
            'tuner_type': 'full',
            'model_type': 'qwen3_5_moe',
            'task_type': 'causal_lm',
            'enable_npu_model_patch': False,
            'add_version': False,
            'resume_from_checkpoint': None,
            'resume_only_model': False,
            'save_only_model': False,
            'sequence_parallel_size': 1,
            'padding_free': False,
            'packing': False,
            'max_steps': 1,
            'gradient_accumulation_steps': 1,
            'optimizer': None,
            'use_galore': False,
            'loss_type': None,
            'enable_dft_loss': False,
            'enable_channel_loss': False,
            'router_aux_loss_coef': 0,
            'use_flash_ckpt': False,
            'attn_impl': None,
        }
        values.update(overrides)
        args = object.__new__(FSDPTurboSftArguments)
        for name, value in values.items():
            setattr(args, name, value)
        return args

    def test_nontrivial_topology_is_valid(self):
        args = FSDPTurboArguments(fsdp_size=2, tp_size=2, ep_size=2, efsdp_size=2, pp_size=1)

        self.assertIsNone(args.validate_fsdpturbo(world_size=8))

    def test_parallel_sizes_must_be_positive(self):
        for field in ('fsdp_size', 'tp_size', 'cp_size', 'ep_size', 'efsdp_size', 'pp_size'):
            for value in (0, -1):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    args = FSDPTurboArguments(**{field: value})
                    args.validate_fsdpturbo(world_size=8)

    def test_unsupported_or_incoherent_topologies_are_rejected(self):
        cases = {
            'pipeline_parallelism': dict(pp_size=2),
            'efsdp_without_ep': dict(efsdp_size=2, ep_size=1),
            'tp_with_custom_fsdp': dict(tp_size=2, fsdp_implementation='custom'),
        }
        for name, kwargs in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                FSDPTurboArguments(**kwargs).validate_fsdpturbo(world_size=8)

    def test_world_size_must_cover_both_mesh_products(self):
        cases = {
            'fsdp_times_tp': dict(fsdp_size=2, tp_size=2, world_size=6),
            'efsdp_times_ep': dict(ep_size=2, efsdp_size=2, world_size=6),
            'product_larger_than_world': dict(fsdp_size=4, tp_size=2, world_size=4),
            'cp_times_tp': dict(cp_size=3, tp_size=2, world_size=8),
        }
        for name, values in cases.items():
            world_size = values.pop('world_size')
            with self.subTest(name=name), self.assertRaises(ValueError):
                FSDPTurboArguments(**values).validate_fsdpturbo(world_size=world_size)

    def test_topology_validation_precedes_sft_initialization(self):
        calls = []

        def validate(_self, world_size=None):
            calls.append(('validate', world_size))

        def init_sft(_self):
            calls.append(('sft', None))
            raise RuntimeError('stop after the parent initializer is reached')

        args = self._bare_sft_args(task_type=None)
        with mock.patch.object(FSDPTurboArguments, 'validate_fsdpturbo', autospec=True, side_effect=validate), \
                mock.patch.object(SftArguments, '__post_init__', autospec=True, side_effect=init_sft), \
                self.assertRaisesRegex(RuntimeError, 'parent initializer'):
            args.__post_init__()

        self.assertEqual([name for name, _ in calls[:2]], ['validate', 'sft'])

    def test_hf_distributed_backends_are_rejected_before_sft_init(self):
        cases = {
            'fsdp': {
                'fsdp': 'full_shard'
            },
            'deepspeed': {
                'deepspeed': 'zero2'
            },
        }
        for name, overrides in cases.items():
            args = self._bare_sft_args(**overrides)
            with self.subTest(name=name), \
                    mock.patch.object(FSDPTurboArguments, 'validate_fsdpturbo', autospec=True), \
                    mock.patch.object(SftArguments, '__post_init__', autospec=True) as init_sft, \
                    self.assertRaises(ValueError):
                args.__post_init__()
            init_sft.assert_not_called()

    def test_swift_npu_model_patch_is_rejected(self):
        args = self._bare_sft_args(enable_npu_model_patch=True)
        with mock.patch.object(FSDPTurboArguments, 'validate_fsdpturbo', autospec=True), \
                mock.patch.object(SftArguments, '__post_init__', autospec=True):
            with self.assertRaises(ValueError):
                args.__post_init__()

    def test_versioned_output_directory_is_rejected(self):
        args = self._bare_sft_args(add_version=True)
        with mock.patch.object(FSDPTurboArguments, 'validate_fsdpturbo', autospec=True), \
                mock.patch.object(SftArguments, '__post_init__', autospec=True):
            with self.assertRaises(ValueError):
                args.__post_init__()

    def test_old_deepseek_backend_and_native_checkpoint_modes_fail_before_parent_init(self):
        cases = [
            ({
                'model_type': 'deepseek_v4'
            }, 'DeepSeek V4'),
            ({
                'model_type': 'deepseek_v41',
                'resume_from_checkpoint': '/checkpoint'
            }, 'checkpoint resume'),
            ({
                'model_type': 'deepseek_v41',
                'save_only_model': True
            }, 'save_only_model'),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides), \
                    mock.patch.object(SftArguments, '__post_init__', autospec=True) as init_sft, \
                    self.assertRaisesRegex(ValueError, message):
                self._bare_sft_args(**overrides).__post_init__()
            init_sft.assert_not_called()

    def test_native_checkpoint_gate_also_checks_the_resolved_model_type(self):
        args = self._bare_sft_args(model_type=None, save_only_model=True)

        def resolve_model_type(args):
            args.model_type = 'deepseek_v41'

        with mock.patch.object(SftArguments, '__post_init__', autospec=True, side_effect=resolve_model_type), \
                self.assertRaisesRegex(ValueError, 'save_only_model'):
            args.__post_init__()

    def test_resolved_native_checkpoint_modes_fail_before_parent_device_initialization(self):
        cases = [({
            'resume_from_checkpoint': '/checkpoint'
        }, 'checkpoint resume'), ({
            'save_only_model': True
        }, 'save_only_model')]

        def resolve_and_initialize(args):
            args.model_type = 'deepseek_v41'
            args._init_device()

        for overrides, message in cases:
            args = self._bare_sft_args(model_type=None, **overrides)
            with self.subTest(overrides=overrides), \
                    mock.patch.object(SftArguments, '__post_init__', autospec=True,
                                      side_effect=resolve_and_initialize), \
                    mock.patch.object(SftArguments, '_init_device', autospec=True) as initialize_device, \
                    self.assertRaisesRegex(ValueError, message):
                args.__post_init__()
            initialize_device.assert_not_called()

    def test_qwen_checkpoint_modes_are_not_rejected_by_the_native_scope_gate(self):
        args = self._bare_sft_args(resume_from_checkpoint='/checkpoint', save_only_model=True)
        args._validate_model_scope()

    def test_unvalidated_checkpoint_and_offloaded_accumulation_paths_are_rejected(self):
        cases = {
            'model_only_resume': {
                'resume_only_model': True
            },
            'gradient_accumulation': {
                'gradient_accumulation_steps': 2,
                'offload_params': True
            },
        }
        for name, overrides in cases.items():
            args = self._bare_sft_args(**overrides)
            with self.subTest(name=name), \
                    mock.patch.object(FSDPTurboArguments, 'validate_fsdpturbo', autospec=True), \
                    mock.patch.object(SftArguments, '__post_init__', autospec=True), \
                    self.assertRaises(ValueError):
                args.__post_init__()


if __name__ == '__main__':
    unittest.main()
