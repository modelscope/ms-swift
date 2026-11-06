# Copyright (c) ModelScope Contributors. All rights reserved.
"""Check the production configuration helper without importing the CUDA stack."""
import ast
import dataclasses
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


@dataclasses.dataclass
class _ModernDDPConfig:
    megatron_fsdp_use_decoupled_grad: bool = False
    grad_reduce_in_fp32: bool = False
    check_for_nan_in_grad: bool = False


@dataclasses.dataclass
class _LegacyDDPConfig:
    grad_reduce_in_fp32: bool = False
    check_for_nan_in_grad: bool = False


def _load_config_helper():
    path = Path(__file__).resolve().parents[2] / 'swift/megatron/utils/megatron_lm_utils.py'
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_build_ddp_config')
    module = types.ModuleType('config_helper')
    module.dataclasses = dataclasses
    module.DistributedDataParallelConfig = _ModernDDPConfig
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), 'exec'), module.__dict__)
    return module


class TestMegatronFSDPPrecisionAwareConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Execute the complete production helper; only its configuration class is mocked.
        cls.utils = _load_config_helper()

    def _build(self, fsdp=True, precision_aware=False, version=1, **extra):
        args = SimpleNamespace(
            use_megatron_fsdp=fsdp,
            use_precision_aware_optimizer=precision_aware,
            megatron_fsdp_version=version,
            **extra,
        )
        return self.utils._build_ddp_config(args)

    def test_precision_aware_fsdp_v1_sets_decoupled_grad(self):
        config = self._build(precision_aware=True, megatron_fsdp_use_decoupled_grad=False)
        self.assertTrue(config.megatron_fsdp_use_decoupled_grad)
        self.assertTrue(config.check_for_nan_in_grad)

    def test_non_precision_aware_fsdp_v1_disables_decoupled_grad(self):
        config = self._build(precision_aware=False, megatron_fsdp_use_decoupled_grad=True)
        self.assertFalse(config.megatron_fsdp_use_decoupled_grad)

    def test_ddp_and_fsdp_v2_preserve_existing_gradient_flag(self):
        for fsdp, version in ((False, 1), (True, 2)):
            for flag in (False, True):
                with self.subTest(fsdp=fsdp, version=version, flag=flag):
                    config = self._build(
                        fsdp=fsdp, version=version, precision_aware=True, megatron_fsdp_use_decoupled_grad=flag)
                    self.assertIs(config.megatron_fsdp_use_decoupled_grad, flag)

    def test_old_mcore_rejects_unsupported_precision_aware_fsdp(self):
        with mock.patch.object(self.utils, 'DistributedDataParallelConfig', _LegacyDDPConfig):
            with self.assertRaisesRegex(ValueError, 'megatron_fsdp_use_decoupled_grad'):
                self._build(precision_aware=True)

    def test_old_mcore_accepts_unaffected_paths(self):
        with mock.patch.object(self.utils, 'DistributedDataParallelConfig', _LegacyDDPConfig):
            self.assertTrue(self._build(precision_aware=False).check_for_nan_in_grad)
            self.assertTrue(self._build(fsdp=False, precision_aware=True).check_for_nan_in_grad)
            self.assertTrue(self._build(version=2, precision_aware=True).check_for_nan_in_grad)

    def test_missing_version_defaults_to_fsdp_v1(self):
        args = SimpleNamespace(use_megatron_fsdp=True, use_precision_aware_optimizer=True)
        self.assertTrue(self.utils._build_ddp_config(args).megatron_fsdp_use_decoupled_grad)

    def test_missing_precision_aware_flag_defaults_to_false(self):
        args = SimpleNamespace(use_megatron_fsdp=True, megatron_fsdp_use_decoupled_grad=True)
        self.assertFalse(self.utils._build_ddp_config(args).megatron_fsdp_use_decoupled_grad)

    def test_fp32_gradient_reduction_mapping_is_preserved(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                config = self._build(accumulate_allreduce_grads_in_fp32=enabled, grad_reduce_in_fp32=not enabled)
                self.assertIs(config.grad_reduce_in_fp32, enabled)


class TestRealMCoreFSDPPrecisionAwareConfig(unittest.TestCase):

    def test_real_mcore_config_when_field_is_available(self):
        try:
            from swift.megatron.utils import megatron_lm_utils
        except (ImportError, RuntimeError) as error:
            self.skipTest(f'Megatron dependencies unavailable: {error}')
        config_type = megatron_lm_utils.DistributedDataParallelConfig
        if 'megatron_fsdp_use_decoupled_grad' not in {f.name for f in dataclasses.fields(config_type)}:
            self.skipTest('The installed MCore does not expose decoupled_grad configuration')
        args = SimpleNamespace(use_megatron_fsdp=True, use_precision_aware_optimizer=True)
        self.assertTrue(megatron_lm_utils._build_ddp_config(args).megatron_fsdp_use_decoupled_grad)


if __name__ == '__main__':
    unittest.main()
