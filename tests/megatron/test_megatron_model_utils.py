import dataclasses
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_IMPORT_ERROR = ''
try:
    from swift.megatron.model import utils
except ImportError as e:
    utils = None
    _IMPORT_ERROR = str(e)


@dataclasses.dataclass
class _FakeProvider:
    mtp_num_layers: int = None
    mtp_use_repeated_layer: bool = False
    num_moe_experts: int = None
    gated_linear_unit: bool = False
    activation_func: object = None
    variable_seq_lengths: bool = False

    def apply_overrides_and_finalize(self, dtype, overrides):
        self.applied_overrides = overrides

    def provide_distributed_model(self, **kwargs):
        return SimpleNamespace(config=SimpleNamespace())


class _FakeAutoBridge:

    @staticmethod
    def supports(hf_config):
        return True


@unittest.skipUnless(utils is not None, f'Megatron dependencies not available: {_IMPORT_ERROR}')
class TestMegatronModelUtils(unittest.TestCase):

    def test_mtp_shared_weights_maps_to_megatron_bridge_field(self):
        auto_bridge_module = types.ModuleType('megatron.bridge.models.conversion.auto_bridge')
        auto_bridge_module.AutoBridge = _FakeAutoBridge
        modules = {
            'megatron': types.ModuleType('megatron'),
            'megatron.bridge': types.ModuleType('megatron.bridge'),
            'megatron.bridge.models': types.ModuleType('megatron.bridge.models'),
            'megatron.bridge.models.conversion': types.ModuleType('megatron.bridge.models.conversion'),
            'megatron.bridge.models.conversion.auto_bridge': auto_bridge_module,
        }
        for shared_weights, provider_default in ((True, False), (False, False), (False, True)):
            with self.subTest(shared_weights=shared_weights, provider_default=provider_default):
                args = SimpleNamespace(
                    mtp_num_layers=3,
                    mtp_shared_weights=shared_weights,
                    torch_dtype=None,
                    router_replay_mode='disabled',
                    megatron_extra_kwargs=None,
                    padding_free=False,
                    use_cpu_initialization=False,
                )
                backend = SimpleNamespace(_bridge=SimpleNamespace(to_megatron_provider=lambda *_args, **_kwargs: None))
                provider = _FakeProvider(mtp_use_repeated_layer=provider_default)

                with patch.dict(sys.modules, modules), patch.object(
                        utils.MegatronBridgeBackend, 'from_hf_config', return_value=backend), patch.object(
                            backend._bridge, 'to_megatron_provider', return_value=provider):
                    models = utils._get_megatron_bridge_model(args, SimpleNamespace())

                self.assertEqual(provider.applied_overrides['mtp_num_layers'], 3)
                if shared_weights:
                    self.assertTrue(provider.applied_overrides['mtp_use_repeated_layer'])
                else:
                    self.assertNotIn('mtp_use_repeated_layer', provider.applied_overrides)
                    self.assertEqual(provider.mtp_use_repeated_layer, provider_default)
                self.assertIs(models[0].config.bridge, backend)

    def test_megatron_bridge_export_accepts_common_skip_flag(self):
        tensor = object()
        auto_bridge = SimpleNamespace(
            export_hf_weights=lambda models, cpu=False: [SimpleNamespace(param_name='weight', weight=tensor)])
        backend = utils.MegatronBridgeBackend(auto_bridge)

        exported = list(backend.export_weights([object()], target_device='cpu', skip_unsupported_export=True))

        self.assertEqual(exported, [('weight', tensor)])

    def test_engram_freeze_is_scoped_to_deepseek_v41_full_grpo(self):
        from swift.megatron.utils import utils as megatron_utils

        model = object()
        base_args = {
            'tuner_type': 'full',
            'freeze_parameters_ratio': 0,
            'freeze_parameters': [],
            'freeze_parameters_regex': None,
            'trainable_parameters': [],
            'trainable_parameters_regex': None,
        }
        cases = (
            ('deepseek_v41', 'grpo', True),
            ('deepseek_v4', 'grpo', False),
            ('deepseek_v41', 'sft', False),
        )
        for model_type, rlhf_type, should_freeze in cases:
            with self.subTest(model_type=model_type, rlhf_type=rlhf_type):
                args = SimpleNamespace(**base_args, model_type=model_type, rlhf_type=rlhf_type)
                with patch.object(megatron_utils, 'freeze_parameters'), patch.object(
                        megatron_utils, '_freeze_engram_parameters') as freeze_engram, patch.object(
                            megatron_utils, 'get_model_parameter_info', return_value=''), patch.object(
                                megatron_utils.dist, 'get_rank', return_value=0), patch.object(
                                    megatron_utils.mpu, 'get_data_parallel_rank', return_value=0):
                    result = megatron_utils.prepare_mcore_model(args, model)

                self.assertIs(result, model)
                self.assertEqual(freeze_engram.called, should_freeze)


if __name__ == '__main__':
    unittest.main()
