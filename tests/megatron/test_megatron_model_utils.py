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

    def test_mcore_mtp_preserves_extra_kwargs_priority(self):
        from mcore_bridge.model.register import ModelMeta

        hf_config = SimpleNamespace(num_hidden_layers=2)
        meta = ModelMeta(
            'test', ['test'], config_cls=lambda **kwargs: SimpleNamespace(**kwargs), mtp_prefix='model.layers')
        cases = (
            ('auto', None, {}, 1, False),
            (3, True, {
                'mtp_num_layers': 0,
                'mtp_shared_weights': False
            }, 0, False),
            (3, True, {
                'mtp_num_layers': None,
                'mtp_shared_weights': False
            }, None, False),
            (1, False, {
                'mtp_num_layers': 3,
                'mtp_shared_weights': None
            }, 3, True),
        )
        for depth, shared, extra, expected_depth, expected_shared in cases:
            with self.subTest(depth=depth, shared=shared, extra=extra):
                args = SimpleNamespace(
                    megatron_model_meta=meta,
                    model_dir='student',
                    mtp_num_layers=depth,
                    mtp_shared_weights=shared,
                    task_type='causal_lm',
                    torch_dtype=None,
                    decoder_first_pipeline_num_layers=None,
                    decoder_last_pipeline_num_layers=None,
                    fp4_param_gather=False,
                    fp8_param_gather=False,
                    router_replay_mode='disabled',
                    megatron_extra_kwargs=extra,
                    padding_free=False,
                )
                with patch.object(
                        utils, 'hf_to_mcore_config', return_value={'hf_config': hf_config}), patch.object(
                            utils,
                            'fields',
                            return_value=[
                                SimpleNamespace(name='mtp_num_layers'),
                                SimpleNamespace(name='mtp_shared_weights')
                            ]), patch(
                                'mcore_bridge.model.mtp_detect._read_weight_keys',
                                return_value=['model.layers.2.eh_proj.weight']), patch.object(
                                    utils, 'is_torch_npu_available', return_value=False), patch.object(
                                        utils, '_check_attention_backend'), patch.object(utils, '_check_padding_free'):
                    config = utils.get_mcore_model_config(args, hf_config)
                self.assertEqual(config.mtp_num_layers, expected_depth)
                self.assertEqual(config.mtp_shared_weights, expected_shared)
                self.assertEqual(args.mtp_num_layers, expected_depth)
                self.assertEqual(args.mtp_shared_weights, expected_shared)
                self.assertIs(config.hf_config, hf_config)
                self.assertEqual(config.mcore_model_type, meta.model_type)
                self.assertEqual(args.megatron_extra_kwargs, extra)

    def test_teacher_mtp_does_not_mutate_student(self):
        from swift.megatron.trainers import rollout_mixin

        student_meta, teacher_meta = object(), object()
        extra = {'mtp_num_layers': 4, 'mtp_shared_weights': True, 'mtp_use_repeated_layer': True, 'other': 7}
        args = SimpleNamespace(
            model_dir='student',
            teacher_model_dir='teacher',
            megatron_model_meta=student_meta,
            teacher_megatron_model_meta=teacher_meta,
            mtp_num_layers=3,
            mtp_shared_weights=True,
            megatron_extra_kwargs=extra,
            use_cpu_initialization=True,
        )
        trainer = SimpleNamespace(
            args=args,
            use_teacher_api=False,
            _is_self_distillation=False,
            _teacher_use_disable_adapter=False,
            offload_teacher_model=False)
        model = unittest.mock.Mock(config=SimpleNamespace(bridge=unittest.mock.Mock()))
        with patch.object(
                rollout_mixin.AutoConfig, 'from_pretrained', return_value=object()), patch.object(
                    rollout_mixin, 'get_mcore_model', return_value=[model]) as build:
            rollout_mixin.MegatronRolloutMixin._load_teacher_model(trainer)
        teacher_args = build.call_args.args[0]
        self.assertIsNot(teacher_args, args)
        self.assertEqual(teacher_args.model_dir, 'teacher')
        self.assertIs(teacher_args.megatron_model_meta, teacher_meta)
        self.assertFalse(teacher_args.mtp_num_layers)
        self.assertFalse(teacher_args.mtp_shared_weights)
        self.assertEqual(teacher_args.megatron_extra_kwargs, {'other': 7})
        self.assertEqual((args.mtp_num_layers, args.mtp_shared_weights), (3, True))
        self.assertIs(args.megatron_model_meta, student_meta)
        self.assertEqual(args.model_dir, 'student')
        self.assertEqual(extra, {
            'mtp_num_layers': 4,
            'mtp_shared_weights': True,
            'mtp_use_repeated_layer': True,
            'other': 7
        })
        model.config.bridge.load_weights.assert_called_once_with([model], 'teacher')

    def test_hf_mtp_num_layers_records_physical_sets(self):
        from swift.megatron.utils import get_hf_mtp_num_layers

        self.assertEqual(get_hf_mtp_num_layers(3, repeated=True), 1)
        self.assertEqual(get_hf_mtp_num_layers(3, repeated=False), 3)
        self.assertEqual(get_hf_mtp_num_layers(1, repeated=True), 1)
        self.assertIsNone(get_hf_mtp_num_layers(None, repeated=True))
        self.assertEqual(get_hf_mtp_num_layers(0, repeated=False), 0)

    def test_mtp_fallback_without_resolver(self):
        with patch.object(utils, 'resolve_mtp_config', None):
            self.assertEqual(utils._resolve_mtp_values('auto', None, None, 'unused', None), (None, False))
            self.assertEqual(utils._resolve_mtp_values(3, None, None, 'unused', None), (3, False))
            self.assertEqual(utils._resolve_mtp_values(3, True, None, 'unused', None), (3, True))

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
