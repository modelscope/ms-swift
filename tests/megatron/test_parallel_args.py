# Copyright (c) ModelScope Contributors. All rights reserved.
import ast
import importlib.util
import itertools
import os
import sys
import types
import unittest
from contextlib import contextmanager
from dataclasses import field, make_dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ARGUMENTS = Path(__file__).resolve().parents[2] / 'swift' / 'megatron' / 'arguments'
spec = importlib.util.spec_from_file_location('_parallel_args', ARGUMENTS / 'parallel_args.py')
parallel_args = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parallel_args)


def make_args(**overrides):
    values = dict(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        virtual_pipeline_model_parallel_size=None)
    values.update(overrides)
    return SimpleNamespace(**values)


class TestParallelArgs(unittest.TestCase):

    def test_defaults(self):
        parallel_args.validate_parallel_args(make_args(), 1)
        parallel_args.validate_parallel_args(make_args(), 8)

    def test_unspecified_expert_tp_uses_megatron_fallback(self):
        args = make_args(tensor_model_parallel_size=2, expert_tensor_parallel_size=None, expert_model_parallel_size=4)
        before = vars(args).copy()
        parallel_args.validate_parallel_args(args, 8)
        self.assertEqual(vars(args), before)
        args.tensor_model_parallel_size = 4
        with self.assertRaisesRegex(ValueError, 'None resolves to tensor_model_parallel_size'):
            parallel_args.validate_parallel_args(args, 8)

    def test_derived_and_virtual_sizes_are_left_to_existing_flow(self):
        for name in ('data_parallel_size', 'expert_data_parallel_size', 'virtual_pipeline_model_parallel_size'):
            for value in (None, 0, -1, 'placeholder'):
                with self.subTest(name=name, value=value):
                    args = make_args(**{name: value})
                    before = vars(args).copy()
                    parallel_args.validate_parallel_args(args, 8)
                    self.assertEqual(vars(args), before)

    def test_no_new_python_type_or_sign_policy(self):

        class ParallelSize(int):
            pass

        # Other backend stages may reject raw inputs; this preflight must not
        # invent a restriction beyond the two checks it brings forward.
        for value in (True, ParallelSize(2), -1, 1.0):
            with self.subTest(value=value):
                parallel_args.validate_parallel_args(make_args(tensor_model_parallel_size=value), 8)

    def test_error_explains_failed_rank_partition(self):
        with self.assertRaisesRegex(ValueError, r'Remainder: 8 % 3 = 2.*dense parallel groups'):
            parallel_args.validate_parallel_args(make_args(tensor_model_parallel_size=3), 8)
        with self.assertRaisesRegex(ValueError, r'Remainder: 8 % 3 = 2.*expert parallel groups'):
            parallel_args.validate_parallel_args(make_args(expert_model_parallel_size=3), 8)

    def test_backend_arithmetic_errors_explain_values(self):
        names = ('tensor_model_parallel_size', 'pipeline_model_parallel_size', 'context_parallel_size',
                 'expert_model_parallel_size', 'expert_tensor_parallel_size')
        for name in names:
            for value in (0, 'invalid'):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    parallel_args.validate_parallel_args(make_args(**{name: value}), 8)
        with self.assertRaisesRegex(ValueError, 'Original error:'):
            parallel_args.validate_parallel_args(make_args(tensor_model_parallel_size=None), 8)

    def test_no_new_world_size_value_policy(self):
        # These raw values are not a claim of valid training. Do not add a new
        # rejection where Megatron's two arithmetic checks themselves do not fail.
        for value in (0, -8, 8.0, True):
            with self.subTest(world_size=value):
                parallel_args.validate_parallel_args(make_args(), value)

    def test_dense_divisibility(self):
        for tp, pp, cp in ((3, 1, 1), (1, 3, 1), (1, 1, 3), (2, 2, 3), (16, 1, 1)):
            with self.subTest(
                    tp=tp, pp=pp, cp=cp), self.assertRaisesRegex(
                        ValueError, r'world_size \(8\).*tensor_model_parallel_size.*context_parallel_size'):
                parallel_args.validate_parallel_args(
                    make_args(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp, context_parallel_size=cp),
                    8)

    def test_expert_divisibility(self):
        for etp, ep, pp in ((1, 3, 1), (3, 1, 1), (4, 4, 1), (2, 4, 2)):
            with self.subTest(
                    etp=etp, ep=ep,
                    pp=pp), self.assertRaisesRegex(ValueError,
                                                   r'expert_tensor_parallel_size.*expert_model_parallel_size'):
                parallel_args.validate_parallel_args(
                    make_args(
                        expert_tensor_parallel_size=etp, expert_model_parallel_size=ep,
                        pipeline_model_parallel_size=pp), 8)

    def test_separate_dense_and_expert_grids(self):
        args = make_args(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            context_parallel_size=2,
            expert_model_parallel_size=4)
        # Dense DP=2, expert DP=2: EP need not divide dense DP, nor multiply CP.
        parallel_args.validate_parallel_args(args, 16)
        # A different expert TP is valid; VPP is not a physical world-size factor.
        args.expert_tensor_parallel_size = 2
        args.virtual_pipeline_model_parallel_size = 3
        parallel_args.validate_parallel_args(args, 16)

    def test_unknown_world_size_defers_to_existing_flow(self):
        parallel_args.validate_parallel_args(make_args(tensor_model_parallel_size=3))
        parallel_args.validate_parallel_args(make_args(expert_model_parallel_size=0))

    def test_validation_does_not_modify_args(self):
        args = make_args(tensor_model_parallel_size=2)
        before = vars(args).copy()
        parallel_args.validate_parallel_args(args, 8)
        self.assertEqual(vars(args), before)

    def test_world_size_from_environment(self):
        for env, expected in (({}, 1), ({
                'WORLD_SIZE': '16'
        }, 16), ({
                '_PATCH_WORLD_SIZE': '8'
        }, 8), ({
                'WORLD_SIZE': '16',
                '_PATCH_WORLD_SIZE': '8'
        }, 16)):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                self.assertEqual(parallel_args.get_parallel_world_size(), expected)

    def test_invalid_environment_world_size(self):
        for name in ('WORLD_SIZE', '_PATCH_WORLD_SIZE'):
            for value in ('abc', '1.5', 'true'):
                with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}, clear=True), \
                        self.assertRaisesRegex(ValueError, name):
                    parallel_args.get_parallel_world_size()

    def test_empty_environment_uses_swift_fallback(self):
        # Match swift.utils.env.get_dist_setting: empty values behave as unset.
        cases = (({
            'WORLD_SIZE': '',
            '_PATCH_WORLD_SIZE': '8'
        }, 8), ({
            'WORLD_SIZE': ''
        }, 1), ({
            '_PATCH_WORLD_SIZE': ''
        }, 1), ({
            'WORLD_SIZE': '',
            '_PATCH_WORLD_SIZE': '',
            'RANK': '0'
        }, None))
        for env, expected in cases:
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                self.assertEqual(parallel_args.get_parallel_world_size(), expected)

    def test_launcher_without_world_size_is_unknown(self):
        for name in ('RANK', 'LOCAL_RANK', 'NPROC_PER_NODE', 'NNODES', 'LOCAL_WORLD_SIZE', 'LOCAL_SIZE'):
            with self.subTest(name=name), patch.dict(os.environ, {name: '2'}, clear=True):
                self.assertIsNone(parallel_args.get_parallel_world_size())

    def test_live_world_size_takes_precedence(self):
        distributed = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, get_world_size=lambda: 16)
        with patch.dict(os.environ, {'WORLD_SIZE': '8'}, clear=True):
            self.assertEqual(parallel_args.get_parallel_world_size(distributed=distributed), 16)
            self.assertIsNone(parallel_args.get_parallel_world_size(True, distributed))

    def test_integer_environment_spellings_match_existing_parser(self):
        for raw in ('8', ' 8 ', '+8', '08'):
            with self.subTest(value=raw), patch.dict(os.environ, {'WORLD_SIZE': raw}, clear=True):
                world = parallel_args.get_parallel_world_size()
                self.assertEqual(world, 8)
                parallel_args.validate_parallel_args(make_args(tensor_model_parallel_size=8), world)

    def test_sft_rejects_before_model_or_distributed_initialization(self):
        # Load the real SFT entry with mocked dependencies, without installing torch/Megatron.
        utils = types.ModuleType('swift.utils')
        for name in ('add_version_to_work_dir', 'get_logger', 'init_process_group', 'is_last_rank', 'to_abspath'):
            setattr(utils, name, Mock())
        base = types.ModuleType('_parallel_args_test.megatron_base_args')
        parent_init = Mock(side_effect=RuntimeError('entered parent initialization'))
        base.MegatronBaseArguments = type('MegatronBaseArguments', (), {'__post_init__': parent_init})
        torch = types.ModuleType('torch')
        torch.distributed = SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False)
        modules = {
            'swift': types.ModuleType('swift'),
            'swift.utils': utils,
            'torch': torch,
            '_parallel_args_test': types.ModuleType('_parallel_args_test'),
            '_parallel_args_test.megatron_base_args': base,
            '_parallel_args_test.parallel_args': parallel_args,
        }
        spec = importlib.util.spec_from_file_location('_parallel_args_test.sft_args', ARGUMENTS / 'sft_args.py')
        module = importlib.util.module_from_spec(spec)
        modules[spec.name] = module
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {'WORLD_SIZE': '8'}, clear=True):
            spec.loader.exec_module(module)
            args = module.MegatronSftArguments.__new__(module.MegatronSftArguments)
            args.skip_megatron_init = False
            args.mcore_model = None
            for overrides in ({
                    'tensor_model_parallel_size': 3
            }, {
                    'expert_model_parallel_size': 3
            }, {
                    'context_parallel_size': 0
            }, {
                    'pipeline_model_parallel_size': 1.5
            }):
                vars(args).update(vars(make_args(**overrides)))
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    args.__post_init__()
            parent_init.assert_not_called()
            utils.to_abspath.assert_not_called()
            utils.init_process_group.assert_not_called()
            vars(args).update(vars(make_args(tensor_model_parallel_size=2)))
            with self.assertRaisesRegex(RuntimeError, 'entered parent initialization'):
                args.__post_init__()
            parent_init.assert_called_once()


@contextmanager
def training_entries(env=None, live_world_size=None):
    """Run real argument constructors while replacing model/distributed dependencies.

    Read the parallel defaults from the real Megatron dataclass, not a second
    hand-maintained set. Model setup and communication are mocks, not hardware tests.
    """
    tree = ast.parse((ARGUMENTS / 'megatron_args.py').read_text(encoding='utf-8'))
    names = set(vars(make_args()))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MegatronArguments')
    defaults = {
        node.target.id: ast.literal_eval(node.value)
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in names
    }
    assert defaults.keys() == names
    parent_init, backend_init, output_init = Mock(), Mock(), Mock()

    def post_init(self):
        parent_init(self)
        if not self.skip_megatron_init:
            self._init_megatron_args()

    fields = [(name, object, value) for name, value in defaults.items()]
    fields += [
        ('skip_megatron_init', bool, False),
        ('mcore_model', object, None),
        ('model', object, 'local-model'),
        ('perform_initialization', bool, False),
        ('dataset', list, field(default_factory=lambda: ['local-data'])),
        ('cached_dataset', list, field(default_factory=list)),
        ('tensorboard_dir', object, None),
        ('output_dir', str, '/mock/output'),
        ('resume_from_checkpoint', object, None),
        ('pipeline_model_parallel_layout', object, None),
        ('sequence_parallel', bool, False),
        ('use_megatron_fsdp', bool, False),
    ]
    base_cls = make_dataclass(
        'MegatronBaseArguments', fields, namespace={
            '__post_init__': post_init,
            '_init_megatron_args': backend_init
        })
    base = types.ModuleType('_parallel_regression.megatron_base_args')
    base.MegatronBaseArguments = base_cls
    utils = types.ModuleType('swift.utils')
    for name in ('add_version_to_work_dir', 'get_logger', 'init_process_group', 'is_last_rank'):
        setattr(utils, name, Mock())
    utils.to_abspath = Mock(side_effect=lambda value, **kwargs: value)
    utils.HfConfigFactory = SimpleNamespace(to_torch_dtype=lambda value: value)
    torch = types.ModuleType('torch')
    torch.distributed = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: live_world_size is not None,
        get_world_size=lambda: live_world_size)
    model_args = types.ModuleType('_parallel_regression.megatron_args')
    model_args.MegatronArguments = SimpleNamespace(load_args_config=lambda ckpt: {})
    modules = {
        'swift': types.ModuleType('swift'),
        'swift.utils': utils,
        'torch': torch,
        '_parallel_regression': types.ModuleType('_parallel_regression'),
        '_parallel_regression.megatron_base_args': base,
        '_parallel_regression.megatron_args': model_args,
        '_parallel_regression.parallel_args': parallel_args,
    }
    entries = {}
    with patch.dict(sys.modules, modules), patch.dict(os.environ, env or {}, clear=True):
        for filename, class_name in (
            ('sft_args', 'MegatronSftArguments'),
            ('pretrain_args', 'MegatronPretrainArguments'),
            ('rlhf_args', 'MegatronRLHFArguments'),
            ('export_args', 'MegatronExportArguments'),
        ):
            name = f'_parallel_regression.{filename}'
            spec = importlib.util.spec_from_file_location(name, ARGUMENTS / f'{filename}.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            entries[filename] = getattr(module, class_name)
        with patch.object(entries['sft_args'], '_init_output_dir', output_init):
            yield entries, SimpleNamespace(
                parent=parent_init, backend=backend_init, output=output_init, utils=utils, defaults=defaults)


class TestNormalTrainingRegression(unittest.TestCase):

    def test_real_defaults_all_training_entries(self):
        with training_entries() as (entries, calls):
            for name in ('sft_args', 'pretrain_args', 'rlhf_args'):
                with self.subTest(entry=name):
                    args = entries[name]()
                    for key, value in calls.defaults.items():
                        self.assertEqual(getattr(args, key), value)
            self.assertEqual(calls.backend.call_count, 3)

    def test_valid_dense_and_expert_configurations_all_entries(self):
        cases = (
            (8, dict(tensor_model_parallel_size=8)),
            (8, dict(pipeline_model_parallel_size=8)),
            (8, dict(context_parallel_size=8)),
            (8, dict(tensor_model_parallel_size=2, pipeline_model_parallel_size=2, context_parallel_size=2)),
            (16,
             dict(
                 tensor_model_parallel_size=2,
                 pipeline_model_parallel_size=2,
                 context_parallel_size=2,
                 expert_model_parallel_size=4)),
            (16,
             dict(
                 tensor_model_parallel_size=2,
                 pipeline_model_parallel_size=2,
                 expert_tensor_parallel_size=4,
                 expert_model_parallel_size=2)),
            (8, dict(pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=3)),
            (8, dict(pipeline_model_parallel_size=2, pipeline_model_parallel_layout='t|t')),
            (8, dict(sequence_parallel=True)),
            (8, dict(use_megatron_fsdp=True)),
            (8, dict(tensor_model_parallel_size=2, expert_tensor_parallel_size=None, expert_model_parallel_size=4)),
            (8,
             dict(
                 pipeline_model_parallel_size=2,
                 pipeline_model_parallel_layout='tt|tt',
                 virtual_pipeline_model_parallel_size=0)),
        )
        for world, values in cases:
            with training_entries({'WORLD_SIZE': str(world)}) as (entries, calls):
                for name in ('sft_args', 'pretrain_args', 'rlhf_args'):
                    with self.subTest(world=world, values=values, entry=name):
                        args = entries[name](**values)
                        for key, value in values.items():
                            self.assertEqual(getattr(args, key), value)
                self.assertEqual(calls.backend.call_count, 3)

    def test_rlhf_algorithms_and_resume_settings(self):
        with training_entries({'WORLD_SIZE': '8'}) as (entries, calls):
            for algorithm in ('dpo', 'kto', 'grpo', 'gkd', 'rm'):
                for load_args in (True, False):
                    with self.subTest(algorithm=algorithm, load_args=load_args):
                        args = entries['rlhf_args'](
                            rlhf_type=algorithm,
                            tensor_model_parallel_size=2,
                            resume_from_checkpoint='/mock/checkpoint-100',
                            load_args=load_args)
                        self.assertEqual(args.tensor_model_parallel_size, 2)
                        self.assertEqual(args.resume_from_checkpoint, '/mock/checkpoint-100')
                        self.assertEqual(args.load_args, load_args)
            self.assertEqual(calls.backend.call_count, 10)

    def test_unknown_driver_world_size_does_not_reject(self):
        for env in ({
                'NPROC_PER_NODE': '8'
        }, {
                'NNODES': '2'
        }, {
                'RANK': '0'
        }, {
                'LOCAL_RANK': '0'
        }, {
                'LOCAL_WORLD_SIZE': '8'
        }, {
                'LOCAL_SIZE': '8'
        }):
            with training_entries(env) as (entries, calls):
                for name in ('sft_args', 'pretrain_args', 'rlhf_args'):
                    with self.subTest(env=env, entry=name):
                        entries[name](tensor_model_parallel_size=3)
                self.assertEqual(calls.backend.call_count, 3)

    def test_ray_driver_does_not_add_an_argument_policy(self):
        with training_entries({'WORLD_SIZE': '1'}) as (entries, calls):
            for name in ('sft_args', 'pretrain_args', 'rlhf_args'):
                with self.subTest(entry=name):
                    entries[name](skip_megatron_init=True, tensor_model_parallel_size=8, expert_model_parallel_size=8)
            calls.backend.assert_not_called()
            calls.output.assert_not_called()

    def test_existing_process_group_overrides_stale_env(self):
        with training_entries({'WORLD_SIZE': '1'}, live_world_size=16) as (entries, calls):
            for name in ('sft_args', 'pretrain_args', 'rlhf_args'):
                entries[name](tensor_model_parallel_size=8)
            self.assertEqual(calls.backend.call_count, 3)

    def test_initialized_world_size_ignores_invalid_stale_env(self):
        with training_entries({'WORLD_SIZE': 'stale'}, live_world_size=8) as (entries, calls):
            for name in ('sft_args', 'pretrain_args', 'rlhf_args'):
                entries[name](tensor_model_parallel_size=8)
            self.assertEqual(calls.backend.call_count, 3)

    def test_worker_correctly_rechecks_after_driver(self):
        with training_entries() as (entries, calls):
            entries['rlhf_args'](skip_megatron_init=True, tensor_model_parallel_size=3)
            calls.backend.assert_not_called()
        with training_entries({'WORLD_SIZE': '8'}) as (entries, calls):
            with self.assertRaisesRegex(ValueError, 'world_size'):
                entries['rlhf_args'](tensor_model_parallel_size=3)
            calls.backend.assert_not_called()

    def test_empty_world_size_fallback_all_entries(self):
        with training_entries({'WORLD_SIZE': '', '_PATCH_WORLD_SIZE': '8'}) as (entries, calls):
            for name in ('sft_args', 'pretrain_args', 'rlhf_args'):
                entries[name](tensor_model_parallel_size=8)
            self.assertEqual(calls.backend.call_count, 3)

    def test_invalid_configs_stop_all_entries_before_initialization(self):
        for values in ({
                'tensor_model_parallel_size': 3
        }, {
                'expert_model_parallel_size': 3
        }, {
                'context_parallel_size': 0
        }, {
                'pipeline_model_parallel_size': 0
        }, {
                'expert_tensor_parallel_size': 2.5
        }, {
                'tensor_model_parallel_size': 'invalid'
        }):
            with training_entries({'WORLD_SIZE': '8'}) as (entries, calls):
                for name in ('sft_args', 'pretrain_args', 'rlhf_args'):
                    with self.subTest(values=values, entry=name), self.assertRaises(ValueError):
                        entries[name](**values)
                calls.parent.assert_not_called()
                calls.output.assert_not_called()
                calls.backend.assert_not_called()
                calls.utils.init_process_group.assert_not_called()
                calls.utils.to_abspath.assert_not_called()

    def test_unparseable_world_size_stops_all_entries(self):
        for value in ('abc', '1.5'):
            with training_entries({'WORLD_SIZE': value}) as (entries, calls):
                for name in ('sft_args', 'pretrain_args', 'rlhf_args'):
                    with self.subTest(world=value, entry=name), self.assertRaises(ValueError):
                        entries[name]()
                calls.parent.assert_not_called()
                calls.output.assert_not_called()
                calls.backend.assert_not_called()

    def test_export_does_not_call_new_validator(self):
        with training_entries({'WORLD_SIZE': '1'}) as (entries, calls):
            with patch.object(entries['export_args'], '_init_megatron_args'), \
                    patch.object(parallel_args, 'validate_parallel_args', side_effect=AssertionError('must not run')):
                entries['export_args'](tensor_model_parallel_size=8)


class TestParallelConfigurationMatrix(unittest.TestCase):
    results = {}

    def test_bounded_configuration_matrix(self):
        # Enumerate a finite test domain, not a claim to test every possible model.
        worlds = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128)
        degrees = (1, 2, 3, 4, 6, 8)
        divisors = {w: {n for n in range(1, w + 1) if w % n == 0} for w in worlds}
        accepted, rejected = 0, 0
        for world in worlds:
            for tp, pp, cp, etp, ep in itertools.product(degrees, repeat=5):
                expected = tp * pp * cp in divisors[world] and etp * ep * pp in divisors[world]
                args = make_args(
                    tensor_model_parallel_size=tp,
                    pipeline_model_parallel_size=pp,
                    context_parallel_size=cp,
                    expert_tensor_parallel_size=etp,
                    expert_model_parallel_size=ep)
                try:
                    parallel_args.validate_parallel_args(args, world)
                    actual = True
                    accepted += 1
                except ValueError:
                    actual = False
                    rejected += 1
                if actual != expected:
                    self.fail(f'Unexpected decision: W={world}, TP={tp}, PP={pp}, CP={cp}, ETP={etp}, EP={ep}')
        type(self).results = dict(total=accepted + rejected, accepted=accepted, rejected=rejected)
        print(f'Configuration matrix: {type(self).results}')


if __name__ == '__main__':
    unittest.main()
