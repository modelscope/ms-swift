# Copyright (c) ModelScope Contributors. All rights reserved.
import types
import unittest
from unittest import mock

import torch
import torch.distributed as dist

try:
    from megatron.core.optimizer import (ParamKey, ParamPredicate, _get_param_groups, get_standard_config_overrides)
    from megatron.core.optimizer.layer_wise_optimizer import is_managed_by_layer_wise_optimizer

    from swift.megatron.arguments.megatron_args import MegatronArguments
    from swift.megatron.trainers.base import BaseMegatronTrainer
    MEGATRON_UNAVAILABLE = None
except Exception as e:
    # CI does not install megatron, and a mismatched mcore dependency can fail with anything from
    # ImportError to AttributeError, so any failure here means the tests below cannot run.
    MEGATRON_UNAVAILABLE = f'Megatron dependencies not available: {e}'


def _make_args(**kwargs):
    args = types.SimpleNamespace(
        optimizer='muon',
        muon_lr=None,
        muon_min_lr=None,
        muon_scalar_optimizer='adam',
        lr=1e-4,
        min_lr=1e-5,
        apply_wd_to_qk_layernorm=False,
        vit_lr=None,
        aligner_lr=None,
    )
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def _make_config(args):
    """The handful of `OptimizerConfig` fields the code under test reads."""
    return types.SimpleNamespace(
        optimizer=args.optimizer,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=0.1,
        decoupled_lr=None,
        decoupled_min_lr=None,
        apply_wd_to_qk_layernorm=args.apply_wd_to_qk_layernorm,
        use_precision_aware_optimizer=False,
    )


def _get_overrides(args):
    trainer = types.SimpleNamespace(args=args)
    return BaseMegatronTrainer._get_muon_config_overrides(trainer, _make_config(args))


def _muon_param_key():
    return ParamKey(predicate=ParamPredicate(name='muon_managed_matrix', fn=is_managed_by_layer_wise_optimizer))


@unittest.skipIf(MEGATRON_UNAVAILABLE is not None, MEGATRON_UNAVAILABLE or '')
class TestMuonConfigOverrides(unittest.TestCase):
    """`_get_muon_config_overrides` turns `--muon_lr`/`--muon_min_lr` into an mcore parameter override."""

    def test_no_override_without_the_arguments(self):
        """Passing nothing lets mcore install its own standard overrides."""
        self.assertEqual(_get_overrides(_make_args()), {})

    def test_only_the_arguments_that_were_set_are_overridden(self):
        for kwargs, expected in [
            ({
                'muon_lr': 1e-2
            }, {
                'max_lr': 1e-2
            }),
            ({
                'muon_min_lr': 1e-3
            }, {
                'min_lr': 1e-3
            }),
            ({
                'muon_lr': 1e-2,
                'muon_min_lr': 1e-3
            }, {
                'max_lr': 1e-2,
                'min_lr': 1e-3
            }),
        ]:
            with self.subTest(**kwargs):
                overrides = _get_overrides(_make_args(**kwargs))['config_overrides']
                # Keying on mcore's own routing predicate is what keeps the override and the split in sync.
                self.assertEqual(overrides[_muon_param_key()], expected)

    def test_mcores_standard_overrides_are_kept(self):
        """mcore drops its standard overrides once the caller passes any, so they have to be carried over.

        Losing them would silently start applying weight decay to biases and LayerNorm weights, and would
        drop `--apply_wd_to_qk_layernorm`, which mcore implements through those very overrides.
        """
        for apply_wd_to_qk_layernorm in [False, True]:
            with self.subTest(apply_wd_to_qk_layernorm=apply_wd_to_qk_layernorm):
                args = _make_args(muon_lr=1e-2, apply_wd_to_qk_layernorm=apply_wd_to_qk_layernorm)
                overrides = _get_overrides(args)['config_overrides']
                standard = get_standard_config_overrides(_make_config(args))
                self.assertTrue(standard)
                for param_key, override in standard.items():
                    self.assertEqual(overrides[param_key], override)

    def test_the_override_is_keyed_on_mcores_own_routing_predicate(self):
        """`ParamPredicate` compares by name only, so assert the predicate really is mcore's routing rule."""
        overrides = _get_overrides(_make_args(muon_lr=1e-2))['config_overrides']
        param_key, = [key for key in overrides if key == _muon_param_key()]
        self.assertIs(param_key.predicate.fn, is_managed_by_layer_wise_optimizer)


@unittest.skipIf(MEGATRON_UNAVAILABLE is not None, MEGATRON_UNAVAILABLE or '')
class TestMuonParamGroups(unittest.TestCase):
    """The override has to reach the parameter groups mcore hands to the optimizers."""

    @classmethod
    def setUpClass(cls):
        # mcore aligns the param groups across ranks with an `all_gather_object`, so a process group has to
        # exist. `HashStore` keeps it in-process: no ports, no files, nothing to clean up but the group itself.
        cls._owns_process_group = not dist.is_initialized()
        if cls._owns_process_group:
            dist.init_process_group(backend='gloo', store=dist.HashStore(), rank=0, world_size=1)

    @classmethod
    def tearDownClass(cls):
        if cls._owns_process_group:
            dist.destroy_process_group()

    @staticmethod
    def _param(shape, is_embedding_or_output=False):
        param = torch.nn.Parameter(torch.empty(shape, device='meta'))
        if is_embedding_or_output:
            param.is_embedding_or_output_parameter = True
        return param

    def _build_param_groups(self, args):
        """Run mcore's real `_get_param_groups` over a model whose parameters cover both routing branches.

        Returns the Muon groups and the scalar-optimizer groups, split the way mcore splits them.
        """
        params = {
            'decoder.layers.0.mlp.linear_fc1.weight': self._param((4, 4)),  # Muon
            'decoder.layers.0.self_attention.linear_qkv.bias': self._param((4, )),  # scalar optimizer
            'embedding.word_embeddings.weight': self._param((4, 4), is_embedding_or_output=True),  # scalar optimizer
        }
        groups = self._param_groups_for(args, params)
        # mcore reads the optimizer of a group with `group.get('optimizer', <the emerging optimizer>)`.
        muon_groups = [group for group in groups if 'optimizer' not in group]
        scalar_groups = [group for group in groups if group.get('optimizer') == args.muon_scalar_optimizer]
        self.assertEqual(len(muon_groups) + len(scalar_groups), len(groups))
        return muon_groups, scalar_groups

    def _param_groups_for(self, args, params):
        model_chunk = mock.MagicMock()
        model_chunk.named_parameters.return_value = list(params.items())
        config = _make_config(args)
        # Mirror mcore: fall back to the standard overrides when the caller passes none, then add the
        # routing override that sends the non-matrix parameters to the scalar optimizer.
        config_overrides = _get_overrides(args).get('config_overrides') or get_standard_config_overrides(config)
        scalar_key = ParamKey(
            predicate=ParamPredicate(
                name='nonlinear_or_embedding', fn=lambda p: not is_managed_by_layer_wise_optimizer(p)))
        config_overrides = dict(config_overrides)
        config_overrides[scalar_key] = {'optimizer': args.muon_scalar_optimizer}
        return _get_param_groups([model_chunk], config, config_overrides)

    def test_muon_and_the_scalar_optimizer_get_separate_learning_rates(self):
        args = _make_args(muon_lr=1e-2, muon_min_lr=1e-3)
        muon_groups, scalar_groups = self._build_param_groups(args)
        self.assertTrue(muon_groups and scalar_groups)
        for group in muon_groups:
            self.assertEqual((group['max_lr'], group['min_lr']), (1e-2, 1e-3))
            # mcore flags a group whose learning rate was overridden; it drives the reported learning rate.
            self.assertFalse(group['default_config'])
        for group in scalar_groups:
            self.assertEqual((group['max_lr'], group['min_lr']), (args.lr, args.min_lr))
            self.assertTrue(group['default_config'])

    def test_without_muon_lr_both_optimizers_share_the_base_learning_rate(self):
        args = _make_args()
        muon_groups, scalar_groups = self._build_param_groups(args)
        self.assertTrue(muon_groups and scalar_groups)
        for group in muon_groups + scalar_groups:
            self.assertEqual((group['max_lr'], group['min_lr']), (args.lr, args.min_lr))
            self.assertTrue(group['default_config'])

    def test_the_weight_decay_skip_survives_the_override(self):
        """End of the chain for the standard overrides: the bias still has to land in a `wd_mult=0.0` group."""
        muon_groups, scalar_groups = self._build_param_groups(_make_args(muon_lr=1e-2))
        self.assertIn(0., [group['wd_mult'] for group in scalar_groups])
        self.assertEqual([group['wd_mult'] for group in muon_groups], [1.])

    def test_apply_wd_to_qk_layernorm_reaches_the_param_groups(self):
        """Under Muon this is mcore's job, so its `s1_not_qkln` rule has to survive into the groups.

        The qk layernorm weights are 1-D, so they only keep their weight decay if that rule applied.
        """
        params = {
            'decoder.layers.0.self_attention.q_layernorm.weight': self._param((4, )),
            'decoder.layers.0.self_attention.linear_qkv.layer_norm_weight': self._param((4, )),
        }
        for apply_wd_to_qk_layernorm, expected in [(False, {0.}), (True, {0., 1.})]:
            with self.subTest(apply_wd_to_qk_layernorm=apply_wd_to_qk_layernorm):
                args = _make_args(muon_lr=1e-2, apply_wd_to_qk_layernorm=apply_wd_to_qk_layernorm)
                groups = self._param_groups_for(args, params)
                self.assertEqual({group['wd_mult'] for group in groups if group['params']}, expected)

    def test_the_scheduler_reads_the_overridden_learning_rate(self):
        """`OptimizerParamScheduler` is the consumer: it must see the per-group values, not just `--lr`."""
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
        args = _make_args(muon_lr=1e-2, muon_min_lr=1e-3)
        muon_groups, scalar_groups = self._build_param_groups(args)
        scheduler = OptimizerParamScheduler(
            optimizer=mock.MagicMock(param_groups=muon_groups + scalar_groups),
            init_lr=0.,
            max_lr=args.lr,
            min_lr=args.min_lr,
            lr_warmup_steps=0,
            lr_decay_steps=10,
            lr_decay_style='constant',
            start_wd=0.1,
            end_wd=0.1,
            wd_incr_steps=10,
            wd_incr_style='constant',
        )
        self.assertEqual([scheduler.get_lr(group) for group in muon_groups], [args.muon_lr])
        self.assertEqual({scheduler.get_lr(group) for group in scalar_groups}, {args.lr})


@unittest.skipIf(MEGATRON_UNAVAILABLE is not None, MEGATRON_UNAVAILABLE or '')
class TestOwnParamGroupsGate(unittest.TestCase):
    """Replacing mcore's parameter grouping costs Muon its routing, so it has to stay as narrow as possible."""

    @staticmethod
    def _needs(optimizer='muon', **kwargs):
        return BaseMegatronTrainer._needs_own_param_groups(_make_args(**kwargs), optimizer)

    def test_muon_keeps_mcores_param_groups_by_default(self):
        self.assertFalse(self._needs())

    def test_apply_wd_to_qk_layernorm_is_left_to_mcore_under_muon(self):
        """mcore implements it in `get_standard_config_overrides`, so the replacement is not needed."""
        self.assertFalse(self._needs(apply_wd_to_qk_layernorm=True))
        self.assertFalse(self._needs(optimizer='dist_muon', apply_wd_to_qk_layernorm=True))

    def test_other_optimizers_keep_using_swifts_param_groups(self):
        self.assertTrue(self._needs(optimizer='adam', apply_wd_to_qk_layernorm=True))

    def test_the_multimodal_learning_rates_leave_no_choice(self):
        for kwargs in [{'vit_lr': 1e-5}, {'aligner_lr': 1e-5}]:
            with self.subTest(**kwargs):
                self.assertTrue(self._needs(**kwargs))


@unittest.skipIf(MEGATRON_UNAVAILABLE is not None, MEGATRON_UNAVAILABLE or '')
class TestReportedLearningRate(unittest.TestCase):
    """With `--muon_lr` the parameter groups no longer share a learning rate, so one has to be picked."""

    @staticmethod
    def _report(*param_groups):
        trainer = types.SimpleNamespace(optimizer=types.SimpleNamespace(param_groups=list(param_groups)))
        return BaseMegatronTrainer._get_reported_learning_rate(trainer)

    def test_the_overridden_group_is_not_reported(self):
        muon = {'params': ['w'], 'lr': 1e-2, 'default_config': False}
        scalar = {'params': ['b'], 'lr': 1e-4, 'default_config': True}
        self.assertEqual(self._report(muon, scalar), 1e-4)

    def test_groups_without_the_flag_count_as_default(self):
        """swift builds its own parameter groups for `--vit_lr`, and those carry no `default_config`."""
        self.assertEqual(self._report({'params': ['w'], 'lr': 1e-5}), 1e-5)

    def test_empty_groups_are_skipped(self):
        self.assertEqual(self._report({'params': [], 'lr': 1e-2}, {'params': ['b'], 'lr': 1e-4}), 1e-4)
        self.assertIsNone(self._report({'params': [], 'lr': 1e-2}))

    def test_falls_back_when_every_group_was_overridden(self):
        self.assertEqual(self._report({'params': ['w'], 'lr': 1e-2, 'default_config': False}), 1e-2)


@unittest.skipIf(MEGATRON_UNAVAILABLE is not None, MEGATRON_UNAVAILABLE or '')
class TestMuonArgumentValidation(unittest.TestCase):
    """`--muon_lr` only makes sense where mcore actually applies the override."""

    @staticmethod
    def _check(**kwargs):
        args = types.SimpleNamespace(
            optimizer='muon',
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            muon_use_nesterov=False,
            muon_lr=None,
            muon_min_lr=None,
            vit_lr=None,
            aligner_lr=None,
            apply_wd_to_qk_layernorm=False,
        )
        for key, value in kwargs.items():
            setattr(args, key, value)
        MegatronArguments._check_muon(args)
        return args

    def test_muon_is_rejected_alongside_the_arguments_that_drop_the_override(self):
        for kwargs in [{'vit_lr': 1e-5}, {'aligner_lr': 1e-5}]:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError) as cm:
                    self._check(**kwargs)
                self.assertIn(next(iter(kwargs)), str(cm.exception))

    def test_apply_wd_to_qk_layernorm_is_accepted(self):
        """mcore implements it through `config_overrides`, so Muon does not have to give it up."""
        self._check(apply_wd_to_qk_layernorm=True, muon_lr=1e-2)

    def test_muon_learning_rates_are_rejected_for_other_optimizers(self):
        with self.assertRaises(ValueError):
            self._check(optimizer='adam', muon_lr=1e-2)

    def test_a_plain_muon_setup_passes(self):
        args = self._check(muon_lr=1e-2, muon_min_lr=1e-3)
        self.assertFalse(args.use_distributed_optimizer)
