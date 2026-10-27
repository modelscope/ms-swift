# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class RankGroup:

    def __init__(self, rank):
        self._rank = rank

    def rank(self):
        return self._rank

    def size(self):
        return 2


class TestExpertParallelCompat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from swift.megatron import init
        cls.compat = init

    def test_expert_filter_keeps_distinct_experts_and_deduplicates_replicas(self):
        from megatron.core import tensor_parallel

        predicate = tensor_parallel.param_is_not_tensor_parallel_duplicate
        expert = torch.nn.Parameter(torch.ones(1))
        expert.allreduce = False
        dense = torch.nn.Parameter(torch.ones(1))
        self.assertTrue(predicate(expert, tp_group=RankGroup(1), expert_tp_group=RankGroup(0)))
        self.assertFalse(predicate(dense, tp_group=RankGroup(1), expert_tp_group=RankGroup(0)))
        self.assertTrue(predicate(dense, tp_group=RankGroup(0), expert_tp_group=RankGroup(1)))
        self.assertFalse(predicate(expert, tp_group=RankGroup(0), expert_tp_group=RankGroup(1)))
        expert.tensor_model_parallel = True
        self.assertTrue(predicate(expert, tp_group=RankGroup(1), expert_tp_group=RankGroup(1)))

    def test_legacy_expert_filter_uses_global_etp_group(self):
        if not self.compat._legacy_expert_topology():
            self.skipTest('Native MCore callers pass the expert TP group explicitly.')
        from megatron.core import parallel_state, tensor_parallel

        expert = torch.nn.Parameter(torch.ones(1))
        expert.allreduce = False
        predicate = tensor_parallel.param_is_not_tensor_parallel_duplicate
        for rank in (0, 1):
            with self.subTest(expert_tp_rank=rank), \
                    patch.object(parallel_state, 'get_expert_tensor_parallel_group', return_value=RankGroup(rank)):
                self.assertEqual(predicate(expert, tp_group=RankGroup(1 - rank)), rank == 0)

    def test_norm_includes_both_independent_experts(self):
        from megatron.core import tensor_parallel

        params = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
        for p, grad in zip(params, (3., 4.)):
            p.allreduce = False
            p.grad = torch.tensor([grad])
        predicate = tensor_parallel.param_is_not_tensor_parallel_duplicate
        selected = [
            p.grad for rank, p in enumerate(params)
            if predicate(p, tp_group=RankGroup(rank), expert_tp_group=RankGroup(0))
        ]
        norm = torch.stack([g.square().sum() for g in selected]).sum().sqrt()
        self.assertEqual(norm.item(), 5.)
        for p in params:
            p.grad.mul_(1. / norm)
        actual = torch.stack([p.grad.square().sum() for p in params]).sum().sqrt()
        self.assertAlmostEqual(actual.item(), 1., places=6)

    def test_legacy_main_param_copies_expert_metadata(self):
        if not self.compat._legacy_expert_topology():
            self.skipTest('Native MCore copies optimizer metadata with a separate helper.')
        from megatron.core import tensor_parallel

        model_param = torch.nn.Parameter(torch.ones(2, 2))
        model_param.allreduce = False
        model_param.tensor_model_parallel = True
        model_param.partition_dim = 0
        model_param.partition_stride = 1
        main_param = model_param.detach().reshape(-1).clone()
        tensor_parallel.copy_tensor_model_parallel_attributes(main_param, model_param)
        self.assertIs(main_param.allreduce, False)
        self.assertIs(main_param.tensor_model_parallel, True)
        self.assertEqual(main_param.partition_dim, 0)

    def test_uninitialized_native_tp_weights_keep_all_shards(self):
        from megatron.core import ModelParallelConfig, tensor_parallel
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding

        config = ModelParallelConfig(
            tensor_model_parallel_size=2, use_cpu_initialization=True, perform_initialization=False)
        group = RankGroup(1)
        column = ColumnParallelLinear(4, 8, config=config, init_method=None, bias=True, stride=2, tp_group=group)
        row = RowParallelLinear(
            4,
            8,
            config=config,
            init_method=None,
            bias=True,
            input_is_parallel=True,
            skip_bias_add=True,
            tp_group=group)
        embedding = VocabParallelEmbedding(8, 4, config=config, init_method=None, tp_group=group)
        model = torch.nn.ModuleList([embedding, column, row])
        self.compat.prepare_expert_parallel_parameters(model)
        predicate = tensor_parallel.param_is_not_tensor_parallel_duplicate
        for module, dim in ((embedding, 0), (column, 0), (row, 1)):
            self.assertTrue(predicate(module.weight, tp_group=group))
            self.assertEqual(module.weight.partition_dim, dim)
            self.assertEqual(module.weight.partition_stride, 2 if module is column else 1)
        self.assertTrue(predicate(column.bias, tp_group=group))
        self.assertFalse(predicate(row.bias, tp_group=group))
        self.compat.prepare_expert_parallel_parameters(model)
        self.assertEqual(row.weight.partition_dim, 1)

    def test_duplicated_te_parameters_are_replicas(self):
        from megatron.core import tensor_parallel

        class LocalLinear(torch.nn.Module):

            def __init__(self, parallel_mode, is_expert=False):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2, 2))
                self.weight.tensor_model_parallel = True

        self.compat._patch_duplicated_te_parameters(LocalLinear)
        constructor = LocalLinear.__init__
        self.compat._patch_duplicated_te_parameters(LocalLinear)
        self.assertIs(LocalLinear.__init__, constructor)
        duplicate = LocalLinear('duplicated')
        self.assertIs(duplicate.weight.tensor_model_parallel, False)
        predicate = tensor_parallel.param_is_not_tensor_parallel_duplicate
        self.assertTrue(predicate(duplicate.weight, tp_group=RankGroup(0)))
        self.assertFalse(predicate(duplicate.weight, tp_group=RankGroup(1)))
        self.assertIs(LocalLinear('column').weight.tensor_model_parallel, True)
        self.assertIs(LocalLinear('duplicated', is_expert=True).weight.tensor_model_parallel, True)

    def test_te_expert_metadata_distinguishes_shards_and_replicas(self):
        for mode in ('column', 'row', None):
            for use_expert_pgs in (False, True):
                with self.subTest(mode=mode, use_expert_pgs=use_expert_pgs):
                    module = torch.nn.Module()
                    for name in ('weight', 'weight12', 'bias', 'bias12', 'weight_scale'):
                        module.register_parameter(name, torch.nn.Parameter(torch.ones(2)))
                    module.adapter = torch.nn.Linear(2, 2)
                    self.compat._set_expert_parameter_attributes(module, mode, use_expert_pgs)
                    for name, param in module.named_parameters(recurse=False):
                        self.assertIs(param.allreduce, not use_expert_pgs)
                        if name == 'weight_scale':
                            self.assertFalse(hasattr(param, 'tensor_model_parallel'))
                            continue
                        partitioned = mode in ('column', 'row') and (name.startswith('weight') or mode == 'column')
                        self.assertIs(param.tensor_model_parallel, partitioned)
                        if partitioned:
                            self.assertEqual(param.partition_dim, 1 if mode == 'row' else 0)
                            self.assertEqual(param.partition_stride, 1)
                    self.assertFalse(hasattr(module.adapter.weight, 'allreduce'))
                    self.assertFalse(hasattr(module.adapter.weight, 'tensor_model_parallel'))

    def test_te_constructor_keeps_original_mode_and_expert_flag(self):

        class LocalGroupedLinear(torch.nn.Module):

            def __init__(self, parallel_mode, config, is_expert=False):
                super().__init__()
                self.config = config
                self.parallel_mode = None  # Explicit expert communication hides the mode from TE.
                self.weight0 = torch.nn.Parameter(torch.ones(2))
                self.bias0 = torch.nn.Parameter(torch.ones(2))
                self.weight0.tensor_model_parallel = False
                self.bias0.tensor_model_parallel = False

        self.compat._patch_te_expert_parameters(LocalGroupedLinear)
        constructor = LocalGroupedLinear.__init__
        self.compat._patch_te_expert_parameters(LocalGroupedLinear)
        self.assertIs(LocalGroupedLinear.__init__, constructor)
        for tp, ep, etp in ((2, 1, 2), (4, 1, 2), (2, 2, 2), (2, 1, 1)):
            config = SimpleNamespace(
                tensor_model_parallel_size=tp, expert_model_parallel_size=ep, expert_tensor_parallel_size=etp)
            for mode in ('column', 'row'):
                with self.subTest(tp=tp, ep=ep, etp=etp, mode=mode):
                    module = LocalGroupedLinear(mode, config, True)
                    self.assertFalse(hasattr(module, 'is_expert'))
                    self.assertIs(module.weight0.tensor_model_parallel, True)
                    self.assertIs(module.bias0.tensor_model_parallel, mode == 'column')
                    self.assertIs(module.weight0.allreduce, ep == 1 and tp == etp)
                    self.assertEqual(module.weight0.partition_dim, 1 if mode == 'row' else 0)
            dense = LocalGroupedLinear('column', config)
            self.assertIs(dense.weight0.tensor_model_parallel, False)
            self.assertFalse(hasattr(dense.weight0, 'allreduce'))

    def test_independent_te_constructor_preserves_stride(self):

        class LocalColumnLinear(torch.nn.Module):

            def __init__(self, config, is_expert=False, stride=1):
                super().__init__()
                self.config = config
                self.weight = torch.nn.Parameter(torch.ones(2))
                self.bias = torch.nn.Parameter(torch.ones(2))

        config = SimpleNamespace(
            tensor_model_parallel_size=4, expert_model_parallel_size=1, expert_tensor_parallel_size=2)
        self.compat._patch_te_expert_parameters(LocalColumnLinear, 'column')
        module = LocalColumnLinear(config, is_expert=True, stride=2)
        for param in module.parameters():
            self.assertIs(param.tensor_model_parallel, True)
            self.assertIs(param.allreduce, False)
            self.assertEqual(param.partition_dim, 0)
            self.assertEqual(param.partition_stride, 2)

    def test_etp_shards_and_row_bias_have_correct_clipping_norm(self):
        from megatron.core import tensor_parallel

        predicate = tensor_parallel.param_is_not_tensor_parallel_duplicate
        for tp, ep, etp in ((2, 1, 2), (4, 1, 2), (2, 2, 2)):
            with self.subTest(tp=tp, ep=ep, etp=etp):
                modules = []
                for rank, grad in enumerate((3., 4.)):
                    module = torch.nn.Module()
                    module.weight0 = torch.nn.Parameter(torch.ones(1))
                    module.weight0.grad = torch.tensor([grad])
                    module.bias0 = torch.nn.Parameter(torch.ones(1))
                    module.bias0.grad = torch.tensor([5.])
                    self.compat._set_expert_parameter_attributes(module, 'row', ep > 1 or etp != tp)
                    modules.append(module)
                selected = [
                    param.grad for rank, module in enumerate(modules) for param in module.parameters()
                    if predicate(param, tp_group=RankGroup(rank), expert_tp_group=RankGroup(rank))
                ]
                # Independent oracle: both weight shards and exactly one replicated bias.
                norm = torch.stack([grad.square().sum() for grad in selected]).sum().sqrt()
                self.assertAlmostEqual(norm.item(), 50.**0.5, places=6)
                for module in modules:
                    for param in module.parameters():
                        param.grad.mul_(1. / (norm + 1e-6))
                actual_squared = sum(module.weight0.grad.square().sum() for module in modules)
                actual_squared += modules[0].bias0.grad.square().sum()
                self.assertAlmostEqual(actual_squared.sqrt().item(), 1., places=6)

    def test_ep_one_still_uses_expert_topology_when_etp_differs(self):
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
        from megatron.core.transformer.moe.moe_layer import MoELayer

        config = SimpleNamespace(
            expert_model_parallel_size=1, tensor_model_parallel_size=2, expert_tensor_parallel_size=1)
        modules = []
        for cls in (ColumnParallelLinear, RowParallelLinear):
            module = cls.__new__(cls)
            torch.nn.Module.__init__(module)
            module.config = config
            module.is_expert = True
            module.register_parameter('weight', torch.nn.Parameter(torch.empty(2, 2)))
            module.register_parameter('bias', torch.nn.Parameter(torch.empty(2)))
            module.weight.tensor_model_parallel = True
            module.weight.partition_dim = 0 if cls is ColumnParallelLinear else 1
            module.bias.tensor_model_parallel = False
            modules.append(module)
        with patch.object(self.compat, '_legacy_expert_topology', return_value=True):
            moe = MoELayer.__new__(MoELayer)
            torch.nn.Module.__init__(moe)
            moe.config = config
            moe.experts = torch.nn.ModuleList(modules)
            self.compat.prepare_expert_parallel_parameters(moe)
        column, row = modules
        self.assertIs(column.weight.allreduce, False)
        self.assertIs(row.weight.allreduce, False)
        self.assertIs(column.bias.tensor_model_parallel, False)
        self.assertIs(row.bias.tensor_model_parallel, False)
        self.assertIs(column.weight.tensor_model_parallel, True)
        self.assertIs(row.weight.tensor_model_parallel, True)
        self.assertEqual(column.weight.partition_dim, 0)
        self.assertEqual(row.weight.partition_dim, 1)

    def test_expert_containers_without_backend_specific_flags(self):
        from megatron.core.transformer.moe import experts as expert_layers
        from megatron.core.transformer.moe.moe_layer import MoELayer

        expert_types = [expert_layers.TEGroupedMLP, expert_layers.SequentialMLP]
        # Native GroupedMLP was removed in MCore 0.19.
        if hasattr(expert_layers, 'GroupedMLP'):
            expert_types.append(expert_layers.GroupedMLP)
        for expert_type in expert_types:
            for tp, ep, etp in ((1, 1, 1), (2, 1, 1), (2, 2, 1)):
                with self.subTest(expert_type=expert_type.__name__, tp=tp, ep=ep, etp=etp):
                    # Parameter ownership, not a GPU/NPU linear's optional is_expert flag,
                    # determines its process group. No device kernels are needed here.
                    experts = expert_type.__new__(expert_type)
                    torch.nn.Module.__init__(experts)
                    experts.register_parameter('weight0', torch.nn.Parameter(torch.ones(2)))
                    experts.weight0.allreduce = True
                    experts.weight0.tensor_model_parallel = False
                    experts.adapter = torch.nn.Linear(2, 2, bias=False)
                    experts.adapter.weight.allreduce = False
                    moe = MoELayer.__new__(MoELayer)
                    torch.nn.Module.__init__(moe)
                    moe.config = SimpleNamespace(
                        tensor_model_parallel_size=tp, expert_model_parallel_size=ep, expert_tensor_parallel_size=etp)
                    moe.experts = experts
                    moe.router = torch.nn.Linear(2, 2)
                    moe.shared_experts = torch.nn.Linear(2, 2)
                    for p in (*moe.router.parameters(), *moe.shared_experts.parameters()):
                        p.allreduce = True
                    with patch.object(self.compat, '_legacy_expert_topology', return_value=True):
                        self.compat.prepare_expert_parallel_parameters(moe)
                    expected = ep == 1 and tp == etp
                    self.assertIs(experts.weight0.allreduce, expected)
                    self.assertIs(experts.adapter.weight.allreduce, expected)
                    self.assertIs(experts.weight0.tensor_model_parallel, False)
                    for p in (*moe.router.parameters(), *moe.shared_experts.parameters()):
                        self.assertIs(p.allreduce, True)

    def test_patches_are_idempotent(self):
        from megatron.core.tensor_parallel import layers

        self.compat.patch_megatron_compatibility()
        predicate = layers.param_is_not_tensor_parallel_duplicate
        copier = layers.copy_tensor_model_parallel_attributes
        self.compat.patch_megatron_compatibility()
        self.assertIs(layers.param_is_not_tensor_parallel_duplicate, predicate)
        self.assertIs(layers.copy_tensor_model_parallel_attributes, copier)

    def test_corrected_filter_and_new_metadata_are_untouched(self):
        from megatron.core.tensor_parallel import layers

        def corrected(param, tp_group=None, expert_tp_group=None):
            return 'native'

        with patch.object(layers, 'param_is_not_tensor_parallel_duplicate', corrected), \
                patch.object(self.compat, '_legacy_expert_topology', return_value=False):
            copier = layers.copy_tensor_model_parallel_attributes
            self.compat.patch_megatron_compatibility()
            self.assertIs(layers.param_is_not_tensor_parallel_duplicate, corrected)
            self.assertIs(layers.copy_tensor_model_parallel_attributes, copier)
            self.compat.prepare_expert_parallel_parameters(None)
