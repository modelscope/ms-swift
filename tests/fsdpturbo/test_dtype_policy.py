import importlib.util
import torch
import unittest
from torch import nn
from types import SimpleNamespace
from unittest import mock


@unittest.skipUnless(importlib.util.find_spec('fsdp_turbo'), 'requires the optional FSDPTurbo package')
class TestFSDPTurboDtypePolicy(unittest.TestCase):

    def test_build_model_preserves_loader_selected_parameter_dtypes(self):
        from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig

        from swift.fsdpturbo.trainer import FSDPTurboTrainer
        from swift.pipelines.train.tuner import TunerMixin

        class ToyModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.regular = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
                self.sensitive = nn.Parameter(torch.ones(2, dtype=torch.float32))
                self.register_buffer('indices', torch.arange(2, dtype=torch.int64))
                self.config = SimpleNamespace(use_cache=True)

        model = ToyModel()
        trainer = object.__new__(FSDPTurboTrainer)
        trainer.args = SimpleNamespace(
            tuner_type='full',
            get_model_processor=mock.Mock(return_value=(model, None)),
            tp_size=1,
            ep_size=1,
            gradient_checkpointing=False,
            torch_dtype=torch.bfloat16,
            offload_params=False,
            resume_from_checkpoint=None,
            router_aux_loss_coef=0,
        )
        trainer.template = SimpleNamespace(model=None)
        trainer.train_dataset = mock.sentinel.dataset
        trainer.processor = mock.sentinel.processor
        trainer.train_args = SimpleNamespace(device='cpu')
        trainer.spec = SimpleNamespace(
            native_model_factory=None,
            frozen_modules=(),
            validate_model=mock.Mock(),
            matching_modules=mock.Mock(return_value={}),
        )
        trainer.config = FSDPTurboConfig()
        original_config = trainer.config

        dtypes = {name: param.dtype for name, param in model.named_parameters()}
        with mock.patch.object(TunerMixin, 'prepare_model', return_value=model), \
                mock.patch('swift.fsdpturbo.loading.load_rank0_model', return_value=(model, dtypes)), \
                mock.patch('swift.fsdpturbo.loading.materialize_sharded_model'), \
                mock.patch('fsdp_turbo.fsdp_turbo.FSDPTurbo',
                           side_effect=lambda _config, wrapped: SimpleNamespace(model=wrapped)):
            result = trainer.build_model()

        self.assertIs(result.model, model)
        self.assertEqual(trainer.config.distributed.fsdp_plan.ignored_params, ['sensitive'])
        self.assertEqual(original_config.distributed.fsdp_plan.ignored_params, [])
        self.assertEqual(model.regular.dtype, torch.bfloat16)
        self.assertEqual(model.sensitive.dtype, torch.float32)
        self.assertEqual(model.indices.dtype, torch.int64)
        self.assertFalse(model.config.use_cache)

    def test_replicated_fp32_gradients_are_world_averaged_before_step(self):
        from swift.fsdpturbo.trainer import FSDPTurboTrainer

        trainer = object.__new__(FSDPTurboTrainer)
        param = nn.Parameter(torch.ones(2, dtype=torch.float32))
        param.grad = torch.full_like(param, 3.0)
        trainer._replicated_parameters = [param]

        def all_reduce(tensor, **_kwargs):
            tensor.mul_(2)

        with mock.patch('swift.fsdpturbo.trainer.dist.is_initialized', return_value=True), \
                mock.patch('swift.fsdpturbo.trainer.dist.get_world_size', return_value=2), \
                mock.patch('swift.fsdpturbo.trainer.dist.all_reduce', side_effect=all_reduce) as reduce:
            trainer._sync_replicated_gradients()

        self.assertEqual(reduce.call_count, 2)
        torch.testing.assert_close(param.grad, torch.full_like(param, 3.0))

    def test_a_remote_gradient_is_not_skipped_when_local_branch_is_unused(self):
        from swift.fsdpturbo.trainer import FSDPTurboTrainer

        trainer = object.__new__(FSDPTurboTrainer)
        param = nn.Parameter(torch.ones(2, dtype=torch.float32))
        trainer._replicated_parameters = [param]

        def all_reduce(tensor, **_kwargs):
            tensor.add_(4 if tensor.is_floating_point() else 1)

        with mock.patch('swift.fsdpturbo.trainer.dist.is_initialized', return_value=True), \
                mock.patch('swift.fsdpturbo.trainer.dist.get_world_size', return_value=2), \
                mock.patch('swift.fsdpturbo.trainer.dist.all_reduce', side_effect=all_reduce):
            trainer._sync_replicated_gradients()

        torch.testing.assert_close(param.grad, torch.full_like(param, 2.0))

    def test_globally_unused_replicas_keep_no_gradient_or_adamw_state(self):
        from swift.fsdpturbo.trainer import FSDPTurboTrainer

        trainer = object.__new__(FSDPTurboTrainer)
        parameter = nn.Parameter(torch.ones(2, dtype=torch.float32))
        trainer._replicated_parameters = [parameter]
        optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.2)
        with mock.patch('swift.fsdpturbo.trainer.dist.is_initialized', return_value=True), \
                mock.patch('swift.fsdpturbo.trainer.dist.get_world_size', return_value=2), \
                mock.patch('swift.fsdpturbo.trainer.dist.all_reduce'):
            trainer._sync_replicated_gradients()
        optimizer.step()

        self.assertIsNone(parameter.grad)
        self.assertFalse(optimizer.state)
        torch.testing.assert_close(parameter, torch.ones_like(parameter))


if __name__ == '__main__':
    unittest.main()
