# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import torch
import unittest
from types import SimpleNamespace
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec('fsdp_turbo'), 'requires the optional FSDPTurbo package')
class TestLossGroupInitialization(unittest.TestCase):

    def test_swift_uses_the_backend_loss_reduction(self):
        from fsdp_turbo.training.trainer import BaseTrainer

        from swift.fsdpturbo.trainer import FSDPTurboTrainer

        self.assertIs(
            getattr(FSDPTurboTrainer.sync_loss, '__func__', FSDPTurboTrainer.sync_loss), BaseTrainer.sync_loss.__func__)

    def make_trainer(self, size):
        from swift.fsdpturbo.trainer import FSDPTurboTrainer

        trainer = object.__new__(FSDPTurboTrainer)
        trainer.train_args = SimpleNamespace(device=torch.device('cpu'))
        trainer.model = SimpleNamespace(
            parallel_state=SimpleNamespace(get_data_group_size=lambda: size, get_data_group=lambda: 'loss-group'))
        return trainer

    def test_collective_initializes_the_loss_group_before_training(self):
        trainer = self.make_trainer(8)
        with patch('torch.distributed.all_reduce') as reduce:
            trainer._initialize_loss_group()
        reduce.assert_called_once()
        self.assertEqual(reduce.call_args.kwargs['group'], 'loss-group')
        self.assertEqual(reduce.call_args.args[0].item(), 0)

    def test_native_warmup_deduplicates_mesh_aliases_and_skips_single_rank_groups(self):
        trainer = self.make_trainer(8)
        trainer.config = object()
        parallel_state = SimpleNamespace(
            device_mesh_map={
                'dense': SimpleNamespace(get_all_groups=lambda: ['dense-group', 'single-group']),
                'dense_alias': SimpleNamespace(get_all_groups=lambda: ['dense-group']),
                'expert': SimpleNamespace(get_all_groups=lambda: ['expert-group']),
            })
        group_sizes = {'single-group': 1, 'dense-group': 8, 'expert-group': 8}
        with patch('fsdp_turbo.distributed.parallel_state.init_parallel_state',
                   return_value=parallel_state) as initialize, \
                patch('torch.distributed.get_world_size', side_effect=group_sizes.__getitem__), \
                patch('torch.distributed.new_group', return_value='loading-ready') as loading_group, \
                patch('torch.distributed.all_reduce') as reduce:
            trainer._initialize_native_groups()
        initialize.assert_called_once_with(trainer.config)
        loading_group.assert_called_once_with(backend='gloo')
        self.assertEqual(trainer._native_load_group, 'loading-ready')
        self.assertEqual([call.kwargs.get('group') for call in reduce.call_args_list],
                         [None, 'dense-group', 'expert-group'])

    def test_single_data_rank_needs_no_collective(self):
        trainer = self.make_trainer(1)
        with patch('torch.distributed.all_reduce') as reduce:
            trainer._initialize_loss_group()
        reduce.assert_not_called()


if __name__ == '__main__':
    unittest.main()
