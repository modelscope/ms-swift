# Copyright (c) ModelScope Contributors. All rights reserved.
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import unittest
from datetime import timedelta
from torch.distributed.device_mesh import init_device_mesh
from transformers import Trainer
from types import SimpleNamespace

from swift.sequence_parallel import sequence_parallel
from swift.trainers.mixin import DataLoaderMixin


class _Trainer(DataLoaderMixin, Trainer):
    """Exercise dataloader construction without initializing a model."""


def _check_sampling_options(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=120))
    try:
        sequence_parallel.device_mesh = init_device_mesh('cpu', (2, 2), mesh_dim_names=('data', 'sequence'))
        sequence_parallel.world_size = 2
        trainer = _Trainer.__new__(_Trainer)
        trainer.template = SimpleNamespace(sequence_parallel_size=2)
        trainer.train_dataset = list(range(16))
        trainer.eval_dataset = list(range(16))
        trainer._train_batch_size = 2
        trainer.data_collator = lambda batch: batch
        trainer.accelerator = SimpleNamespace(device=None)
        trainer.args = SimpleNamespace(
            remove_unused_columns=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            dataloader_persistent_workers=False,
            dataloader_prefetch_factor=None,
            dataloader_drop_last=False,
            eval_batch_size=2,
            deepspeed=None,
            group_by_length=False,
            process_index=rank)

        def collect(loader):
            return [item for batch in loader for item in batch]

        def expected_order(shuffle, seed, dp_rank, dp_size):
            if shuffle:
                generator = torch.Generator().manual_seed(seed)
                indices = torch.randperm(16, generator=generator).tolist()
            else:
                indices = list(range(16))
            return indices[dp_rank::dp_size]

        eval_order = expected_order(True, 42, rank // 2, 2)
        for shuffle, seed in ((True, 42), (False, 42), (True, 123), (True, 0), (True, None)):
            trainer.args.train_dataloader_shuffle = shuffle
            trainer.args.data_seed = seed
            effective_seed = 0 if seed is None else seed
            for sp_size in (2, 1):
                trainer.template.sequence_parallel_size = sp_size
                dp_rank, dp_size = (rank // 2, 2) if sp_size == 2 else (rank, 4)
                expected = expected_order(shuffle, effective_seed, dp_rank, dp_size)
                actual = collect(trainer.get_train_dataloader())
                assert actual == expected, (rank, sp_size, shuffle, seed, actual, expected)
                # Rebuilding the loader must reproduce the configured order, including when resuming.
                assert collect(trainer.get_train_dataloader()) == expected
                assert collect(trainer.get_train_dataloader(skip_batches=1)) == expected[2:]
            trainer.template.sequence_parallel_size = 2
            # Training-only options must not alter the shared evaluation construction path.
            assert collect(trainer.get_eval_dataloader()) == eval_order
        # Keep the rendezvous store alive until every rank finishes using the mesh.
        dist.barrier()
    finally:
        sequence_parallel.world_size = None
        sequence_parallel.device_mesh = None
        dist.destroy_process_group()


def _check_unaligned_dataset(rank, rendezvous):
    """A dataset that is not divisible by the world size must shard identically with and without SP."""
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=120))
    try:
        sequence_parallel.device_mesh = init_device_mesh('cpu', (2, 2), mesh_dim_names=('data', 'sequence'))
        sequence_parallel.world_size = 2
        trainer = _Trainer.__new__(_Trainer)
        trainer.template = SimpleNamespace(sequence_parallel_size=1)
        trainer.train_dataset = list(range(18))
        trainer.eval_dataset = list(range(18))
        trainer._train_batch_size = 1
        trainer.data_collator = lambda batch: batch
        trainer.accelerator = SimpleNamespace(device=None)
        trainer.args = SimpleNamespace(
            remove_unused_columns=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            dataloader_persistent_workers=False,
            dataloader_prefetch_factor=None,
            dataloader_drop_last=False,
            eval_batch_size=1,
            deepspeed=None,
            group_by_length=False,
            train_dataloader_shuffle=False,
            data_seed=42,
            process_index=rank)

        def collect(loader):
            return [item for batch in loader for item in batch]

        def step_sets(batches, accumulation_steps, num_steps):
            result = []
            for step in range(num_steps):
                local = set(batches[step * accumulation_steps:(step + 1) * accumulation_steps])
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, local)
                result.append(set().union(*gathered))
            return result

        for shuffle in (False, True):
            trainer.args.train_dataloader_shuffle = shuffle
            trainer.template.sequence_parallel_size = 1
            dp_batches = collect(trainer.get_train_dataloader())
            trainer.template.sequence_parallel_size = 2
            sp_batches = collect(trainer.get_train_dataloader())

            # N=18, world=4 -> both paths keep the first 16 samples and drop the tail.
            assert len(dp_batches) == 4, (rank, dp_batches)
            assert len(sp_batches) == 8, (rank, sp_batches)
            dp_steps = step_sets(dp_batches, 1, 4)
            sp_steps = step_sets(sp_batches, 2, 4)
            assert dp_steps == sp_steps, (rank, shuffle, dp_steps, sp_steps)
            for step in dp_steps:
                assert len(step) == 4
                assert 16 not in step and 17 not in step
        # Keep the rendezvous store alive until every rank finishes using the mesh.
        dist.barrier()
    finally:
        sequence_parallel.world_size = None
        sequence_parallel.device_mesh = None
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo is required')
class TestSPSamplingOptions(unittest.TestCase):

    def test_training_options_preserve_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_check_sampling_options, args=(f'file://{directory}/rendezvous', ), nprocs=4, join=True)

    def test_unaligned_dataset_matches_dp_sharding(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_check_unaligned_dataset, args=(f'file://{directory}/rendezvous', ), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()
