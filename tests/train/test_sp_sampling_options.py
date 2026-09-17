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
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo is required')
class TestSPSamplingOptions(unittest.TestCase):

    def test_training_options_preserve_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_check_sampling_options, args=(f'file://{directory}/rendezvous', ), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()
