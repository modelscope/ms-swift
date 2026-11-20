# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import unittest
from transformers import default_data_collator
from types import SimpleNamespace
from unittest import mock

from swift.trainers.mixin import DataLoaderMixin


class TestSpDataloaderPrefetch(unittest.TestCase):

    def test_map_dataset_prefetch_factor(self):
        dataset = [{'input_ids': [1]}, {'input_ids': [2]}]
        for num_workers, prefetch_factor, expected in ((1, 16, 16), (1, None, 2), (0, None, None)):
            with self.subTest(num_workers=num_workers, prefetch_factor=prefetch_factor):
                trainer = DataLoaderMixin()
                trainer.args = SimpleNamespace(
                    dataloader_num_workers=num_workers,
                    dataloader_pin_memory=False,
                    dataloader_persistent_workers=False,
                    dataloader_prefetch_factor=prefetch_factor,
                    dataloader_drop_last=False,
                    dataloader_multiprocessing_context=None)
                trainer.accelerator = SimpleNamespace(device=torch.device('cpu'))
                trainer.data_collator = default_data_collator
                trainer._get_collator_with_removed_columns = mock.Mock(return_value=default_data_collator)
                with mock.patch(
                        'swift.trainers.mixin.SequenceParallelSampler',
                        return_value=torch.utils.data.SequentialSampler(dataset)), mock.patch(
                            'swift.trainers.mixin.sequence_parallel', SimpleNamespace(dp_rank=0)):
                    loader = trainer.get_sp_dataloader(dataset, batch_size=1)
                self.assertEqual(loader.prefetch_factor, expected)
                self.assertEqual([batch['input_ids'].item() for batch in loader], [1, 2])


if __name__ == '__main__':
    unittest.main()
