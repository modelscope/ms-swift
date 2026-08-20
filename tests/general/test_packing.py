# Copyright (c) ModelScope Contributors. All rights reserved.
import multiprocessing as mp
import unittest

from swift.dataset import IterablePackingDataset


class _Template:

    max_length = 8

    def encode(self, data, return_length=False):
        import torch
        return {
            'input_ids': [data['input_id']],
            'labels': [data['input_id']],
            'start_method': mp.get_start_method(),
            'cuda_initialized': torch.cuda.is_initialized(),
        }


class TestIterablePackingDataset(unittest.TestCase):

    def test_worker_uses_spawn_context(self):
        dataset = IterablePackingDataset(
            _Template(),
            [{'input_id': 1}, {'input_id': 2}],
            num_proc=2,
            packing_interval=2,
            packing_length=8,
        )
        try:
            packed = list(dataset)
            rows = [row for group in packed for row in group]
            self.assertEqual([row['start_method'] for row in rows], ['spawn', 'spawn'])
            self.assertTrue(all(not row['cuda_initialized'] for row in rows))
        finally:
            for worker in dataset.workers:
                worker.terminate()
                worker.join()


if __name__ == '__main__':
    unittest.main()
