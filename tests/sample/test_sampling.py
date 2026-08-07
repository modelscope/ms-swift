import unittest
from types import SimpleNamespace
from unittest.mock import patch

from swift.pipelines.sampling.sampling import SwiftSampling


class FakeDataset:

    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def select(self, indices):
        return FakeDataset([self.values[i] for i in indices])


class TestSampling(unittest.TestCase):

    @patch('swift.pipelines.sampling.sampling.load_dataset')
    def test_data_range_partitions_cover_dataset(self, mock_load_dataset):
        args = SimpleNamespace(
            dataset=['test-dataset'],
            dataset_shuffle=False,
            get_dataset_kwargs=lambda: {},
        )

        for dataset_size, expected_sizes in [(10, [4, 3, 3]), (2, [1, 1, 0])]:
            with self.subTest(dataset_size=dataset_size):
                dataset = FakeDataset(list(range(dataset_size)))
                mock_load_dataset.return_value = dataset, None
                shards = []
                for shard_index in range(3):
                    sampling = SwiftSampling.__new__(SwiftSampling)
                    sampling.args = args
                    sampling.cur_piece = shard_index
                    sampling.total_piece = 3
                    shards.append(sampling._get_dataset().values)

                self.assertEqual([len(shard) for shard in shards], expected_sizes)
                self.assertEqual([item for shard in shards for item in shard], list(range(dataset_size)))


if __name__ == '__main__':
    unittest.main()
