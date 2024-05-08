import tempfile
import unittest

from datasets import Dataset as HfDataset

from swift.llm import DatasetName, get_dataset


class TestDataset(unittest.TestCase):

    @unittest.skip('fix citest')
    def test_dataset(self):
        train_dataset, val_dataset = get_dataset([DatasetName.leetcode_python_en, DatasetName.blossom_math_zh])
        assert isinstance(train_dataset, HfDataset) and val_dataset is None
        totol_len = 12359
        assert len(train_dataset) == totol_len

        train_dataset, val_dataset = get_dataset([DatasetName.leetcode_python_en, DatasetName.blossom_math_zh], 0.01)
        assert isinstance(train_dataset, HfDataset) and isinstance(train_dataset, HfDataset)
        assert len(train_dataset) + len(val_dataset) == totol_len


if __name__ == '__main__':
    unittest.main()
