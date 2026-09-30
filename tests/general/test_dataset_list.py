# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import unittest
from unittest.mock import patch

from swift.dataset import DATASET_MAPPING, DatasetMeta, get_dataset_list, register_dataset


class TestDatasetList(unittest.TestCase):

    def test_builtin_named_dataset(self):
        for use_hf, dataset_id in [('0', 'swift/self-cognition'), ('1', 'modelscope/self-cognition')]:
            with self.subTest(use_hf=use_hf), patch.dict(os.environ, {'USE_HF': use_hf}):
                self.assertIn(dataset_id, get_dataset_list())

    def test_named_and_unnamed_datasets_use_selected_hub(self):
        with patch.dict(DATASET_MAPPING, clear=True):
            register_dataset(DatasetMeta(dataset_name='named', ms_dataset_id='ms/named', hf_dataset_id='hf/named'))
            register_dataset(DatasetMeta(dataset_name='x', ms_dataset_id='ms/short', hf_dataset_id='hf/short'))
            register_dataset(DatasetMeta(ms_dataset_id='ms/unnamed', hf_dataset_id='hf/unnamed'))
            register_dataset(DatasetMeta(dataset_name='ms-only', ms_dataset_id='ms/only'))
            register_dataset(DatasetMeta(dataset_name='hf-only', hf_dataset_id='hf/only'))
            register_dataset(DatasetMeta(dataset_path='/local/data.jsonl'))

            for use_hf, prefix in [('0', 'ms'), ('1', 'hf')]:
                with self.subTest(use_hf=use_hf), patch.dict(os.environ, {'USE_HF': use_hf}):
                    self.assertEqual(get_dataset_list(),
                                     [f'{prefix}/named', f'{prefix}/short', f'{prefix}/unnamed', f'{prefix}/only'])


if __name__ == '__main__':
    unittest.main()
