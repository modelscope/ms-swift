import unittest
from unittest.mock import patch

from datasets import Dataset as HfDataset

from swift.dataset import DatasetMeta, DatasetSyntax, load_dataset
from swift.dataset.loader import DatasetLoader


class TestDatasetSourceSelection(unittest.TestCase):

    def test_dataset_syntax_parses_explicit_hub_prefixes(self):
        self.assertIs(DatasetSyntax.parse('hf::example-org/dataset').use_hf, True)
        self.assertIs(DatasetSyntax.parse('ms::example-org/dataset').use_hf, False)

    def test_explicit_ms_source_overrides_global_hf_default(self):
        observed = []

        def fake_load_repo_dataset(self, dataset_id, subset, *, use_hf=None, revision=None):
            observed.append((dataset_id, use_hf))
            return HfDataset.from_dict({
                'messages': [[{
                    'role': 'user',
                    'content': 'hello',
                }, {
                    'role': 'assistant',
                    'content': 'world',
                }]]
            })

        with patch.object(DatasetSyntax, 'get_dataset_meta', return_value=DatasetMeta()), \
                patch.object(DatasetLoader, '_load_repo_dataset', new=fake_load_repo_dataset):
            train_dataset, val_dataset = load_dataset('ms::example-org/private-dataset', use_hf=True)

        self.assertEqual(observed, [('example-org/private-dataset', False)])
        self.assertEqual(len(train_dataset), 1)
        self.assertIsNone(val_dataset)


if __name__ == '__main__':
    unittest.main()
