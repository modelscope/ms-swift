import os
import unittest
from contextlib import nullcontext
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from swift.dataset import DatasetLoader
from swift.dataset.register import SubsetDataset


class FakeHub:

    def __init__(self, error=None):
        self.error = error
        self.load_calls = []

    def load_dataset(self, dataset_id, subset, split, **kwargs):
        self.load_calls.append((dataset_id, subset, split))
        if self.error is not None:
            raise self.error
        # A folder containing `dataset_infos.json` must stay invisible to the loader.
        if os.path.exists(os.path.join(dataset_id, 'dataset_infos.json')):
            raise AssertionError('dataset_infos.json must be hidden while the local folder is loaded')
        return SimpleNamespace(**{'_hf_ds': 'dataset'})


class TestLocalDatasetInfos(unittest.TestCase):

    def _make_loader(self):
        loader = object.__new__(DatasetLoader)
        loader.streaming = False
        loader.hub_token = None
        loader.download_mode = 'reuse_dataset_if_exists'
        loader.num_proc = 1
        loader.columns = None
        loader.remove_unused_columns = False
        loader.strict = False
        loader.disable_auto_column_mapping = False
        loader.load_from_cache_file = True
        return loader

    def _make_subset(self):
        subset = SubsetDataset(subset='default')
        subset.split = ['train']
        subset.preprocess_func = lambda dataset, **kwargs: dataset
        return subset

    def _run(self, error=None):
        with TemporaryDirectory() as tmp_dir:
            self.data_dir = tmp_dir
            with open(os.path.join(tmp_dir, 'train.jsonl'), 'w') as f:
                f.write('{"messages": []}\n')
            with open(os.path.join(tmp_dir, 'dataset_infos.json'), 'w') as f:
                f.write('{"default": {}}')
            hub = FakeHub(error)
            loader = self._make_loader()
            with patch('swift.dataset.loader.get_hub', return_value=hub), \
                    patch('swift.dataset.loader.safe_ddp_context', new=lambda *a, **kw: nullcontext()):
                if error is not None:
                    with self.assertRaises(type(error)):
                        loader._load_repo_dataset(tmp_dir, self._make_subset())
                else:
                    loader._load_repo_dataset(tmp_dir, self._make_subset())
            # The file is only hidden while loading, the user's directory must be left untouched.
            self.assertFalse(os.path.exists(os.path.join(tmp_dir, 'dataset_infos.json_bak')))
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, 'dataset_infos.json')))
            return hub

    def test_dataset_infos_is_restored_after_a_successful_load(self):
        hub = self._run()
        self.assertEqual(hub.load_calls, [(self.data_dir, 'default', 'train')])

    def test_dataset_infos_is_restored_when_the_load_fails(self):
        self._run(error=RuntimeError('hub unreachable'))

    def test_dataset_infos_is_renamed_back_when_the_folder_has_no_infos(self):
        with TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'train.jsonl'), 'w') as f:
                f.write('{"messages": []}\n')
            self.assertEqual(sorted(os.listdir(tmp_dir)), ['train.jsonl'])
            hub = FakeHub()
            loader = self._make_loader()
            with patch('swift.dataset.loader.get_hub', return_value=hub), \
                    patch('swift.dataset.loader.safe_ddp_context', new=lambda *a, **kw: nullcontext()):
                loader._load_repo_dataset(tmp_dir, self._make_subset())
            self.assertEqual(sorted(os.listdir(tmp_dir)), ['train.jsonl'])


if __name__ == '__main__':
    unittest.main()
