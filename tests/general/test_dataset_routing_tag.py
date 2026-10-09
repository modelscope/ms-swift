"""Tests for the multi-teacher ``dataset`` routing tag added by ``load_dataset``.

The tag is a constant column, but ``Dataset.add_column`` materializes the dataset first whenever it
carries an indices table: it falls back to ``flatten_indices()``, which is a ``map()``. ``map()``
derives its cache file path from the dataset fingerprint, so every rank computes the same path. When
the dataset directory lives on a filesystem shared between ranks -- the normal setup for multi-node
training -- all ranks write that file concurrently and race on the ``shutil.move()`` -> ``os.chmod()``
at the end of ``Dataset._map_single``, which fails with ``FileNotFoundError`` on the losing ranks.

Sampling (``dataset#n``) and ``split_dataset_ratio`` both go through ``select()``, so the indices
table is present in ordinary training runs. Tagging before ``post_process()`` keeps ``add_column()``
on the in-memory path, which is what these tests pin down.
"""
import contextlib
import json
import os
import tempfile
import unittest
from datasets import Dataset as HfDataset
from unittest import mock

from swift.dataset import DatasetMeta, load_dataset, register_dataset
from swift.dataset.loader import _inject_dataset_routing_tag

_ORIGINAL_FLATTEN_INDICES = HfDataset.flatten_indices


@contextlib.contextmanager
def _track_flatten_indices():
    """Count flatten_indices() calls while still running the real implementation."""
    with mock.patch.object(
            HfDataset, 'flatten_indices', autospec=True, side_effect=_ORIGINAL_FLATTEN_INDICES) as tracker:
        yield tracker


class TestDatasetRoutingTag(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        tmp_path = self._tmp_dir.name
        self.dataset_path = os.path.join(tmp_path, 'routing_tag.jsonl')
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            for i in range(100):
                messages = [{
                    'role': 'user',
                    'content': f'question {i}'
                }, {
                    'role': 'assistant',
                    'content': f'answer {i}'
                }]
                f.write(json.dumps({'messages': messages}, ensure_ascii=False) + '\n')

        self.dataset_name = f'test-routing-tag-{id(self)}'
        register_dataset(DatasetMeta(dataset_path=self.dataset_path, dataset_name=self.dataset_name), exist_ok=True)

        cache_patch = mock.patch.dict(os.environ, {'MODELSCOPE_CACHE': os.path.join(tmp_path, 'cache')})
        cache_patch.start()
        self.addCleanup(cache_patch.stop)
        self.addCleanup(self._tmp_dir.cleanup)

    def test_routing_tag_is_added_without_flattening(self):
        # flatten_indices() is what writes the rank-colliding cache file; tagging must not need it.
        with _track_flatten_indices() as flatten_indices:
            train_dataset, _ = load_dataset([f'{self.dataset_name}#50'], shuffle=True, seed=42)

        flatten_indices.assert_not_called()
        self.assertEqual(len(train_dataset), 50)
        self.assertIn('dataset', train_dataset.features)
        # A dataset registered by path is rewritten to that path by load_dataset(), so the tag
        # carries the path rather than the registered name.
        self.assertEqual(set(train_dataset['dataset']), {self.dataset_path})

    def test_routing_tag_survives_train_val_split(self):
        train_dataset, val_dataset = load_dataset([self.dataset_name], split_dataset_ratio=0.2, seed=42)

        self.assertEqual(set(train_dataset['dataset']), {self.dataset_path})
        self.assertEqual(set(val_dataset['dataset']), {self.dataset_path})
        self.assertEqual(len(train_dataset) + len(val_dataset), 100)

    def test_existing_dataset_column_in_local_file(self):
        with open(self.dataset_path, encoding='utf-8') as f:
            rows = [json.loads(line) for line in f]
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            for i, row in enumerate(rows):
                row.update(dataset=f'previous-source-{i % 2}', row_id=i)
                f.write(json.dumps(row) + '\n')

        for streaming in [False, True]:
            for remove_unused_columns in [False, True]:
                with self.subTest(streaming=streaming, remove_unused_columns=remove_unused_columns):
                    with _track_flatten_indices() as flatten_indices:
                        train_dataset, val_dataset = load_dataset(
                            f'{self.dataset_name}#50',
                            streaming=streaming,
                            remove_unused_columns=remove_unused_columns,
                            split_dataset_ratio=0.2,
                            seed=42)
                    flatten_indices.assert_not_called()
                    train_rows, val_rows = list(train_dataset), list(val_dataset)
                    self.assertEqual(len(train_rows), 40)
                    self.assertEqual(len(val_rows), 10)
                    for row in train_rows + val_rows:
                        self.assertEqual(row['dataset'], self.dataset_path)
                        self.assertEqual('row_id' in row, not remove_unused_columns)
                        self.assertEqual(len(row['messages']), 2)

    def test_replacing_tag_preserves_input_dataset(self):
        original = HfDataset.from_dict({'dataset': ['old-a', 'old-b'], 'value': [1, 2]})
        tagged = _inject_dataset_routing_tag(original, 'current-source')
        self.assertEqual(list(original['dataset']), ['old-a', 'old-b'])
        self.assertEqual(list(tagged['dataset']), ['current-source'] * 2)
        self.assertEqual(list(tagged['value']), [1, 2])


if __name__ == '__main__':
    unittest.main()
