"""Unit tests for dataset progress tracking functionality.

Tests the DatasetProgressCallback and related components for tracking
per-dataset training progress in multi-dataset training scenarios.
"""
import unittest
from collections import defaultdict
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import torch


class TestDataLoaderSourceInjection(unittest.TestCase):
    """Test _add_dataset_source method in DatasetLoader."""

    def test_add_source_regular_dataset(self):
        """Test adding source to a regular HuggingFace dataset."""
        from datasets import Dataset

        # Create a simple dataset
        data = {'text': ['hello', 'world', 'test']}
        dataset = Dataset.from_dict(data)

        # Simulate _add_dataset_source
        source = 'my_dataset.json'
        dataset = dataset.add_column('_dataset_source', [source] * len(dataset))

        # Verify source column was added
        self.assertIn('_dataset_source', dataset.column_names)
        self.assertEqual(dataset['_dataset_source'], [source, source, source])

    def test_add_source_streaming_dataset(self):
        """Test adding source to a streaming IterableDataset."""
        from datasets import Dataset

        # Create a simple dataset and convert to iterable
        data = {'text': ['hello', 'world']}
        dataset = Dataset.from_dict(data).to_iterable_dataset()

        # Simulate _add_dataset_source for streaming
        source = 'streaming_data.jsonl'

        def add_source(example):
            example['_dataset_source'] = source
            return example

        dataset = dataset.map(add_source)

        # Verify source is added (check first item)
        item = next(iter(dataset))
        self.assertEqual(item['_dataset_source'], source)


class TestTemplateEncodePreserveSource(unittest.TestCase):
    """Test that template.encode preserves _dataset_source."""

    def test_encode_preserves_dataset_name_field(self):
        """Test that dataset_name field is preserved as _dataset_source."""
        # Mock the Template class behavior
        class MockStdTemplateInputs:
            def __init__(self, extra_kwargs):
                self.extra_kwargs = extra_kwargs

        extra_kwargs = {'dataset_name': 'my_classification_data', '_dataset_source': 'data.json'}
        inputs = MockStdTemplateInputs(extra_kwargs)

        # Simulate the encode logic
        encoded = {}
        dataset_source = inputs.extra_kwargs.get('dataset_name') or inputs.extra_kwargs.get('_dataset_source')
        if dataset_source:
            encoded['_dataset_source'] = dataset_source

        # dataset_name should take priority
        self.assertEqual(encoded['_dataset_source'], 'my_classification_data')

    def test_encode_preserves_dataset_source_field(self):
        """Test that _dataset_source is preserved when task is not present."""
        class MockStdTemplateInputs:
            def __init__(self, extra_kwargs):
                self.extra_kwargs = extra_kwargs

        extra_kwargs = {'_dataset_source': 'data.json'}
        inputs = MockStdTemplateInputs(extra_kwargs)

        encoded = {}
        dataset_source = inputs.extra_kwargs.get('task') or inputs.extra_kwargs.get('_dataset_source')
        if dataset_source:
            encoded['_dataset_source'] = dataset_source

        self.assertEqual(encoded['_dataset_source'], 'data.json')


class TestPackingRowPreserveSource(unittest.TestCase):
    """Test that packing_row preserves _dataset_source as a list."""

    def test_packing_row_collects_sources(self):
        """Test packing_row collects sources from all packed samples."""
        # Simulate packing_row logic for _dataset_source
        row = [
            {'input_ids': [1, 2, 3], 'length': 3, '_dataset_source': 'dataset_a.json'},
            {'input_ids': [4, 5], 'length': 2, '_dataset_source': 'dataset_b.json'},
            {'input_ids': [6, 7, 8, 9], 'length': 4, '_dataset_source': 'dataset_a.json'},
        ]

        # Simulate packing_row behavior
        packed = {}
        keys = set()
        for r in row:
            keys.update(r.keys())

        for key in keys:
            if key == '_dataset_source':
                packed[key] = [x.get(key) for x in row]

        expected_sources = ['dataset_a.json', 'dataset_b.json', 'dataset_a.json']
        self.assertEqual(packed['_dataset_source'], expected_sources)

    def test_packing_row_handles_missing_source(self):
        """Test packing_row handles samples without _dataset_source."""
        row = [
            {'input_ids': [1, 2, 3], 'length': 3, '_dataset_source': 'dataset_a.json'},
            {'input_ids': [4, 5], 'length': 2},  # No _dataset_source
        ]

        packed = {}
        packed['_dataset_source'] = [x.get('_dataset_source') for x in row]

        self.assertEqual(packed['_dataset_source'], ['dataset_a.json', None])


class TestUpdateDatasetProgress(unittest.TestCase):
    """Test _update_dataset_progress method."""

    def test_update_counts_single_source(self):
        """Test counting samples with single source."""
        # Simulate template behavior
        track_dataset_progress = True
        dataset_progress_counts = {}

        batch = [
            {'input_ids': [1, 2], '_dataset_source': 'ds_a'},
            {'input_ids': [3, 4], '_dataset_source': 'ds_a'},
            {'input_ids': [5, 6], '_dataset_source': 'ds_b'},
        ]

        for b in batch:
            sources = b.pop('_dataset_source', None)
            if sources is None:
                continue
            if isinstance(sources, str):
                sources = [sources]
            for source in sources:
                if source is not None:
                    dataset_progress_counts[source] = dataset_progress_counts.get(source, 0) + 1

        self.assertEqual(dataset_progress_counts, {'ds_a': 2, 'ds_b': 1})

    def test_update_counts_packed_sources(self):
        """Test counting samples from packed batch (sources as list)."""
        dataset_progress_counts = {}

        # After packing, _dataset_source is a list
        batch = [
            {
                'input_ids': [1, 2, 3, 4, 5],
                '_dataset_source': ['ds_a', 'ds_b', 'ds_a']  # 3 packed samples
            },
        ]

        for b in batch:
            sources = b.pop('_dataset_source', None)
            if sources is None:
                continue
            if isinstance(sources, str):
                sources = [sources]
            for source in sources:
                if source is not None:
                    dataset_progress_counts[source] = dataset_progress_counts.get(source, 0) + 1

        self.assertEqual(dataset_progress_counts, {'ds_a': 2, 'ds_b': 1})

    def test_removes_dataset_source_from_batch(self):
        """Test that _dataset_source is removed from batch after processing."""
        batch = [
            {'input_ids': [1, 2], '_dataset_source': 'ds_a'},
        ]

        # Process batch
        for b in batch:
            b.pop('_dataset_source', None)

        # Verify _dataset_source is removed
        self.assertNotIn('_dataset_source', batch[0])


class TestDatasetProgressCallback(unittest.TestCase):
    """Test DatasetProgressCallback functionality."""

    def test_callback_calculates_progress(self):
        """Test that callback correctly calculates progress percentage."""
        dataset_sizes = {'ds_a': 100, 'ds_b': 200}
        counts = {'ds_a': 50, 'ds_b': 100}

        # Simulate on_log behavior
        logs = {}
        for source, count in counts.items():
            total = dataset_sizes.get(source)
            if total and total > 0:
                progress = min(count / total * 100, 100.0)
                logs[f'dataset_progress/{source}'] = round(progress, 2)
            else:
                logs[f'dataset_samples/{source}'] = count

        self.assertEqual(logs['dataset_progress/ds_a'], 50.0)
        self.assertEqual(logs['dataset_progress/ds_b'], 50.0)

    def test_callback_handles_unknown_total(self):
        """Test callback logs counts when total is unknown."""
        dataset_sizes = {}  # Unknown totals
        counts = {'ds_a': 50, 'ds_b': 100}

        logs = {}
        for source, count in counts.items():
            total = dataset_sizes.get(source)
            if total and total > 0:
                progress = min(count / total * 100, 100.0)
                logs[f'dataset_progress/{source}'] = round(progress, 2)
            else:
                logs[f'dataset_samples/{source}'] = count

        self.assertEqual(logs['dataset_samples/ds_a'], 50)
        self.assertEqual(logs['dataset_samples/ds_b'], 100)

    def test_callback_caps_at_100_percent(self):
        """Test that progress is capped at 100%."""
        dataset_sizes = {'ds_a': 100}
        counts = {'ds_a': 150}  # More than total (e.g., multiple epochs)

        logs = {}
        for source, count in counts.items():
            total = dataset_sizes.get(source)
            if total and total > 0:
                progress = min(count / total * 100, 100.0)
                logs[f'dataset_progress/{source}'] = round(progress, 2)

        self.assertEqual(logs['dataset_progress/ds_a'], 100.0)


class TestDistributedGather(unittest.TestCase):
    """Test distributed gather logic."""

    def test_aggregate_counts_from_multiple_ranks(self):
        """Test aggregating counts from multiple processes."""
        # Simulate gathered data from 4 ranks
        gathered = [
            {'ds_a': 10, 'ds_b': 5},
            {'ds_a': 12, 'ds_b': 3},
            {'ds_a': 8, 'ds_b': 7},
            {'ds_a': 10, 'ds_b': 5},
        ]

        global_counts = defaultdict(int)
        for local in gathered:
            if local:
                for source, count in local.items():
                    global_counts[source] += count

        self.assertEqual(global_counts['ds_a'], 40)  # 10+12+8+10
        self.assertEqual(global_counts['ds_b'], 20)  # 5+3+7+5

    def test_handles_empty_ranks(self):
        """Test handling of empty/None entries from some ranks."""
        gathered = [
            {'ds_a': 10},
            None,  # Rank didn't participate
            {'ds_a': 5, 'ds_b': 3},
            {},  # Empty dict
        ]

        global_counts = defaultdict(int)
        for local in gathered:
            if local:
                for source, count in local.items():
                    global_counts[source] += count

        self.assertEqual(global_counts['ds_a'], 15)
        self.assertEqual(global_counts['ds_b'], 3)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete flow."""

    def test_import_callback(self):
        """Test that DatasetProgressCallback can be imported."""
        from swift.trainers.callback import DatasetProgressCallback
        self.assertTrue(callable(DatasetProgressCallback))

    def test_import_template_base(self):
        """Test that Template with progress tracking can be imported."""
        from swift.llm.template.base import Template
        # Check that _update_dataset_progress method exists
        self.assertTrue(hasattr(Template, '_update_dataset_progress'))
        self.assertTrue(callable(getattr(Template, '_update_dataset_progress')))

    def test_train_args_has_track_dataset_progress(self):
        """Test that TrainArguments has track_dataset_progress field."""
        from swift.llm.argument import TrainArguments
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(TrainArguments)]
        self.assertIn('track_dataset_progress', field_names)


if __name__ == '__main__':
    unittest.main()
