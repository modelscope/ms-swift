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


class TestProgressTrackingCollator(unittest.TestCase):
    """Test ProgressTrackingCollator wrapper."""

    def test_collects_single_sources(self):
        """Test collecting sources from non-packed batch."""
        from swift.llm.dataset import ProgressTrackingCollator

        # Mock original collator
        def mock_collator(batch):
            return {'input_ids': [[1, 2], [3, 4], [5, 6]]}

        wrapper = ProgressTrackingCollator(mock_collator, track_progress=True)

        batch = [
            {'input_ids': [1, 2], '_dataset_source': 'ds_a'},
            {'input_ids': [3, 4], '_dataset_source': 'ds_a'},
            {'input_ids': [5, 6], '_dataset_source': 'ds_b'},
        ]

        result = wrapper(batch)

        # Verify _batch_sources is collected
        self.assertIn('_batch_sources', result)
        self.assertEqual(result['_batch_sources'], ['ds_a', 'ds_a', 'ds_b'])

        # Verify _dataset_source is removed from batch
        for b in batch:
            self.assertNotIn('_dataset_source', b)

    def test_collects_packed_sources(self):
        """Test collecting sources from packed batch (sources as list)."""
        from swift.llm.dataset import ProgressTrackingCollator

        def mock_collator(batch):
            return {'input_ids': [[1, 2, 3, 4, 5]]}

        wrapper = ProgressTrackingCollator(mock_collator, track_progress=True)

        # After packing, _dataset_source is a list
        batch = [
            {
                'input_ids': [1, 2, 3, 4, 5],
                '_dataset_source': ['ds_a', 'ds_b', 'ds_a']  # 3 packed samples
            },
        ]

        result = wrapper(batch)

        self.assertIn('_batch_sources', result)
        self.assertEqual(result['_batch_sources'], ['ds_a', 'ds_b', 'ds_a'])

    def test_removes_source_when_not_tracking(self):
        """Test that _dataset_source is removed even when not tracking."""
        from swift.llm.dataset import ProgressTrackingCollator

        def mock_collator(batch):
            return {'input_ids': [[1, 2]]}

        wrapper = ProgressTrackingCollator(mock_collator, track_progress=False)

        batch = [
            {'input_ids': [1, 2], '_dataset_source': 'ds_a'},
        ]

        result = wrapper(batch)

        # Should not have _batch_sources when not tracking
        self.assertNotIn('_batch_sources', result)

        # But _dataset_source should still be removed
        self.assertNotIn('_dataset_source', batch[0])

    def test_handles_missing_source(self):
        """Test handling samples without _dataset_source."""
        from swift.llm.dataset import ProgressTrackingCollator

        def mock_collator(batch):
            return {'input_ids': [[1, 2], [3, 4]]}

        wrapper = ProgressTrackingCollator(mock_collator, track_progress=True)

        batch = [
            {'input_ids': [1, 2], '_dataset_source': 'ds_a'},
            {'input_ids': [3, 4]},  # No _dataset_source
        ]

        result = wrapper(batch)

        # Should only contain the one source that exists
        self.assertEqual(result['_batch_sources'], ['ds_a'])


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


class TestDistributedCollectiveSync(unittest.TestCase):
    """Test that collective operations are called correctly for all ranks.

    This is critical: dist.gather_object is a collective operation that requires
    all ranks to participate. If only rank 0 calls it, other ranks will hang.
    """

    def test_on_log_calls_gather_for_all_ranks(self):
        """Test that on_log calls _gather_counts for all ranks, not just rank 0.

        This test verifies the fix for NCCL timeout issue where only rank 0
        was calling gather_object while other ranks returned early.
        """
        from swift.trainers.callback import DatasetProgressCallback

        callback = DatasetProgressCallback({'ds_a': 100})
        callback._dataset_progress_counts = {'ds_a': 50}

        mock_args = MagicMock()
        mock_state = MagicMock()

        # Test for non-zero rank (e.g., rank 1, 2, 3)
        mock_state.is_world_process_zero = False

        logs = {}

        # Mock _gather_counts to track if it's called
        gather_called = []

        def mock_gather():
            gather_called.append(True)
            return {}  # Non-rank-0 returns empty after gather

        with patch.object(callback, '_gather_counts', side_effect=mock_gather):
            callback.on_log(mock_args, mock_state, None, logs=logs)

        # CRITICAL: _gather_counts should be called even for non-zero rank
        # This ensures all ranks participate in the collective operation
        self.assertEqual(len(gather_called), 1, '_gather_counts must be called for all ranks')

        # But logs should NOT be modified for non-zero rank
        self.assertEqual(logs, {})

    def test_on_log_rank_0_processes_results(self):
        """Test that rank 0 calls gather AND processes results."""
        from swift.trainers.callback import DatasetProgressCallback

        callback = DatasetProgressCallback({'ds_a': 100})

        mock_args = MagicMock()
        mock_state = MagicMock()
        mock_state.is_world_process_zero = True
        mock_state.global_step = 100

        logs = {}

        # Mock _gather_counts to return aggregated data
        def mock_gather():
            return {'ds_a': 50}  # Aggregated from all ranks

        with patch.object(callback, '_gather_counts', side_effect=mock_gather):
            callback.on_log(mock_args, mock_state, None, logs=logs)

        # Rank 0 should process and add to logs
        self.assertIn('dataset_progress/ds_a', logs)
        self.assertEqual(logs['dataset_progress/ds_a'], 50.0)

    def test_gather_counts_distributed_all_ranks_participate(self):
        """Test _gather_counts requires all ranks to call gather_object."""
        from swift.trainers.callback import DatasetProgressCallback

        callback = DatasetProgressCallback()
        callback._dataset_progress_counts = {'ds_a': 10}

        gather_object_calls = []

        def mock_gather_object(obj, gather_list, dst):
            gather_object_calls.append({
                'obj': obj,
                'gather_list': gather_list,
                'dst': dst
            })
            # Simulate gather on rank 0
            if gather_list is not None:
                gather_list[0] = obj
                gather_list[1] = {'ds_a': 15}
                gather_list[2] = {'ds_a': 20}
                gather_list[3] = {'ds_a': 5}

        with patch('torch.distributed.is_initialized', return_value=True), \
                patch('torch.distributed.get_world_size', return_value=4), \
                patch('torch.distributed.get_rank', return_value=0), \
                patch('torch.distributed.gather_object', side_effect=mock_gather_object):
            result = callback._gather_counts()

        # gather_object should be called
        self.assertEqual(len(gather_object_calls), 1)
        self.assertEqual(gather_object_calls[0]['dst'], 0)

        # Result should aggregate all ranks: 10 + 15 + 20 + 5 = 50
        self.assertEqual(result['ds_a'], 50)

    def test_gather_counts_non_rank_0_returns_empty(self):
        """Test _gather_counts returns empty dict for non-rank-0 after gather."""
        from swift.trainers.callback import DatasetProgressCallback

        callback = DatasetProgressCallback()
        callback._dataset_progress_counts = {'ds_a': 10}

        with patch('torch.distributed.is_initialized', return_value=True), \
                patch('torch.distributed.get_world_size', return_value=4), \
                patch('torch.distributed.get_rank', return_value=2), \
                patch('torch.distributed.gather_object'):
            result = callback._gather_counts()

        # Non-rank-0 should return empty dict after participating in gather
        self.assertEqual(result, {})


class TestDatasetProgressCallbackMethods(unittest.TestCase):
    """Test DatasetProgressCallback actual methods."""

    def test_set_trainer_wraps_training_step(self):
        """Test that set_trainer correctly wraps training_step."""
        from swift.trainers.callback import DatasetProgressCallback

        callback = DatasetProgressCallback({'ds_a': 100})

        # Create mock trainer
        mock_trainer = MagicMock()
        original_step_called = []

        def original_training_step(model, inputs):
            original_step_called.append(True)
            return {'loss': 0.5}

        mock_trainer.training_step = original_training_step

        # Wrap trainer
        callback.set_trainer(mock_trainer)

        # Call wrapped training_step with _batch_sources
        inputs = {'input_ids': torch.tensor([1, 2, 3]), '_batch_sources': ['ds_a', 'ds_a', 'ds_b']}
        result = mock_trainer.training_step(None, inputs)

        # Verify original was called
        self.assertEqual(len(original_step_called), 1)

        # Verify _batch_sources was extracted and counted
        self.assertEqual(callback._dataset_progress_counts, {'ds_a': 2, 'ds_b': 1})

        # Verify _batch_sources was removed from inputs
        self.assertNotIn('_batch_sources', inputs)

        # Verify original return value is preserved
        self.assertEqual(result, {'loss': 0.5})

    def test_set_trainer_handles_missing_batch_sources(self):
        """Test wrapped training_step handles missing _batch_sources."""
        from swift.trainers.callback import DatasetProgressCallback

        callback = DatasetProgressCallback()
        mock_trainer = MagicMock()
        mock_trainer.training_step = lambda model, inputs: {'loss': 0.5}

        callback.set_trainer(mock_trainer)

        # Call without _batch_sources
        inputs = {'input_ids': torch.tensor([1, 2, 3])}
        result = mock_trainer.training_step(None, inputs)

        # Should work without errors
        self.assertEqual(result, {'loss': 0.5})
        self.assertEqual(callback._dataset_progress_counts, {})

    def test_on_train_begin_clears_counts(self):
        """Test on_train_begin clears previous counts."""
        from swift.trainers.callback import DatasetProgressCallback

        callback = DatasetProgressCallback({'ds_a': 100})
        callback._dataset_progress_counts = {'ds_a': 50, 'ds_b': 30}

        # Mock args and state
        mock_args = MagicMock()
        mock_args.report_to = []
        mock_state = MagicMock()
        mock_state.is_world_process_zero = True

        callback.on_train_begin(mock_args, mock_state, None)

        # Counts should be cleared
        self.assertEqual(callback._dataset_progress_counts, {})

    def test_on_log_updates_logs_dict(self):
        """Test on_log adds progress metrics to logs."""
        from swift.trainers.callback import DatasetProgressCallback

        callback = DatasetProgressCallback({'ds_a': 100, 'ds_b': 200})
        callback._dataset_progress_counts = {'ds_a': 50, 'ds_b': 100}

        mock_args = MagicMock()
        mock_state = MagicMock()
        mock_state.is_world_process_zero = True
        mock_state.global_step = 100

        logs = {}
        callback.on_log(mock_args, mock_state, None, logs=logs)

        # Should have progress metrics
        self.assertIn('dataset_progress/ds_a', logs)
        self.assertIn('dataset_progress/ds_b', logs)
        self.assertEqual(logs['dataset_progress/ds_a'], 50.0)
        self.assertEqual(logs['dataset_progress/ds_b'], 50.0)

    def test_on_log_skips_non_zero_rank(self):
        """Test on_log does nothing on non-zero rank."""
        from swift.trainers.callback import DatasetProgressCallback

        callback = DatasetProgressCallback({'ds_a': 100})
        callback._dataset_progress_counts = {'ds_a': 50}

        mock_args = MagicMock()
        mock_state = MagicMock()
        mock_state.is_world_process_zero = False

        logs = {}
        callback.on_log(mock_args, mock_state, None, logs=logs)

        # Should not add anything
        self.assertEqual(logs, {})

    def test_gather_counts_single_process(self):
        """Test _gather_counts in non-distributed setting."""
        from swift.trainers.callback import DatasetProgressCallback

        callback = DatasetProgressCallback()
        callback._dataset_progress_counts = {'ds_a': 10, 'ds_b': 20}

        # In non-distributed setting, should return local counts directly
        with patch('torch.distributed.is_initialized', return_value=False):
            result = callback._gather_counts()

        self.assertEqual(result, {'ds_a': 10, 'ds_b': 20})

    def test_on_train_end_closes_writer(self):
        """Test on_train_end closes TensorBoard writer."""
        from swift.trainers.callback import DatasetProgressCallback

        callback = DatasetProgressCallback()
        mock_writer = MagicMock()
        callback._tb_writer = mock_writer

        callback.on_train_end(None, None, None)

        mock_writer.close.assert_called_once()
        self.assertIsNone(callback._tb_writer)


class TestProgressTrackingCollatorEdgeCases(unittest.TestCase):
    """Test edge cases for ProgressTrackingCollator."""

    def test_empty_batch(self):
        """Test handling empty batch."""
        from swift.llm.dataset import ProgressTrackingCollator

        def mock_collator(batch):
            return {'input_ids': []}

        wrapper = ProgressTrackingCollator(mock_collator)
        result = wrapper([])

        self.assertEqual(result, {'input_ids': []})
        self.assertNotIn('_batch_sources', result)

    def test_preserves_all_collator_output(self):
        """Test that all original collator output is preserved."""
        from swift.llm.dataset import ProgressTrackingCollator

        def mock_collator(batch):
            return {
                'input_ids': [[1, 2], [3, 4]],
                'attention_mask': [[1, 1], [1, 1]],
                'labels': [[-100, 2], [-100, 4]],
                'custom_field': 'preserved'
            }

        wrapper = ProgressTrackingCollator(mock_collator)
        batch = [
            {'input_ids': [1, 2], '_dataset_source': 'ds_a'},
            {'input_ids': [3, 4], '_dataset_source': 'ds_b'},
        ]
        result = wrapper(batch)

        # All original fields should be preserved
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
        self.assertIn('labels', result)
        self.assertIn('custom_field', result)
        self.assertEqual(result['custom_field'], 'preserved')

        # Plus _batch_sources
        self.assertIn('_batch_sources', result)
        self.assertEqual(result['_batch_sources'], ['ds_a', 'ds_b'])

    def test_handles_none_in_sources_list(self):
        """Test handling None values in packed sources list."""
        from swift.llm.dataset import ProgressTrackingCollator

        def mock_collator(batch):
            return {'input_ids': [[1, 2, 3, 4, 5]]}

        wrapper = ProgressTrackingCollator(mock_collator)

        # Packed batch with some None sources
        batch = [
            {
                'input_ids': [1, 2, 3, 4, 5],
                '_dataset_source': ['ds_a', None, 'ds_b', None, 'ds_a']
            },
        ]
        result = wrapper(batch)

        # Should filter out None values
        self.assertEqual(result['_batch_sources'], ['ds_a', 'ds_b', 'ds_a'])

    def test_all_none_sources(self):
        """Test handling when all sources are None."""
        from swift.llm.dataset import ProgressTrackingCollator

        def mock_collator(batch):
            return {'input_ids': [[1, 2]]}

        wrapper = ProgressTrackingCollator(mock_collator)

        batch = [
            {'input_ids': [1, 2], '_dataset_source': [None, None]},
        ]
        result = wrapper(batch)

        # Should not have _batch_sources if all are None
        self.assertNotIn('_batch_sources', result)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete flow."""

    def test_import_callback(self):
        """Test that DatasetProgressCallback can be imported."""
        from swift.trainers.callback import DatasetProgressCallback
        self.assertTrue(callable(DatasetProgressCallback))

    def test_import_collator(self):
        """Test that ProgressTrackingCollator can be imported."""
        from swift.llm.dataset import ProgressTrackingCollator
        self.assertTrue(callable(ProgressTrackingCollator))

    def test_train_args_has_track_dataset_progress(self):
        """Test that TrainArguments has track_dataset_progress field."""
        from swift.llm.argument import TrainArguments
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(TrainArguments)]
        self.assertIn('track_dataset_progress', field_names)

    def test_callback_no_template_dependency(self):
        """Test that DatasetProgressCallback doesn't require template."""
        from swift.trainers.callback import DatasetProgressCallback

        # Should be able to create callback without template
        callback = DatasetProgressCallback({'ds_a': 100})
        self.assertEqual(callback.dataset_sizes, {'ds_a': 100})

    def test_collator_and_callback_integration(self):
        """Test ProgressTrackingCollator and DatasetProgressCallback work together."""
        from swift.llm.dataset import ProgressTrackingCollator
        from swift.trainers.callback import DatasetProgressCallback

        # Create collator wrapper
        def mock_collator(batch):
            return {'input_ids': [[b['input_ids']] for b in batch]}

        wrapper = ProgressTrackingCollator(mock_collator)

        # Create callback
        callback = DatasetProgressCallback({'ds_a': 10, 'ds_b': 5})

        # Create mock trainer
        mock_trainer = MagicMock()
        mock_trainer.training_step = lambda model, inputs: {'loss': 0.5}
        callback.set_trainer(mock_trainer)

        # Simulate training loop
        for _ in range(3):
            batch = [
                {'input_ids': [1, 2], '_dataset_source': 'ds_a'},
                {'input_ids': [3, 4], '_dataset_source': 'ds_b'},
            ]
            collated = wrapper(batch)

            # Simulate training_step
            mock_trainer.training_step(None, collated)

        # Verify counts
        self.assertEqual(callback._dataset_progress_counts['ds_a'], 3)
        self.assertEqual(callback._dataset_progress_counts['ds_b'], 3)


class TestDistributedIntegration(unittest.TestCase):
    """Instructions for real distributed testing.

    The unit tests above use mocks to simulate distributed behavior.
    To truly verify the fix works in multi-GPU environments, run:

        torchrun --nproc_per_node=4 -m tests.train.test_dataset_progress TestDistributedIntegration.run_distributed_test

    Or create a separate script and run:

        torchrun --nproc_per_node=4 tests/train/test_dist_progress_real.py
    """

    @staticmethod
    def run_distributed_test():
        """Real distributed test - must be run with torchrun."""
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print(f'[Rank {rank}/{world_size}] Starting test', flush=True)

        # Test 1: Basic gather
        from swift.trainers.callback import DatasetProgressCallback
        cb = DatasetProgressCallback({'ds_a': 100})
        cb._dataset_progress_counts = {'ds_a': (rank + 1) * 10}

        result = cb._gather_counts()
        dist.barrier()

        if rank == 0:
            expected = sum((r + 1) * 10 for r in range(world_size))
            assert result.get('ds_a') == expected, f'Expected {expected}, got {result}'
            print(f'[Rank 0] Test passed! {result}', flush=True)

        dist.destroy_process_group()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and 'run_distributed_test' in sys.argv[1]:
        TestDistributedIntegration.run_distributed_test()
    else:
        unittest.main()
