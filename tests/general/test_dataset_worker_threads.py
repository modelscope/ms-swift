"""Tests for capping the torch threads of the dataset preprocessing worker processes.

With `dataset_num_proc > 1`, every `datasets.map` worker used to keep one torch thread per CPU core,
so the CPU ops of the image/video processors oversubscribed the cores (`num_proc * cpu_count` threads).
"""
import os
import tempfile
import torch
import unittest
from datasets import Dataset
from unittest import mock

from swift.dataset import RowPreprocessor
from swift.dataset.packing import IterablePackingDataset
from swift.utils import limit_worker_num_threads


def _cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class ThreadsRecorder(RowPreprocessor):

    def preprocess(self, row):
        row['num_threads'] = torch.get_num_threads()
        row['pid'] = os.getpid()
        return row


SAMPLES = [{
    'messages': [{
        'role': 'user',
        'content': str(i)
    }, {
        'role': 'assistant',
        'content': str(i)
    }]
} for i in range(8)]


class TestLimitWorkerNumThreads(unittest.TestCase):

    def setUp(self):
        self.num_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, self.num_threads)

    def _limit(self, num_workers, cpu_count, num_threads, env=None):
        torch.set_num_threads(num_threads)
        with mock.patch.dict(os.environ, env or {}, clear=False), \
                mock.patch('os.sched_getaffinity', return_value=set(range(cpu_count)), create=True):
            if env is None:
                os.environ.pop('OMP_NUM_THREADS', None)
            limit_worker_num_threads(num_workers)
        return torch.get_num_threads()

    def test_split_cores_between_workers(self):
        self.assertEqual(self._limit(4, cpu_count=16, num_threads=16), 4)
        self.assertEqual(self._limit(3, cpu_count=16, num_threads=16), 5)
        self.assertEqual(self._limit(32, cpu_count=16, num_threads=16), 1)

    def test_single_worker_is_unchanged(self):
        self.assertEqual(self._limit(1, cpu_count=16, num_threads=16), 16)

    def test_never_increase_threads(self):
        self.assertEqual(self._limit(2, cpu_count=16, num_threads=2), 2)

    def test_respect_omp_num_threads(self):
        self.assertEqual(self._limit(4, cpu_count=16, num_threads=16, env={'OMP_NUM_THREADS': '16'}), 16)


@unittest.skipIf('OMP_NUM_THREADS' in os.environ, 'An explicit OMP_NUM_THREADS disables the cap.')
class TestRowPreprocessorWorkerThreads(unittest.TestCase):

    def setUp(self):
        # In-memory datasets are mapped into `MODELSCOPE_CACHE`; keep every test isolated.
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        patcher = mock.patch.dict(os.environ, {'MODELSCOPE_CACHE': temp_dir.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _map(self, num_proc):
        return ThreadsRecorder()(
            Dataset.from_list(SAMPLES), num_proc=num_proc, batch_size=2, load_from_cache_file=False)

    def test_map_workers_split_cores(self):
        num_proc = 2
        dataset = self._map(num_proc)
        self.assertEqual(len(dataset), len(SAMPLES))
        self.assertNotIn(os.getpid(), set(dataset['pid']))
        expected = min(torch.get_num_threads(), max(_cpu_count() // num_proc, 1))
        self.assertEqual(set(dataset['num_threads']), {expected})

    def test_single_process_is_unchanged(self):
        num_threads = torch.get_num_threads()
        dataset = self._map(1)
        self.assertEqual(set(dataset['num_threads']), {num_threads})
        self.assertEqual(torch.get_num_threads(), num_threads)


class TestIterablePackingWorkerThreads(unittest.TestCase):

    def test_worker_jobs_pass_num_proc(self):
        dataset = IterablePackingDataset.__new__(IterablePackingDataset)
        dataset._in_queue, dataset._out_queue = object(), object()
        dataset.template, dataset.strict, dataset.num_proc = object(), False, 3
        jobs = dataset._worker_jobs()
        self.assertEqual(len(jobs), 3)
        self.assertTrue(all(job[-1] == 3 for job in jobs))


if __name__ == '__main__':
    unittest.main()
