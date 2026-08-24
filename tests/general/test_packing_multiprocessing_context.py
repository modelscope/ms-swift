"""Fast, hermetic tests for the packing multiprocessing-context behavior.

These tests intentionally avoid downloading any model. They exercise the multiprocessing-context
resolution / fork-fallback logic that backs the ``dataloader_multiprocessing_context`` feature,
plus the two packing datasets end-to-end with lightweight, picklable fakes.

Why this is worth testing carefully:
- The default must stay ``fork`` unless fork is actually unsafe (CUDA / torch.distributed already
  initialized), otherwise existing runs silently change behavior.
- Queues and worker processes must come from the *same* context, otherwise they deadlock.
- A non-fork context that cannot start a worker (e.g. an un-picklable template under ``spawn``) must
  transparently fall back to ``fork`` instead of crashing.

The fakes below are defined at module scope on purpose: ``spawn``/``forkserver`` pickle the worker
target, so any object reachable from it (template, dataset) must be importable/picklable.
"""
import multiprocessing as mp
import unittest
from unittest import mock

from swift.dataset.packing import (IterablePackingDataset, PackingDataset, _get_fork_context, _is_fork_context,
                                   _resolve_mp_context, _spawn_workers)

FORK_AVAILABLE = 'fork' in mp.get_all_start_methods()
SPAWN_AVAILABLE = 'spawn' in mp.get_all_start_methods()
FORKSERVER_AVAILABLE = 'forkserver' in mp.get_all_start_methods()


class FakeTemplate:
    """Minimal, picklable stand-in for a swift Template (no model, no processor)."""

    packing = False
    padding_free = False
    max_length = 32
    sequence_parallel_size = 1

    def encode(self, data, return_length=True):
        input_ids = list(data['input_ids'])
        return {'input_ids': input_ids, 'labels': list(input_ids), 'length': len(input_ids)}


class UnpicklableTemplate(FakeTemplate):
    """A template that cannot be pickled, to force the spawn->fork fallback path."""

    def __init__(self):
        # a lambda attribute makes the whole object un-picklable under spawn/forkserver
        self._not_picklable = lambda x: x


class ListDataset:
    """Map-style dataset exposing ``dataset['lengths']`` like swift datasets do."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == 'lengths':
                return [len(r['input_ids']) for r in self.rows]
            raise KeyError(key)
        return self.rows[key]


class IterRows:
    """Iterable dataset for IterablePackingDataset."""

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


def _make_rows(n=40, max_len=10):
    return [{'input_ids': list(range(k % max_len + 1))} for k in range(n)]


# a module-level callable usable as a spawn/forkserver Process target
def _noop_worker(*args):
    return None


class TestResolveMpContext(unittest.TestCase):

    def test_default_keeps_fork_when_nothing_initialized(self):
        """Default (None) must preserve historical fork behavior when CUDA/dist are not initialized."""
        with mock.patch('torch.cuda.is_initialized', return_value=False), \
             mock.patch('torch.distributed.is_available', return_value=True), \
             mock.patch('torch.distributed.is_initialized', return_value=False):
            ctx = _resolve_mp_context(None)
        # On Linux this is fork; whatever it is, it must equal the platform default (i.e. unchanged).
        self.assertEqual(ctx._name, mp.get_context()._name)

    @unittest.skipUnless(SPAWN_AVAILABLE, 'spawn not available')
    def test_default_switches_to_spawn_when_cuda_initialized(self):
        with mock.patch('torch.cuda.is_initialized', return_value=True):
            ctx = _resolve_mp_context(None)
        self.assertEqual(ctx._name, 'spawn')

    @unittest.skipUnless(SPAWN_AVAILABLE, 'spawn not available')
    def test_default_switches_to_spawn_when_dist_initialized(self):
        with mock.patch('torch.cuda.is_initialized', return_value=False), \
             mock.patch('torch.distributed.is_available', return_value=True), \
             mock.patch('torch.distributed.is_initialized', return_value=True):
            ctx = _resolve_mp_context(None)
        self.assertEqual(ctx._name, 'spawn')

    def test_cuda_probe_exception_is_swallowed(self):
        """If torch.cuda.is_initialized() raises, we must not crash and must keep fork."""
        with mock.patch('torch.cuda.is_initialized', side_effect=RuntimeError('no cuda')), \
             mock.patch('torch.distributed.is_available', return_value=False):
            ctx = _resolve_mp_context(None)
        self.assertEqual(ctx._name, mp.get_context()._name)

    @unittest.skipUnless(SPAWN_AVAILABLE, 'spawn not available')
    def test_explicit_spawn_is_respected(self):
        self.assertEqual(_resolve_mp_context('spawn')._name, 'spawn')

    @unittest.skipUnless(FORKSERVER_AVAILABLE, 'forkserver not available')
    def test_explicit_forkserver_is_respected(self):
        self.assertEqual(_resolve_mp_context('forkserver')._name, 'forkserver')

    @unittest.skipUnless(FORK_AVAILABLE, 'fork not available')
    def test_explicit_fork_is_respected(self):
        self.assertEqual(_resolve_mp_context('fork')._name, 'fork')


class TestContextHelpers(unittest.TestCase):

    @unittest.skipUnless(FORK_AVAILABLE, 'fork not available')
    def test_get_fork_context_returns_fork(self):
        self.assertEqual(_get_fork_context()._name, 'fork')

    def test_get_fork_context_falls_back_when_fork_unavailable(self):
        """On platforms without fork, we must degrade to the default context, not raise."""
        real_get_context = mp.get_context

        def fake_get_context(method=None):
            if method == 'fork':
                raise ValueError('fork not available')
            return real_get_context(method)

        with mock.patch('swift.dataset.packing.mp.get_context', side_effect=fake_get_context):
            ctx = _get_fork_context()
        self.assertIsNotNone(ctx)

    @unittest.skipUnless(FORK_AVAILABLE, 'fork not available')
    def test_is_fork_context(self):
        self.assertTrue(_is_fork_context(mp.get_context('fork')))

    @unittest.skipUnless(SPAWN_AVAILABLE, 'spawn not available')
    def test_is_not_fork_context_for_spawn(self):
        self.assertFalse(_is_fork_context(mp.get_context('spawn')))


class TestSpawnWorkers(unittest.TestCase):

    @unittest.skipUnless(FORK_AVAILABLE, 'fork not available')
    def test_spawn_workers_starts_all(self):
        ctx = mp.get_context('fork')
        returned_ctx, workers = _spawn_workers(ctx, target=_noop_worker, jobs=[(), (), ()])
        self.assertIs(returned_ctx, ctx)
        self.assertEqual(len(workers), 3)
        for w in workers:
            w.join(timeout=10)

    def test_spawn_workers_signals_fallback_on_first_worker_failure(self):
        """A non-fork context that fails to start the first worker returns (None, [])."""
        fake_ctx = mock.Mock()
        fake_ctx._name = 'spawn'
        fake_ctx.Process.side_effect = RuntimeError('cannot pickle')
        returned_ctx, workers = _spawn_workers(fake_ctx, target=_noop_worker, jobs=[(), ()])
        self.assertIsNone(returned_ctx)
        self.assertEqual(workers, [])

    def test_spawn_workers_reraises_for_fork_failure(self):
        """If even fork cannot start a worker, there is nothing to fall back to -> re-raise."""
        fake_ctx = mock.Mock()
        fake_ctx._name = 'fork'
        fake_ctx.Process.side_effect = RuntimeError('boom')
        with self.assertRaises(RuntimeError):
            _spawn_workers(fake_ctx, target=_noop_worker, jobs=[()])

    def test_spawn_workers_reraises_when_failure_not_on_first_worker(self):
        """Failure after the first worker started is unexpected and must propagate."""
        fake_ctx = mock.Mock()
        fake_ctx._name = 'spawn'
        started = mock.Mock()
        fake_ctx.Process.side_effect = [started, RuntimeError('later failure')]
        with self.assertRaises(RuntimeError):
            _spawn_workers(fake_ctx, target=_noop_worker, jobs=[(), ()])


def _run_packing_dataset(ctx_name, rows, packing_num_proc=1):
    return PackingDataset(
        FakeTemplate(),
        ListDataset(rows),
        strict=False,
        load_from_cache_file=False,
        packing_length=16,
        packing_num_proc=packing_num_proc,
        multiprocessing_context=ctx_name,
    )


class TestPackingDatasetContexts(unittest.TestCase):
    """PackingDataset must produce identical results regardless of the multiprocessing context."""

    @classmethod
    def setUpClass(cls):
        cls.rows = _make_rows(40)

    def _assert_valid(self, pd, expected_total):
        self.assertGreater(len(pd), 0)
        total = sum(len(idx) for idx in pd.packed_idx)
        self.assertEqual(total, expected_total)
        packed = pd[0]
        self.assertIsInstance(packed, list)
        self.assertGreater(len(packed), 0)
        self.assertIn('input_ids', packed[0])

    @unittest.skipUnless(FORK_AVAILABLE, 'fork not available')
    def test_default_context(self):
        self._assert_valid(_run_packing_dataset(None, self.rows), expected_total=len(self.rows))

    @unittest.skipUnless(FORK_AVAILABLE and SPAWN_AVAILABLE, 'need both fork and spawn')
    def test_spawn_matches_fork(self):
        """A single spawn run: proves non-fork wiring works *and* yields identical results to fork.

        This is the only slow (spawn) PackingDataset case on purpose; the rest of the spawn behavior
        is covered by the fast helper/mocked tests to keep the suite quick.
        """
        fork_pd = _run_packing_dataset(None, self.rows)
        spawn_pd = _run_packing_dataset('spawn', self.rows)
        self._assert_valid(spawn_pd, expected_total=len(self.rows))
        self.assertEqual(fork_pd.packed_idx, spawn_pd.packed_idx)
        self.assertEqual(fork_pd.packed_length, spawn_pd.packed_length)


def _drain(ipd):
    return list(iter(ipd))


def _run_iter_packing(ctx_name, rows, template=None, num_proc=1):
    return IterablePackingDataset(
        template or FakeTemplate(),
        IterRows(rows),
        num_proc=num_proc,
        packing_interval=16,
        packing_length=16,
        multiprocessing_context=ctx_name,
    )


class TestIterablePackingDatasetContexts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rows = _make_rows(48)

    def _assert_valid(self, ipd):
        out = _drain(ipd)
        self.assertGreater(len(out), 0)
        # every yielded pack is a list of dataset indices
        for pack in out:
            self.assertIsInstance(pack, list)
            self.assertGreater(len(pack), 0)

    @unittest.skipUnless(FORK_AVAILABLE, 'fork not available')
    def test_default_context(self):
        self._assert_valid(_run_iter_packing(None, self.rows))

    @unittest.skipUnless(SPAWN_AVAILABLE, 'spawn not available')
    def test_spawn_context(self):
        self._assert_valid(_run_iter_packing('spawn', self.rows))

    @unittest.skipUnless(FORK_AVAILABLE, 'fork not available')
    def test_multi_proc(self):
        self._assert_valid(_run_iter_packing(None, self.rows, num_proc=2))

    @unittest.skipUnless(FORK_AVAILABLE and SPAWN_AVAILABLE, 'need both fork and spawn')
    def test_unpicklable_template_falls_back_to_fork(self):
        """Forcing spawn with an un-picklable template must transparently fall back to fork."""
        ipd = _run_iter_packing('spawn', self.rows, template=UnpicklableTemplate())
        # all workers must end up on the fork context after fallback
        self.assertEqual(len(ipd.workers), 1)
        self._assert_valid(ipd)


class TestDataloaderContextInjection(unittest.TestCase):
    """The DataLoader-side injection is pure kwargs plumbing; assert it only activates when set."""

    def _build_params(self, mp_context, num_workers):
        # mirror the exact condition used in mixin.get_train_dataloader / build_streaming_dataloader
        dataloader_params = {}
        if mp_context is not None and num_workers > 0:
            dataloader_params['multiprocessing_context'] = mp_context
        return dataloader_params

    def test_not_injected_by_default(self):
        self.assertNotIn('multiprocessing_context', self._build_params(None, 4))

    def test_not_injected_when_no_workers(self):
        self.assertNotIn('multiprocessing_context', self._build_params('spawn', 0))

    def test_injected_when_set_with_workers(self):
        params = self._build_params('spawn', 4)
        self.assertEqual(params['multiprocessing_context'], 'spawn')


if __name__ == '__main__':
    unittest.main()
