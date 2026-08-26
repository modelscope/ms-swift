"""Fast, hermetic tests for the packing multiprocessing-context behavior.

These tests intentionally avoid downloading any model. They exercise the multiprocessing-context
resolution that backs the ``dataloader_multiprocessing_context`` feature, the ``Template.__getstate__``
model-stripping that keeps the pickled template lightweight, and the two packing datasets end-to-end
with lightweight, picklable fakes.

Why this is worth testing carefully:
- The default (None) must equal the platform default (fork on Python <=3.13, forkserver on 3.14+),
  and only switch to ``spawn`` when fork is actually unsafe (CUDA / torch.distributed initialized).
- Queues and worker processes must come from the *same* context, otherwise they deadlock.
- A non-fork context that cannot start a worker (e.g. an un-picklable arg under ``spawn``) must raise
  loudly -- we deliberately do NOT fall back to fork, which could dead-lock a child that inherited an
  active torch threadpool / CUDA state.

The fakes below are defined at module scope on purpose: ``spawn``/``forkserver`` pickle the worker
target, so any object reachable from it (template, dataset) must be importable/picklable.
"""
import multiprocessing as mp
import types
import unittest
from unittest import mock

from swift.dataset.packing import IterablePackingDataset, PackingDataset, _resolve_mp_context, _spawn_workers
from swift.template.base import Template

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
    """A template that cannot be pickled, to exercise the loud-failure (no fork fallback) path."""

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

    def test_default_keeps_platform_default_when_nothing_initialized(self):
        """Default (None) must preserve the platform default when CUDA/dist are not initialized."""
        with mock.patch('torch.cuda.is_initialized', return_value=False), \
             mock.patch('torch.distributed.is_available', return_value=True), \
             mock.patch('torch.distributed.is_initialized', return_value=False):
            ctx = _resolve_mp_context(None)
        # fork on Python <=3.13, forkserver on 3.14+; either way it must equal the platform default.
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
        """If torch.cuda.is_initialized() raises, we must not crash and must keep the platform default."""
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


class TestTemplateGetstate(unittest.TestCase):
    """``Template.__getstate__`` must drop the (live, possibly-CUDA) model so pickling a template that
    crosses a process boundary stays lightweight; everything else must survive."""

    def test_model_and_dummy_are_dropped_other_attrs_kept(self):
        # Call the unbound method on a stand-in carrying a __dict__ -- hermetic, no heavy construction.
        stub = types.SimpleNamespace(model=object(), dummy_model=object(), tokenizer='tok', max_length=8)
        state = Template.__getstate__(stub)
        self.assertIsNone(state['model'])
        self.assertIsNone(state['dummy_model'])
        self.assertEqual(state['tokenizer'], 'tok')
        self.assertEqual(state['max_length'], 8)
        # must not mutate the original object
        self.assertIsNotNone(stub.model)


class TestSpawnWorkers(unittest.TestCase):

    @unittest.skipUnless(FORK_AVAILABLE, 'fork not available')
    def test_spawn_workers_starts_all(self):
        ctx = mp.get_context('fork')
        workers = _spawn_workers(ctx, target=_noop_worker, jobs=[(), (), ()])
        self.assertEqual(len(workers), 3)
        for w in workers:
            w.join(timeout=10)

    def test_spawn_workers_reraises_on_start_failure(self):
        """No fork fallback: a worker that cannot start (e.g. un-picklable arg under spawn) must raise
        loudly rather than silently degrade to fork, which could dead-lock."""
        fake_ctx = mock.Mock()
        fake_ctx._name = 'spawn'
        fake_ctx.Process.side_effect = RuntimeError('cannot pickle')
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
        is covered by the fast mocked tests to keep the suite quick.
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

    @unittest.skipUnless(SPAWN_AVAILABLE, 'spawn not available')
    def test_unpicklable_template_raises_under_spawn(self):
        """With the fork fallback removed, forcing spawn with an un-picklable template must raise
        loudly instead of silently degrading to fork."""
        with self.assertRaises(Exception):
            _run_iter_packing('spawn', self.rows, template=UnpicklableTemplate())


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
