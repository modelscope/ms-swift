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
import importlib
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import tempfile
import torch
import types
import unittest
from torch.utils.data import DataLoader
from unittest import mock

import swift.utils.utils as utils_module
from swift.dataset.packing import IterablePackingDataset, PackingDataset, _resolve_mp_context, _spawn_workers
from swift.template.base import Template
from swift.utils import get_external_files, import_external_file, patch_dataloader_external_plugins

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


class _TemplateLike:
    """Picklable stand-in that goes through the real ``Template.__getstate__`` under test."""

    __getstate__ = Template.__getstate__


def _hooked_model():
    """A model carrying the kind of hook ``enable_input_require_grads`` installs: a local closure.

    transformers registers exactly this shape, so anything holding a reference to a hooked model cannot be
    pickled -- which is what forkserver/spawn dataloader workers hit on Python 3.14.
    """
    model = torch.nn.Linear(2, 2)

    def make_inputs_require_grads(module, args, output):
        output.requires_grad_(True)

    return model, model.register_forward_hook(make_inputs_require_grads)


def _stashed_deepspeed_initialize(*args, **kwargs):
    """Stands in for the original ``deepspeed.initialize`` that zero3 makes the template stash."""
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

    def test_hook_bookkeeping_is_dropped_so_no_model_is_reachable(self):
        """``_handles`` pairs every hook with its live model, so keeping it re-pickles the very model that
        ``model = None`` drops -- and a hooked model reaches un-picklable local closures."""
        model, handle = _hooked_model()
        template = _TemplateLike()
        template.model = model
        template.dummy_model = None
        template._handles = [(model, handle)]
        template._deepspeed_initialize = None
        template.max_length = 8

        restored = pickle.loads(pickle.dumps(template))

        self.assertIsNone(restored.model)
        self.assertEqual(restored._handles, [])
        self.assertEqual(restored.max_length, 8)

    def test_the_live_template_keeps_its_hooks(self):
        """Workers are spawned mid-run, so pickling must leave ``remove_post_encode_hook`` able to undo."""
        model, handle = _hooked_model()
        template = _TemplateLike()
        template.model = model
        template.dummy_model = None
        template._handles = [(model, handle)]
        template._deepspeed_initialize = _stashed_deepspeed_initialize

        Template.__getstate__(template)

        self.assertEqual(template._handles, [(model, handle)])
        self.assertIs(template.model, model)
        self.assertIs(template._deepspeed_initialize, _stashed_deepspeed_initialize)

    def test_stashed_deepspeed_initialize_is_dropped(self):
        """Under zero3 the template stashes the original ``deepspeed.initialize`` and rebinds the module
        attribute to a wrapper. pickle stores functions by reference and refuses one whose module attribute
        no longer points back at it, so the stash cannot cross a process boundary either."""
        template = _TemplateLike()
        template.model = None
        template.dummy_model = None
        template._handles = []
        template._deepspeed_initialize = _stashed_deepspeed_initialize

        with mock.patch(f'{__name__}._stashed_deepspeed_initialize', lambda *args, **kwargs: None):
            # the hazard is real: the stashed original is no longer what its module name resolves to
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(template._deepspeed_initialize)
            restored = pickle.loads(pickle.dumps(template))

        self.assertIsNone(restored._deepspeed_initialize)


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


# --- external plugin replay in workers -------------------------------------------------------------
#
# A swift plugin takes effect purely through import side effects (``import_external_file`` just execs
# the file). Under fork the worker inherits those side effects for free; under forkserver/spawn it
# starts clean, so anything the worker looks up by name afterwards -- an extra PIL codec being the
# motivating case -- is silently missing unless the import is replayed. Every test below pairs the
# patched case with an unpatched control, so a regression shows up as the two agreeing.

CODEC_MODULE = '_swift_test_codec'


def _codec_registry_size():
    """Count the plugin's own entries in the stand-in registry, 0 if the plugin never ran here.

    Counts only ``JXL`` so that a caller-supplied worker_init_fn writing its own marker into the same
    registry does not inflate the number. Deliberately importable-or-zero rather than raising: that is
    exactly the shape of the bug, where a worker silently lacks a codec instead of failing loudly.
    """
    try:
        module = importlib.import_module(CODEC_MODULE)
    except ImportError:
        return 0
    return module.REGISTRY.count('JXL')


def _record_worker_init(worker_id):
    """A caller-supplied worker_init_fn; the patch must delegate to it, not replace it."""
    importlib.import_module(CODEC_MODULE).REGISTRY.append(f'inner-{worker_id}')


class CodecProbeTemplate(FakeTemplate):
    """Reports, through ``length``, whether the plugin took effect in the packing worker."""

    def encode(self, data, return_length=True):
        return {'input_ids': [1], 'labels': [1], 'length': _codec_registry_size()}


class CodecProbeDataset:
    """``__getitem__`` runs in the dataloader worker, like ``Template.encode`` -> ``load_image`` does."""

    def __len__(self):
        return 2

    def __getitem__(self, index):
        try:
            registry = importlib.import_module(CODEC_MODULE).REGISTRY
        except ImportError:
            registry = []
        return registry.count('JXL'), int(any(str(entry).startswith('inner-') for entry in registry))


class TestExternalPluginReplayInWorkers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        # stands in for a third-party lib whose global registry is consulted inside the worker
        with open(os.path.join(cls.tmpdir, f'{CODEC_MODULE}.py'), 'w') as f:
            f.write('REGISTRY = []\n')
        # stands in for the user's one-line plugin: `import pillow_jxl`
        cls.plugin = os.path.join(cls.tmpdir, '_swift_test_plugin.py')
        with open(cls.plugin, 'w') as f:
            f.write(f'import {CODEC_MODULE}\n{CODEC_MODULE}.REGISTRY.append("JXL")\n')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        self._saved_external = list(utils_module._external_files)
        self._saved_modules = set(sys.modules)
        self._saved_path = list(sys.path)
        self._saved_init = DataLoader.__init__
        self._was_patched = getattr(DataLoader, '_swift_external_plugins', False)
        import_external_file(self.plugin)  # what BaseArguments._import_external_plugins does
        self.assertEqual(_codec_registry_size(), 1, 'plugin should have taken effect in the parent')

    def tearDown(self):
        DataLoader.__init__ = self._saved_init
        DataLoader._swift_external_plugins = self._was_patched
        utils_module._external_files[:] = self._saved_external
        for name in set(sys.modules) - self._saved_modules:
            del sys.modules[name]
        sys.path[:] = self._saved_path

    def test_get_external_files_records_the_plugin(self):
        self.assertIn(self.plugin, get_external_files())

    def test_get_external_files_does_not_duplicate(self):
        import_external_file(self.plugin)
        self.assertEqual(get_external_files().count(self.plugin), 1)

    def _dataloader_sees(self, ctx_name, worker_init_fn=None):
        loader = DataLoader(
            CodecProbeDataset(),
            batch_size=2,
            num_workers=1,
            multiprocessing_context=ctx_name,
            worker_init_fn=worker_init_fn,
        )
        plugin, inner = next(iter(loader))
        return plugin.tolist(), inner.tolist()

    @unittest.skipUnless(FORK_AVAILABLE, 'fork not available')
    def test_fork_inherits_even_unpatched(self):
        self.assertEqual(self._dataloader_sees('fork')[0], [1, 1])

    @unittest.skipUnless(SPAWN_AVAILABLE, 'spawn not available')
    def test_spawn_needs_the_patch(self):
        self.assertEqual(self._dataloader_sees('spawn')[0], [0, 0], 'control: spawn must start clean')
        patch_dataloader_external_plugins()
        self.assertEqual(self._dataloader_sees('spawn')[0], [1, 1])

    @unittest.skipUnless(FORKSERVER_AVAILABLE, 'forkserver not available')
    def test_forkserver_needs_the_patch(self):
        self.assertEqual(self._dataloader_sees('forkserver')[0], [0, 0], 'control: forkserver starts clean')
        patch_dataloader_external_plugins()
        self.assertEqual(self._dataloader_sees('forkserver')[0], [1, 1])

    @unittest.skipUnless(SPAWN_AVAILABLE, 'spawn not available')
    def test_patch_delegates_to_a_caller_supplied_worker_init_fn(self):
        patch_dataloader_external_plugins()
        plugin, inner = self._dataloader_sees('spawn', worker_init_fn=_record_worker_init)
        self.assertEqual(plugin, [1, 1])
        self.assertEqual(inner, [1, 1], 'the original worker_init_fn must still run')

    def test_patch_is_idempotent_and_does_not_nest(self):
        patch_dataloader_external_plugins()
        patched_init = DataLoader.__init__
        patch_dataloader_external_plugins()
        self.assertIs(DataLoader.__init__, patched_init, 'patching twice must not re-wrap')
        loader = DataLoader(CodecProbeDataset(), batch_size=2, num_workers=1, worker_init_fn=_record_worker_init)
        self.assertIs(loader.worker_init_fn.inner, _record_worker_init, 'the wrapper must not stack')

    def test_patch_leaves_single_process_loaders_alone(self):
        patch_dataloader_external_plugins()
        loader = DataLoader(CodecProbeDataset(), batch_size=2, num_workers=0)
        self.assertIsNone(loader.worker_init_fn, 'nothing to replay without workers')

    def test_packing_worker_job_carries_the_plugin_paths(self):
        # IterablePackingDataset drives raw ctx.Process workers, not a DataLoader, so the constructor
        # patch cannot reach them; the paths ride along in the job tuple instead.
        ipd = _run_iter_packing(None, _make_rows(4), template=CodecProbeTemplate())
        try:
            self.assertEqual(ipd._worker_jobs()[0][-1], get_external_files())
        finally:
            for worker in ipd.workers:
                worker.terminate()

    def _packing_worker_sees(self, ctx_name, external_files):
        ctx = mp.get_context(ctx_name)
        in_queue, out_queue = ctx.Queue(), ctx.Queue()
        worker = ctx.Process(
            target=IterablePackingDataset._processor,
            args=(in_queue, out_queue, CodecProbeTemplate(), False, external_files),
            daemon=True)
        worker.start()
        try:
            in_queue.put((0, {'input_ids': [1]}))
            return out_queue.get(timeout=60)[1]['length']
        finally:
            worker.terminate()
            worker.join(timeout=10)

    @unittest.skipUnless(SPAWN_AVAILABLE, 'spawn not available')
    def test_packing_worker_spawn_needs_replay(self):
        self.assertEqual(self._packing_worker_sees('spawn', ()), 0, 'control: spawn must start clean')
        self.assertEqual(self._packing_worker_sees('spawn', get_external_files()), 1)

    @unittest.skipUnless(FORKSERVER_AVAILABLE, 'forkserver not available')
    def test_packing_worker_forkserver_needs_replay(self):
        self.assertEqual(self._packing_worker_sees('forkserver', ()), 0, 'control: forkserver starts clean')
        self.assertEqual(self._packing_worker_sees('forkserver', get_external_files()), 1)


if __name__ == '__main__':
    unittest.main()
