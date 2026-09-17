import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import swift
import swift.utils.utils as utils_module
from swift.utils import get_external_files, import_external_file, patch_dataloader_external_plugins


class TestExternalPlugins(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.saved_path = sys.path[:]
        self.saved_external = get_external_files()

    def tearDown(self):
        sys.path[:] = self.saved_path
        utils_module._external_files[:] = self.saved_external
        for name, module in list(sys.modules.items()):
            if str(getattr(module, '__file__', '')).startswith(self.tmp.name + os.sep):
                sys.modules.pop(name, None)

    def plugin(self, directory, source, filename='plugin.py'):
        path = Path(self.tmp.name) / directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf-8')
        return str(path)

    def test_fresh_interpreter_unpickles_same_named_plugins(self):
        modules = []
        for directory in ('first', 'second'):
            path = self.plugin(
                f'目录 {directory}',
                f'''
from dataclasses import dataclass
VALUE = {directory!r}

@dataclass
class Record:
    value: int

    def render(self):
        return [VALUE, self.value]

def label():
    return VALUE
''',
                filename='my-plugin.v1.py')
            module = import_external_file(path)
            self.assertIs(import_external_file(path), module)
            modules.append(module)
        first, second = modules
        payload = pickle.dumps((first.Record(1), second.Record(2), first.Record, first.label, second.Record(3).render))
        result = subprocess.run([
            sys.executable, '-c', '''
import json, pickle, sys
a, b, cls, function, method = pickle.loads(sys.stdin.buffer.read())
assert type(a) is cls
assert type(a) is not type(b)
print(json.dumps([a.render(), b.render(), cls(4).render(), function(), method()]))
'''
        ],
                                input=payload,
                                capture_output=True,
                                timeout=60,
                                cwd=Path(swift.__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(json.loads(result.stdout), [['first', 1], ['second', 2], ['first', 4], 'first', ['second', 3]])

    def test_path_normalization_and_non_identifier_filename(self):
        path = self.plugin('目录 with spaces', 'VALUE = object()\n', filename='my-plugin.v1.py')
        module = import_external_file(path)
        relative = os.path.relpath(path)
        self.assertIs(import_external_file(relative), module)
        self.assertIs(import_external_file(os.path.join(os.path.dirname(path), '.', os.path.basename(path))), module)
        self.assertEqual(get_external_files().count(path), 1)
        self.assertIs(importlib.import_module(module.__name__), module)

    def test_failed_import_can_be_retried(self):
        path = self.plugin('broken', 'raise RuntimeError("plugin failed")\n')
        with self.assertRaisesRegex(RuntimeError, 'plugin failed'):
            import_external_file(path)
        self.assertFalse(any(getattr(module, '__file__', None) == path for module in list(sys.modules.values())))
        Path(path).write_text('VALUE = "recovered"\n')
        self.assertEqual(import_external_file(path).VALUE, 'recovered')

    def test_package_relative_imports_still_work(self):
        self.plugin('package', 'VALUE = 42\n', filename='helper.py')
        path = self.plugin('package', 'from .helper import VALUE\n', filename='__init__.py')
        module = import_external_file(path)
        self.assertEqual(module.VALUE, 42)
        self.assertEqual(importlib.import_module(f'{module.__name__}.helper').VALUE, 42)

    def test_legacy_pickle_after_importing_plugin(self):
        path = self.plugin('legacy', '''
class Record:
    def __init__(self, value):
        self.value = value
''')
        # Produce an actual pickle using the former import_external_file module naming scheme.
        legacy_name = f'_swift_external_{hashlib.sha256(path.encode()).hexdigest()}'
        spec = importlib.util.spec_from_file_location(legacy_name, path)
        legacy_module = importlib.util.module_from_spec(spec)
        sys.modules[legacy_name] = legacy_module
        spec.loader.exec_module(legacy_module)
        payload = pickle.dumps(legacy_module.Record(7))
        del sys.modules[legacy_name]

        module = import_external_file(path)
        restored = pickle.loads(payload)
        self.assertIs(type(restored), module.Record)
        self.assertEqual(restored.value, 7)

    def test_legacy_package_pickles_preserve_submodule_identity(self):
        for eager in (False, True):
            for submodule in ('helper', 'nested.helper'):
                for legacy_first in (False, True):
                    with self.subTest(eager=eager, submodule=submodule, legacy_first=legacy_first):
                        directory = f'package-{eager}-{submodule}-{legacy_first}'
                        counter = Path(self.tmp.name) / directory / 'imports.txt'
                        self.plugin(
                            directory,
                            f'''
with open({str(counter)!r}, 'a') as f:
    f.write('imported\\n')

class Record:
    def __init__(self, value):
        self.value = value
''',
                            filename=submodule.replace('.', '/') + '.py')
                        if '.' in submodule:
                            self.plugin(directory, '', filename='nested/__init__.py')
                        source = f'from .{submodule} import Record\n' if eager else ''
                        path = self.plugin(directory, source, filename='__init__.py')
                        legacy = f'_swift_external_{hashlib.sha256(path.encode()).hexdigest()}'
                        spec = importlib.util.spec_from_file_location(legacy, path)
                        old = importlib.util.module_from_spec(spec)
                        sys.modules[legacy] = old
                        spec.loader.exec_module(old)
                        old_child = importlib.import_module(f'{legacy}.{submodule}')
                        payload = pickle.dumps(old_child.Record(7))
                        for name in list(sys.modules):
                            if name == legacy or name.startswith(legacy + '.'):
                                del sys.modules[name]
                        counter.write_text('')

                        package = import_external_file(path)
                        canonical_name = f'{package.__name__}.{submodule}'
                        if legacy_first:
                            restored = pickle.loads(payload)
                            child = importlib.import_module(canonical_name)
                        else:
                            child = importlib.import_module(canonical_name)
                            restored = pickle.loads(payload)
                        self.assertIs(type(restored), child.Record)
                        self.assertEqual(restored.value, 7)
                        self.assertIs(importlib.import_module(f'{legacy}.{submodule}'), child)
                        self.assertEqual(counter.read_text(), 'imported\n')
                        self.assertEqual(child.__name__, canonical_name)
                        self.assertEqual(child.__spec__.name, canonical_name)
                        self.assertEqual(child.__loader__.name, canonical_name)
                        self.assertEqual(child.__package__, canonical_name.rpartition('.')[0])

    def test_legacy_submodule_import_failure_can_be_retried(self):
        path = self.plugin('retry-package', '', filename='__init__.py')
        helper = self.plugin('retry-package', 'raise RuntimeError("helper failed")\n', filename='helper.py')
        package = import_external_file(path)
        legacy = f'_swift_external_{hashlib.sha256(path.encode()).hexdigest()}.helper'
        canonical = f'{package.__name__}.helper'
        with self.assertRaisesRegex(RuntimeError, 'helper failed'):
            importlib.import_module(legacy)
        self.assertNotIn(legacy, sys.modules)
        self.assertNotIn(canonical, sys.modules)
        Path(helper).write_text('VALUE = 42\n')
        child = importlib.import_module(legacy)
        self.assertIs(child, importlib.import_module(canonical))
        self.assertEqual(child.VALUE, 42)

    def test_dataloader_unpickles_plugin_objects_before_worker_init(self):
        from torch.utils.data import DataLoader

        path = self.plugin(
            'loader', '''
from torch.utils.data import Dataset
READY = False

class Records(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        assert READY, 'worker_init_fn was not called'
        return index

class Collator:
    def collate(self, batch):
        return [value + 10 for value in batch]

def initialize(worker_id):
    global READY
    READY = True
''')
        module = import_external_file(path)
        saved_init = DataLoader.__init__
        was_patched = getattr(DataLoader, '_swift_external_plugins', False)
        try:
            patch_dataloader_external_plugins()
            for context in ('spawn', 'forkserver'):
                if context not in mp.get_all_start_methods():
                    continue
                with self.subTest(context=context):
                    loader = DataLoader(
                        module.Records(),
                        batch_size=2,
                        num_workers=1,
                        multiprocessing_context=context,
                        collate_fn=module.Collator().collate,
                        worker_init_fn=module.initialize,
                        timeout=60)
                    self.assertEqual(list(loader), [[10, 11]])
        finally:
            DataLoader.__init__ = saved_init
            DataLoader._swift_external_plugins = was_patched


if __name__ == '__main__':
    unittest.main()
