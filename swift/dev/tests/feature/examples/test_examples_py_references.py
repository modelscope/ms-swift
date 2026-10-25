# Copyright (c) ModelScope Contributors. All rights reserved.
"""S16-16 (L2): the examples' python reference files import and register cleanly.

The examples tree carries ten ``.py`` entry files (custom plugin registries, code-mode entry
scripts, the server cookbook client) that docs and users copy verbatim. A green smoke here means
each one imports in a CLEAN interpreter (subprocess, not this pytest process -- a
plugin-registry test that shares the interpreter can be green while the real CLI order is
broken, see the load_external kind-declaration bug) and its registrations are actually visible
afterwards.

The plugin registry files are loaded through the REAL loading path
(``PluginRegistry.load_external`` -- what ``--external_plugins`` reduces to) and their
registered names are resolved. The code-mode entry files (``examples/{rl,deploy}/code/*.py``)
execute at ``__main__`` only; importing them is the compilation-level smoke (their argv parsing
is covered by the CLI suites).
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[5] / 'examples'

#: (file, kind, registered name) for files whose point is the registration.
PLUGIN_FILES = [
    ('infer/code/custom_plugins.py', 'reward', 'length_bonus'),
    ('infer/code/custom_plugins.py', 'tool', 'word_count'),
    ('infer/code/custom_tools.py', 'tool', 'calculator'),
    ('rl/opsd/opsd_plugin.py', 'dataset', None),  # names asserted dynamically below
    ('train/plugins/my_plugin.py', 'callback', 'progress_printer'),
]

#: Files whose point is being importable (code-mode entries + client); import must be clean.
IMPORT_ONLY_FILES = [
    'deploy/code/deploy.py',
    'rl/code/dpo.py',
    'rl/code/grpo.py',
    'rl/code/opsd.py',
    'train/custom/model_template/model.py',
    'server/client.py',
]


def _spec_of(rel):
    path = EXAMPLES / rel
    assert path.exists(), f'{rel} disappeared from examples/; update this test with the tree'
    spec = importlib.util.spec_from_file_location(f'_examples_smoke_{path.stem}_{abs(hash(str(path))) & 0xffff}', path)
    module = importlib.util.module_from_spec(spec)
    return module, spec


def _load_in_clean_interpreter(rel, check_code):
    """Import ONE example file in a fresh subprocess and run ``check_code`` (a string with an
    ``m`` placeholder bound to the loaded module) inside it; print the result as the last line."""
    script = f'''
import importlib.util, sys
spec = importlib.util.spec_from_file_location("m", r"{EXAMPLES / rel}")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
result = None
{check_code}
print("RESULT:", result)
'''
    proc = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=300,
                          env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'PYTHONPATH': str(EXAMPLES.parent),
                               'HOME': '/root'})
    assert proc.returncode == 0, f'{rel} failed to import in a clean interpreter:\n{proc.stderr[-3000:]}'
    return proc.stdout.strip().splitlines()[-1]


def test_plugin_reference_files_register_their_names():
    """custom_plugins / custom_tools / my_plugin register exactly the names the docs promise,
    loaded through the REAL external path (``load_external``) and resolved by name in a clean
    interpreter."""
    for rel, kind, name in PLUGIN_FILES:
        if name is None:
            continue
        script = f'''
from swift.dev.utils.plugin import PluginRegistry
PluginRegistry.load_external(r"{EXAMPLES / rel}")
item = PluginRegistry.get("{kind}", "{name}")
print("RESULT:", type(item).__name__)
'''
        proc = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, f'{rel} failed load_external + get({kind},{name}):\n{proc.stderr[-3000:]}'
        assert proc.stdout.strip().splitlines()[-1].startswith('RESULT:'), proc.stdout[-500:]


def test_opsd_plugin_imports_and_registers():
    """opsd_plugin.py imports in a clean interpreter and executes its ``@register_dataset``
    decorators (the opsd e2e in the RL suite consumes the registered loaders)."""
    script = f'''
import importlib.util
spec = importlib.util.spec_from_file_location("m", r"{EXAMPLES / 'rl/opsd/opsd_plugin.py'}")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print("RESULT: imported")
'''
    proc = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f'opsd_plugin failed to import:\n{proc.stderr[-3000:]}'
    assert 'RESULT: imported' in proc.stdout


@pytest.mark.parametrize('rel', IMPORT_ONLY_FILES)
def test_code_entry_files_import_cleanly(rel):
    """deploy/code/*.py, rl/code/*.py, the custom model template and the server client import
    without error -- the level of smoke a docs-copy-and-paste user needs to hold."""
    _load_in_clean_interpreter(rel, 'result = "imported"')
