# Copyright (c) ModelScope Contributors. All rights reserved.
"""S0 -- the test suite's own foundations (ut_plan S0-01..S0-10).

Every later number builds on five pieces: conftest's tmp-reclaim + runtime isolation,
``reset_twinkle_runtime``, ``TinyModel``/``TinyData``, ``tiny_loader`` and the torchrun runners. A
fault in any of them poisons whole families of tests downstream, so each gets its own oracle here
rather than being trusted. Each test's docstring names the ut_plan id it closes.
"""
import json
import subprocess
import time
from pathlib import Path

import pytest

from swift.dev.tests._runners import Runners
from swift.dev.tests._twinkle_runtime import reset_twinkle_runtime
from swift.dev.tests.tiny import TinyData, TinyModel

#: The repo root, where editable-installed ``swift.dev`` resolves from and where subprocess pytest
#: runs must sit so the working tree shadows the installed copy.
REPO_ROOT = Path(__file__).resolve().parents[3]
#: Interpreters are pinned by absolute path: a shell with conda's ``(base)`` active would otherwise
#: resolve ``pytest``/``torchrun`` to a 3.14 without twinkle installed.
PYTEST_BIN = '/usr/local/bin/pytest'
TORCHRUN_BIN = '/usr/local/bin/torchrun'

_PASS_CASE = """\
import pytest


@pytest.mark.slow
def test_heavy(tmp_path):
    (tmp_path / 'sentinel.txt').write_text('keep-me')
"""

_FAIL_CASE = """\
import pytest


@pytest.mark.slow
def test_heavy(tmp_path):
    (tmp_path / 'sentinel.txt').write_text('keep-me')
    assert 0, 'deliberate failure'
"""


def _run_reclaim_subprocess(tmp_path: Path, *, fail: bool) -> Path:
    """Run one slow sentinel test in a fresh pytest; return its basetemp tree for inspection.

    The case file lives OUTSIDE swift/dev/tests, so the tree's conftest only reaches it because the
    subprocess loads it explicitly as a plugin (``-p``); that is exactly what makes this a real
    end-to-end check of the fixture rather than a re-implementation of its logic.
    """
    case_file = tmp_path / 'case_reclaim' / 'test_reclaim.py'
    case_file.parent.mkdir(parents=True)
    case_file.write_text(_FAIL_CASE if fail else _PASS_CASE)
    basetemp = tmp_path / 'basetemp'
    proc = subprocess.run([
        PYTEST_BIN, str(case_file), '-p', 'swift.dev.tests.conftest', '--basetemp', str(basetemp),
        '-q', '--no-header', '-p', 'no:cacheprovider'],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    expected_code = 1 if fail else 0
    assert proc.returncode == expected_code, f'unexpected exit {proc.returncode}\n{proc.stdout[-800:]}'
    return basetemp / 'test_heavy0'


@pytest.mark.parametrize('fail', [False, True], ids=['pass-reclaims', 'fail-keeps'])
def test_slow_tmp_reclaim(tmp_path, fail):
    """S0-01: a PASSING slow test's tmp_path is deleted, a FAILING one's is kept.

    The reclaim exists because a slow Megatron run writes 12-15 GB into tmp_path and pytest keeps the
    last 3 basetemp trees; the failure mode it kills is ENOSPC misread as a checkpointing bug. Kill
    direction check: a reclaim that never fires leaves the passing tree behind (asserted gone), one
    that fires on failures destroys the only evidence for debugging (asserted kept).
    """
    heavy_tmp = _run_reclaim_subprocess(tmp_path, fail=fail)
    if fail:
        assert (heavy_tmp / 'sentinel.txt').read_text() == 'keep-me'
    else:
        assert not heavy_tmp.exists()


def test_reset_restores_infra_globals():
    """S0-02: stale ray-era infra globals are cleared, and TWINKLE_MODE decides the restored mode.

    Without the reset a local-mode construction after a ray run inherits ``_mode='ray'`` and the
    ``@remote_class`` path raises "Unsupported mode". The restored default must be the import-time
    one, not None -- None breaks runs that never called initialize().
    """
    import twinkle.infra as infra
    infra._mode = 'ray'
    infra._device_group, infra._device_mesh = object(), object()
    reset_twinkle_runtime()
    assert infra._mode == 'local'
    assert infra._device_group is None and infra._device_mesh is None
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('TWINKLE_MODE', 'ray')
        infra._mode = 'local'
        reset_twinkle_runtime()
        assert infra._mode == 'ray'


def test_reset_allows_reinit_with_different_precision():
    """S0-03: after the reset, a second Accelerator with a DIFFERENT mixed_precision builds.

    accelerate's state is a process-global singleton keyed on the first setting; without the reset
    the second construction dies with "AcceleratorState has already been initialized and cannot be
    changed", which reads as a product bug but is test-harness leakage.
    """
    from accelerate import Accelerator
    Accelerator(mixed_precision='no')
    reset_twinkle_runtime()
    Accelerator(mixed_precision='bf16')
    reset_twinkle_runtime()


@pytest.fixture(scope='module')
def tiny_dir(tmp_path_factory):
    """One shared TinyModel build for the two consumers (S0-04 load-back, S0-05 loss anchor)."""
    return TinyModel.build(tmp_path_factory.mktemp('tiny') / 'tiny')


def test_tiny_model_build_produces_loadable_dir(tiny_dir):
    """S0-04: the built directory reloads through AutoModel with the shrunk geometry intact."""
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(tiny_dir)
    assert config.num_hidden_layers == 4
    assert config.hidden_size == 128
    model = AutoModelForCausalLM.from_pretrained(tiny_dir)
    assert model.__class__.__name__ == 'Qwen2ForCausalLM'


def test_first_loss_anchors_ln_vocab(tiny_dir):
    """S0-05: a randomly-initialised model's first loss must start near ``ln(vocab_size)``.

    This is the oracle every e2e loss assertion leans on: a shifted, masked or double-normalised
    loss lands far away from it immediately, where "is it finite" stays green. Measured deviation on
    this build is ~0.03, so the 0.5 band still kills any labels misalignment.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    expected = TinyModel.initial_loss(tiny_dir)
    model = AutoModelForCausalLM.from_pretrained(tiny_dir)
    tokenizer = AutoTokenizer.from_pretrained(tiny_dir)
    batch = tokenizer(['What is 2+2?', 'Name a colour.'], return_tensors='pt', padding=True)
    loss = model(**batch, labels=batch['input_ids']).loss.item()
    assert abs(loss - expected) < 0.5, f'first loss {loss:.4f} vs ln(vocab) {expected:.4f}'


def test_loader_builder_keeps_family_hooks(tmp_path):
    """S0-06 (dense half): loader_builder('qwen2') walks the REGISTERED loader's full chain.

    config -> processor -> model, each through the family's hooks, then shrinks -- so the tiny
    checkpoint is still a qwen2 built by qwen2's own loader, not a bare from_config.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from swift.dev.tests.tiny_loader import loader_builder
    model_dir = TinyModel.build(tmp_path / 'dense', builder=loader_builder('qwen2'))
    config = AutoConfig.from_pretrained(model_dir)
    assert config.model_type == 'qwen2'
    assert config.num_hidden_layers == 4 and config.hidden_size == 128
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    assert model.__class__.__name__ == 'Qwen2ForCausalLM'


def test_loader_builder_shrinks_moe_geometry(tmp_path):
    """S0-06 (MoE half): loader_builder('qwen3_moe') shrinks a REAL MoE geometry, not an invented one.

    The config starts from the real Qwen3-30B-A3B snapshot, so ``num_experts=4`` is a shrink of a
    family the loader genuinely knows -- this is the row capability tests will name.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from swift.dev.tests.tiny_loader import loader_builder
    model_dir = TinyModel.build(
        tmp_path / 'moe',
        tokenizer_id='Qwen/Qwen3-30B-A3B',
        builder=loader_builder('qwen3_moe'),
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128)
    config = AutoConfig.from_pretrained(model_dir)
    assert config.model_type == 'qwen3_moe'
    assert config.num_experts == 4
    assert config.num_hidden_layers == 4
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    assert 'Moe' in model.__class__.__name__


def test_shrink_truncates_per_layer_lists(tmp_path):
    """S0-07: per-layer lists follow the layer count down; unrelated lists stay untouched.

    transformers>=4.54 validates ``len(layer_types) == num_hidden_layers`` at SAVE time, so the
    assertion is the save itself, not just the attribute state.
    """
    from transformers import Qwen3Config
    from swift.dev.tests.tiny_loader import shrink
    config = Qwen3Config(num_hidden_layers=24)
    config.layer_types = ['full_attention', 'sliding_attention'] * 12
    config.some_unrelated = [1, 2, 3]
    shrink(config, {'num_hidden_layers': 4})
    assert config.layer_types == ['full_attention', 'sliding_attention'] * 2
    assert config.some_unrelated == [1, 2, 3]
    config.save_pretrained(str(tmp_path))  # the length check that used to fire late fires here if broken


def test_launch_timeout_kills_the_process_group():
    """S0-08: Runners.launch's timeout actually fires and kills the whole session.

    The failure it guards against is unbounded: torchrun's grandchildren inherit the pipes, so a
    plain subprocess.run(timeout=...) keeps waiting for EOF and one run hung for 12 hours. The kill
    must land well under a minute for the harness to stay usable, hence the wall-clock bound.
    """
    start = time.monotonic()
    with pytest.raises(AssertionError, match='exceeded'):
        Runners.launch(['sleep', '600'], timeout=3)
    assert time.monotonic() - start < 60


def test_rendezvous_port_zero_two_ranks():
    """S0-09: the c10d ``port=0`` rendezvous stands up two real ranks on this box.

    NOT ``--master_port=0``: that exports MASTER_PORT=0 so rank 1 dials port 0 and hangs. Every
    two-GPU runner later depends on this contract, and both ranks printing after one all-reduce
    proves both made it through rendezvous AND the collective.
    """
    res = Runners.launch([
        TORCHRUN_BIN, '--nproc_per_node=2', *Runners.RENDEZVOUS, Runners.path('rendezvous_smoke')],
        timeout=300)
    assert res.returncode == 0, f'torchrun failed\nstdout tail:\n{res.stdout[-800:]}\nstderr tail:\n{res.stderr[-2000:]}'
    assert 'rank0 rendezvous-ok' in res.stdout
    assert 'rank1 rendezvous-ok' in res.stdout


_ROW_SHAPES = {
    'sft': lambda row: (
        set(row) == {'messages'}
        and [turn['role'] for turn in row['messages']] == ['user', 'assistant']),
    'pretrain': lambda row: (
        set(row) == {'messages'}
        and [turn['role'] for turn in row['messages']] == ['assistant']),
    'preference': lambda row: (
        set(row) == {'messages', 'rejected_response'}
        and [turn['role'] for turn in row['messages']] == ['user', 'assistant']
        and isinstance(row['rejected_response'], str)),
    'prompt_only': lambda row: (
        set(row) == {'messages'}
        and [turn['role'] for turn in row['messages']] == ['user']),
    'seq_cls': lambda row: (
        set(row) == {'messages', 'label'}
        and [turn['role'] for turn in row['messages']] == ['user']
        and isinstance(row['label'], int)),
    'embedding': lambda row: (
        set(row) == {'messages', 'positive_messages', 'negative_messages'}
        and [turn['role'] for turn in row['messages']] == ['user']
        and all(isinstance(group, list) and group for group in
                (row['positive_messages'], row['negative_messages']))),
}


@pytest.mark.parametrize('kind', sorted(_ROW_SHAPES))
def test_tiny_data_row_shapes(kind, tmp_path):
    """S0-10: each TinyData writer emits the exact row shape its family's loader expects.

    The shapes are copied from registered loaders (embedding's lists-of-conversations, seq_cls's
    integer label), so a drift here breaks every family e2e that builds its dataset from TinyData --
    the point of the per-kind assertion rather than "is it valid json".
    """
    path = getattr(TinyData, kind)(tmp_path / f'{kind}.jsonl', n=4)
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 4
    assert all(_ROW_SHAPES[kind](row) for row in rows)
