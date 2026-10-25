# Copyright (c) ModelScope Contributors. All rights reserved.
"""Writer / checkpoint / candidate-cache tests (tmp_path, no GPU, no model).

Covers the "write to file" half of the pipeline: the plain incremental jsonl writer (and its append
contract that ``test_cli_entrypoints.py`` also pins), the four-file checkpointed resume scheme behind a
'dpo' / resumed run, and the prompt-keyed candidate cache (legacy's ``cache_files``).
"""
import json
import os

import pytest

from swift.dev.components.infer.component import _CandidateCache
from swift.dev.dataset.storage.writers import (StoreWriter, _CheckpointPaths, store_output_complete,
                                               store_writer_kind)
from swift.dev.tests.infer.conftest import write_cache_file


def _read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


# --- StoreWriter, incremental mode (jsonl) ---------------------------------------------


def test_incremental_writer_appends_across_batches(tmp_path):
    """An incremental format flushes each finished batch; the hard contract: consecutive writes
    append, never overwrite."""
    path = str(tmp_path / 'out.jsonl')
    writer = StoreWriter(path, 'jsonl')
    assert writer.kind == 'incremental'
    writer.write([{'response': 'a'}])
    writer.write([{'response': 'b'}])
    assert _read_jsonl(path) == [{'response': 'a'}, {'response': 'b'}]


def test_incremental_writer_appends_to_a_preexisting_file(tmp_path):
    """A pre-existing output file means "continue it": rows append to whatever is already there."""
    path = str(tmp_path / 'out.jsonl')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'response': 'old'}) + '\n')
    writer = StoreWriter(path, 'jsonl')
    writer.write([{'response': 'new'}])
    assert _read_jsonl(path) == [{'response': 'old'}, {'response': 'new'}]


def test_writer_kind_policy_is_decided_in_one_place():
    """store_writer_kind folds format capability with run intent: no path -> memory, a
    non-incremental format -> batch, an incremental format + checkpoint -> checkpoint."""
    assert store_writer_kind(None, 'jsonl') == 'memory'
    assert store_writer_kind('out.jsonl', 'jsonl') == 'incremental'
    assert store_writer_kind('out.jsonl', 'arrow') == 'batch'
    assert store_writer_kind('out.jsonl', 'jsonl', checkpoint=True) == 'checkpoint'


def test_memory_writer_is_a_noop():
    writer = StoreWriter(None, 'jsonl')
    assert writer.kind == 'memory'
    writer.write([{'response': 'a'}])  # must not raise
    writer.finish([{'response': 'a'}])
    assert writer.skip(0) is False  # memory never skips


# --- _CheckpointPaths -----------------------------------------------------------------


def test_checkpoint_paths_fresh_start_clears_stale_files(tmp_path):
    paths = _CheckpointPaths(str(tmp_path), 'out.jsonl')
    # stale artifacts from a crashed run
    for p in (paths.tmp, paths.resume, paths.state):
        with open(p, 'w') as f:
            f.write('stale')
    last, mode = paths.prepare(resume=False)
    assert (last, mode) == (-1, 'w')
    assert not os.path.exists(paths.tmp) and not os.path.exists(paths.resume)
    assert not os.path.exists(paths.state)


def test_checkpoint_paths_checkpoint_then_resume_roundtrip(tmp_path):
    paths = _CheckpointPaths(str(tmp_path), 'out.jsonl')
    paths.prepare(resume=False)
    with open(paths.tmp, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'response': 'a'}) + '\n')
    paths.checkpoint(3)  # snapshot tmp -> resume, then record batch_index 3
    assert os.path.exists(paths.resume) and os.path.exists(paths.state)
    # a crash leaves tmp half-written; resume copies the clean snapshot back over it
    with open(paths.tmp, 'w', encoding='utf-8') as f:
        f.write('{"trunc')
    last, mode = paths.prepare(resume=True)
    assert (last, mode) == (3, 'a')
    assert _read_jsonl(paths.tmp) == [{'response': 'a'}]  # restored from the snapshot


def test_checkpoint_paths_finalize_publishes_final(tmp_path):
    paths = _CheckpointPaths(str(tmp_path), 'out.jsonl')
    paths.prepare(resume=False)
    with open(paths.tmp, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'response': 'a'}) + '\n')
    paths.checkpoint(0)
    paths.finalize()
    assert os.path.exists(paths.final)
    assert _read_jsonl(paths.final) == [{'response': 'a'}]
    assert not os.path.exists(paths.tmp) and not os.path.exists(paths.state)


def test_checkpoint_paths_finalize_keeps_previous_under_timestamp(tmp_path):
    paths = _CheckpointPaths(str(tmp_path), 'out.jsonl')
    with open(paths.final, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'response': 'previous'}) + '\n')
    paths.prepare(resume=False)
    with open(paths.tmp, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'response': 'new'}) + '\n')
    paths.finalize()
    assert _read_jsonl(paths.final) == [{'response': 'new'}]
    # the previous final was moved aside, not overwritten
    backups = [f for f in os.listdir(str(tmp_path)) if f.startswith('out.jsonl.')]
    assert backups, 'previous final was not preserved under a timestamp'


# --- StoreWriter, checkpoint mode (the resumable four-file scheme) ----------------------


def test_checkpoint_writer_writes_then_finalizes(tmp_path):
    path = str(tmp_path / 'out.jsonl')
    writer = StoreWriter(path, 'jsonl', checkpoint=True)
    assert writer.kind == 'checkpoint'
    assert writer.skip(0) is False  # resume_from == -1, so nothing is skipped on a fresh run
    writer.write([{'response': 'a'}, {'response': 'b'}])
    writer.checkpoint(0)
    writer.finish([])
    assert _read_jsonl(path) == [{'response': 'a'}, {'response': 'b'}]
    assert store_output_complete(path) is True


def test_checkpoint_writer_resume_skips_finished_batches(tmp_path):
    """A crash mid-run leaves the snapshot + state; the resumed writer skips the finished batch and
    appends the rest, so the final output has both runs' rows without duplicating the finished batch."""
    path = str(tmp_path / 'out.jsonl')
    # first run: write batch 0, checkpoint it, then "crash" (close without finish -> no final file)
    w1 = StoreWriter(path, 'jsonl', checkpoint=True)
    w1.write([{'response': 'a'}])
    w1.checkpoint(0)
    w1._f.close()
    assert not os.path.exists(path)  # finalize never ran, so no published output yet
    # resumed run: batch 0 is recorded finished -> skip(0) True, and the snapshot is restored into tmp
    w2 = StoreWriter(path, 'jsonl', checkpoint=True, resume=True)
    assert w2.skip(0) is True
    assert w2.skip(1) is False
    w2.write([{'response': 'b'}])
    w2.checkpoint(1)
    w2.finish([])
    rows = _read_jsonl(path)
    assert {'response': 'a'} in rows and {'response': 'b'} in rows


# --- StoreWriter, batch mode (arrow) ----------------------------------------------------


def _rows():
    """Two infer-emit-shaped rows: a semantic column plus the token/rollout columns infer produces."""
    return [
        {'response': 'a', 'messages': [{'role': 'user', 'content': 'q'}], 'id': 0},
        {'response': 'b', 'messages': [{'role': 'user', 'content': 'q'}], 'id': 1},
    ]


def test_batch_writer_serialises_once_at_finish(tmp_path):
    """``write()`` is a no-op for a batch format (nothing lands until the run ends); ``finish()``
    writes one Arrow ``save_to_disk`` table, readable by the same reader the export half writes for."""
    from swift.dev.dataset.storage.base import load_dataset_store
    path = str(tmp_path / 'out.jsonl')
    writer = StoreWriter(path, 'arrow')
    assert writer.kind == 'batch'
    writer.write(_rows())  # buffered, not flushed
    assert not os.path.exists(writer.store_path)
    writer.finish(_rows())
    assert os.path.isdir(writer.store_path)  # an Arrow store is a directory
    ds = load_dataset_store(writer.store_path)
    assert len(ds) == 2 and ds['response'] == ['a', 'b']


def test_batch_writer_derives_path_from_jsonl(tmp_path):
    """The Arrow dir sits beside the jsonl path it was derived from: ``.jsonl`` is stripped, ``.arrow``
    appended, so the sidecar-relative ``rollout_tokens`` paths embedded in the rows stay valid."""
    assert StoreWriter(str(tmp_path / 'run.jsonl'), 'arrow').store_path == str(tmp_path / 'run.arrow')
    # a path with no .jsonl suffix just gains .arrow (it is not stripped from the middle of a name)
    assert StoreWriter(str(tmp_path / 'run'), 'arrow').store_path == str(tmp_path / 'run.arrow')


def test_batch_writer_store_fields_restricts_columns(tmp_path):
    """``store_fields`` is forwarded as the column allow-list, so only the named columns are
    persisted (an infer emit row carries more than a training cache wants)."""
    from swift.dev.dataset.storage.base import load_dataset_store
    writer = StoreWriter(str(tmp_path / 'out.jsonl'), 'arrow', store_fields=['response', 'id'])
    writer.finish(_rows())
    ds = load_dataset_store(writer.store_path)
    assert set(ds.column_names) == {'response', 'id'}  # 'messages' dropped


def test_batch_writer_skip_and_checkpoint_are_noops(tmp_path):
    """The batch writer never resumes (resume is refused at construction), so ``skip`` is always
    False and ``checkpoint`` does nothing -- it shares the emit surface without the semantics."""
    writer = StoreWriter(str(tmp_path / 'out.jsonl'), 'arrow')
    assert writer.skip(0) is False and writer.skip(5) is False
    writer.checkpoint(3)  # must not raise or touch disk
    assert not os.path.exists(writer.store_path)


def test_batch_writer_refuses_resume(tmp_path):
    """A one-shot-at-the-end format cannot resume; the refusal is loud, not a silent half-run."""
    with pytest.raises(ValueError, match='cannot resume'):
        StoreWriter(str(tmp_path / 'out.jsonl'), 'arrow', resume=True)


# --- _CandidateCache ------------------------------------------------------------------


def test_candidate_cache_lookup_by_prompt(tmp_path):
    prompt = [{'role': 'user', 'content': 'What is 2+2?'}]
    cache_path = write_cache_file(str(tmp_path / 'cache.jsonl'), [(prompt, ['4', 'five', '4!'])])
    cache = _CandidateCache([cache_path])
    # asking for <= the cached count returns that many candidates
    hits = cache.lookup({'messages': prompt}, 2)
    assert hits is not None and [c.decoded for c in hits] == ['4', 'five']
    # asking for more than the cache holds is a MISS (it would silently shrink the group)
    assert cache.lookup({'messages': prompt}, 4) is None
    # an uncovered prompt misses
    assert cache.lookup({'messages': [{'role': 'user', 'content': 'other'}]}, 1) is None


def test_candidate_cache_empty_when_no_files():
    cache = _CandidateCache([])
    assert cache.lookup({'messages': [{'role': 'user', 'content': 'x'}]}, 1) is None


def test_candidate_cache_skips_corrupt_and_nonassistant_lines(tmp_path):
    path = str(tmp_path / 'cache.jsonl')
    prompt = [{'role': 'user', 'content': 'q'}]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'messages': prompt + [{'role': 'assistant', 'content': 'good'}]}) + '\n')
        f.write('{"truncated json\n')  # a crashed producer's partial line
        f.write(json.dumps({'messages': prompt}) + '\n')  # no trailing assistant -> not a candidate
    cache = _CandidateCache([path])
    hits = cache.lookup({'messages': prompt}, 1)
    assert hits is not None and [c.decoded for c in hits] == ['good']


def test_candidate_cache_ignores_missing_file(tmp_path):
    cache = _CandidateCache([str(tmp_path / 'does_not_exist.jsonl')])
    assert cache.lookup({'messages': [{'role': 'user', 'content': 'q'}]}, 1) is None
