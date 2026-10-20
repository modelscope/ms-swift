# Copyright (c) Modelscope Contributors. All rights reserved.
"""S3 additions -- the plan's newly-derived rows the v5-port files do not cover.

Covers: S3-13 import_cls lazy spec, S3-25 SwiftDataset bounds, S3-35 idempotent conversion,
S3-50 empty/single-row stores, S3-51 large fields, S3-54 numeric/nested fidelity, S3-61
note_row_error budget, S3-67/68 streaming split semantics.

N/A rows, recorded so the plan marking stays honest:
- S3-15 (strict column-mapping gate): aliases never reject -- resolve_aliases keeps unknown names
  (test_format_converter pins the alias rules instead).
- S3-49 (concurrent same-name write lock): the writer layer is single-driver by construction
  (store_writer_kind's checkpoint mode assumes one driver process); there is no lock to pin.
- S3-60 (lengths requirement): pinned by test_api.py::TestGroupByLength (refusal) and the packing
  suite (the measure pass that produces lengths).
"""
from __future__ import annotations

import json
import os

import pytest
from datasets import Dataset as HfDataset


# --- S3-13: import_cls lazy 'module:attr' spec ------------------------------------------------


def test_import_cls_resolves_a_module_attr_spec():
    from swift.dev.dataset.format_converter import OpenAIConverter
    from swift.dev.dataset.loader.base import DatasetLoader

    assert DatasetLoader.import_cls('swift.dev.dataset.format_converter:OpenAIConverter') \
        is OpenAIConverter
    # a module never imported until now resolves just the same -- the spec is fully lazy
    assert DatasetLoader.import_cls('json:loads') is json.loads


def test_import_cls_names_the_missing_piece():
    from swift.dev.dataset.loader.base import DatasetLoader

    with pytest.raises(ModuleNotFoundError):
        DatasetLoader.import_cls('no_such_module_xyz:Thing')
    with pytest.raises(AttributeError):
        DatasetLoader.import_cls('json:no_such_attr_xyz')
    with pytest.raises(ModuleNotFoundError):
        # no ':' in the spec: partition puts the whole string in the module half
        DatasetLoader.import_cls('jsonloads')


# --- S3-25: SwiftDataset getitem bounds ------------------------------------------------------


class _PassthroughTemplate:
    """The smallest template that can stand in: encode returns a fixed row, measure never runs."""

    def encode(self, row, **kwargs):
        return {'input_ids': [1], 'length': 1}


def test_swiftdataset_getitem_bounds():
    from swift.dev.dataset import SwiftDataset

    ds = SwiftDataset(HfDataset.from_list([{'messages': [{'role': 'user', 'content': 'a'}]},
                                           {'messages': [{'role': 'user', 'content': 'b'}]}]),
                      _PassthroughTemplate())
    assert ds[0]['input_ids'] == [1] and ds[1]['input_ids'] == [1]
    # negative index keeps the underlying HF Dataset's python semantics: the LAST row
    assert ds[-1]['input_ids'] == [1]
    # out of range propagates: the reader must not serve another row for a bad index
    with pytest.raises(IndexError):
        ds[2]
    with pytest.raises(IndexError):
        ds[-3]


# --- S3-35: conversion is idempotent on standard rows -----------------------------------------


def test_standard_rows_convert_to_themselves():
    """A standard row re-run through the default Preprocessor must come back unchanged, asserted
    against the preset literal (not a round-trip self-comparison, which a broken pair could pass).
    Rows are key-homogeneous: datasets' from_list infers columns from the rows themselves and a
    row carrying a key the others lack loses it at the INPUT, before any conversion runs."""
    from swift.dev.dataset import Preprocessor

    standard = [{
        'messages': [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a'},
        ],
        'images': [{'bytes': None, 'path': 'http://x/a.png'}],
    }, {
        'messages': [{'role': 'user', 'content': 'q2'}],
        'images': [{'bytes': None, 'path': 'http://x/b.png'}],
    }]
    rows = [dict(r) for r in Preprocessor()(HfDataset.from_list(standard), load_from_cache_file=False)]
    assert rows == standard


def test_converted_rows_are_stable_on_a_second_pass():
    """alpaca -> standard, then standard -> standard: the second pass changes nothing (the
    converter must not re-interpret its own output, e.g. promote an aliased column again)."""
    from swift.dev.dataset import Preprocessor

    alpaca = [{'instruction': 'q', 'input': 'ctx', 'output': 'a'}]
    first = [dict(r) for r in Preprocessor()(HfDataset.from_list(alpaca), load_from_cache_file=False)]
    expected = [{'messages': [{'role': 'user', 'content': 'q\nctx'}, {'role': 'assistant', 'content': 'a'}]}]
    assert first == expected
    second = [dict(r) for r in
              Preprocessor()(HfDataset.from_list(first), load_from_cache_file=False)]
    assert second == expected


# --- S3-50 / S3-51 / S3-54: store edges, large fields, type fidelity --------------------------


@pytest.mark.parametrize('backend', ['arrow', 'jsonl'])
def test_store_empty_and_single_row_roundtrips(tmp_path, backend):
    """The degenerate edges of the store: ZERO rows (an export of an empty split -- load_from_disk
    IndexErrors and the json builder StopIterates on it, so the storage layer answers 0 rows) and
    ONE row (the minimum real store) both round-trip. jsonl takes a suffixed path: the suffix is
    the caller's, the writer never renames."""
    from swift.dev.dataset.storage import load_dataset_store, write_dataset_store

    suffix = '.jsonl' if backend == 'jsonl' else ''
    empty = write_dataset_store([], str(tmp_path / 'empty') + suffix, backend=backend)
    assert len(load_dataset_store(empty)) == 0

    one = write_dataset_store([{'input_ids': [1, 2], 'labels': [-100, 2], 'lengths': [2]}],
                              str(tmp_path / 'one') + suffix, backend=backend)
    loaded = load_dataset_store(one)
    assert len(loaded) == 1
    assert loaded[0]['input_ids'] == [1, 2] and loaded[0]['lengths'] == [2]


def test_store_large_field_roundtrips_untruncated(tmp_path):
    """A multi-megabyte text field must survive the jsonl write byte-for-byte: a truncated or
    re-encoded field would silently corrupt training rows."""
    from swift.dev.dataset.storage import load_dataset_store, write_dataset_store

    payload = ('swift' * 1_000_000)  # 5 MB
    path = write_dataset_store([{'input_ids': list(range(64)), 'text': payload}],
                               str(tmp_path / 'big') + '.jsonl', backend='jsonl')
    loaded = load_dataset_store(path)
    assert len(loaded) == 1 and loaded[0]['text'] == payload


def test_store_numeric_and_nested_columns_keep_their_values(tmp_path):
    """Type fidelity through a store round-trip: the float keeps its exact value (no float->int
    downgrade), the int stays an int, and a nested list-of-struct column comes back structurally
    identical -- arrow types it, jsonl re-parses it; both must agree with the input."""
    from swift.dev.dataset.storage import load_dataset_store, write_dataset_store

    rows = [{
        'input_ids': [7, 11],
        'lengths': [2],
        'score': 0.125,
        'counts': 3,
        'tags': [{'name': 'a', 'weight': 0.5}, {'name': 'b', 'weight': 1.5}],
    }]
    path = write_dataset_store(rows, str(tmp_path / 'types'), backend='arrow')
    row = load_dataset_store(path)[0]
    assert row['score'] == 0.125 and row['counts'] == 3 and isinstance(row['counts'], int)
    assert row['tags'] == rows[0]['tags']
    assert row['input_ids'] == [7, 11]

    jsonl_path = write_dataset_store(rows, str(tmp_path / 'types2') + '.jsonl', backend='jsonl')
    row = load_dataset_store(jsonl_path)[0]
    assert row['score'] == 0.125 and row['counts'] == 3
    assert row['tags'] == rows[0]['tags']


# --- S3-61: note_row_error's traceback budget -------------------------------------------------


class _Owner:
    traceback_limit = 2

    def __init__(self):
        self._traceback_counter = 0


def test_note_row_error_counts_and_respects_the_budget():
    from swift.dev.dataset._diagnostics import note_row_error

    owner = _Owner()
    for _ in range(5):
        try:
            raise ValueError('row is bad')
        except ValueError:
            note_row_error(owner, 'dropped')
    # the budget caps LOGGING, not the knowledge: the counter stops at the limit
    assert owner._traceback_counter == 2

    owner = _Owner()
    owner.traceback_limit = None  # None disables the budget entirely
    for _ in range(3):
        try:
            raise ValueError('row is bad')
        except ValueError:
            note_row_error(owner, 'dropped')
    assert owner._traceback_counter == 3


def test_malformed_rows_are_dropped_and_counted_not_silent():
    """strict=False drops a row the message check rejects and keeps the rest; the in-process map
    (num_proc<=1) lets the drop be OBSERVED on the preprocessor's own counter. strict=True
    re-raises the original check error -- the 'malformed row' note is only a warning."""
    from swift.dev.dataset import Preprocessor

    rows = [  # key-homogeneous: from_list infers columns per row set and drops stragglers
        {'messages': [{'role': 'user', 'content': 'good'}]},  # converts
        {'messages': [{'role': 'bogus_role', 'content': 'x'}]},  # rejected by the message check
        {'messages': [{'role': 'user', 'content': 'good2'}]},  # converts
    ]
    pre = Preprocessor(traceback_limit=5)
    out = pre(HfDataset.from_list(rows), load_from_cache_file=False)
    kept = [dict(r) for r in out]
    assert len(kept) == 2
    assert kept[0]['messages'] == [{'role': 'user', 'content': 'good'}]
    assert kept[1]['messages'] == [{'role': 'user', 'content': 'good2'}]
    assert pre._traceback_counter == 1, 'the dropped row must be counted, not silently swallowed'

    with pytest.raises(AssertionError, match='bogus_role'):
        Preprocessor(strict=True)(HfDataset.from_list(rows), load_from_cache_file=False)


# --- S3-67 / S3-68: streaming split semantics -------------------------------------------------


class _CountingStream:
    """A minimal stand-in stream: ``take``/``skip`` are the only ops ``split_streaming`` uses, so
    generator wrappers are enough to pin the split arithmetic (they chain: take(N).skip(k))."""

    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return iter([{'i': i} for i in range(self.n)])

    def take(self, k):
        return _Sliced(self, take=k)

    def skip(self, k):
        return _Sliced(self, skip=k)


class _Sliced:
    def __init__(self, src, *, take=None, skip=None):
        self._src = src
        self._take = take
        self._skip = skip

    def take(self, k):
        return _Sliced(self, take=k)

    def skip(self, k):
        return _Sliced(self, skip=k)

    def __iter__(self):
        rows = list(self._src)[self._skip or 0:]
        return iter(rows[:self._take] if self._take is not None else rows)


def test_streaming_split_semantics():
    """S3-67+S3-68: a stream has no length, so a FRACTIONAL ratio needs a ``#N`` budget (refused
    without one); with ``#N`` the first ``N*ratio`` rows become validation and the rest training;
    the degenerate ratios 0/1 answer without a budget."""
    from swift.dev.dataset.loader.base import DatasetLoader

    # no budget: the degenerate ratios have an answer, a fraction does not
    stream = _CountingStream(100)
    train, val = DatasetLoader.split_streaming(stream, None, 0.0)
    assert train is stream and val is None
    train, val = DatasetLoader.split_streaming(stream, None, 1.0)
    assert train is None and val is stream
    with pytest.raises(ValueError, match='#N'):
        DatasetLoader.split_streaming(stream, None, 0.2)

    # with a #N budget of 100 and ratio 0.2: first 20 rows are val, the rest train
    train, val = DatasetLoader.split_streaming(_CountingStream(100), 100, 0.2)
    assert [r['i'] for r in val] == list(range(20))
    assert [r['i'] for r in train] == list(range(20, 100))

    # a budget whose ratio floors to zero rows: no val split at all
    train, val = DatasetLoader.split_streaming(_CountingStream(100), 10, 0.05)
    assert val is None and len(list(train)) == 10
# --- S3-65: a streaming SFT run, end to end (L2) ---------------------------------------------


@pytest.mark.slow
@pytest.mark.accel(1)
def test_streaming_sft_end_to_end(tmp_path):
    """S3-65: a streaming dataset trains end to end. The whole chain only a real run can prove:
    the jsonl loads as an IterableDataset, the derived loader workers apply, the twinkle loader
    iterates it (an iterable has no len, so steps come from max_steps), the forward encodes on the
    fly, and the loop stops at max_steps with finite losses. TinyModel keeps it to seconds."""
    from swift.dev.config import (
        CheckpointConfig,
        DatasetConfig,
        DistributedConfig,
        ModelConfig,
        TemplateConfig,
        TrainConfig,
    )
    from swift.dev.recipe.sft.run_sft import run_sft
    from swift.dev.tests.tiny import TinyData, TinyModel

    model_dir = TinyModel.build(tmp_path / 'tiny')
    data_path = TinyData.sft(tmp_path / 'sft.jsonl', n=16)
    history = run_sft(
        ModelConfig(model=model_dir, model_type='qwen2'),
        TemplateConfig(template='qwen2_5', max_length=512),
        DatasetConfig(dataset=[data_path], streaming=True, dataset_shuffle=True, data_seed=42),
        TrainConfig(learning_rate=1e-4, optim='adamw', lr_scheduler='cosine', warmup_ratio=0.0,
                    per_device_train_batch_size=2, gradient_accumulation_steps=1, max_steps=2,
                    max_grad_norm=1.0),
        DistributedConfig(),
        CheckpointConfig(),
        output_dir=str(tmp_path / 'out'),
        _save_final=False,
    )
    assert len(history) == 2, f'expected the 2 max_steps, got {len(history)}'
    losses = [r['loss'] for r in history]
    assert all(abs(loss) != float('inf') for loss in losses), f'non-finite loss: {losses}'


def test_streaming_megatron_keeps_the_exact_batch_invariant():
    """S3-66 (L1 anchor): megatron's global-batch invariant under streaming -- _drop_last is a
    backend decision (megatron drops the trailing partial batch even on an iterable, where the
    stream simply ends mid-batch) and _twinkle_loader_layout sizes the batch by the DP width.
    The 2-rank streaming megatron TRAIN lands with the S13 feature batch."""
    from swift.dev.builders.dataset import _drop_last, _twinkle_loader_layout
    from swift.dev.config import DistributedConfig

    megatron = DistributedConfig(backend='megatron', mode='local', nproc_per_node=2)
    assert _drop_last(megatron) is True, 'a streaming megatron run must drop the partial tail batch'
    global_batch, _ = _twinkle_loader_layout(megatron, 4)
    assert global_batch == 8, 'the loader batch is the per-device size scaled by the DP width'
# --- S3-52 / S3-53: cached_dataset x fresh-encode mixing -------------------------------------


def _hf(rows):
    return HfDataset.from_list(rows)


def test_concat_with_cached_refuses_lazy_and_stream_mixing():
    """S3-52: a lazy/stream wrapper cannot concatenate with a map-style cached_dataset -- the
    refusal names the fix (eager-encode or cache only); a cache ALONE in those modes is fine."""
    from swift.dev.builders.dataset import _concat_with_cached

    cached = [_hf([{'messages': [{'role': 'user', 'content': 'cached'}]}])]
    for mode in ('lazy', 'stream'):
        with pytest.raises(ValueError, match='cached_dataset cannot be combined'):
            _concat_with_cached(object(), cached, encode_mode=mode)
    # cache-only is legal in any mode (nothing to concatenate against)
    merged = _concat_with_cached(None, cached, encode_mode='lazy')
    assert merged[0]['messages'] == [{'role': 'user', 'content': 'cached'}]


def test_concat_with_cached_puts_fresh_rows_before_cached_rows():
    """S3-53: the fresh split comes FIRST, the cache after -- order is observable training data,
    so it is pinned, not incidental."""
    from swift.dev.builders.dataset import _concat_with_cached

    fresh = _hf([{'messages': [{'role': 'user', 'content': f'fresh{i}'}]} for i in range(2)])
    cached = [_hf([{'messages': [{'role': 'user', 'content': f'cached{i}'}]} for i in range(3)])]
    merged = _concat_with_cached(fresh, cached, encode_mode='eager')
    contents = [row['messages'][0]['content'] for row in merged]
    assert contents == ['fresh0', 'fresh1', 'cached0', 'cached1', 'cached2']


def test_concat_with_cached_aligns_schemas_onto_the_cache():
    """S3-52 (schema half): the fresh split and the cache may carry the same columns with different
    arrow types (struct vs json content, depending on how each was produced); concatenation casts
    the FRESH side onto the cache's schema, while a column-set mismatch is refused by name."""
    from swift.dev.builders.dataset import _concat_with_cached

    fresh = _hf([{'messages': [{'role': 'user', 'content': 'f'}]}])
    # a cache whose messages column is json-typed (as an incremental jsonl round-trip produces)
    from datasets.features import Json
    from datasets.features import List as ListFeature

    cache_feats = fresh.features.copy()
    cache_feats['messages'] = ListFeature(Json())
    cached = [fresh.cast(cache_feats)]

    merged = _concat_with_cached(fresh, cached, encode_mode='eager')
    assert len(merged) == 2
    assert [row['messages'][0]['content'] for row in merged] == ['f', 'f']

    other = _hf([{'other_column': [1]}])
    with pytest.raises(ValueError, match='do not match the training split'):
        _concat_with_cached(fresh, [other], encode_mode='eager')


# --- S3-62: max_pixels bounds the measured image grid -----------------------------------------


MM_MODEL = os.environ.get('SWIFT_TEST_MODEL', '/mnt/workspace/yzhao/tastelikefeet/Qwen3.5-4B-CM-v2')
MM_MODEL_TYPE = os.environ.get('SWIFT_TEST_MODEL_TYPE', 'qwen3_5')


@pytest.mark.skipif(not os.path.isdir(MM_MODEL), reason=f'no local multimodal model at {MM_MODEL}')
def test_max_pixels_bounds_the_measured_image_grid(tmp_path):
    """S3-62: max_pixels caps the preprocessed image resolution, so the measured grid (and the
    number of image placeholder tokens) shrinks with it -- the multi-modal length-measurement path
    honors the budget BEFORE any token is counted."""
    import torch

    from swift.model import get_model_processor
    from swift.template import get_template

    from swift.dev.tests.component.dataset.test_multimodal_e2e import make_image, mm_row

    processor = get_model_processor(MM_MODEL, load_model=False, model_type=MM_MODEL_TYPE)[1]
    row = mm_row('what is this', [make_image(tmp_path / 'img.png', size=448)])

    grids = {}
    for max_pixels in (None, 32 * 28 * 4):  # full resolution, then a tiny budget
        template = get_template(processor, template_type=MM_MODEL_TYPE, max_length=2048)
        if max_pixels is not None:
            template.max_pixels = max_pixels
        with torch.device('meta'):
            template.model = get_model_processor(
                MM_MODEL, model_type=MM_MODEL_TYPE, return_dummy_model=True)[0]
        template.set_mode('train')
        encoded = template.encode(row)
        grids[max_pixels] = encoded['image_grid_thw']
    assert grids[None] is not None and len(grids[None]) > 0
    assert grids[32 * 28 * 4].sum() < grids[None].sum(), \
        f'max_pixels did not shrink the grid: {grids}'


# --- S3-64: the message-repair preprocessor ---------------------------------------------------


def test_messages_repair_preprocessor_fixes_and_drops(tmp_path):
    """S3-64: a repair preprocessor fixes a broken dialogue before conversion (here: single-quoted
    almost-JSON), and a repair that yields nothing drops the row (None -> no standard row)."""
    from swift.dev.dataset import Preprocessor
    from swift.dev.dataset.preprocessor.base import MessagesRepairPreprocessor

    class SingleQuoteRepair(MessagesRepairPreprocessor):

        def repair(self, messages):
            if not messages:
                return None
            return messages.replace("'", '"')

    rows = [
        {'messages': "[{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]"},
        {'messages': ''},  # nothing to repair -> dropped
    ]
    out = SingleQuoteRepair()(HfDataset.from_list(rows), load_from_cache_file=False)
    kept = [dict(r) for r in out]
    assert kept == [{'messages': [{'role': 'user', 'content': 'q'},
                                  {'role': 'assistant', 'content': 'a'}]}]


def test_messages_repair_base_is_abstract():
    """The base repair is an explicit override point: forgetting it fails loudly, not silently."""
    from swift.dev.dataset.preprocessor.base import MessagesRepairPreprocessor

    with pytest.raises(NotImplementedError):
        MessagesRepairPreprocessor().repair([])
