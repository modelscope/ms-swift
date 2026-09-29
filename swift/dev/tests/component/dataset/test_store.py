# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for the unified dataset store (``swift.dev.dataset.store``).

This is the one serialiser ``swift export --to_cached_dataset`` and ``swift infer`` both write through,
so its contract is tested directly and offline: no model, no GPU, no network. Every backend, the field
allow-list, and the reader's per-path mechanics (the ``length``->``lengths`` rename, the
``truncation_strategy='delete'`` filter, the ``#N`` row budget) are exercised end to end -- write a store,
read it back through :func:`load_dataset_store`, and assert on the columns and values that survive.

The recipe-level chains that drive this writer (``export_cached_dataset`` and ``run_infer``) are covered
in ``test_cached_dataset_store.py`` and the ``infer`` suite; here the primitives are pinned so a regression
in the shared module is caught at its source rather than only through one of its two callers.
"""
import logging
import os

import pytest
from datasets import Dataset as HfDataset

from swift.dev.dataset.store import (CORE_STORE_FIELDS, KNOWN_STORE_FIELDS, load_dataset_store,
                                     validate_store_fields, write_dataset_store)


def _rows():
    """Two encoded-shaped rows: the core three columns plus a semantic column that rides along."""
    return [
        {
            'input_ids': [1, 2, 3],
            'labels': [-100, 2, 3],
            'lengths': 3,
            'messages': [{
                'role': 'user',
                'content': 'hi'
            }]
        },
        {
            'input_ids': [4, 5, 6, 7],
            'labels': [-100, -100, 6, 7],
            'lengths': 4,
            'messages': [{
                'role': 'user',
                'content': 'hello'
            }]
        },
    ]


# --- field registry -------------------------------------------------------------------


def test_core_fields_are_the_materialised_anchor():
    """``lengths`` is the anchor every length-aware consumer reads, so it is a core column alongside the
    tokens; the registry is a tuple (ordered, documented) not a set."""
    assert CORE_STORE_FIELDS == ('input_ids', 'labels', 'lengths')
    assert set(CORE_STORE_FIELDS) <= KNOWN_STORE_FIELDS


def test_known_fields_cover_the_infer_emit_and_recorder_payload():
    """The vocabulary names the columns infer's emit and rollout recorder produce, so a ``store_fields``
    entry for any of them validates rather than being rejected as unrecognised."""
    for field in ('completion_mask', 'response_token_ids', 'response_loss_mask', 'rollout_logprobs', 'messages',
                  'all_messages', 'completions', 'rejected_response', 'rollout_tokens', 'id', 'num_generations',
                  'scores', 'reward_score', 'images', 'videos', 'audios'):
        assert field in KNOWN_STORE_FIELDS, field


# --- validate_store_fields ------------------------------------------------------------


def test_validate_returns_deduped_keep_list():
    keep = validate_store_fields(['input_ids', 'labels', 'labels', 'lengths'], ['input_ids', 'labels', 'lengths'])
    assert keep == ['input_ids', 'labels', 'lengths']  # order preserved, duplicate collapsed


def test_validate_rejects_unknown_field_name():
    """A misspelt / non-store name is refused rather than silently dropped -- the whole point of the guard."""
    with pytest.raises(ValueError, match='unrecognised'):
        validate_store_fields(['input_ids', 'typo_field'], ['input_ids', 'labels', 'lengths'])


def test_validate_rejects_known_field_absent_from_this_data():
    """A real store field that THIS encoding did not produce is a different error than an unknown name:
    the message says it is known-but-absent so the caller knows the command, not the spelling, is wrong."""
    with pytest.raises(ValueError, match='not produced by this encoding'):
        validate_store_fields(['rollout_logprobs'], ['input_ids', 'labels', 'lengths'])


def test_validate_warns_when_lengths_dropped(caplog):
    """Dropping ``lengths`` is legal (a tokens-only store) but disables every length-aware consumer, so it
    is surfaced rather than passed silently."""
    with caplog.at_level(logging.WARNING):
        keep = validate_store_fields(['input_ids', 'labels'], ['input_ids', 'labels', 'lengths'])
    assert keep == ['input_ids', 'labels']
    assert any('lengths' in rec.message for rec in caplog.records)


def test_validate_no_warning_when_data_has_no_lengths(caplog):
    with caplog.at_level(logging.WARNING):
        validate_store_fields(['input_ids'], ['input_ids', 'labels'])
    assert not any('lengths' in rec.message for rec in caplog.records)


# --- write/load: arrow backend --------------------------------------------------------


def test_arrow_roundtrip_from_list_of_dicts(tmp_path):
    path = str(tmp_path / 'store')
    returned = write_dataset_store(_rows(), path, backend='arrow')
    assert returned == path
    assert os.path.isdir(path)  # an arrow store is a save_to_disk directory
    ds = load_dataset_store(path)
    assert len(ds) == 2
    assert set(ds.column_names) == {'input_ids', 'labels', 'lengths', 'messages'}
    assert ds[0]['input_ids'] == [1, 2, 3]
    assert ds[1]['labels'] == [-100, -100, 6, 7]
    assert ds[1]['lengths'] == 4


def test_arrow_roundtrip_from_hf_dataset(tmp_path):
    """The exporter hands over an HF ``Dataset``; infer hands over ``list[dict]`` -- both are accepted."""
    path = str(tmp_path / 'store')
    write_dataset_store(HfDataset.from_list(_rows()), path, backend='arrow')
    ds = load_dataset_store(path)
    assert len(ds) == 2 and ds[0]['input_ids'] == [1, 2, 3]


def test_arrow_default_backend_is_arrow(tmp_path):
    """``backend`` defaults to arrow, so the exporter's ``write_dataset_store(enc, path)`` writes a dir."""
    path = str(tmp_path / 'store')
    write_dataset_store(_rows(), path)
    assert os.path.isdir(path)
    assert len(load_dataset_store(path)) == 2


def test_arrow_fields_restriction_keeps_only_requested(tmp_path):
    path = str(tmp_path / 'store')
    write_dataset_store(_rows(), path, backend='arrow', fields=['input_ids', 'labels', 'lengths'])
    ds = load_dataset_store(path)
    assert set(ds.column_names) == {'input_ids', 'labels', 'lengths'}  # 'messages' dropped


def test_arrow_fields_restriction_rejects_unknown_column(tmp_path):
    path = str(tmp_path / 'store')
    with pytest.raises(ValueError, match='unrecognised'):
        write_dataset_store(_rows(), path, backend='arrow', fields=['input_ids', 'nope'])


# --- write/load: jsonl backend --------------------------------------------------------


def test_jsonl_roundtrip_from_list_of_dicts(tmp_path):
    path = str(tmp_path / 'store.jsonl')
    returned = write_dataset_store(_rows(), path, backend='jsonl')
    assert returned == path
    assert os.path.isfile(path)  # a jsonl store is a single file
    ds = load_dataset_store(path)
    assert len(ds) == 2
    assert ds[0]['input_ids'] == [1, 2, 3]
    assert ds[1]['messages'] == [{'role': 'user', 'content': 'hello'}]


def test_jsonl_roundtrip_from_hf_dataset(tmp_path):
    path = str(tmp_path / 'store.jsonl')
    write_dataset_store(HfDataset.from_list(_rows()), path, backend='jsonl')
    ds = load_dataset_store(path)
    assert len(ds) == 2 and ds[0]['labels'] == [-100, 2, 3]


def test_jsonl_fields_restriction_keeps_only_requested(tmp_path):
    path = str(tmp_path / 'store.jsonl')
    write_dataset_store(_rows(), path, backend='jsonl', fields=['input_ids', 'lengths'])
    ds = load_dataset_store(path)
    assert set(ds.column_names) == {'input_ids', 'lengths'}


# --- backend dispatch guards ----------------------------------------------------------


def test_bin_backend_raises_not_implemented(tmp_path):
    """``bin`` is a named later phase: it raises rather than silently falling back to another container."""
    with pytest.raises(NotImplementedError, match='later phase'):
        write_dataset_store(_rows(), str(tmp_path / 'x'), backend='bin')


def test_unknown_backend_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match='unknown store backend'):
        write_dataset_store(_rows(), str(tmp_path / 'x'), backend='parquet')


# --- reader mechanics -----------------------------------------------------------------


def test_load_renames_legacy_length_to_lengths(tmp_path):
    """ms-swift 3.x wrote the token count as ``length``; the reader normalises it to ``lengths`` so a
    cache exported by either side loads into the same contract."""
    path = str(tmp_path / 'legacy')
    rows = [{'input_ids': [1, 2], 'labels': [-100, 2], 'length': 2},
            {'input_ids': [3, 4, 5], 'labels': [-100, 4, 5], 'length': 3}]
    write_dataset_store(rows, path, backend='arrow')
    ds = load_dataset_store(path)
    assert 'lengths' in ds.column_names and 'length' not in ds.column_names
    assert ds['lengths'] == [2, 3]


def test_load_keeps_lengths_when_already_named_lengths(tmp_path):
    path = str(tmp_path / 'store')
    write_dataset_store(_rows(), path, backend='arrow')
    ds = load_dataset_store(path)
    assert ds['lengths'] == [3, 4]


def test_delete_filter_drops_over_length_scalar(tmp_path):
    """``truncation_strategy='delete'`` + ``max_length`` drops rows longer than the budget before serving."""
    path = str(tmp_path / 'store')
    rows = [{'input_ids': list(range(n)), 'labels': list(range(n)), 'lengths': n} for n in (10, 200, 50)]
    write_dataset_store(rows, path, backend='arrow')
    ds = load_dataset_store(path, max_length=100, truncation_strategy='delete')
    assert ds['lengths'] == [10, 50]  # the 200-token row is gone


def test_delete_filter_uses_longest_of_packed_list(tmp_path):
    """A packed cache stores ``lengths`` as a list of the packed segments' counts; the filter is on the
    real sequence length, i.e. the longest segment, not the list length or its sum."""
    path = str(tmp_path / 'store')
    rows = [
        {'input_ids': [1], 'labels': [1], 'lengths': [30, 40]},   # longest 40 -> kept at max_length=100
        {'input_ids': [2], 'labels': [2], 'lengths': [60, 70]},   # longest 70 -> kept
        {'input_ids': [3], 'labels': [3], 'lengths': [50, 120]},  # longest 120 -> dropped
    ]
    write_dataset_store(rows, path, backend='arrow')
    ds = load_dataset_store(path, max_length=100, truncation_strategy='delete')
    assert len(ds) == 2
    assert ds['lengths'] == [[30, 40], [60, 70]]


def test_delete_filter_keeps_unmeasurable_empty_lengths(tmp_path):
    """A row that could not be encoded carries ``lengths=[]`` (see MeasurePreprocessor, which keeps the
    column ``List[int]`` on both branches); it is treated as length 0 and kept, to be substituted at
    access time rather than dropped here. The measurable rows are list-typed too, as ``template.encode``
    always emits ``lengths`` as a list."""
    path = str(tmp_path / 'store')
    rows = [{'input_ids': [1], 'labels': [1], 'lengths': [5]}, {'input_ids': [], 'labels': [], 'lengths': []}]
    write_dataset_store(rows, path, backend='arrow')
    ds = load_dataset_store(path, max_length=3, truncation_strategy='delete')
    assert len(ds) == 1  # only the empty-lengths row survives; the 5-token row exceeds max_length=3
    assert ds[0]['lengths'] == []


def test_no_delete_filter_without_max_length(tmp_path):
    """``truncation_strategy='delete'`` alone (no max_length) cannot filter, so every row is kept."""
    path = str(tmp_path / 'store')
    rows = [{'input_ids': list(range(n)), 'labels': list(range(n)), 'lengths': n} for n in (10, 200)]
    write_dataset_store(rows, path, backend='arrow')
    ds = load_dataset_store(path, truncation_strategy='delete')
    assert len(ds) == 2


def test_row_budget_hash_n_subsamples(tmp_path):
    """A trailing ``#N`` on a path that does not itself exist is a row budget, honoured after the read."""
    path = str(tmp_path / 'store')
    rows = [{'input_ids': [i], 'labels': [i], 'lengths': 1} for i in range(5)]
    write_dataset_store(rows, path, backend='arrow')
    ds = load_dataset_store(path + '#2')
    assert len(ds) == 2


def test_row_budget_is_deterministic_with_seed(tmp_path):
    path = str(tmp_path / 'store')
    rows = [{'input_ids': [i], 'labels': [i], 'lengths': 1} for i in range(10)]
    write_dataset_store(rows, path, backend='arrow')
    a = load_dataset_store(path + '#4', data_seed=7, shuffle=True)
    b = load_dataset_store(path + '#4', data_seed=7, shuffle=True)
    assert a['input_ids'] == b['input_ids']


def test_existing_path_with_hash_is_read_as_is(tmp_path):
    """When the literal path exists, the ``#N`` is part of the name, not a budget -- an existing store is
    read as-is (matching the ``dataset#N`` syntax used everywhere else)."""
    dir_path = str(tmp_path / 'weird#3')
    rows = [{'input_ids': [i], 'labels': [i], 'lengths': 1} for i in range(5)]
    write_dataset_store(rows, dir_path, backend='arrow')
    ds = load_dataset_store(dir_path)  # the name exists -> no split, all 5 rows
    assert len(ds) == 5


def test_jsonl_reader_roundtrips_through_load(tmp_path):
    """The reader detects the backend from the path: a file is jsonl, a directory is arrow."""
    path = str(tmp_path / 'store.jsonl')
    write_dataset_store(_rows(), path, backend='jsonl')
    assert not os.path.isdir(path)
    ds = load_dataset_store(path)
    assert len(ds) == 2 and ds[0]['input_ids'] == [1, 2, 3]
