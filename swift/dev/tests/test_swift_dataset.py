# Copyright (c) ModelScope Contributors. All rights reserved.
"""Parity tests for :class:`SwiftDataset` against the two components it replaces.

Uses a real tokenizer rather than a mock: what is being checked is that the lengths this class measures
and the rows it serves are the same ones the old pair produced, and a mock ``encode`` would only compare
the mock against itself.
"""
import copy
import os

import pytest
from datasets import Dataset as HfDataset

MODEL = os.environ.get('SWIFT_TEST_MODEL', '/mnt/workspace/yzhao/tastelikefeet/Qwen3.5-4B-CM-v2')
MODEL_TYPE = os.environ.get('SWIFT_TEST_MODEL_TYPE', 'qwen3_5')

pytestmark = pytest.mark.skipif(not os.path.isdir(MODEL), reason=f'no local model at {MODEL}')


def make_rows(n=24):
    """Rows carrying a grounding dataset's ``objects``: nested, heterogeneous, and Arrow-hostile."""
    return [{
        'messages': [{
            'role': 'user',
            'content': f'question number {i} ' + 'padding words ' * (i % 7)
        }, {
            'role': 'assistant',
            'content': f'answer {i} ' + 'more words ' * (i % 5)
        }],
        'objects': {
            'ref': [f'obj{i}'],
            'bbox': [[10 + i, 20, 30, 40 + i]],
            'bbox_type': 'real',
            'image_id': [0],
        },
    } for i in range(n)]


@pytest.fixture(scope='module')
def processor():
    from swift.model import get_model_processor
    return get_model_processor(MODEL, load_model=False, model_type=MODEL_TYPE)[1]


@pytest.fixture
def rows():
    return make_rows()


@pytest.fixture
def dataset(rows):
    return HfDataset.from_list(copy.deepcopy(rows))


def build_template(processor, max_length=8192):
    from swift.template import get_template
    template = get_template(processor, template_type='qwen2_5', max_length=max_length)
    template.set_mode('train')
    return template


def test_lengths_match_add_length_preprocessor(processor, dataset):
    """Compared against *legacy*'s pass, which is the behaviour equivalence has to be shown against."""
    from swift.dataset.utils import AddLengthPreprocessor
    from swift.dev.dataset import SwiftDataset
    legacy = AddLengthPreprocessor(build_template(processor))(dataset, load_from_cache_file=False)
    expected = [sum(x) if isinstance(x, list) else x for x in legacy['lengths']]

    ds = SwiftDataset(dataset, build_template(processor), load_from_cache_file=False)
    assert ds.lengths == expected
    assert len(ds.lengths) == len(ds) == len(dataset)


def test_rows_match_lazy_llm_dataset(processor, dataset):
    from swift.dev.dataset import LazyLLMDataset, SwiftDataset
    lazy = LazyLLMDataset(dataset, build_template(processor).encode, random_state=42)
    ds = SwiftDataset(dataset, build_template(processor), load_from_cache_file=False, random_state=42)
    for i in range(len(dataset)):
        got, want = ds[i], lazy[i]
        assert got.keys() == want.keys()
        assert all(got[k] == want[k] for k in got)


def test_substitution_matches_lazy_llm_dataset(processor, dataset):
    """The same test at a length that rejects *some* rows, so the substitution path is compared too.

    Both walk one permutation of the same seed with the same cursor, so a substituted row has to be the
    same substituted row -- and building the permutation on first failure rather than up front must not
    change which one it is. 48 is chosen against the real distribution (40-60 tokens): tight enough that
    rows are rejected, loose enough that a substitute can be found.
    """
    from swift.dev.dataset import LazyLLMDataset, SwiftDataset
    lazy = LazyLLMDataset(dataset, build_template(processor, max_length=48).encode, random_state=42)
    ds = SwiftDataset(dataset, build_template(processor, max_length=48), load_from_cache_file=False, random_state=42)

    assert ds._idx_list is None, 'nothing should be permuted before a failure happens'
    served = [(ds[i], lazy[i]) for i in range(len(dataset))]
    assert ds._idx_list is not None, 'this length must reject some rows, or the test proves nothing'
    for got, want in served:
        assert got.keys() == want.keys()
        assert all(got[k] == want[k] for k in got)


def test_getitem_returns_one_dict(processor, dataset):
    from swift.dev.dataset import SwiftDataset
    row = SwiftDataset(dataset, build_template(processor), load_from_cache_file=False)[0]
    assert isinstance(row, dict) and 'input_ids' in row


def test_nothing_is_measured_until_asked(processor, dataset):
    from swift.dev.dataset import SwiftDataset
    ds = SwiftDataset(dataset, build_template(processor), load_from_cache_file=False)
    assert ds._lengths is None, 'constructing should not measure'
    assert ds._idx_list is None, 'constructing should not permute'

    for i in range(len(dataset)):
        ds[i]
    assert ds._lengths is None, 'serving rows should not measure'
    assert ds._idx_list is None, 'serving rows that all encode should not permute'

    measured = ds.lengths
    assert ds._lengths is not None
    assert ds.lengths is measured, 'a second ask should reuse the first'


def test_objects_column_survives_measuring(processor, dataset, rows):
    """Grounding's ``objects`` is a nested dict of mixed types -- the column Arrow is most likely to
    reshape. It has to come back byte-identical, or a bbox is normalised against the wrong image."""
    from swift.dev.dataset import SwiftDataset
    ds = SwiftDataset(dataset, build_template(processor), load_from_cache_file=False)
    assert ds.lengths, 'measuring should produce a length for every row'

    assert dataset.to_list() == rows, 'the source dataset must not be modified'
    for i in (0, 7, len(rows) - 1):
        assert ds.dataset[i]['objects'] == rows[i]['objects']


def test_encode_does_not_mutate_the_row(processor, dataset):
    """``normalize_bbox`` writes width/height into ``objects`` and pops ``bbox_type``. If that reached
    the stored row, the second encode -- the one that actually serves the sample -- would see an
    already-normalised bbox."""
    row = dataset[3]
    before = copy.deepcopy(row)
    build_template(processor).encode(row, return_length=True)
    assert row == before


def test_unusable_rows_are_marked_not_dropped(processor, dataset):
    from swift.dev.dataset import SwiftDataset
    ds = SwiftDataset(dataset, build_template(processor, max_length=12), load_from_cache_file=False)
    assert len(ds) == len(dataset), 'measuring must not change the row count'
    assert any(length == 0 for length in ds.lengths), 'an unmeasurable row should read 0'
    assert len(ds.lengths) == len(dataset), 'lengths must stay aligned with indices'


# ---- EncodedDataset: rows an earlier pass already encoded ------------------------------------


def encode_ahead(processor, dataset):
    from swift.dev.dataset import EncodePreprocessor
    return EncodePreprocessor(build_template(processor))(dataset, load_from_cache_file=False)


def test_encoded_dataset_serves_stored_rows(processor, dataset):
    from swift.dev.dataset import EncodedDataset
    encoded = encode_ahead(processor, dataset)
    ds = EncodedDataset(encoded, build_template(processor))
    assert len(ds) == len(encoded)
    assert ds[0] == encoded[0], 'an already-encoded row should come back untouched'


def test_encoded_dataset_reads_stored_lengths_without_encoding(processor, dataset):
    """The override pair: measuring reads the recorded count, so nothing re-encodes an encoded row."""
    from swift.dev.dataset import EncodedDataset
    encoded = encode_ahead(processor, dataset)
    template = build_template(processor)
    ds = EncodedDataset(encoded, template)

    calls = []
    original = template.encode
    template.encode = lambda *a, **k: calls.append(1) or original(*a, **k)
    try:
        lengths = ds.lengths
    finally:
        template.encode = original
    assert not calls, 'measuring an encoded dataset must not call template.encode'
    assert lengths == [sum(x) if isinstance(x, list) else x for x in encoded['lengths']]


def test_encoded_dataset_rejects_standard_rows(processor, dataset):
    """The failure the old pairing produced silently -- `KeyError: 'messages'` -- said up front."""
    from swift.dev.dataset import EncodedDataset
    with pytest.raises(ValueError, match='already encoded'):
        EncodedDataset(dataset, build_template(processor))


# ---- PackingDataset: now inherits instead of wrapping ----------------------------------------


def test_packing_plans_from_its_own_lengths(processor, dataset):
    """Takes standard rows and encodes them itself, so no lazily-encoding wrapper sits underneath."""
    from swift.dev.dataset import PackingDataset, SwiftDataset
    template = build_template(processor, max_length=256)
    ds = PackingDataset(template, dataset, packing_length=128, load_from_cache_file=False)

    assert isinstance(ds, SwiftDataset)
    assert len(ds) == len(ds.packed_idx) >= 1
    assert sum(len(pack) for pack in ds.packed_idx) == len(dataset), 'every usable row belongs to a pack'


def test_packing_serves_encoded_rows_per_group(processor, dataset):
    from swift.dev.dataset import PackingDataset
    ds = PackingDataset(build_template(processor, max_length=256), dataset, packing_length=128,
                        load_from_cache_file=False)
    group = ds[0]
    assert isinstance(group, list) and group, 'a group is a list of rows for the collator to concatenate'
    assert all(isinstance(row, dict) and 'input_ids' in row for row in group)
    assert len(group) == len(ds.packed_idx[0])


def test_packing_respects_packing_length(processor, dataset):
    from swift.dev.dataset import PackingDataset
    ds = PackingDataset(build_template(processor, max_length=256), dataset, packing_length=128,
                        load_from_cache_file=False)
    for index, pack in enumerate(ds.packed_idx):
        planned = sum(ds.lengths[i] for i in pack)
        assert planned <= 128, f'pack {index} planned {planned} tokens'
        served = sum(len(row['input_ids']) for row in ds[index])
        assert served == planned, f'pack {index} served {served} but planned {planned}'


def test_packing_leaves_out_unusable_rows(processor, dataset):
    """A row that reads 0 cannot be served, so reserving room for it would misplan the pack.

    48 rejects part of the real 40-60 distribution rather than all of it: with every row rejected the
    plan would be empty and the check would hold for the wrong reason.
    """
    from swift.dev.dataset import PackingDataset
    ds = PackingDataset(build_template(processor, max_length=48), dataset, packing_length=256,
                        load_from_cache_file=False)
    unusable = {i for i, length in enumerate(ds.lengths) if not length}
    packed = {i for pack in ds.packed_idx for i in pack}
    assert unusable, 'this max_length must reject some rows, or the test proves nothing'
    assert packed, 'and it must leave some rows usable, or the plan is empty for the wrong reason'
    assert not (packed & unusable), f'unusable rows were planned into packs: {sorted(packed & unusable)}'
    assert packed | unusable == set(range(len(dataset))), 'every row is either planned or marked unusable'
