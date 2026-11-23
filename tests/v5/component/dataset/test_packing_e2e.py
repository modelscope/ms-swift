# Copyright (c) ModelScope Contributors. All rights reserved.
"""End-to-end packing tests: a real file on disk all the way to a packed, concatenated sequence.

The existing coverage stops on both sides of the seam this file closes:

- ``processor/test_packing.py`` drives the *collate* side with synthetic rows
  (``{'input_ids': [1, 2, 3]}``), proving the InputProcessor flattens a packed group and injects the
  multiple-0 ``position_ids``. It never builds a ``PackingDataset``, so the planning pass is untouched.
- ``test_api.py`` mocks ``PackingDataset`` / ``IterablePackingDataset`` to check wiring -- which args
  reach them -- not what they produce.

What is left unproven is the two halves together, which is what a user actually runs:

    a file -> load_dataset -> PackingDataset (measures every row's length, then plans groups)
           -> __getitem__ returns the encoded rows of one group
           -> the template's own data_collator concatenates them into ONE sequence whose position_ids
              reset per member.

If this runs, packing is usable end to end. The map-style path (planned globally from known lengths) and
the streaming path (packed over a sliding window, lengths known only as rows arrive) are both driven,
because they are different code with different information available.
"""
import json
import os
from itertools import chain

import pytest

MODEL = os.environ.get('SWIFT_TEST_MODEL', '/mnt/workspace/yzhao/tastelikefeet/Qwen3.5-4B-CM-v2')
MODEL_TYPE = os.environ.get('SWIFT_TEST_MODEL_TYPE', 'qwen3_5')

needs_model = pytest.mark.skipif(not os.path.isdir(MODEL), reason=f'no local model at {MODEL}')

# Short, varied-length rows. Lengths differ so a packing plan has a real choice to make, and are small
# enough that a modest ``packing_length`` still groups several of them together.
ROWS = [
    {'query': 'hello there, how are you doing today', 'response': 'fine'},
    {'query': 'hi', 'response': 'ok'},
    {'query': 'give me a somewhat longer answer please', 'response': 'sure, here is a longer answer for you'},
    {'query': 'and one more', 'response': 'done'},
    {'query': 'count to three', 'response': 'one two three'},
    {'query': 'name a colour', 'response': 'blue'},
    {'query': 'what is the capital of france', 'response': 'paris'},
    {'query': 'say something short', 'response': 'yes'},
]

# The rows encode to ~33-47 tokens each (system prompt included). A budget of 100 fits two or three
# short rows per pack -- small enough to force real grouping, large enough that no single row overflows
# it, so every row is packable and the plan partitions the whole dataset.
PACKING_LENGTH = 100


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return str(path)


@pytest.fixture(scope='module')
def processor():
    from swift.model import get_model_processor
    return get_model_processor(MODEL, load_model=False, model_type=MODEL_TYPE)[1]


def build_template(processor, max_length=8192):
    from swift.template import get_template
    template = get_template(processor, template_type='qwen2_5', max_length=max_length)
    template.set_mode('train')
    return template


def assert_packed_concatenation(template, group):
    """One packed group -> data_collator -> a single concatenated sequence with per-member resets.

    ``group`` is what ``PackingDataset.__getitem__`` returns: a ``list`` of encoded rows. The collator
    flattens it (packing is on) and ``packing_row`` concatenates the members, so the result is one
    sequence whose ``position_ids`` restart at 0 for each member -- the layout flash-attn reads as
    separate sequences. A regression that padded instead of packed, or that lost the reset, fails here.
    """
    assert len(group) > 1, 'this group must hold several members, or packing is proven nothing'
    batch = template.data_collator([group])

    lengths = [row['length'] for row in group]
    total = sum(lengths)
    # padding_free packs the whole group into one row: shape is (1, total), not (members, ...).
    assert batch['input_ids'].shape == (1, total)
    assert batch['labels'].shape == (1, total)

    # position_ids is the concatenation of range(len) per member -- one 0-reset per member.
    position_ids = batch['position_ids'].flatten().tolist()
    expected = list(chain.from_iterable(range(length) for length in lengths))
    assert position_ids == expected
    assert position_ids.count(0) == len(group)


# ---- the map-style path: plan globally from measured lengths (needs the model) ----------------


@needs_model
def test_packing_dataset_partitions_and_plans(processor, tmp_path):
    """PackingDataset measures every row, then groups the indices so nothing is lost or doubled."""
    from swift.dev.dataset import PackingDataset, load_dataset

    path = write_jsonl(tmp_path / 'a6_map.jsonl', ROWS)
    train, _ = load_dataset([path])
    template = build_template(processor)
    pds = PackingDataset(template, train, load_from_cache_file=False, packing_length=PACKING_LENGTH)

    # Packing must have been switched on downstream, or the collator would pad instead of concatenating.
    assert template.packing is True and template.padding_free is True
    assert len(pds) == len(pds.packed_idx) > 0
    # Every encodable row lands in exactly one group: no row dropped, none counted twice.
    assert sorted(chain.from_iterable(pds.packed_idx)) == list(range(len(train)))
    # And packing actually packed -- at least one group holds more than one member.
    assert max(len(group) for group in pds.packed_idx) > 1
    # Each group's planned length is the sum of its members' measured lengths, within the budget.
    for group_idx, members in enumerate(pds.packed_idx):
        assert pds.packed_length[group_idx] == sum(pds.lengths[i] for i in members)


@needs_model
def test_packing_dataset_group_collates_into_one_concatenated_sequence(processor, tmp_path):
    """The hand-off: __getitem__ returns a group, and the template concatenates it into one sequence."""
    from swift.dev.dataset import PackingDataset, load_dataset

    path = write_jsonl(tmp_path / 'a6_collate.jsonl', ROWS)
    train, _ = load_dataset([path])
    template = build_template(processor)
    pds = PackingDataset(template, train, load_from_cache_file=False, packing_length=PACKING_LENGTH)

    group_index = next(i for i, members in enumerate(pds.packed_idx) if len(members) > 1)
    group = pds[group_index]
    # A group is a list of encoded rows, each already carrying tokens and its own length.
    assert isinstance(group, list) and all('input_ids' in row for row in group)
    assert_packed_concatenation(template, group)


@needs_model
def test_sequential_strategy_preserves_dataset_order(processor, tmp_path):
    """'sequential' next-fit keeps sample order; 'binpack' is free to reorder. Pin the difference.

    A sequential sampler needs the packed order to follow the dataset order, which is exactly what the
    sequential strategy guarantees and binpack does not. Flattening the sequential plan must read back
    as 0..N-1.
    """
    from swift.dev.dataset import PackingDataset, load_dataset

    path = write_jsonl(tmp_path / 'a6_seq.jsonl', ROWS)
    train, _ = load_dataset([path])
    template = build_template(processor)
    pds = PackingDataset(
        template, train, load_from_cache_file=False, packing_length=PACKING_LENGTH, packing_strategy='sequential')
    assert list(chain.from_iterable(pds.packed_idx)) == list(range(len(train)))


# ---- the streaming path: pack over a sliding window, lengths known only as rows arrive ---------


@needs_model
def test_iterable_packing_dataset_packs_a_stream(processor, tmp_path):
    """A stream cannot be planned, so IterablePackingDataset packs a window as rows arrive.

    Encoding happens in the workers here (a stream has no separate measuring pass), so iteration yields
    groups of already-encoded rows -- the same shape the map-style __getitem__ returns, and it must
    collate the same way.
    """
    from swift.dev.dataset import IterablePackingDataset, load_dataset

    path = write_jsonl(tmp_path / 'a6_stream.jsonl', ROWS)
    train, _ = load_dataset([path], streaming=True)
    template = build_template(processor)
    ipds = IterablePackingDataset(
        template, train, num_proc=1, packing_interval=4, packing_length=PACKING_LENGTH)

    groups = list(ipds)
    assert groups, 'the stream produced no packed groups'
    # Every yielded group is a list of encoded rows; at least one holds several members.
    assert all(isinstance(group, list) and all('input_ids' in row for row in group) for group in groups)
    multi = next(group for group in groups if len(group) > 1)
    assert_packed_concatenation(template, multi)
