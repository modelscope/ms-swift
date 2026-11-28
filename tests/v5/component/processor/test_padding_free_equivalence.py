"""padding_free must be a LAYOUT change only: the supervised tokens cannot move.

Why this file exists: ``test_packing.py`` pins the packed layout (0-reset position_ids,
cu_seqlens, ``_is_packed_position_ids``), but nothing asserted that the concatenated stream still
carries the *same* tokens and labels as the padded batch. Packing's failure mode is silently-wrong
numerics: a dropped/duplicated token or a shifted label keeps the shape valid and the loss finite,
so a layout-only suite stays green. These tests compare the two forms token by token.

The rows below are hand-built (no tokenizer, no weights) on purpose -- the relay under test is
``InputProcessor.prepare_inputs -> collate_fn``, which takes already-encoded features.
"""

from __future__ import annotations

import pytest

from swift.dev.processor import InputProcessor

# Variable lengths on purpose: equal-length rows would hide an off-by-one in the concatenation.
ROWS = [
    {"input_ids": [11, 12, 13, 14], "labels": [-100, -100, 13, 14]},
    {"input_ids": [21, 22], "labels": [-100, 22]},
    {"input_ids": [31, 32, 33], "labels": [-100, -100, 33]},
]


def _micro_batch(padding_free: bool):
    processor = InputProcessor(framework="transformers", padding_free=padding_free)
    collated = processor.collate_fn(processor.prepare_inputs([dict(row) for row in ROWS]))
    return collated[0]


def _padded_strip(batch, key):
    """Right-padded rows -> the same tokens/positions in row order.

    Slicing by each row's own length (not by a pad-token value) keeps the assertion honest even if a
    real token id equals the pad id.
    """
    max_len = max(len(row["input_ids"]) for row in ROWS)
    flat = batch[key].flatten().tolist()
    stripped = []
    for i, row in enumerate(ROWS):
        start = i * max_len
        stripped += flat[start : start + len(row["input_ids"])]
    return stripped


def _packed_strip(batch, key):
    return batch[key].flatten().tolist()


@pytest.mark.accel(1)
def test_padding_free_keeps_the_padded_token_stream():
    padded = _micro_batch(False)
    packed = _micro_batch(True)

    packed_ids = _packed_strip(packed, "input_ids")
    padded_ids = _padded_strip(padded, "input_ids")
    assert packed_ids == padded_ids, f"padding_free changed the token stream: {packed_ids} != {padded_ids}"
    assert padded_ids == [tok for row in ROWS for tok in row["input_ids"]]


@pytest.mark.accel(1)
def test_padding_free_keeps_every_label_and_its_mask():
    padded = _micro_batch(False)
    packed = _micro_batch(True)

    # labels are compared position by position, INCLUDING -100: a shifted label that keeps the same
    # count would still change which token is supervised, which is the silent failure packing risks.
    packed_labels = _packed_strip(packed, "labels")
    padded_labels = _padded_strip(padded, "labels")
    assert packed_labels == padded_labels, (
        f"padding_free moved the supervision mask: {packed_labels} != {padded_labels}"
    )
    assert packed_labels.count(-100) == 5  # rows: 2 + 1 + 2


@pytest.mark.accel(1)
def test_padding_free_position_resets_match_row_boundaries():
    packed = _micro_batch(True)
    positions = packed["position_ids"].flatten().tolist()
    expected = []
    for row in ROWS:
        expected += list(range(len(row["input_ids"])))
    assert positions == expected, f"position reset boundaries wrong: {positions} != {expected}"

    params = InputProcessor._get_packed_seq_params(packed["position_ids"])
    boundaries = [0]
    for row in ROWS:
        boundaries.append(boundaries[-1] + len(row["input_ids"]))
    assert params.cu_seqlens_q.tolist() == boundaries, (
        f"cu_seqlens {params.cu_seqlens_q.tolist()} != row boundaries {boundaries}"
    )
    assert int(params.max_seqlen_q) == max(len(row["input_ids"]) for row in ROWS)


@pytest.mark.accel(1)
def test_padded_form_still_carries_padding_that_packing_drops():
    """Negative control: if the processor silently packed under padding_free=False, the padded batch
    would have no padding cells and the tests above would pass for the wrong reason."""
    padded = _micro_batch(False)
    max_len = max(len(row["input_ids"]) for row in ROWS)
    assert padded["input_ids"].numel() == max_len * len(ROWS), "padded batch lost its padding cells"
    assert max_len * len(ROWS) > sum(len(row["input_ids"]) for row in ROWS)
    assert InputProcessor._is_packed_position_ids(padded["position_ids"]) is False
