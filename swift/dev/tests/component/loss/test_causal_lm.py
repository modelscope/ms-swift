# Copyright (c) ModelScope Contributors. All rights reserved.
"""``CustomCrossEntropyLoss``: the ``--loss_type cross_entropy`` SFT loss.

Two things are pinned. The denominator: only positions whose *next* label is not ``-100`` count, so a
padded batch is not diluted by its padding -- and ``num_items_in_batch``, which the Trainer supplies
when gradient-accumulation batches are fused, overrides that count. The shift: the loss is computed by
*rolling* the labels rather than slicing the logits, so when ``labels[0]`` is not ``-100`` the final
position trains toward the first token. Standard templates mask the prompt, so ``labels[0]`` is -100
and the wrap-around never fires -- but it is the actual behavior, and it is what a refactor to logit
slicing would silently change.
"""
import types

import pytest
import torch
import torch.nn.functional as F

from swift.loss.causal_lm import CustomCrossEntropyLoss


class _Trainer:
    pass


def _loss():
    return CustomCrossEntropyLoss(args=None, trainer=_Trainer())


def _outputs(logits):
    return types.SimpleNamespace(logits=logits)


# One sample, four positions, vocab of 4. Position i predicts labels[i+1]; positions 2 and 3 target
# padding.
LOGITS = torch.full((1, 4, 4), -10.0)
LOGITS[0, 0, 2] = 10.0  # predicts labels[1] = 2
LOGITS[0, 1, 3] = 10.0  # predicts labels[2] = 3
LOGITS[0, 2, 0] = 10.0  # target is -100: ignored
LABELS = torch.tensor([[-100, 2, 3, -100]])


def test_a_batch_that_predicts_every_target_scores_zero():
    assert float(_loss()(_outputs(LOGITS), LABELS)) == pytest.approx(0.0, abs=1e-4)


def test_a_wrong_prediction_is_divided_by_the_number_of_real_targets():
    logits = LOGITS.clone()
    logits[0, 0, 2] = -10.0  # position 0 now predicts token 1...
    logits[0, 0, 1] = 10.0  # ...while the target is token 2
    loss = _loss()(_outputs(logits), LABELS)
    expected = F.cross_entropy(torch.tensor([[-10.0, 10.0, -10.0, -10.0]]), torch.tensor([2]))
    assert float(loss) == pytest.approx(float(expected) / 2, abs=1e-3)  # two valid targets


def test_num_items_in_batch_overrides_the_denominator():
    """HF passes the *global* token count when accumulation steps are fused into one batch."""
    logits = LOGITS.clone()
    logits[0, 0, 2] = -10.0
    logits[0, 0, 1] = 10.0
    loss = _loss()(_outputs(logits), LABELS, num_items_in_batch=4)
    expected = F.cross_entropy(torch.tensor([[-10.0, 10.0, -10.0, -10.0]]), torch.tensor([2]))
    assert float(loss) == pytest.approx(float(expected) / 4, abs=1e-3)


def test_padding_positions_neither_contribute_nor_dilute():
    """Three of four positions are padding; the loss is set by the one real target alone."""
    labels = torch.tensor([[-100, -100, -100, 2]])
    logits = torch.full((1, 4, 4), -10.0)
    logits[0, 2, 2] = 10.0  # position 2 predicts labels[3] = 2 correctly
    assert float(_loss()(_outputs(logits), labels)) == pytest.approx(0.0, abs=1e-4)


def test_the_labels_are_rolled_so_the_last_position_predicts_the_first_label():
    labels = torch.tensor([[2, 3, -100, -100]])
    logits = torch.full((1, 4, 4), -10.0)
    logits[0, 0, 3] = 10.0  # predicts labels[1] = 3 correctly
    logits[0, 3, 0] = 10.0  # the final position predicts token 0, but the rolled target is labels[0] = 2
    loss = _loss()(_outputs(logits), labels)
    # the denominator is labels[:, 1:], which holds exactly one real target
    expected = F.cross_entropy(torch.tensor([[-10.0, -10.0, -10.0, 10.0]]), torch.tensor([2]))
    assert float(loss) == pytest.approx(float(expected), abs=1e-3)
