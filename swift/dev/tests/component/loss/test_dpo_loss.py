# Copyright (c) ModelScope Contributors. All rights reserved.
"""``DPOTrainer.dpo_loss``: swift's own 14 preference-loss variants.

``dpo_loss`` is a bound method, but the state it reads is a handful of scalars on ``self`` plus the
accelerator's device, so it is exercised here with a stand-in trainer rather than a constructed
``DPOTrainer`` (which would need a model, a collator and an ``Accelerator``). The canonical batch is
built so that ``beta * logits == 0.1`` on every entry, which is what the pinned numbers encode; the
variants differ only in how they turn that logit into a loss.
"""
import types

import pytest
import torch

from swift.rlhf_trainers.dpo_trainer import DPOTrainer

BETA = 0.1


class _Accelerator:
    device = torch.device('cpu')


class _RunningMoments:
    """The part of TRL's ``RunningMoments`` that ``bco_pair`` reads."""

    def __init__(self):
        self._mean = 0.0

    def update(self, value):
        self._mean = float(value)

    @property
    def mean(self):
        return self._mean


def _trainer(**overrides):
    state = {
        'accelerator': _Accelerator(),
        'args': types.SimpleNamespace(discopop_tau=0.05),
        'beta': BETA,
        'f_divergence_type': '',
        'f_divergence_params': None,
        'label_smoothing': 0.0,
        'loss_type': 'sigmoid',
        'reference_free': False,
        'running': _RunningMoments(),
    }
    state.update(overrides)
    return types.SimpleNamespace(**state)


# chosen - ref_chosen = 0.5 and rejected - ref_rejected = -0.5, so every logit is beta * 1.0 = 0.1.
CHOSEN = torch.tensor([-1.0, -2.0])
REJECTED = torch.tensor([-3.0, -4.0])
REF_CHOSEN = torch.tensor([-1.5, -2.5])
REF_REJECTED = torch.tensor([-2.5, -3.5])


def _dpo(loss_type='sigmoid', model_output=None, **overrides):
    return DPOTrainer.dpo_loss(
        _trainer(**overrides), CHOSEN, REJECTED, REF_CHOSEN, REF_REJECTED,
        loss_type=loss_type, model_output=model_output)


@pytest.mark.parametrize('loss_type,expected', [
    ('sigmoid', 0.6443966627120972),
    ('robust', 0.6443966627120972),
    ('exo_pair', 2.589954137802124),
    ('hinge', 0.8999999761581421),
    ('ipo', 16.0),
    ('bco_pair', 1.3369193077087402),
    ('sppo_hard', 40.5),
    ('nca_pair', 1.3619192838668823),
    ('aot', 0.6443966627120972),
    ('aot_unpaired', 0.6443966627120972),
    ('apo_zero', 0.9750052690505981),
    ('apo_down', 0.9875181913375854),
    ('discopop', 0.8737921118736267),
])
def test_each_variant_matches_its_pinned_value(loss_type, expected):
    losses, _, _ = _dpo(loss_type)
    assert [float(x) for x in losses] == pytest.approx([expected, expected], rel=1e-4)


def test_sft_variant_uses_the_batch_nll():
    """``sft`` forwards the batch's supervised NLL instead of the preference logits. The reward fields
    are still the usual ``beta * (logps - ref_logps)``: the zeros that the branch assigns are
    overwritten by the shared tail of the function."""
    losses, chosen_rewards, rejected_rewards = _dpo('sft', model_output={'nll_loss': torch.tensor(2.0)})
    assert [float(x) for x in losses] == pytest.approx([2.0, 2.0])
    assert [float(x) for x in chosen_rewards] == pytest.approx([BETA * 0.5, BETA * 0.5])
    assert [float(x) for x in rejected_rewards] == pytest.approx([-BETA * 0.5, -BETA * 0.5])


def test_an_unknown_variant_is_refused():
    with pytest.raises(ValueError, match='Unknown loss type'):
        _dpo('does_not_exist')


def test_rewards_are_beta_times_the_log_ratio_shift():
    _, chosen_rewards, rejected_rewards = _dpo()
    assert [float(x) for x in chosen_rewards] == pytest.approx([BETA * 0.5, BETA * 0.5])
    assert [float(x) for x in rejected_rewards] == pytest.approx([-BETA * 0.5, -BETA * 0.5])


def test_reference_free_ignores_the_reference_logps():
    """With the flag set, the reference terms are dropped, so their values cannot matter."""
    free = _dpo(reference_free=True)[0]
    zeros = DPOTrainer.dpo_loss(
        _trainer(reference_free=True), CHOSEN, REJECTED, torch.zeros(2), torch.zeros(2), loss_type='sigmoid')[0]
    assert torch.allclose(free, zeros)


def test_label_smoothing_changes_the_sigmoid_loss():
    smoothed = _dpo(label_smoothing=0.01)[0]
    assert [float(x) for x in smoothed] == pytest.approx([0.6453966498374939] * 2, rel=1e-4)


@pytest.mark.parametrize('divergence,expected', [
    ('alpha_divergence', 0.6423947215080261),
    ('js_divergence', 0.6684596538543701),
])
def test_an_f_divergence_shifts_the_logits(divergence, expected):
    """``f_divergence_type`` rewrites the logits before the preference loss, which moves the value
    even though the preference ordering is unchanged."""
    losses, _, _ = _dpo(f_divergence_type=divergence, loss_type='sigmoid')
    assert [float(x) for x in losses] == pytest.approx([expected] * 2, rel=1e-4)


def test_the_alpha_divergence_coefficient_is_configurable():
    """The coefficient comes from ``f_divergence_params`` when present, and changes the loss."""
    losses, _, _ = _dpo(
        f_divergence_type='alpha_divergence',
        f_divergence_params={'alpha_divergence_coef': 0.5},
        loss_type='sigmoid')
    assert [float(x) for x in losses] == pytest.approx([0.6439003944396973] * 2, rel=1e-4)
