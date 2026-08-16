import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from swift.rlhf_trainers.dpo_trainer import DPOTrainer


class DummyModel:

    def __call__(self, input_ids, **kwargs):
        return SimpleNamespace(logits=torch.empty((*input_ids.shape, 1)))


def _make_forward_trainer(*, padding_free, packing, sequence_parallel_size, per_token_logps, loss_mask, cu_seqlens):
    trainer = object.__new__(DPOTrainer)
    trainer.template = SimpleNamespace(
        padding_free=padding_free, packing=packing, sequence_parallel_size=sequence_parallel_size)
    trainer.args = SimpleNamespace(ld_alpha=None)
    trainer.is_encoder_decoder = False
    trainer.aux_loss_enabled = False
    trainer.loss_type = ['sigmoid', 'ipo']
    trainer.label_pad_token_id = -100
    trainer.get_use_logits_to_keep = lambda _: False
    trainer.get_per_token_logps = lambda *args, **kwargs: (per_token_logps.clone(), torch.zeros_like(per_token_logps),
                                                           loss_mask.clone())
    if cu_seqlens is not None:
        trainer.get_cu_seqlens = lambda *args, **kwargs: cu_seqlens
    return trainer


def _get_forward_inputs(padding_free):
    if padding_free:
        per_token_logps = torch.tensor([[0., 0., -1.5, -1.5, 0., 0., -1., -1., -1., -1., -1., -1.]])
        loss_mask = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
        position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7]])
        cu_seqlens = torch.tensor([0, 4, 12])
    else:
        per_token_logps = torch.tensor([
            [0., 0., -1.5, -1.5, 0., 0., 0., 0.],
            [0., 0., -1., -1., -1., -1., -1., -1.],
        ])
        loss_mask = torch.tensor([
            [0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1],
        ], dtype=torch.bool)
        position_ids = torch.arange(8).repeat(2, 1)
        cu_seqlens = None
    batch = {
        'input_ids': torch.zeros_like(per_token_logps, dtype=torch.long),
        'labels': torch.zeros_like(per_token_logps, dtype=torch.long),
        'text_position_ids': position_ids,
    }
    return per_token_logps, loss_mask, cu_seqlens, batch


@torch.no_grad()
@pytest.mark.parametrize(
    'padding_free,packing,sequence_parallel_size',
    [
        (False, False, 1),
        (True, False, 1),
        (True, True, 1),
        (True, False, 2),
    ],
)
def test_dpo_forward_returns_sequence_sums_and_completion_token_counts(padding_free, packing, sequence_parallel_size):
    per_token_logps, loss_mask, cu_seqlens, batch = _get_forward_inputs(padding_free)
    trainer = _make_forward_trainer(
        padding_free=padding_free,
        packing=packing,
        sequence_parallel_size=sequence_parallel_size,
        per_token_logps=per_token_logps,
        loss_mask=loss_mask,
        cu_seqlens=cu_seqlens,
    )

    output = trainer.concatenated_forward(DummyModel(), batch)

    torch.testing.assert_close(output['chosen_logps'], torch.tensor([-3.]))
    torch.testing.assert_close(output['rejected_logps'], torch.tensor([-6.]))
    torch.testing.assert_close(output['chosen_completion_token_counts'], torch.tensor([2]))
    torch.testing.assert_close(output['rejected_completion_token_counts'], torch.tensor([6]))


def _make_loss_trainer(loss_type, loss_weights=None, completion_token_counts=(2, 6)):
    trainer = object.__new__(DPOTrainer)
    trainer.loss_type = loss_type
    trainer.loss_weights = loss_weights
    trainer.beta = 0.1
    trainer.label_smoothing = 0.0
    trainer.reference_free = False
    trainer.f_divergence_type = 'reverse_kl'
    trainer.f_divergence_params = {}
    trainer.f_alpha_divergence_coef = 0.5
    trainer.use_weighting = False
    trainer.aux_loss_enabled = False
    trainer.args = SimpleNamespace(rpo_alpha=None)
    trainer.accelerator = SimpleNamespace(device=torch.device('cpu'), gather_for_metrics=lambda tensor: tensor)
    trainer.concatenated_forward = lambda *args, **kwargs: {
        'chosen_logps': torch.tensor([-3.]),
        'rejected_logps': torch.tensor([-6.]),
        'chosen_completion_token_counts': torch.tensor([completion_token_counts[0]]),
        'rejected_completion_token_counts': torch.tensor([completion_token_counts[1]]),
        'mean_chosen_logits': torch.tensor(0.),
        'mean_rejected_logits': torch.tensor(0.),
        'nll_loss': torch.tensor(0.),
    }
    return trainer


def _get_loss_and_metrics(loss_type, loss_weights=None, completion_token_counts=(2, 6)):
    trainer = _make_loss_trainer(loss_type, loss_weights, completion_token_counts)
    batch = {
        # Precomputed reference log-probs remain sequence sums and are normalized only for IPO.
        'ref_chosen_logps': torch.tensor([-1.]),
        'ref_rejected_logps': torch.tensor([-2.]),
    }
    return trainer.get_batch_loss_metrics(None, batch)


def _get_loss(loss_type, loss_weights=None, completion_token_counts=(2, 6)):
    loss, _ = _get_loss_and_metrics(loss_type, loss_weights, completion_token_counts)
    return loss


def test_mixed_dpo_and_ipo_losses_use_sum_and_mean_logps_independently():
    sigmoid_loss = _get_loss('sigmoid')
    ipo_loss = _get_loss('ipo')
    weights = [0.25, 0.75]
    mixed_loss = _get_loss(['sigmoid', 'ipo'], weights)

    sigmoid_logit = ((-3.) - (-6.)) - ((-1.) - (-2.))
    expected_sigmoid_loss = -F.logsigmoid(torch.tensor(0.1 * sigmoid_logit))
    ipo_logit = ((-3. / 2) - (-6. / 6)) - ((-1. / 2) - (-2. / 6))
    expected_ipo_loss = (torch.tensor(ipo_logit) - 1 / (2 * 0.1))**2

    torch.testing.assert_close(sigmoid_loss, expected_sigmoid_loss)
    torch.testing.assert_close(ipo_loss, expected_ipo_loss)
    torch.testing.assert_close(mixed_loss, weights[0] * sigmoid_loss + weights[1] * ipo_loss)


def test_ipo_clamps_empty_completion_token_count():
    loss = _get_loss('ipo', completion_token_counts=(0, 0))
    expected_logit = ((-3.) - (-6.)) - ((-1.) - (-2.))
    expected_loss = (torch.tensor(expected_logit) - 1 / (2 * 0.1))**2
    torch.testing.assert_close(loss, expected_loss)
