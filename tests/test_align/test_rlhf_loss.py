import pytest
import torch
from types import SimpleNamespace
from unittest.mock import patch

from swift.rlhf_trainers import rlhf_mixin


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('reduction', ['mean', 'sum'])
def test_sequence_parallel_selective_log_softmax(dtype, reduction):
    trainer = object.__new__(rlhf_mixin.RLHFTrainerMixin)
    trainer.template = SimpleNamespace(sequence_parallel_size=2)
    labels = torch.tensor([[1, 2, -100], [3, 4, 5]])
    logits = torch.randn(2, 3, 7, dtype=dtype, requires_grad=True)
    expected_logits = logits.detach().clone().requires_grad_(True)
    expected_mask = labels != -100
    expected_labels = labels.masked_fill(~expected_mask, 0)
    expected_logps = torch.gather(
        expected_logits.log_softmax(-1), dim=-1, index=expected_labels.unsqueeze(-1)).squeeze(-1) * expected_mask
    expected_reduced_logits = getattr(expected_logits, reduction)(-1)
    expected_logps.sum().backward()

    with (
            patch.object(rlhf_mixin.GatherLoss, 'apply', side_effect=lambda logps, mask, *_: (logps, mask)),
            patch.object(rlhf_mixin.sequence_parallel, 'gather', side_effect=lambda tensor, **_: tensor),
            patch.object(rlhf_mixin.sequence_parallel, 'extra_kwargs', {}),
            patch.object(rlhf_mixin, 'selective_log_softmax', wraps=rlhf_mixin.selective_log_softmax) as
            selective_log_softmax,
    ):
        actual_logps, actual_reduced_logits, actual_mask = trainer.get_per_token_logps(
            logits, labels, reduction=reduction)

    selective_log_softmax.assert_called_once()
    torch.testing.assert_close(actual_logps, expected_logps)
    torch.testing.assert_close(actual_reduced_logits, expected_reduced_logits)
    torch.testing.assert_close(actual_mask, expected_mask)
    actual_logps.sum().backward()
    torch.testing.assert_close(logits.grad, expected_logits.grad)
