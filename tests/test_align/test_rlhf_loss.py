import os
import pytest
import torch
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from swift.rlhf_trainers import rlhf_mixin
from swift.rlhf_trainers.dpo_trainer import DPOTrainer
from swift.rlhf_trainers.kto_trainer import KTOTrainer
from swift.rlhf_trainers.rlhf_mixin import RLHFTrainerMixin
from swift.rlhf_trainers.utils import pad_logps_back_to_batch
from swift.trainers.mixin import SwiftMixin
from swift.utils import get_packed_seq_params


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


def _test_devices():
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))
    return devices


def _reference_compact_cu_seqlens(cu_seqlens, keep_mask):
    boundaries = cu_seqlens.cpu().tolist()
    compact_boundaries = [0]
    kept = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        kept += int(keep_mask[start:end].sum().cpu())
        compact_boundaries.append(kept)
    return torch.tensor(compact_boundaries, dtype=cu_seqlens.dtype, device=cu_seqlens.device)


def _reference_segment_sum(values, lengths):
    outputs = []
    offset = 0
    for length in lengths.cpu().tolist():
        outputs.append(values[offset:offset + length].sum(dim=0))
        offset += length
    return torch.stack(outputs)


def _reference_pad_logps(logps_rmpad, seq_lengths, logits_to_keep, dtype=None, pad_value=-1e10):
    if dtype is None:
        dtype = logps_rmpad.dtype
    lengths = seq_lengths.cpu().tolist()
    device = logps_rmpad.device
    output = torch.full((len(lengths), logits_to_keep), pad_value, dtype=dtype, device=device)
    valid_mask = torch.zeros_like(output, dtype=torch.float32)
    flat = logps_rmpad.flatten().to(dtype)
    offset = 0
    for i, seq_len in enumerate(lengths):
        actual_len = min(max(flat.numel() - offset, 0), seq_len)
        if actual_len <= 0:
            offset += seq_len
            continue
        pad_len = logits_to_keep - (actual_len if actual_len < seq_len else seq_len)
        output[i, pad_len:] = flat[offset:offset + actual_len]
        valid_mask[i, pad_len:] = 1.0
        offset += seq_len
    return output, valid_mask


def _reference_dpo_sum(values, lengths, num_examples, ld_alpha=None, is_ref_model=False):
    lengths_list = lengths.cpu().tolist()
    public_lengths = [min(lengths_list[i], lengths_list[i + num_examples]) for i in range(num_examples)]
    outputs = []
    offset = 0
    for i, length in enumerate(lengths_list):
        public_length = public_lengths[i % num_examples]
        if ld_alpha is not None and not is_ref_model and length > public_length:
            front = values[offset:offset + public_length].sum()
            rear = values[offset + public_length:offset + length].sum()
            outputs.append(front + ld_alpha * rear)
        else:
            outputs.append(values[offset:offset + length].sum())
        offset += length
    return torch.stack(outputs)


class _ModelStub:

    def __init__(self, logits):
        self.logits = logits

    def __call__(self, **kwargs):
        return SimpleNamespace(logits=self.logits)


class _LogitsToKeepModel:

    def __init__(self, logits):
        self.logits = logits

    def __call__(self, logits_to_keep=None, **kwargs):
        logits = self.logits
        if isinstance(logits_to_keep, torch.Tensor):
            logits = logits[:, logits_to_keep]
        return SimpleNamespace(logits=logits)


class _PaddingFreeDPOStub:

    def __init__(self, ld_alpha=0.5):
        self.args = SimpleNamespace(ld_alpha=ld_alpha, use_logits_to_keep=True)
        self.template = SimpleNamespace(sequence_parallel_size=1, padding_free=True)
        self.aux_loss_enabled = False
        self.is_encoder_decoder = False
        self.label_pad_token_id = -100
        self.loss_type = ['sigmoid']

    def get_use_logits_to_keep(self, default_value=True):
        return True

    prepare_logits_to_keep = SwiftMixin.prepare_logits_to_keep
    get_cu_seqlens = SwiftMixin.get_cu_seqlens
    get_per_token_logps = RLHFTrainerMixin.get_per_token_logps
    _packed_sequence_sum = staticmethod(RLHFTrainerMixin._packed_sequence_sum)


class TestPackedRLHFReduction(unittest.TestCase):

    def test_padding_free_dpo_logits_to_keep_ld_integration(self):
        sequence_lengths = [4, 3, 5, 2]
        label_mask_values = [False, True, False, True, True, True, False, True, True, True, False, True, True, False]

        for device in _test_devices():
            with self.subTest(device=device):
                torch.manual_seed(23)
                position_ids = torch.cat([torch.arange(length, device=device)
                                          for length in sequence_lengths]).unsqueeze(0)
                label_mask = torch.tensor(label_mask_values, dtype=torch.bool, device=device)
                token_ids = torch.arange(1, 15, device=device, dtype=torch.long) % 6 + 1
                labels = token_ids.masked_fill(~label_mask, -100).unsqueeze(0)
                actual_logits = torch.randn(1, labels.shape[1], 7, device=device, requires_grad=True)
                expected_logits = actual_logits.detach().clone().requires_grad_(True)

                actual_trainer = _PaddingFreeDPOStub()
                actual_batch = {
                    'labels': labels.clone(),
                    'position_ids': position_ids.clone(),
                    'text_position_ids': position_ids.clone(),
                }
                with patch.dict(os.environ, {'SWIFT_SINGLE_DEVICE_MODE': '1'}):
                    actual_output = DPOTrainer.concatenated_forward(actual_trainer, _LogitsToKeepModel(actual_logits),
                                                                    actual_batch)

                expected_trainer = _PaddingFreeDPOStub()
                expected_batch = {
                    'labels': labels.clone(),
                    'position_ids': position_ids.clone(),
                    'text_position_ids': position_ids.clone(),
                }
                with patch.dict(os.environ, {'SWIFT_SINGLE_DEVICE_MODE': '1'}):
                    SwiftMixin.prepare_logits_to_keep(expected_trainer, expected_batch)
                self.assertTrue(expected_batch['logits_to_keep'].dtype == torch.bool)
                expected_labels = torch.roll(expected_batch['labels'], shifts=-1, dims=1)
                selected_logits = expected_logits[:, expected_batch['logits_to_keep']]
                expected_logps, expected_mean_logits, expected_loss_mask = RLHFTrainerMixin.get_per_token_logps(
                    expected_trainer, selected_logits, expected_labels)
                expected_cu_seqlens = SwiftMixin.get_cu_seqlens(expected_trainer, position_ids,
                                                                expected_batch['logits_to_keep'])
                expected_lengths = expected_cu_seqlens[1:] - expected_cu_seqlens[:-1]
                self.assertEqual(expected_lengths.cpu().tolist(), [3, 2, 4, 1])
                expected_all_logps = _reference_dpo_sum(
                    expected_logps.flatten(), expected_lengths, num_examples=2, ld_alpha=0.5)
                num_tokens = int(expected_cu_seqlens[2].item())
                expected_nll_loss = -expected_logps[:, :num_tokens][expected_loss_mask[:, :num_tokens]].mean()
                expected_chosen_logits = expected_mean_logits[:, :num_tokens][expected_loss_mask[:, :num_tokens]].mean()
                expected_rejected_logits = expected_mean_logits[:, num_tokens:][expected_loss_mask[:,
                                                                                                   num_tokens:]].mean()

                actual_all_logps = torch.cat((actual_output['chosen_logps'], actual_output['rejected_logps']))
                torch.testing.assert_close(actual_all_logps, expected_all_logps, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(actual_output['nll_loss'], expected_nll_loss, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(
                    actual_output['mean_chosen_logits'], expected_chosen_logits, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(
                    actual_output['mean_rejected_logits'], expected_rejected_logits, rtol=1e-5, atol=1e-6)

                actual_objective = actual_all_logps.sum() + actual_output['nll_loss']
                expected_objective = expected_all_logps.sum() + expected_nll_loss
                actual_objective.backward()
                expected_objective.backward()
                torch.testing.assert_close(actual_logits.grad, expected_logits.grad, rtol=1e-5, atol=1e-6)

    def test_get_cu_seqlens(self):
        sequence_lengths = [4, 3, 5, 2]
        keep_masks = [
            [True] * sum(sequence_lengths),
            [True, False, True, False, False, False, False, True, True, False, True, False, False, True],
            [False] * sum(sequence_lengths),
        ]
        trainer = object.__new__(SwiftMixin)

        for device in _test_devices():
            position_ids = torch.cat([torch.arange(length, device=device) for length in sequence_lengths]).unsqueeze(0)
            original = get_packed_seq_params(position_ids)['cu_seq_lens_q']
            for keep_values in keep_masks:
                with self.subTest(device=device, keep_values=keep_values):
                    keep_mask = torch.tensor(keep_values, dtype=torch.bool, device=device)
                    expected = _reference_compact_cu_seqlens(original, keep_mask)
                    actual = trainer.get_cu_seqlens(position_ids, keep_mask)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(actual.dtype, original.dtype)
                    self.assertEqual(actual.device, original.device)

            with self.subTest(device=device, logits_to_keep=None):
                actual = trainer.get_cu_seqlens(position_ids, None)
                torch.testing.assert_close(actual, original, rtol=0, atol=0)

            with self.subTest(device=device, logits_to_keep=11):
                expected = original.clone()
                expected[1:] -= position_ids.shape[-1] + 1 - 11
                actual = trainer.get_cu_seqlens(position_ids, 11)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_pad_logps_back_to_batch(self):
        cases = [
            ('normal', [4, 3, 5, 2], 14, 16),
            ('empty', [3, 0, 2, 5], 10, 8),
            ('all_empty', [0, 0, 0], 0, 4),
            ('truncated', [3, 5, 2], 5, 6),
            ('extra', [3, 2, 1], 10, 5),
            ('small_fast_path', [1, 2], 3, 4),
        ]
        for device in _test_devices():
            for dtype in (torch.float32, torch.bfloat16):
                for name, lengths_list, source_tokens, logits_to_keep in cases:
                    with self.subTest(device=device, dtype=dtype, case=name):
                        lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)
                        logps = torch.arange(source_tokens, dtype=dtype, device=device).reshape(1, -1)
                        expected = _reference_pad_logps(logps, lengths, logits_to_keep)
                        actual = pad_logps_back_to_batch(
                            logps, batch_size=len(lengths_list), seq_lengths=lengths, logits_to_keep=logits_to_keep)
                        torch.testing.assert_close(actual[0].cpu(), expected[0].cpu())
                        torch.testing.assert_close(actual[1].cpu(), expected[1].cpu())
                        self.assertEqual(actual[0].dtype, dtype)
                        self.assertEqual(actual[0].device, logps.device)

                lengths = torch.tensor([4, 3, 5, 2], dtype=torch.int32, device=device)
                logps = torch.arange(14, dtype=torch.bfloat16, device=device).reshape(1, -1)
                expected = _reference_pad_logps(logps, lengths, 16, dtype=torch.float32)
                actual = pad_logps_back_to_batch(
                    logps, batch_size=4, seq_lengths=lengths, logits_to_keep=16, dtype=torch.float32)
                torch.testing.assert_close(actual[0].cpu(), expected[0].cpu())
                torch.testing.assert_close(actual[1].cpu(), expected[1].cpu())

        position_ids = torch.cat([torch.arange(length) for length in [4, 3, 5, 2]]).unsqueeze(0)
        logps = torch.arange(14, dtype=torch.float32).reshape(1, -1)
        lengths = torch.tensor([4, 3, 5, 2], dtype=torch.int32)
        expected = _reference_pad_logps(logps, lengths, 16)
        actual = pad_logps_back_to_batch(logps, batch_size=4, position_ids=position_ids, logits_to_keep=16)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

        lengths = torch.tensor([4, 3, 5, 2], dtype=torch.int32)
        logps = torch.randn(1, 14, requires_grad=True)
        expected_input = logps.detach().clone().requires_grad_(True)
        actual = pad_logps_back_to_batch(logps, batch_size=4, seq_lengths=lengths, logits_to_keep=8, pad_value=0.0)
        expected = _reference_pad_logps(expected_input, lengths, 8, pad_value=0.0)
        (actual[0].square().sum() + actual[1].sum()).backward()
        (expected[0].square().sum() + expected[1].sum()).backward()
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        torch.testing.assert_close(logps.grad, expected_input.grad, rtol=0, atol=0)

    def test_packed_sequence_sum_forward_and_backward(self):
        lengths = [0, 3, 2, 0, 4]
        total_tokens = sum(lengths)

        for device in _test_devices():
            dtypes = [torch.float32, torch.bfloat16]
            if device.type == 'cuda':
                dtypes.append(torch.float16)
            for dtype in dtypes:
                for trailing_shape in [(), (2, )]:
                    with self.subTest(device=device, dtype=dtype, trailing_shape=trailing_shape):
                        torch.manual_seed(42)
                        shape = (total_tokens, *trailing_shape)
                        actual_values = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
                        expected_values = actual_values.detach().clone().requires_grad_(True)
                        device_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)

                        actual = RLHFTrainerMixin._packed_sequence_sum(actual_values, device_lengths)
                        expected = _reference_segment_sum(expected_values, device_lengths)
                        torch.testing.assert_close(actual, expected)
                        self.assertEqual(actual.dtype, dtype)

                        grad = torch.randn_like(actual)
                        actual.backward(grad)
                        expected.backward(grad)
                        torch.testing.assert_close(actual_values.grad, expected_values.grad, rtol=0, atol=0)

    def test_dpo_packed_aggregation(self):
        lengths_list = [3, 5, 0, 5, 2, 4]
        num_examples = len(lengths_list) // 2
        total_tokens = sum(lengths_list)
        cases = [(None, False), (0.0, False), (0.3, False), (0.3, True)]

        for device in _test_devices():
            for dtype in [torch.float32, torch.bfloat16]:
                lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)
                cu_seqlens = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=lengths.dtype)))
                for ld_alpha, is_ref_model in cases:
                    with self.subTest(device=device, dtype=dtype, ld_alpha=ld_alpha, is_ref_model=is_ref_model):
                        torch.manual_seed(7)
                        actual_values = torch.randn(1, total_tokens, dtype=dtype, device=device, requires_grad=True)
                        expected_values = actual_values.detach().clone().requires_grad_(True)
                        reduced_logits = torch.randn_like(actual_values)
                        loss_mask = torch.ones_like(actual_values, dtype=torch.bool)
                        logits = torch.zeros(1, total_tokens, 2, dtype=dtype, device=device)

                        trainer = SimpleNamespace(
                            get_use_logits_to_keep=lambda _: False,
                            aux_loss_enabled=False,
                            is_encoder_decoder=False,
                            template=SimpleNamespace(sequence_parallel_size=1, padding_free=True),
                            label_pad_token_id=-100,
                            loss_type=['sigmoid'],
                            args=SimpleNamespace(ld_alpha=ld_alpha),
                            get_cu_seqlens=lambda *_: cu_seqlens,
                            get_per_token_logps=lambda *_, **__: (actual_values, reduced_logits, loss_mask),
                            _packed_sequence_sum=RLHFTrainerMixin._packed_sequence_sum,
                        )
                        batch = {
                            'labels': torch.ones(1, total_tokens, dtype=torch.long, device=device),
                            'position_ids': torch.arange(total_tokens, device=device).unsqueeze(0),
                        }

                        output = DPOTrainer.concatenated_forward(
                            trainer, _ModelStub(logits), batch, is_ref_model=is_ref_model)
                        actual = torch.cat((output['chosen_logps'], output['rejected_logps']))
                        expected = _reference_dpo_sum(expected_values.flatten(), lengths, num_examples, ld_alpha,
                                                      is_ref_model)
                        torch.testing.assert_close(actual, expected)

                        actual.sum().backward()
                        expected.sum().backward()
                        torch.testing.assert_close(actual_values.grad, expected_values.grad, rtol=0, atol=0)

    def test_kto_packed_aggregation(self):
        lengths_list = [0, 4, 2, 5]
        total_tokens = sum(lengths_list)

        for device in _test_devices():
            for dtype in [torch.float32, torch.bfloat16]:
                with self.subTest(device=device, dtype=dtype):
                    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=device)
                    cu_seqlens = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=lengths.dtype)))
                    torch.manual_seed(11)
                    actual_logps = torch.randn(1, total_tokens, dtype=dtype, device=device, requires_grad=True)
                    actual_logits = torch.randn(1, total_tokens, dtype=dtype, device=device, requires_grad=True)
                    expected_logps = actual_logps.detach().clone().requires_grad_(True)
                    expected_logits = actual_logits.detach().clone().requires_grad_(True)
                    loss_mask = torch.ones_like(actual_logps, dtype=torch.bool)
                    model_logits = torch.zeros(1, total_tokens, 2, dtype=dtype, device=device)

                    trainer = SimpleNamespace(
                        is_encoder_decoder=False,
                        template=SimpleNamespace(sequence_parallel_size=1, padding_free=True),
                        label_pad_token_id=-100,
                        get_cu_seqlens=lambda *_: cu_seqlens,
                        get_per_token_logps=lambda *_, **__: (actual_logps, actual_logits, loss_mask),
                        _packed_sequence_sum=RLHFTrainerMixin._packed_sequence_sum,
                    )
                    inputs = {
                        'text_position_ids': torch.arange(total_tokens, device=device).unsqueeze(0),
                        'position_ids': torch.arange(total_tokens, device=device).unsqueeze(0),
                    }
                    labels = torch.ones(1, total_tokens, dtype=torch.long, device=device)

                    output_logps, output_logits = KTOTrainer.get_batch_logps(trainer, inputs, model_logits, labels)
                    reference_logps = _reference_segment_sum(expected_logps.flatten(), lengths)
                    reference_logits = _reference_segment_sum(expected_logits.flatten(), lengths)
                    torch.testing.assert_close(output_logps, reference_logps)
                    torch.testing.assert_close(output_logits, reference_logits)

                    (output_logps.sum() + 0.25 * output_logits.sum()).backward()
                    (reference_logps.sum() + 0.25 * reference_logits.sum()).backward()
                    torch.testing.assert_close(actual_logps.grad, expected_logps.grad, rtol=0, atol=0)
                    torch.testing.assert_close(actual_logits.grad, expected_logits.grad, rtol=0, atol=0)
