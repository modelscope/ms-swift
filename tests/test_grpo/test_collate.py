"""Tests for swift.rlhf_trainers.utils.collate_to_micro_batch and helpers.

Uses a minimal fake template so the shared collate can be exercised without
loading a real model/tokenizer.
"""
import torch

from swift.rl_core.data import OnPolicySample
from swift.rlhf_trainers.utils import build_completion_mask_and_seq_lengths, build_rollout_logps, collate_to_micro_batch

DEVICE = 'cpu'


class FakeTemplate:
    """Right-pads a list of per-sample encoded dicts into a batch (non-padding-free)."""

    padding_free = False
    padding_side = 'right'

    def data_collator(self, encoded_list, padding_to=None):
        max_len = max(len(e['input_ids']) for e in encoded_list)
        if padding_to:
            max_len = max(max_len, padding_to)
        input_ids, labels, attn = [], [], []
        for e in encoded_list:
            ids, lbl = list(e['input_ids']), list(e['labels'])
            pad = max_len - len(ids)
            input_ids.append(ids + [0] * pad)
            labels.append(lbl + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attn, dtype=torch.long),
        }


def _sample(input_ids, labels, **kw):
    return OnPolicySample(
        messages=[{
            'role': 'user',
            'content': 'x'
        }],
        prompt_id=kw.pop('prompt_id', 'p'),
        request_id=kw.pop('request_id', 'r'),
        encoded={
            'input_ids': input_ids,
            'labels': labels,
            'length': len(input_ids)
        },
        **kw,
    )


class TestCollateToMicroBatch:

    def test_basic_shapes_and_separation(self):
        samples = [
            _sample([1, 2, 3, 4], [-100, -100, 3, 4], finish_reason='stop'),
            _sample([5, 6, 7], [-100, 6, 7], finish_reason='length'),
        ]
        model_inputs, grpo_batch = collate_to_micro_batch(samples, FakeTemplate(), device=DEVICE)
        # model_inputs is a clean whitelist
        assert set(model_inputs) == {'input_ids', 'labels', 'attention_mask'}
        assert 'completion_mask' not in model_inputs
        assert 'advantages' not in model_inputs
        # grpo_batch carries RL signals
        B, T = 2, 4
        assert grpo_batch.completion_mask.shape == (B, T)
        assert grpo_batch.truncated_mask.tolist() == [False, True]
        # backend-specific signals left for caller
        assert grpo_batch.old_per_token_logps is None
        assert grpo_batch.advantages is None

    def test_seq_lengths_from_attention_mask(self):
        samples = [_sample([1, 2, 3, 4], [-100, -100, 3, 4]), _sample([5, 6, 7], [-100, 6, 7])]
        _, grpo_batch = collate_to_micro_batch(samples, FakeTemplate(), device=DEVICE)
        assert grpo_batch.seq_lengths.tolist() == [4, 3]

    def test_rollout_logps_none_when_absent(self):
        samples = [_sample([1, 2, 3], [-100, 2, 3]), _sample([4, 5, 6], [-100, 5, 6])]
        _, grpo_batch = collate_to_micro_batch(samples, FakeTemplate(), device=DEVICE)
        assert grpo_batch.rollout_per_token_logps is None

    def test_rollout_logps_aligned_when_present(self):
        # completion tokens (label != -100, after roll) -> 2 per sample
        samples = [
            _sample([1, 2, 3], [-100, 2, 3], rollout_logprobs=[[-0.5, -0.6]]),
            _sample([4, 5, 6], [-100, 5, 6], rollout_logprobs=[[-0.7, -0.8]]),
        ]
        _, grpo_batch = collate_to_micro_batch(samples, FakeTemplate(), device=DEVICE)
        rlp = grpo_batch.rollout_per_token_logps
        assert rlp is not None and rlp.shape == (2, 3)
        # nonzero entries match the completion positions count
        assert int((rlp != 0).sum().item()) == grpo_batch.completion_mask.sum().item()


class TestBuildRolloutLogps:

    def test_list_interface_and_count_mismatch(self):
        mask = torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.bool)
        # matching counts
        out = build_rollout_logps([[[-0.1, -0.2]], [[-0.3, -0.4]]], mask, DEVICE)
        assert out is not None and out.shape == (2, 3)
        # one extra logprob (N+1) is trimmed
        out2 = build_rollout_logps([[[-0.1, -0.2, -0.9]], [[-0.3, -0.4]]], mask, DEVICE)
        assert out2 is not None
        # missing -> None
        assert build_rollout_logps([None, [[-0.3, -0.4]]], mask, DEVICE) is None


def _old_hf_region(labels):
    """Reference: the pre-refactor HF non-padding-free region-frame algorithm."""
    ltk = int((labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item())
    return labels[:, -ltk:] != -100, ltk


class TestFrameParam:
    """build_completion_mask_and_seq_lengths frame-width parameter (non-padding-free)."""

    def test_full_frame_megatron(self):
        # logits_to_keep=None -> full-sequence frame + roll
        labels = torch.tensor([[-100, -100, 3, 4], [-100, 5, 6, 7]])
        cm, seq_lengths, max_seq_len = build_completion_mask_and_seq_lengths(
            labels, batch_size=2, padding_free=False, encoded_batch={}, device=DEVICE, logits_to_keep=None)
        assert max_seq_len == labels.shape[-1]
        assert cm.shape == labels.shape
        assert torch.equal(cm, torch.roll(labels, -1, -1) != -100)

    def test_region_frame_matches_old_hf(self):
        # logits_to_keep set -> completion-region frame, no roll; must match old HF bit-level
        labels = torch.tensor([[-100, -100, 3, 4], [-100, 5, 6, 7]])
        ref_cm, ltk = _old_hf_region(labels)
        cm, seq_lengths, max_seq_len = build_completion_mask_and_seq_lengths(
            labels, batch_size=2, padding_free=False, encoded_batch={}, device=DEVICE, logits_to_keep=ltk)
        assert max_seq_len == ltk
        assert cm.shape == (labels.shape[0], ltk)
        assert torch.equal(cm, ref_cm)

    def test_collate_region_frame_end_to_end(self):
        samples = [
            _sample([1, 2, 3, 4], [-100, -100, 3, 4], finish_reason='stop'),
            _sample([5, 6, 7, 8], [-100, 6, 7, 8], finish_reason='length'),
        ]
        _, grpo_batch = collate_to_micro_batch(samples, FakeTemplate(), device=DEVICE, use_logits_to_keep=True)
        labels = torch.tensor([[-100, -100, 3, 4], [-100, 6, 7, 8]])
        ref_cm, ltk = _old_hf_region(labels)
        assert grpo_batch.logits_to_keep == ltk
        assert grpo_batch.completion_mask.shape == (2, ltk)
        assert torch.equal(grpo_batch.completion_mask, ref_cm)
