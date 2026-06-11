"""Tests for swift.grpo.data — GRPO batch data structures."""
import pytest
import torch

from swift.grpo.data import GRPO_RL_KEYS, GRPORLBatch

B, T = 4, 8
DEVICE = 'cpu'


def _make_rl_batch(**overrides):
    defaults = {
        'completion_mask': torch.ones(B, T, dtype=torch.bool, device=DEVICE),
        'truncated_mask': torch.zeros(B, dtype=torch.bool, device=DEVICE),
        'seq_lengths': torch.full((B,), T, dtype=torch.long, device=DEVICE),
    }
    defaults.update(overrides)
    return GRPORLBatch(**defaults)


class TestGRPORLBatch:

    def test_required_fields(self):
        rl = _make_rl_batch()
        assert rl.completion_mask.shape == (B, T)
        assert rl.truncated_mask.shape == (B,)
        assert rl.old_per_token_logps is None
        assert rl.advantages is None

    def test_optional_fields(self):
        rl = _make_rl_batch(
            old_per_token_logps=torch.randn(B, T),
            ref_per_token_logps=torch.randn(B, T),
            advantages=torch.randn(B),
            num_items_in_batch=torch.tensor(24.0),
            logits_to_keep=6,
        )
        assert rl.old_per_token_logps.shape == (B, T)
        assert rl.advantages.shape == (B,)
        assert rl.logits_to_keep == 6

    def test_to_dict(self):
        rl = _make_rl_batch(advantages=torch.randn(B))
        d = rl.to_dict()
        assert 'completion_mask' in d
        assert 'advantages' in d
        assert 'old_per_token_logps' not in d

    def test_from_dict(self):
        mixed = {
            'input_ids': torch.randint(0, 100, (B, T)),
            'attention_mask': torch.ones(B, T),
            'completion_mask': torch.ones(B, T, dtype=torch.bool),
            'truncated_mask': torch.zeros(B, dtype=torch.bool),
            'seq_lengths': torch.full((B,), T),
            'old_per_token_logps': torch.randn(B, T),
            'advantages': torch.randn(B),
        }
        rl, model_inputs = GRPORLBatch.from_dict(mixed)
        assert isinstance(rl, GRPORLBatch)
        assert rl.completion_mask.shape == (B, T)
        assert rl.advantages.shape == (B,)
        assert 'input_ids' in model_inputs
        assert 'attention_mask' in model_inputs
        assert 'completion_mask' not in model_inputs
        assert 'advantages' not in model_inputs

    def test_grpo_rl_keys_complete(self):
        expected = {
            'completion_mask', 'truncated_mask', 'seq_lengths',
            'old_per_token_logps', 'ref_per_token_logps',
            'rollout_per_token_logps', 'advantages',
            'num_items_in_batch', 'logits_to_keep',
        }
        assert GRPO_RL_KEYS == expected

    def test_mutable_after_creation(self):
        rl = _make_rl_batch()
        assert rl.advantages is None
        rl.advantages = torch.randn(B)
        assert rl.advantages.shape == (B,)
        rl.num_items_in_batch = torch.tensor(32.0)
        assert rl.num_items_in_batch.item() == 32.0
