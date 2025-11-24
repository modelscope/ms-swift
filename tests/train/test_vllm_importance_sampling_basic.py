"""
Basic tests for vLLM Importance Sampling implementation

This test file verifies the core functionality of the vLLM IS correction.
"""

import torch


class MockGRPOTrainer:
    """Mock GRPO trainer for testing IS methods"""

    def __init__(self, mode='token_truncate', threshold=2.0):
        self.rollout_importance_sampling_mode = mode
        self.rollout_importance_sampling_threshold = threshold
        self.template = MockTemplate()

    def _apply_rollout_importance_sampling(self, vllm_log_ratio, completion_mask, lengths=None):
        """Copy of the implementation from grpo_trainer.py"""
        mode = self.rollout_importance_sampling_mode
        threshold = self.rollout_importance_sampling_threshold

        is_ratio = torch.exp(vllm_log_ratio)

        if mode == 'token_truncate':
            is_weights = torch.clamp(is_ratio, max=threshold)

        elif mode == 'token_mask':
            is_weights = torch.where(is_ratio <= threshold, is_ratio, torch.zeros_like(is_ratio))

        elif mode == 'sequence_truncate':
            if self.template.padding_free:
                ratio_list = torch.split(is_ratio.squeeze(0), lengths.tolist())
                mask_list = torch.split(completion_mask.squeeze(0), lengths.tolist())

                seq_ratios = []
                for ratio, mask in zip(ratio_list, mask_list):
                    log_ratio = torch.log(ratio.clamp(min=1e-10))
                    seq_ratio = torch.exp((log_ratio * mask).sum() / mask.sum().clamp(min=1.0))
                    seq_ratios.append(seq_ratio)

                seq_ratios = torch.stack(seq_ratios)
                clipped_seq_ratios = torch.clamp(seq_ratios, max=threshold)
                is_weights = torch.repeat_interleave(clipped_seq_ratios, lengths).unsqueeze(0)
            else:
                log_ratio = torch.log(is_ratio.clamp(min=1e-10))
                seq_log_ratios = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
                seq_ratios = torch.exp(seq_log_ratios)
                clipped_seq_ratios = torch.clamp(seq_ratios, max=threshold)
                is_weights = clipped_seq_ratios.unsqueeze(-1).expand_as(is_ratio)

        elif mode == 'sequence_mask':
            if self.template.padding_free:
                ratio_list = torch.split(is_ratio.squeeze(0), lengths.tolist())
                mask_list = torch.split(completion_mask.squeeze(0), lengths.tolist())

                seq_ratios = []
                for ratio, mask in zip(ratio_list, mask_list):
                    log_ratio = torch.log(ratio.clamp(min=1e-10))
                    seq_ratio = torch.exp((log_ratio * mask).sum() / mask.sum().clamp(min=1.0))
                    seq_ratios.append(seq_ratio)

                seq_ratios = torch.stack(seq_ratios)
                seq_mask = (seq_ratios <= threshold).float()
                is_weights = torch.repeat_interleave(seq_mask, lengths).unsqueeze(0)
            else:
                log_ratio = torch.log(is_ratio.clamp(min=1e-10))
                seq_log_ratios = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
                seq_ratios = torch.exp(seq_log_ratios)
                seq_mask = (seq_ratios <= threshold).float()
                is_weights = seq_mask.unsqueeze(-1).expand_as(is_ratio)
        else:
            raise ValueError(f'Unknown mode: {mode}')

        return is_weights


class MockTemplate:

    def __init__(self, padding_free=False):
        self.padding_free = padding_free


class TestVLLMImportanceSampling:
    """Test suite for vLLM Importance Sampling"""

    def test_token_truncate_basic(self):
        """Test token-level truncated IS"""
        trainer = MockGRPOTrainer(mode='token_truncate', threshold=2.0)

        # Create mock data: [batch=2, seq_len=4]
        # Log ratios that will produce ratios [0.5, 1.5, 3.0, 5.0]
        vllm_log_ratio = torch.log(torch.tensor([[0.5, 1.5, 3.0, 5.0], [0.8, 1.2, 2.5, 4.0]]))
        completion_mask = torch.ones_like(vllm_log_ratio, dtype=torch.bool)

        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)

        # Check truncation at threshold=2.0
        assert is_weights.shape == vllm_log_ratio.shape
        assert torch.allclose(is_weights[0, 0], torch.tensor(0.5), atol=1e-5)
        assert torch.allclose(is_weights[0, 1], torch.tensor(1.5), atol=1e-5)
        assert torch.allclose(is_weights[0, 2], torch.tensor(2.0), atol=1e-5)  # Truncated
        assert torch.allclose(is_weights[0, 3], torch.tensor(2.0), atol=1e-5)  # Truncated

    def test_token_mask_basic(self):
        """Test token-level masked IS"""
        trainer = MockGRPOTrainer(mode='token_mask', threshold=2.0)

        vllm_log_ratio = torch.log(torch.tensor([[0.5, 1.5, 3.0, 5.0]]))
        completion_mask = torch.ones_like(vllm_log_ratio, dtype=torch.bool)

        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)

        # Check masking: ratio > threshold should be 0
        assert torch.allclose(is_weights[0, 0], torch.tensor(0.5), atol=1e-5)
        assert torch.allclose(is_weights[0, 1], torch.tensor(1.5), atol=1e-5)
        assert torch.allclose(is_weights[0, 2], torch.tensor(0.0), atol=1e-5)  # Masked
        assert torch.allclose(is_weights[0, 3], torch.tensor(0.0), atol=1e-5)  # Masked

    def test_sequence_truncate_basic(self):
        """Test sequence-level truncated IS"""
        trainer = MockGRPOTrainer(mode='sequence_truncate', threshold=2.0)

        # First sequence has high ratios, second has low ratios
        vllm_log_ratio = torch.log(torch.tensor([
            [3.0, 3.0, 3.0, 3.0],  # avg=3.0 > 2.0
            [1.0, 1.0, 1.0, 1.0]
        ]))  # avg=1.0 < 2.0
        completion_mask = torch.ones_like(vllm_log_ratio, dtype=torch.bool)

        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)

        # First sequence should be truncated to 2.0 for all tokens
        assert torch.allclose(is_weights[0, :], torch.tensor(2.0), atol=1e-5)
        # Second sequence should remain 1.0
        assert torch.allclose(is_weights[1, :], torch.tensor(1.0), atol=1e-5)

    def test_sequence_mask_basic(self):
        """Test sequence-level masked IS"""
        trainer = MockGRPOTrainer(mode='sequence_mask', threshold=2.0)

        vllm_log_ratio = torch.log(torch.tensor([
            [3.0, 3.0, 3.0, 3.0],  # avg=3.0 > 2.0
            [1.0, 1.0, 1.0, 1.0]
        ]))  # avg=1.0 < 2.0
        completion_mask = torch.ones_like(vllm_log_ratio, dtype=torch.bool)

        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)

        # First sequence should be completely masked (0)
        assert torch.allclose(is_weights[0, :], torch.tensor(0.0), atol=1e-5)
        # Second sequence should remain 1.0
        assert torch.allclose(is_weights[1, :], torch.tensor(1.0), atol=1e-5)

    def test_padding_free_mode(self):
        """Test padding-free mode"""
        trainer = MockGRPOTrainer(mode='token_truncate', threshold=2.0)
        trainer.template.padding_free = True

        # Simulate padding-free: [1, total_tokens] = [1, 6] for two sequences of len 4 and 2
        vllm_log_ratio = torch.log(torch.tensor([[0.5, 1.5, 3.0, 5.0, 0.8, 1.2]]))
        completion_mask = torch.ones_like(vllm_log_ratio, dtype=torch.bool)
        lengths = torch.tensor([4, 2])  # Two sequences: len=4 and len=2

        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask, lengths)

        # Should have same shape as input
        assert is_weights.shape == vllm_log_ratio.shape
        # Check truncation: first sequence tokens 2,3 should be truncated to 2.0
        assert torch.allclose(is_weights[0, 2], torch.tensor(2.0), atol=1e-5)
        assert torch.allclose(is_weights[0, 3], torch.tensor(2.0), atol=1e-5)
        # Check second sequence: only one token should be truncated if > threshold
        # 0.8 < 2.0, so should remain 0.8
        assert torch.allclose(is_weights[0, 4], torch.tensor(0.8), atol=1e-5)

    def test_threshold_sensitivity(self):
        """Test different threshold values"""
        vllm_log_ratio = torch.log(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        completion_mask = torch.ones_like(vllm_log_ratio, dtype=torch.bool)

        # Test threshold=1.5
        trainer_low = MockGRPOTrainer(mode='token_truncate', threshold=1.5)
        is_weights_low = trainer_low._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)

        # Test threshold=3.5
        trainer_high = MockGRPOTrainer(mode='token_truncate', threshold=3.5)
        is_weights_high = trainer_high._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)

        # Lower threshold should truncate more
        truncated_low = (is_weights_low < torch.exp(vllm_log_ratio)).sum()
        truncated_high = (is_weights_high < torch.exp(vllm_log_ratio)).sum()
        assert truncated_low > truncated_high

    def test_completion_mask(self):
        """Test that completion mask is respected"""
        trainer = MockGRPOTrainer(mode='sequence_truncate', threshold=2.0)

        vllm_log_ratio = torch.log(torch.tensor([[3.0, 3.0, 3.0, 3.0]]))
        # Mask out last two tokens
        completion_mask = torch.tensor([[True, True, False, False]])

        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)

        # Should only consider masked tokens for sequence ratio calculation
        # With only first two tokens (both 3.0), avg=3.0, truncated to 2.0
        assert torch.allclose(is_weights[0, :2], torch.tensor(2.0), atol=1e-5)

    def test_edge_cases(self):
        """Test edge cases"""
        trainer = MockGRPOTrainer(mode='token_truncate', threshold=2.0)

        # Case 1: All ratios below threshold
        vllm_log_ratio = torch.log(torch.tensor([[0.5, 1.0, 1.5]]))
        completion_mask = torch.ones_like(vllm_log_ratio, dtype=torch.bool)
        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)
        assert torch.allclose(is_weights, torch.exp(vllm_log_ratio), atol=1e-5)

        # Case 2: All ratios above threshold
        vllm_log_ratio = torch.log(torch.tensor([[3.0, 4.0, 5.0]]))
        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)
        assert torch.allclose(is_weights, torch.tensor(2.0), atol=1e-5)

        # Case 3: Empty mask
        vllm_log_ratio = torch.log(torch.tensor([[1.0, 2.0, 3.0]]))
        completion_mask = torch.zeros_like(vllm_log_ratio, dtype=torch.bool)
        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)
        # Should still compute but result may not be meaningful
        assert is_weights.shape == vllm_log_ratio.shape


if __name__ == '__main__':
    # Run tests manually
    import sys

    test_instance = TestVLLMImportanceSampling()

    test_methods = [
        'test_token_truncate_basic', 'test_token_mask_basic', 'test_sequence_truncate_basic',
        'test_sequence_mask_basic', 'test_padding_free_mode', 'test_threshold_sensitivity', 'test_completion_mask',
        'test_edge_cases'
    ]

    failed_tests = []
    for method_name in test_methods:
        try:
            print(f'Running {method_name}...')
            getattr(test_instance, method_name)()
            print(f'✓ {method_name} passed')
        except Exception as e:
            print(f'✗ {method_name} failed: {e}')
            failed_tests.append(method_name)

    if failed_tests:
        print(f'\nFailed tests: {failed_tests}')
        sys.exit(1)
    else:
        print('\nAll tests passed!')
