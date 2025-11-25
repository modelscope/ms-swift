"""
Basic tests for vLLM Importance Sampling implementation

This test file verifies the core functionality of the vLLM IS correction,
including the IS weight computation and metrics calculation.

Reference: verl/verl/trainer/ppo/rollout_corr_helper.py
"""

import torch


class MockAccelerator:
    """Mock accelerator for testing metrics gathering"""

    def __init__(self, device='cpu'):
        self.device = device

    def gather_for_metrics(self, tensor):
        # In testing, just return the tensor as-is
        return tensor


class MockGRPOTrainer:
    """Mock GRPO trainer for testing IS methods"""

    def __init__(self, mode='token_truncate', threshold=2.0):
        self.rollout_importance_sampling_mode = mode
        self.rollout_importance_sampling_threshold = threshold
        self.accelerator = MockAccelerator()

    def _compute_sequence_level_ratios(self, is_ratio: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
        """
        Helper function to compute sequence-level importance sampling ratios.

        Args:
            is_ratio: Token-level IS ratios, shape [B, T]
            completion_mask: Boolean mask for completion tokens, shape [B, T]

        Returns:
            Sequence-level ratios as geometric mean of token-level ratios
        """
        log_ratio = torch.log(is_ratio.clamp(min=1e-10))
        seq_log_ratios = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        seq_ratios = torch.exp(seq_log_ratios)

        return seq_ratios

    def _apply_rollout_importance_sampling(self, rollout_log_ratio: torch.Tensor,
                                           completion_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply vLLM importance sampling correction using one of four modes.

        Args:
            rollout_log_ratio: log(π_θ / π_rollout) per token, shape [B, T]
            completion_mask: Boolean mask for completion tokens, shape [B, T]

        Returns:
            IS weights to multiply with loss, same shape as rollout_log_ratio
        """
        mode = self.rollout_importance_sampling_mode
        threshold = self.rollout_importance_sampling_threshold

        # Clamp log_ratio to prevent numerical overflow from padding values (-1e10)
        # A log_ratio of 20 corresponds to exp(20) ≈ 485 million, which is already extreme
        SAFETY_BOUND = 20.0
        rollout_log_ratio_safe = torch.clamp(rollout_log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)

        # Compute importance sampling ratios: exp(log_ratio)
        is_ratio = torch.exp(rollout_log_ratio_safe)

        if mode == 'token_truncate':
            # Token-level truncated IS: clip ratios from above at threshold
            is_weights = torch.clamp(is_ratio, max=threshold)

        elif mode == 'token_mask':
            # Token-level masked IS: mask out tokens with ratio > threshold
            is_weights = torch.where(is_ratio <= threshold, is_ratio, torch.zeros_like(is_ratio))

        elif mode == 'sequence_truncate':
            # Sequence-level truncated IS: compute sequence-level ratio and clip
            seq_ratios = self._compute_sequence_level_ratios(is_ratio, completion_mask)
            clipped_seq_ratios = torch.clamp(seq_ratios, max=threshold)

            is_weights = clipped_seq_ratios.unsqueeze(-1).expand_as(is_ratio)

        elif mode == 'sequence_mask':
            # Sequence-level masked IS: mask entire sequences with ratio > threshold
            seq_ratios = self._compute_sequence_level_ratios(is_ratio, completion_mask)
            seq_mask = (seq_ratios <= threshold).float()

            # Apply mask to original token-level ratios
            is_weights = is_ratio * seq_mask.unsqueeze(-1)
        else:
            return is_ratio

        return is_weights

    def _compute_is_correction_metrics(
        self,
        vllm_log_ratio: torch.Tensor,
        is_weights: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> dict:
        """
        Compute importance sampling correction metrics (ess, clipped_frac, is_weight_mean).
        Only called when rollout_importance_sampling_mode is enabled.

        Args:
            vllm_log_ratio: Log ratio log(π_policy / π_rollout), shape [B, T]
            is_weights: Importance sampling weights after correction, shape [B, T]
            completion_mask: Boolean mask for completion tokens, shape [B, T]

        Returns:
            Dictionary with IS-specific metrics:
                - is_weight_mean: Mean of IS weights
                - ess: Effective Sample Size = 1 / E[(w_i / E[w_i])²]
                - clipped_frac: Fraction of clipped/masked samples
        """
        metrics = {}
        SAFETY_BOUND = 20.0
        threshold = self.rollout_importance_sampling_threshold
        threshold_lower = 1.0 / threshold  # Default lower threshold (reciprocal of upper)

        # Helper function for masked mean
        def masked_mean(x, mask):
            return (x * mask).sum() / mask.sum().clamp(min=1.0)

        # Compute IS ratio with safety bounds
        log_ratio_safe = torch.clamp(vllm_log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        is_ratio = torch.exp(log_ratio_safe)

        # 1. IS weight statistics
        mean_is_weight = masked_mean(is_weights, completion_mask)
        metrics['is_weight_mean'] = self.accelerator.gather_for_metrics(mean_is_weight).nanmean().item()

        # 2. Compute Effective Sample Size (ESS) for IS weights
        # ESS = 1 / E[(w_i / E[w_i])²] (using clamped weights for stability)
        # This measures how many "effective" independent samples we have after IS weighting
        weights_for_ess = is_weights.clamp(min=threshold_lower, max=threshold)
        mean_for_ess = masked_mean(weights_for_ess, completion_mask)
        is_weights_normalized = weights_for_ess / (mean_for_ess + 1e-8)  # Avoid division by zero
        ess = 1.0 / masked_mean(is_weights_normalized.square(), completion_mask).clamp(min=1e-10)
        metrics['ess'] = self.accelerator.gather_for_metrics(ess).nanmean().item()

        # 3. Fraction of clipped/masked samples
        if self.rollout_importance_sampling_mode in ['token_truncate', 'token_mask']:
            # Token-level
            if self.rollout_importance_sampling_mode == 'token_truncate':
                clipped_frac = masked_mean((is_ratio > threshold).float(), completion_mask)
            else:  # token_mask
                clipped_frac = masked_mean((is_weights == 0).float(), completion_mask)
            metrics['clipped_frac'] = self.accelerator.gather_for_metrics(clipped_frac).nanmean().item()
        else:
            # Sequence-level (both truncate and mask)
            seq_ratios = self._compute_sequence_level_ratios(is_ratio, completion_mask)
            clipped_frac = (seq_ratios > threshold).float().mean()
            metrics['clipped_frac'] = self.accelerator.gather_for_metrics(clipped_frac).nanmean().item()

        return metrics


class TestVLLMImportanceSampling:
    """Test suite for vLLM Importance Sampling"""

    def test_token_truncate_basic(self):
        """Test token-level truncated IS"""
        trainer = MockGRPOTrainer(mode='token_truncate', threshold=2.0)

        # Create mock data: [batch=2, seq_len=4]
        # Log ratios that will produce ratios [0.5, 1.5, 3.0, 5.0]
        vllm_log_ratio = torch.log(torch.tensor([[0.5, 1.5, 3.0, 5.0], [0.8, 1.2, 2.5, 4.0]]))
        completion_mask = torch.ones_like(vllm_log_ratio)

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
        completion_mask = torch.ones_like(vllm_log_ratio)

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
        vllm_log_ratio = torch.log(
            torch.tensor([
                [3.0, 3.0, 3.0, 3.0],  # geometric mean=3.0 > 2.0
                [1.0, 1.0, 1.0, 1.0]
            ]))  # geometric mean=1.0 < 2.0
        completion_mask = torch.ones_like(vllm_log_ratio)

        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)

        # First sequence should be truncated to 2.0 for all tokens
        assert torch.allclose(is_weights[0, :], torch.tensor(2.0), atol=1e-5)
        # Second sequence should remain 1.0
        assert torch.allclose(is_weights[1, :], torch.tensor(1.0), atol=1e-5)

    def test_sequence_mask_basic(self):
        """Test sequence-level masked IS"""
        trainer = MockGRPOTrainer(mode='sequence_mask', threshold=2.0)

        vllm_log_ratio = torch.log(
            torch.tensor([
                [3.0, 3.0, 3.0, 3.0],  # geometric mean=3.0 > 2.0
                [1.0, 1.0, 1.0, 1.0]
            ]))  # geometric mean=1.0 < 2.0
        completion_mask = torch.ones_like(vllm_log_ratio)

        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)

        # First sequence should be completely masked (0)
        # Note: sequence_mask multiplies is_ratio by 0, so all tokens become 0
        assert torch.allclose(is_weights[0, :], torch.tensor(0.0), atol=1e-5)
        # Second sequence should keep original ratios (1.0 * 1.0 = 1.0)
        assert torch.allclose(is_weights[1, :], torch.tensor(1.0), atol=1e-5)

    def test_threshold_sensitivity(self):
        """Test different threshold values"""
        vllm_log_ratio = torch.log(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        completion_mask = torch.ones_like(vllm_log_ratio)

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
        completion_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)

        # Should only consider masked tokens for sequence ratio calculation
        # With only first two tokens (both 3.0), geometric mean=3.0, truncated to 2.0
        assert torch.allclose(is_weights[0, :2], torch.tensor(2.0), atol=1e-5)

    def test_edge_cases(self):
        """Test edge cases"""
        trainer = MockGRPOTrainer(mode='token_truncate', threshold=2.0)

        # Case 1: All ratios below threshold
        vllm_log_ratio = torch.log(torch.tensor([[0.5, 1.0, 1.5]]))
        completion_mask = torch.ones_like(vllm_log_ratio)
        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)
        assert torch.allclose(is_weights, torch.exp(vllm_log_ratio), atol=1e-5)

        # Case 2: All ratios above threshold
        vllm_log_ratio = torch.log(torch.tensor([[3.0, 4.0, 5.0]]))
        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask[:, :3])
        assert torch.allclose(is_weights, torch.tensor(2.0), atol=1e-5)

        # Case 3: Empty mask
        vllm_log_ratio = torch.log(torch.tensor([[1.0, 2.0, 3.0]]))
        completion_mask = torch.zeros_like(vllm_log_ratio)
        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)
        # Should still compute but result may not be meaningful
        assert is_weights.shape == vllm_log_ratio.shape

    def test_safety_bound(self):
        """Test that extreme log ratios are clamped"""
        trainer = MockGRPOTrainer(mode='token_truncate', threshold=2.0)

        # Create extreme log ratios that would overflow without clamping
        vllm_log_ratio = torch.tensor([[100.0, -100.0, 0.0]])  # exp(100) would overflow
        completion_mask = torch.ones_like(vllm_log_ratio)

        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)

        # Should not have inf or nan
        assert torch.isfinite(is_weights).all()
        # Large positive log_ratio should be clamped to threshold
        assert is_weights[0, 0] <= 2.0
        # Large negative log_ratio should result in small positive value
        assert is_weights[0, 1] > 0


class TestISCorrectionMetrics:
    """Test suite for IS correction metrics"""

    def test_ess_uniform_weights(self):
        """Test ESS with uniform weights (should be close to 1.0)"""
        trainer = MockGRPOTrainer(mode='token_truncate', threshold=2.0)

        # Uniform weights of 1.0
        vllm_log_ratio = torch.zeros((2, 4))  # exp(0) = 1.0
        completion_mask = torch.ones_like(vllm_log_ratio)
        is_weights = torch.ones_like(vllm_log_ratio)

        metrics = trainer._compute_is_correction_metrics(vllm_log_ratio, is_weights, completion_mask)

        # ESS should be 1.0 for uniform weights
        assert abs(metrics['ess'] - 1.0) < 0.01
        # Mean weight should be 1.0
        assert abs(metrics['is_weight_mean'] - 1.0) < 0.01
        # No clipping for uniform weights
        assert metrics['clipped_frac'] == 0.0

    def test_ess_varied_weights(self):
        """Test ESS with varied weights (should be < 1.0)"""
        trainer = MockGRPOTrainer(mode='token_truncate', threshold=2.0)

        # Varied weights
        vllm_log_ratio = torch.log(torch.tensor([[0.5, 1.0, 1.5, 2.0]]))
        completion_mask = torch.ones_like(vllm_log_ratio)
        is_weights = torch.tensor([[0.5, 1.0, 1.5, 2.0]])

        metrics = trainer._compute_is_correction_metrics(vllm_log_ratio, is_weights, completion_mask)

        # ESS should be less than 1.0 for non-uniform weights
        assert metrics['ess'] < 1.0
        assert metrics['ess'] > 0.0

    def test_clipped_frac_token_truncate(self):
        """Test clipped_frac for token_truncate mode"""
        trainer = MockGRPOTrainer(mode='token_truncate', threshold=2.0)

        # 2 out of 4 tokens exceed threshold
        vllm_log_ratio = torch.log(torch.tensor([[0.5, 1.5, 3.0, 5.0]]))
        completion_mask = torch.ones_like(vllm_log_ratio)
        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)

        metrics = trainer._compute_is_correction_metrics(vllm_log_ratio, is_weights, completion_mask)

        # 2/4 = 0.5 tokens clipped
        assert abs(metrics['clipped_frac'] - 0.5) < 0.01

    def test_clipped_frac_token_mask(self):
        """Test clipped_frac for token_mask mode"""
        trainer = MockGRPOTrainer(mode='token_mask', threshold=2.0)

        # 2 out of 4 tokens exceed threshold
        vllm_log_ratio = torch.log(torch.tensor([[0.5, 1.5, 3.0, 5.0]]))
        completion_mask = torch.ones_like(vllm_log_ratio)
        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)

        metrics = trainer._compute_is_correction_metrics(vllm_log_ratio, is_weights, completion_mask)

        # 2/4 = 0.5 tokens masked (is_weights == 0)
        assert abs(metrics['clipped_frac'] - 0.5) < 0.01

    def test_clipped_frac_sequence_level(self):
        """Test clipped_frac for sequence-level modes"""
        trainer = MockGRPOTrainer(mode='sequence_truncate', threshold=2.0)

        # First sequence exceeds threshold, second doesn't
        vllm_log_ratio = torch.log(torch.tensor([[3.0, 3.0, 3.0, 3.0], [1.0, 1.0, 1.0, 1.0]]))
        completion_mask = torch.ones_like(vllm_log_ratio)
        is_weights = trainer._apply_rollout_importance_sampling(vllm_log_ratio, completion_mask)

        metrics = trainer._compute_is_correction_metrics(vllm_log_ratio, is_weights, completion_mask)

        # 1/2 = 0.5 sequences clipped
        assert abs(metrics['clipped_frac'] - 0.5) < 0.01


class TestOffpolicyMetrics:
    """Test suite for off-policy diagnostic metrics"""

    def test_kl_divergence_same_policy(self):
        """Test KL divergence when policies are identical"""
        # When per_token_logps == rollout_per_token_logps, KL should be 0
        per_token_logps = torch.tensor([[-1.0, -2.0, -1.5, -0.5]])
        rollout_per_token_logps = per_token_logps.clone()
        completion_mask = torch.ones_like(per_token_logps)

        # Helper function for masked mean
        def masked_mean(x, mask, axis=None):
            if axis is None:
                return (x * mask).sum() / mask.sum().clamp(min=1.0)
            else:
                return (x * mask).sum(axis) / mask.sum(axis).clamp(min=1.0)

        # KL = E[log(π_rollout) - log(π_training)]
        kl = masked_mean(rollout_per_token_logps - per_token_logps, completion_mask)

        assert abs(kl.item()) < 1e-6

    def test_k3_kl_estimator(self):
        """Test K3 KL estimator"""
        per_token_logps = torch.tensor([[-1.0, -2.0, -1.5, -0.5]])
        rollout_per_token_logps = torch.tensor([[-1.1, -1.9, -1.6, -0.4]])
        completion_mask = torch.ones_like(per_token_logps)

        def masked_mean(x, mask, axis=None):
            if axis is None:
                return (x * mask).sum() / mask.sum().clamp(min=1.0)
            else:
                return (x * mask).sum(axis) / mask.sum(axis).clamp(min=1.0)

        # K3 estimator: E[exp(log_ratio) - log_ratio - 1]
        log_ratio = per_token_logps - rollout_per_token_logps
        log_ratio *= completion_mask
        k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
        k3_kl = masked_mean(k3_kl_matrix, completion_mask)

        # K3 KL should be non-negative
        assert k3_kl.item() >= 0

    def test_chi2_divergence(self):
        """Test χ² divergence calculation"""
        per_token_logps = torch.tensor([[-1.0, -2.0]])
        rollout_per_token_logps = torch.tensor([[-1.5, -1.5]])
        completion_mask = torch.ones_like(per_token_logps)

        def masked_mean(x, mask, axis=None):
            if axis is None:
                return (x * mask).sum() / mask.sum().clamp(min=1.0)
            else:
                return (x * mask).sum(axis) / mask.sum(axis).clamp(min=1.0)

        SAFETY_BOUND = 20.0
        log_ratio = per_token_logps - rollout_per_token_logps
        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_token = torch.exp(log_ratio_safe)
        rho_squared_token = rho_token.square()
        chi2_token = masked_mean(rho_squared_token, completion_mask) - 1.0

        # χ² should be >= -1 (can be negative if E[ρ²] < 1)
        assert chi2_token.item() >= -1.0


if __name__ == '__main__':
    # Run tests manually
    import sys

    test_classes = [
        ('TestVLLMImportanceSampling', TestVLLMImportanceSampling),
        ('TestISCorrectionMetrics', TestISCorrectionMetrics),
        ('TestOffpolicyMetrics', TestOffpolicyMetrics),
    ]

    failed_tests = []

    for class_name, test_class in test_classes:
        print(f'\n=== {class_name} ===')
        test_instance = test_class()

        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]

        for method_name in test_methods:
            try:
                print(f'Running {method_name}...')
                getattr(test_instance, method_name)()
                print(f'✓ {method_name} passed')
            except Exception as e:
                print(f'✗ {method_name} failed: {e}')
                failed_tests.append(f'{class_name}.{method_name}')

    if failed_tests:
        print(f'\nFailed tests: {failed_tests}')
        sys.exit(1)
    else:
        print('\nAll tests passed!')
