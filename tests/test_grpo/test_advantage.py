"""Tests for swift.grpo.advantage — advantage computation pure functions."""
import pytest
import torch

from swift.grpo.advantage import (
    RewardMetrics,
    compute_advantages,
    compute_advantages_dynamic,
    compute_reward_metrics,
)

DEVICE = 'cpu'


class TestComputeAdvantages:

    def _make_rewards(self, N=8, n_funcs=3, seed=42):
        torch.manual_seed(seed)
        rpf = torch.randn(N, n_funcs, device=DEVICE)
        rw = torch.ones(n_funcs, device=DEVICE)
        return rpf, rw

    @pytest.mark.parametrize('estimator', ['grpo', 'rloo', 'reinforce_plus_plus'])
    def test_estimator_shapes(self, estimator):
        rpf, rw = self._make_rewards()
        adv, rew = compute_advantages(rpf, rw, num_generations=4, advantage_estimator=estimator)
        assert adv.shape == (8,)
        assert rew.shape == (8,)

    @pytest.mark.parametrize('scale', ['group', 'batch', 'none', 'gdpo'])
    def test_scale_rewards(self, scale):
        rpf, rw = self._make_rewards()
        adv, rew = compute_advantages(rpf, rw, num_generations=4, scale_rewards=scale)
        assert adv.shape == (8,)
        assert torch.isfinite(adv).all()

    def test_grpo_group_advantages_sum_zero_per_group(self):
        rpf, rw = self._make_rewards()
        adv, _ = compute_advantages(rpf, rw, num_generations=4, scale_rewards='none')
        grouped = adv.view(-1, 4)
        group_sums = grouped.sum(dim=1)
        torch.testing.assert_close(group_sums, torch.zeros_like(group_sums), atol=1e-5, rtol=0)

    def test_rloo_leave_one_out(self):
        rpf, rw = self._make_rewards(N=4, n_funcs=1)
        adv, rew = compute_advantages(rpf, rw, num_generations=4, advantage_estimator='rloo', scale_rewards='none')
        K = 4
        mean_all = rew.mean()
        expected_0 = rew[0] * K / (K - 1) - mean_all * K / (K - 1)
        torch.testing.assert_close(adv[0], expected_0, atol=1e-5, rtol=0)

    def test_kl_in_reward(self):
        rpf, rw = self._make_rewards()
        kl = torch.ones(8, device=DEVICE) * 0.5
        adv_no_kl, rew_no_kl = compute_advantages(rpf, rw, num_generations=4, kl_in_reward=False, beta=0.04)
        adv_kl, rew_kl = compute_advantages(rpf, rw, num_generations=4, kl_in_reward=True, beta=0.04, kl_values=kl)
        assert not torch.allclose(rew_no_kl, rew_kl)

    def test_single_generation(self):
        rpf, rw = self._make_rewards(N=4)
        adv, _ = compute_advantages(rpf, rw, num_generations=1, scale_rewards='none')
        torch.testing.assert_close(adv, torch.zeros(4, device=DEVICE), atol=1e-5, rtol=0)


class TestTeacherKL:
    """Tests for OPD teacher KL injection (post-normalization)."""

    def _make_rewards(self, N=8, n_funcs=2, seed=42):
        torch.manual_seed(seed)
        rpf = torch.randn(N, n_funcs, device=DEVICE)
        rw = torch.ones(n_funcs, device=DEVICE)
        return rpf, rw

    def test_teacher_kl_changes_advantages(self):
        rpf, rw = self._make_rewards()
        teacher_kl = torch.randn(8, device=DEVICE) * 0.5
        adv_base, _ = compute_advantages(rpf, rw, num_generations=4)
        adv_kl, _ = compute_advantages(rpf, rw, num_generations=4, teacher_kl=teacher_kl, teacher_kl_coef=1.0)
        assert not torch.allclose(adv_base, adv_kl)

    def test_teacher_kl_is_post_normalization(self):
        """Teacher KL should be subtracted AFTER advantage normalization."""
        rpf, rw = self._make_rewards()
        teacher_kl = torch.ones(8, device=DEVICE) * 0.1
        adv_base, _ = compute_advantages(rpf, rw, num_generations=4, scale_rewards='group')
        adv_kl, _ = compute_advantages(
            rpf, rw, num_generations=4, scale_rewards='group', teacher_kl=teacher_kl, teacher_kl_coef=1.0)
        # adv_kl = adv_base - 1.0 * 0.1
        torch.testing.assert_close(adv_kl, adv_base - 0.1, atol=1e-5, rtol=0)

    def test_opd_only_reward_zeroes_base(self):
        """Pure distillation: base advantages should be zero, only KL drives."""
        rpf, rw = self._make_rewards()
        teacher_kl = torch.randn(8, device=DEVICE).abs()
        adv, _ = compute_advantages(
            rpf, rw, num_generations=4, teacher_kl=teacher_kl, teacher_kl_coef=0.5, opd_only_reward=True)
        expected = -0.5 * teacher_kl
        torch.testing.assert_close(adv, expected, atol=1e-5, rtol=0)

    def test_opd_only_reward_without_teacher_kl(self):
        """opd_only_reward without teacher_kl should return normal advantages (no-op)."""
        rpf, rw = self._make_rewards()
        adv_normal, _ = compute_advantages(rpf, rw, num_generations=4)
        adv_opd, _ = compute_advantages(rpf, rw, num_generations=4, opd_only_reward=True)
        torch.testing.assert_close(adv_normal, adv_opd)

    def test_teacher_kl_zero_coef_is_noop(self):
        rpf, rw = self._make_rewards()
        teacher_kl = torch.randn(8, device=DEVICE)
        adv_base, _ = compute_advantages(rpf, rw, num_generations=4)
        adv_zero, _ = compute_advantages(rpf, rw, num_generations=4, teacher_kl=teacher_kl, teacher_kl_coef=0.0)
        torch.testing.assert_close(adv_base, adv_zero)

    def test_ref_kl_and_teacher_kl_orthogonal(self):
        """Ref KL (pre-norm) and teacher KL (post-norm) should be independent."""
        rpf, rw = self._make_rewards()
        ref_kl = torch.ones(8, device=DEVICE) * 0.3
        teacher_kl = torch.ones(8, device=DEVICE) * 0.1

        adv_both, _ = compute_advantages(
            rpf, rw, num_generations=4,
            kl_in_reward=True, beta=0.04, kl_values=ref_kl,
            teacher_kl=teacher_kl, teacher_kl_coef=1.0)

        adv_ref_only, _ = compute_advantages(
            rpf, rw, num_generations=4,
            kl_in_reward=True, beta=0.04, kl_values=ref_kl)

        # Teacher KL is additive post-normalization
        torch.testing.assert_close(adv_both, adv_ref_only - 0.1, atol=1e-5, rtol=0)

    @pytest.mark.parametrize('estimator', ['grpo', 'rloo', 'reinforce_plus_plus'])
    def test_teacher_kl_works_with_all_estimators(self, estimator):
        rpf, rw = self._make_rewards()
        teacher_kl = torch.randn(8, device=DEVICE).abs()
        adv, _ = compute_advantages(
            rpf, rw, num_generations=4, advantage_estimator=estimator,
            teacher_kl=teacher_kl, teacher_kl_coef=0.5)
        assert adv.shape == (8,)
        assert torch.isfinite(adv).all()


class TestTeacherKLDynamic:
    """Teacher KL injection in request-aware mode."""

    def test_teacher_kl_dynamic(self):
        torch.manual_seed(42)
        rpf = torch.randn(6, 2, device=DEVICE)
        rw = torch.ones(2, device=DEVICE)
        prompt_ids = ['p1', 'p1', 'p1', 'p2', 'p2', 'p2']
        request_ids = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']
        teacher_kl = torch.ones(6, device=DEVICE) * 0.2
        adv_base, _ = compute_advantages_dynamic(rpf, rw, prompt_ids, request_ids)
        adv_kl, _ = compute_advantages_dynamic(
            rpf, rw, prompt_ids, request_ids, teacher_kl=teacher_kl, teacher_kl_coef=1.0)
        torch.testing.assert_close(adv_kl, adv_base - 0.2, atol=1e-5, rtol=0)

    def test_opd_only_dynamic(self):
        torch.manual_seed(42)
        rpf = torch.randn(4, 1, device=DEVICE)
        rw = torch.ones(1, device=DEVICE)
        prompt_ids = ['p1', 'p1', 'p2', 'p2']
        request_ids = ['r1', 'r2', 'r3', 'r4']
        teacher_kl = torch.tensor([0.1, 0.2, 0.3, 0.4], device=DEVICE)
        adv, _ = compute_advantages_dynamic(
            rpf, rw, prompt_ids, request_ids,
            teacher_kl=teacher_kl, teacher_kl_coef=2.0, opd_only_reward=True)
        expected = -2.0 * teacher_kl
        torch.testing.assert_close(adv, expected, atol=1e-5, rtol=0)


class TestComputeAdvantagesDynamic:

    def test_basic(self):
        torch.manual_seed(42)
        rpf = torch.randn(6, 2, device=DEVICE)
        rw = torch.ones(2, device=DEVICE)
        prompt_ids = ['p1', 'p1', 'p1', 'p2', 'p2', 'p2']
        request_ids = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6']
        adv, rew = compute_advantages_dynamic(rpf, rw, prompt_ids, request_ids)
        assert adv.shape == (6,)

    def test_duplicate_request_ids(self):
        torch.manual_seed(42)
        rpf = torch.randn(4, 1, device=DEVICE)
        rw = torch.ones(1, device=DEVICE)
        prompt_ids = ['p1', 'p1', 'p1', 'p1']
        request_ids = ['r1', 'r1', 'r2', 'r2']
        adv, rew = compute_advantages_dynamic(rpf, rw, prompt_ids, request_ids)
        assert adv.shape == (4,)
        assert adv[0] == adv[1]
        assert adv[2] == adv[3]


class TestComputeRewardMetrics:

    def test_basic(self):
        torch.manual_seed(42)
        N, K, n_funcs = 8, 4, 3
        rpf = torch.randn(N, n_funcs, device=DEVICE)
        rw = torch.ones(n_funcs, device=DEVICE)
        rewards = (rpf * rw.unsqueeze(0)).sum(dim=1)
        rm = compute_reward_metrics(rewards, rpf, ['r1', 'r2', 'r3'], K, 'group')
        assert isinstance(rm, RewardMetrics)
        assert isinstance(rm.reward_mean, float)
        assert isinstance(rm.reward_std, float)
        assert len(rm.per_func_mean) == 3
        assert len(rm.per_func_std) == 3

    def test_single_generation(self):
        rpf = torch.randn(4, 2, device=DEVICE)
        rw = torch.ones(2, device=DEVICE)
        rewards = (rpf * rw.unsqueeze(0)).sum(dim=1)
        rm = compute_reward_metrics(rewards, rpf, ['a', 'b'], 1, 'group')
        assert rm.reward_std == 0.0
