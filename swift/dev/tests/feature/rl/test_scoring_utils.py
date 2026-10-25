"""S8a — device-group planning + the reward-weighting/scoring pure functions (L1).

``plan_device_groups`` is the placement truth source (colocate overlays the sampler on the trainer's
GPUs; disaggregated gives the sampler its own disjoint ranks; auxiliary groups never overlap), and
the scoring half (``weighted_reward`` / ``build_reward_weights`` / ``weighted_rewards`` /
``accumulate_reward_metric``) is the [N, n_scorers] -> per-sample-scalar collapse every RL family
shares. All hand-computed; no Ray, no GPU.

Maps to ut_plan S8-01/02/03 (device groups) and S8-21/22/23/24 (scoring utils).
"""
import torch
import pytest


class TestPlanDeviceGroups:

    def test_colocate_overlays_the_sampler(self):
        """S8-01: colocate reuses the trainer's GPUs -- the sampler group is a PREFIX of the model
        group (same ranks, overlaid), and the colocate flag is True."""
        from swift.dev.recipe.utils import plan_device_groups

        groups, colocate = plan_device_groups(4, 'colocate', 2, [])
        assert groups == [('model', [0, 1, 2, 3]), ('sampler', [0, 1])]
        assert colocate is True

    def test_disaggregated_sampler_gets_disjoint_ranks(self):
        """S8-02: disaggregated gives the sampler its own ranks AFTER the trainer's -- no overlap."""
        from swift.dev.recipe.utils import plan_device_groups

        groups, colocate = plan_device_groups(8, 'disaggregated', 4, [])
        assert groups == [('model', list(range(8))), ('sampler', [8, 9, 10, 11])]
        assert colocate is False
        model_ranks = set(groups[0][1])
        assert not model_ranks & set(groups[1][1])

    def test_auxiliary_groups_are_disjoint_too(self):
        from swift.dev.recipe.utils import plan_device_groups

        groups, _ = plan_device_groups(8, 'disaggregated', 4, [('extra', 2)])
        assert groups[-1] == ('extra', [12, 13])
        all_ranks = [r for _, ranks in groups for r in ranks]
        assert len(all_ranks) == len(set(all_ranks)), 'no rank may serve two groups'

    def test_sampler_wider_than_trainers_rejected(self):
        """S8-03: a colocate sampler that does not fit within the trainer's GPUs is rejected loudly,
        not silently pinned to ranks the trainer does not own."""
        from swift.dev.recipe.utils import plan_device_groups

        with pytest.raises(ValueError, match='colocate'):
            plan_device_groups(4, 'colocate', 8, [])


class TestWeightedReward:
    """S8-21: the single-row collapse -- nansum over weighted per-scorer columns (a judge NaN must
    not poison the whole sample)."""

    def test_weighted_nansum(self):
        from swift.dev.recipe.grpo.assembly import weighted_reward

        # 8*0.25 + nan*2.0 -- the None judge score contributes nothing, not NaN.
        assert weighted_reward([8.0, None], torch.tensor([0.25, 2.0])) == pytest.approx(2.0)

    def test_all_nan_is_zero(self):
        from swift.dev.recipe.grpo.assembly import weighted_reward

        assert weighted_reward([None, None], torch.tensor([1.0, 1.0])) == pytest.approx(0.0)


class TestBuildRewardWeights:
    """S8-22: None -> all-ones; a mismatched length is rejected (weights align with the resolved
    scorer count, and a silent truncation would reweight the wrong scorers)."""

    def test_none_is_all_ones(self):
        from swift.dev.utils.scoring import build_reward_weights

        assert build_reward_weights(None, 3).tolist() == [1.0, 1.0, 1.0]

    def test_explicit_weights_pass_through(self):
        from swift.dev.utils.scoring import build_reward_weights

        assert build_reward_weights([0.25, 2.0], 2).tolist() == [0.25, 2.0]

    def test_length_mismatch_rejected(self):
        from swift.dev.utils.scoring import build_reward_weights

        with pytest.raises(ValueError, match='length 2 != number of reward funcs 3'):
            build_reward_weights([1.0, 2.0], 3)


class TestWeightedRewards:
    """S8-24: the [N, n_scorers] matrix -> [N] weighted sum, nansum down the columns."""

    def test_column_collapse_with_nan(self):
        from swift.dev.utils.scoring import weighted_rewards

        matrix = torch.tensor([[1.0, 2.0], [3.0, float('nan')]])
        out = weighted_rewards(matrix, [0.5, 0.5])
        assert out.tolist() == pytest.approx([1.5, 1.5])  # row 2: 3*0.5 + nan->0

    def test_default_weights_sum(self):
        from swift.dev.utils.scoring import weighted_rewards

        matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert weighted_rewards(matrix, None).tolist() == pytest.approx([3.0, 7.0])


class TestAccumulateRewardMetric:
    """S8-23: reset-then-accumulate, per-scorer columns by name + the weighted total."""

    class _Metric:

        def __init__(self):
            self.reset()

        def reset(self):
            self.batches = []
            self.accumulated = False

        def accumulate(self, *, rewards, completion_lengths):
            self.accumulated = True
            self.batches.append((rewards, completion_lengths))

    class _Sample:

        def __init__(self, n):
            self.response_token_ids = list(range(n))

    def test_reset_then_accumulate_with_named_columns(self):
        from swift.dev.utils.scoring import accumulate_reward_metric

        metric = self._Metric()
        metric.batches.append('stale')  # a previous batch's data
        samples = [self._Sample(2), self._Sample(3)]
        rewards = torch.tensor([[1.0, 4.0], [2.0, 6.0]])
        accumulate_reward_metric(metric, samples, rewards,
                                 scorer_names=['acc', 'fmt'], reward_weights=[1.0, 1.0])
        assert 'stale' not in metric.batches  # the stale batch is gone after reset
        rewards_dict, lengths = metric.batches[0]
        assert rewards_dict == {'acc': [1.0, 2.0], 'fmt': [4.0, 6.0], 'total': [5.0, 8.0]}
        assert lengths == [2, 3]

    def test_unnamed_columns_report_total_only(self):
        """A single-callable fallback resolves no names -- only the weighted total is reported."""
        from swift.dev.utils.scoring import accumulate_reward_metric

        metric = self._Metric()
        accumulate_reward_metric(metric, [self._Sample(1)], torch.tensor([[2.0, 3.0]]),
                                 scorer_names=[], reward_weights=None)
        rewards_dict, _ = metric.batches[0]
        assert rewards_dict == {'total': [5.0]}
