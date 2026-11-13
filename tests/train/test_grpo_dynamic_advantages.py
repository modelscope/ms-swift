# Copyright (c) ModelScope Contributors. All rights reserved.
import torch

from swift.rl_core.advantage import compute_advantages, compute_advantages_dynamic


def _rewards(num_prompts=3, num_generations=4, num_funcs=2, seed=0):
    torch.manual_seed(seed)
    rewards_per_func = torch.randn(num_prompts * num_generations, num_funcs)
    rewards_per_func[1, 1] = float('nan')  # a reward function that skipped one completion
    reward_weights = torch.tensor([1.0, 0.5])[:num_funcs]
    prompt_ids = [f'p{i // num_generations}' for i in range(num_prompts * num_generations)]
    request_ids = [f'r{i}' for i in range(num_prompts * num_generations)]
    return rewards_per_func, reward_weights, prompt_ids, request_ids


def test_dynamic_gdpo_matches_fixed_group_gdpo():
    rewards_per_func, reward_weights, prompt_ids, request_ids = _rewards()
    expected, _ = compute_advantages(rewards_per_func, reward_weights, num_generations=4, scale_rewards='gdpo')
    advantages, _ = compute_advantages_dynamic(
        rewards_per_func, reward_weights, prompt_ids, request_ids, scale_rewards='gdpo')
    torch.testing.assert_close(advantages, expected)


def test_dynamic_gdpo_maps_repeated_requests_back():
    rewards_per_func, reward_weights, prompt_ids, request_ids = _rewards()
    expected, _ = compute_advantages_dynamic(
        rewards_per_func, reward_weights, prompt_ids, request_ids, scale_rewards='gdpo')
    # a multi-turn request contributes two rows with the same rewards
    rewards_per_func = torch.cat([rewards_per_func, rewards_per_func[5:6]])
    advantages, _ = compute_advantages_dynamic(
        rewards_per_func,
        reward_weights,
        prompt_ids + [prompt_ids[5]],
        request_ids + [request_ids[5]],
        scale_rewards='gdpo')
    torch.testing.assert_close(advantages[:-1], expected)
    torch.testing.assert_close(advantages[-1], expected[5])
