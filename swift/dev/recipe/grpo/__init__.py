# Copyright (c) ModelScope Contributors. All rights reserved.
# GRPO's pilot recipe: the visible five-segment main (assembly builds the cluster, run_grpo trains).
from __future__ import annotations

from swift.dev.recipe.grpo.assembly import (
    Cluster, build_driver, greedy_params, group_variance, grpo_group_requests, grpo_on_group, initialize_cluster)
from swift.dev.recipe.grpo.run_grpo import run_grpo
from swift.dev.recipe.utils import (
    initialize_rl_devices, plan_device_groups, prompt_rows_from_dataset, resolve_rollout_max_steps,
    rollout_step_budget)

__all__ = [
    'Cluster', 'build_driver', 'grpo_group_requests', 'grpo_on_group', 'group_variance', 'greedy_params',
    'initialize_cluster', 'initialize_rl_devices', 'plan_device_groups', 'prompt_rows_from_dataset',
    'resolve_rollout_max_steps', 'rollout_step_budget', 'run_grpo',
]
