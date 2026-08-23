"""Minimal end-to-end GRPO training loop, peer of SFTLoop.

Data flow per step:
    rollout (RolloutEngine.generate) -> reward -> group-relative advantages
    -> GRPOLoss forward_backward (GA) -> clip_grad_and_step.

Scope (T1 wired):
  - reward: `swift.dev.reward` (L1 API over swift `orms`); a single toy callable is also accepted
    for smoke tests. Multiple reward funcs + reward_weights supported.
  - advantage: `swift.dev.advantage` (L1 API over rl_core: grpo/rloo/reinforce++,
    group/batch/none/gdpo); estimator/scale read from RLHFConfig.
  - only forward_backward / forward_only are used (backend-agnostic).

Deliberately NOT wired (bundled with later items, see doc P2-3c / T0):
  - weight-sync: the RolloutEngine's vLLM keeps its INITIAL weights, so the behavior policy never
    tracks the trained policy. A green run here is "pipeline works", never "GRPO is correct".
  - ref/KL (beta): old_logps come from the rollout logprobs, and ref-KL needs a ref forward; both
    must land WITH weight-sync (else ratio is systematically off, hidden by the step0 same-weights
    test). DAPO dynamic sampling, reward models, async rewards are separate items.
"""
# TODO: not implemented yet
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional

from swift.dev.advantage import compute_advantages
from swift.dev.reward import compute_rewards_per_func, get_reward_funcs
from swift.utils import get_logger

if TYPE_CHECKING:
    from swift.dev.configs import RLHFConfig
    from swift.dev.model import TrainableModel

logger = get_logger()


def compute_group_advantages(rewards: List[float], num_generations: int, scale: str = 'group') -> List[float]:
    """Group-relative advantages for a flat reward list (thin alias over `swift.dev.advantage`).

    Kept for the single-scalar-reward smoke path + existing tests. rewards are ordered so that each
    consecutive block of `num_generations` belongs to one prompt group; returns a flat list aligned
    with rewards.
    """
    return compute_advantages(rewards, num_generations, scale_rewards=scale)


def toy_length_reward(sample: Any) -> float:
    """Deterministic toy reward: normalized completion length. Reward quality is out of scope
    here — we only need a real, non-constant advantage signal so the loss is non-trivial.

    ``response_token_ids`` is per-turn (``List[List[int]]``), so total length sums inner turns.
    """
    turns = getattr(sample, 'response_token_ids', None) or []
    return float(sum(len(turn) for turn in turns))


class GRPOLoop:
    """Minimal GRPO loop. See module docstring for the scope (no weight-sync).

    Reward is specified one of two ways (mutually exclusive):
      - reward_funcs: list of registered `orms` names and/or callables (the T1 path; resolved via
        swift.dev.reward.get_reward_funcs, scored per-func, combined by reward_weights, then
        advantage via swift.dev.advantage). This is the real path.
      - reward_fn: a single callable RolloutSample -> float (the smoke path; kept for existing
        tests). Ignored when reward_funcs is given.
    """

    def __init__(self,
                 model: TrainableModel,
                 rollout_engine: Any,
                 prompts: List[List[dict]],
                 *,
                 num_generations: int = 4,
                 reward_funcs: Optional[List[Any]] = None,
                 reward_weights: Optional[List[float]] = None,
                 advantage_estimator: str = 'grpo',
                 scale_rewards: str = 'group',
                 rlhf_config: Optional['RLHFConfig'] = None,
                 reward_fn: Callable[[Any], float] = toy_length_reward,
                 max_steps: int = 3,
                 gradient_accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 sampling_params: Optional[dict] = None):
        self.model = model
        self.rollout = rollout_engine
        self.prompts = prompts
        self.num_generations = num_generations
        self.advantage_estimator = advantage_estimator
        self.scale_rewards = scale_rewards
        self.reward_weights = reward_weights
        # Resolve reward funcs once (name -> orms instance). Empty list -> fall back to reward_fn.
        self.reward_funcs, self.reward_func_names = (
            get_reward_funcs(reward_funcs, rlhf_config) if reward_funcs else ([], []))
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.sampling_params = sampling_params
        self.global_step = 0
        self.micro_step = 0
        self.history: list = []

    def _active_group(self):
        return self.model.optimizer_group[self.model._get_default_group()]

    def _rollout_step(self):
        """One rollout: generate, reward, group-advantage. Returns list of (sample, advantage).

        Two reward paths (see class docstring):
          - reward_funcs set -> per-func reward matrix + advantage via the L1 reward/advantage APIs
            (estimator/scale from config; supports rloo/reinforce++/gdpo, reward_weights).
          - else -> single scalar reward_fn + group-mean advantage (smoke path).
        Samples are grouped by prompt (num_generations each), so rewards/advantages stay in group
        order and each block of num_generations is one prompt group.
        """
        samples = self.rollout.generate(
            self.prompts, num_samples=self.num_generations, sampling_params=self.sampling_params)
        if self.reward_funcs:
            # Loop-side glue only: project samples onto the L1 reward API's inputs (completion text +
            # batched dataset columns). The scoring/advantage math itself lives in swift.dev.*.
            completions = [s.decoded for s in samples]
            column_keys = {k for s in samples for k in (getattr(s, 'extra', None) or {})}
            columns = {k: [(getattr(s, 'extra', None) or {}).get(k) for s in samples] for k in column_keys}
            rewards_per_func = compute_rewards_per_func(completions, self.reward_funcs, columns)
            advantages = compute_advantages(
                rewards_per_func,
                self.num_generations,
                reward_weights=self.reward_weights,
                advantage_estimator=self.advantage_estimator,
                scale_rewards=self.scale_rewards)
        else:
            rewards = [self.reward_fn(s) for s in samples]
            advantages = compute_group_advantages(rewards, self.num_generations, scale=self.scale_rewards)
        return list(zip(samples, advantages))

    def fit(self) -> list:
        """Run max_steps GRPO steps. Each step: rollout -> one GA window of forward_backward.

        NOTE the rollout policy is NEVER updated (no weight-sync),
        so this is a pipeline smoke, not a correct GRPO training run.
        """
        ga = self.gradient_accumulation_steps
        group = self._active_group()
        for _ in range(self.max_steps):
            batch = self._rollout_step()
            # Feed each generated trajectory as one micro-batch (1 sequence each): GA correctness
            # requires an equal number of sequences per micro-batch -> 1 seq per micro.
            for sample, advantage in batch:
                self.micro_step += 1
                inputs = [sample.input_feature]
                self.model.forward_backward(
                    inputs=inputs, gradient_accumulation_steps=ga, advantages=[advantage], old_logps=[sample.old_logps])
                is_boundary = group.do_grad_sync(ga)
                self.model.clip_grad_and_step(max_grad_norm=self.max_grad_norm, gradient_accumulation_steps=ga)
                if is_boundary:
                    self.global_step += 1
                    metrics = group.calculate_metrics(True)
                    loss = float(metrics['loss']) if metrics.get('loss') is not None else float('nan')
                    self.history.append({'step': self.global_step, 'loss': loss})
        return self.history
