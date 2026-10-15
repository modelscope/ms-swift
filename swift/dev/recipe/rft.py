"""RFT's select + reward-metric component for the flat ``run_rft`` recipe.

``RFTSelector`` is the rejection-sampling half of RFT, decoupled from the loop it used to live on:
``run_rft`` blocks on one whole-prompt-set rollout per round (no driver -- see ``run_rft``), scores the
completions through the shared :mod:`~swift.dev.recipe._scoring` chokepoint, and hands the scored samples
to this selector, which keeps the per-prompt winners and returns them as a samples-only
:class:`~swift.dev.recipe._scoring.TrainingBatch` for plain cross-entropy SFT.

* :meth:`accumulate_metric` -- feeds one round's whole generated rollout (BEFORE rejection) into the
  driver-held ``CompletionRewardMetric``, so the logged ``train/total_reward`` tracks the policy's
  improving sample quality; the SFT loss on the kept subset alone hides the convergence signal.
* :meth:`select` -- the rejection filter (best-of-N / reward threshold / top-K, capped by
  ``rft_max_samples_per_prompt``), applied group-wise across every prompt's ``num_generations``
  completions.
* :meth:`step_telemetry` -- the per-step metric fold read by the recipe body's ``StepRunner``.

RFT has no advantage estimator, no reference model and no importance ratio: the kept completions are
trained with plain SFT, so the batch carries no advantage / log-prob columns.
"""
from __future__ import annotations

from typing import Any, List, Optional

from twinkle.metric import CompletionRewardMetric

from swift.dev.recipe._scoring import TrainingBatch, accumulate_reward_metric


def _rft_log_line(record: dict, loss: float) -> str:
    """Console head signal for RFT: loss plus this round's mean total reward when one was computed.

    The reward is what actually shows rejection sampling working -- the SFT loss on the kept completions
    starts near its floor and stays flat -- so the step line leads with ``reward`` when the round produced
    one. Injected into ``StepRunner`` as its ``log_formatter``.
    """
    reward = record.get('train/total_reward')
    reward_str = f'  reward={reward:.4f}' if reward is not None else ''
    return f'step {record.get("step")}  loss={loss:.4f}{reward_str}'


class RFTSelector:
    """The rejection-sampling selector + reward-metric holder injected into the flat ``run_rft`` body.

    ``num_generations`` is the per-prompt sample count N (``rft_num_samples`` or ``num_generations``):
    every block of N samples is one prompt's generation set, and the filter runs per block. The selector
    owns the ``CompletionRewardMetric`` the recipe feeds once per round, mirroring the reward-metric
    placement of :class:`~swift.dev.recipe.grpo.GRPOAssembler`.
    """

    def __init__(self,
                 *,
                 num_generations: int,
                 rft_select: str = 'best_of_n',
                 rft_threshold: float = 0.0,
                 rft_top_k: int = 1,
                 rft_max_samples_per_prompt: Optional[int] = None,
                 reward_scorer_names: Optional[List[str]] = None,
                 reward_weights: Optional[List[float]] = None):
        if rft_select not in ('best_of_n', 'threshold', 'top_k'):
            raise ValueError(f"rft_select must be one of 'best_of_n'/'threshold'/'top_k', got {rft_select!r}.")
        if rft_max_samples_per_prompt is not None and rft_max_samples_per_prompt < 1:
            raise ValueError(f'rft_max_samples_per_prompt must be >= 1 or None, got {rft_max_samples_per_prompt}.')
        self.num_generations = num_generations
        self.rft_select = rft_select
        self.rft_threshold = rft_threshold
        self.rft_top_k = max(1, rft_top_k)
        self.rft_max_samples_per_prompt = rft_max_samples_per_prompt
        #: The outcome-channel roster (``--orm`` + ``--reward_model`` labels), for the metric's per-scorer
        #: columns; an empty roster reports only the weighted ``total``.
        self.reward_scorer_names = list(reward_scorer_names or [])
        self.reward_weights = reward_weights
        #: Driver-side reward component fed once per round and read back by :meth:`step_telemetry`.
        self._reward_metric = CompletionRewardMetric()

    # --- the per-round metric fold -------------------------------------------------------------------------

    def accumulate_metric(self, samples: List[Any], rewards_per_func: Any) -> None:
        """Feed one round's WHOLE generated rollout into the reward metric (reset-then-accumulate).

        Called before rejection so the logged reward tracks the policy's improving sample quality --
        filtering first would make the metric read only the kept winners and hide the trend.
        """
        accumulate_reward_metric(self._reward_metric, samples, rewards_per_func,
                                 scorer_names=self.reward_scorer_names, reward_weights=self.reward_weights)

    def step_telemetry(self) -> dict:
        """The per-optimizer-step metric fold read by the recipe body's ``StepRunner``.

        Every step of one round reports the same round-level reward (RFT re-samples only between rounds,
        so the reward is naturally one value per round); ``calculate`` is non-resetting and returns an
        empty mapping when nothing was accumulated.
        """
        return dict(self._reward_metric.calculate())

    # --- the rejection filter ------------------------------------------------------------------------------

    def _select_keep(self, group_rewards: List[float]) -> List[int]:
        """Local indices (into one prompt's N samples) to keep, ordered best-first."""
        order = sorted(range(len(group_rewards)), key=lambda i: group_rewards[i], reverse=True)
        if self.rft_select == 'best_of_n':
            keep = order[:1]
        elif self.rft_select == 'top_k':
            keep = order[:self.rft_top_k]
        else:  # threshold: keep every sample at/above the threshold, best-first.
            keep = [i for i in order if group_rewards[i] >= self.rft_threshold]
        if self.rft_max_samples_per_prompt is not None:
            keep = keep[:self.rft_max_samples_per_prompt]
        return keep

    def select(self, samples: List[Any], rewards: List[float]) -> TrainingBatch:
        """Apply the rejection filter across every prompt group, returning the kept samples.

        Returns a samples-only ``TrainingBatch``: RFT trains the kept completions with plain cross-entropy,
        so it carries no advantage / log-prob columns (they stay ``None``) and only ``.samples`` is consumed
        downstream.
        """
        n = self.num_generations
        if len(samples) % n != 0:
            raise RuntimeError(f'{len(samples)} samples is not a multiple of num_generations={n}; '
                               'group-wise rejection sampling is impossible.')
        selected: List[Any] = []
        for start in range(0, len(samples), n):
            group = samples[start:start + n]
            group_rewards = rewards[start:start + n]
            for local in self._select_keep(group_rewards):
                selected.append(group[local])
        return TrainingBatch(samples=selected)
