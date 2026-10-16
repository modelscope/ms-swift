"""PPO's score + assemble + encode component for the flat ``run_ppo`` recipe.

``PPOAssembler`` is the per-sample counterpart of :class:`~swift.dev.recipe.grpo.assembler.GRPOAssembler`: the
driver (``twinkle_agentic.async_rl.streaming_driver.StreamingDriver``) owns the rollout mechanism and
knows nothing about PPO; this class supplies the PPO-specific callables the driver invokes, plus the
two-model encoding half the flat recipe body calls:

* :meth:`score_sample` -- the per-sample **score seam**, run by the driver between collect and consume. It
  computes each trajectory's raw outcome reward (``sample.orm_reward``, one float per resolved ORM scorer)
  and returns ``None`` so the driver keeps the in-place-mutated sample.
* :meth:`assemble` -- the **assemble seam**, run by the driver once a per-sample pull is ready. It turns
  the scored samples into a :class:`~swift.dev.recipe.common.scoring.TrainingBatch` with the GAE anchors computed
  ONCE: two whole-batch forwards (the critic's per-token values, the reference's per-token logps for the
  KL penalty), then per-token rewards ``r_t = -kl_coef * (logp_t - ref_logp_t)`` with the outcome channel's
  weighted scalar added at the final token, then twinkle's ``GAEAdvantage`` backwards over the response.
* :meth:`forward_kwargs` / :meth:`critic_kwargs` -- the **encoding half**, called by the recipe body per
  mini-batch to build the policy's clipped-surrogate and the critic's clipped-value-loss
  ``forward_backward`` kwargs.

The structural differences from :class:`~swift.dev.recipe.grpo.assembler.GRPOAssembler` (confirmed with the
maintainer):

* advantage source: GRPO computes group-relative scalar advantages from the outcome matrix (group
  assembly boundary, no model forwards in ``assemble``, batch carries features + advantages, and a
  group-completion hook); PPO runs per-token GAE against the learned critic (per-sample assembly
  boundary, ``assemble`` contains TWO chunked model forwards, and the batch additionally carries
  values/returns/old_logps with no group hook).
* shared surface: the outcome scoring and the mini-batch splitting are the same
  :mod:`~swift.dev.recipe.common.scoring` pure functions :class:`~swift.dev.recipe.grpo.assembler.GRPOAssembler`
  delegates to; the two assemblers are independent classes with no shared base -- the common code lives
  in the pure functions, not an inheritance tree.

CHORD / PRM / teacher-signal reweighting are absent because PPO's per-token GAE defines its own reward
and anchor pipeline; the outcome-reward half is the one shared chokepoint.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from twinkle.advantage import GAEAdvantage
from twinkle.metric import CompletionRewardMetric

from swift.dev.recipe.common.scoring import (
    TrainingBatch,
    batched_response_scores,
    response_positions,
    score_samples,
    split_mini_batches,
    weighted_rewards,
)
from swift.dev.utils import get_logger

if TYPE_CHECKING:
    from twinkle_agentic.async_rl.streaming_driver import BufferRecord

    from swift.dev.config import RLConfig
    from swift.dev.model import TrainableModel

logger = get_logger()


def _ppo_log_line(record: dict, loss: float) -> str:
    """Console head signal for PPO: reward plus the critic's explained_variance.

    The reward shows the policy improving; explained_variance (how well the critic predicts returns)
    shows the value function keeping up -- together the two quantities that reveal PPO converging, which
    the clipped-surrogate loss alone hides. Both are optional so a degenerate rollout still logs loss.
    Injected into ``StepRunner`` as its ``log_formatter``.
    """
    parts = [f'step {record.get("step")}', f'loss={loss:.4f}']
    value_loss = record.get('value_loss')
    if value_loss is not None:
        parts.append(f'value_loss={value_loss:.4f}')
    reward = record.get('train/total_reward')
    if reward is not None:
        parts.append(f'reward={reward:.4f}')
    explained_variance = record.get('train/explained_variance')
    if explained_variance is not None:
        parts.append(f'explained_variance={explained_variance:.4f}')
    return '  '.join(parts)


class PPOAssembler:
    """The PPO score + assemble + two-model encode component injected into the per-sample streaming driver.

    The outcome-reward scoring is the shared :mod:`~swift.dev.recipe.common.scoring` chokepoint
    (:func:`score_samples` / :func:`weighted_rewards`), so PPO scores exactly like GRPO/RFT instead of
    down a separate raw-model path. With no ``--orm`` and no ``--reward_model`` the channel scores a zero
    column, so the reward is the per-token KL penalty alone. Per-token GAE runs against the critic
    (``task='value'`` keeps the seq_cls head's per-token output ``V(s_t)``); advantages/old_logps/
    returns/old_values are captured ONCE per rollout, so the ``num_ppo_epochs`` replay passes are genuine
    PPO batch re-uses of the same anchors. The critic is never published to the sampler, so no rollout
    regime touches ``value_model``.
    """

    def __init__(self,
                 model: 'TrainableModel',
                 rollout: Any,
                 *,
                 value_model: 'TrainableModel',
                 reference: Any = None,
                 reward_scorers: Optional[List[Any]] = None,
                 reward_scorer_names: Optional[List[str]] = None,
                 reward_weights: Optional[List[float]] = None,
                 reward_fn: Any = None,
                 train_batch_size: int = 1,
                 gradient_accumulation_steps: int = 1,
                 rl_config: Optional['RLConfig'] = None):
        # Validate the outcome-reward roster up front: the recipe resolved ``--orm`` rules + ``--reward_model``
        # scorers into ONE ordered list; this only checks the display names align one-to-one with the scorers.
        resolved_scorers = list(reward_scorers or [])
        scorer_names = list(reward_scorer_names or [])
        if scorer_names and len(scorer_names) != len(resolved_scorers):
            raise ValueError('reward_scorer_names must align one-to-one with reward_scorers.')

        self.model = model
        self.rollout = rollout
        #: The trainable critic (a seq_cls num_labels=1 value model), stepped alongside the policy each
        #: micro-step; never published to the sampler.
        self.value_model = value_model
        self.reference = reference
        #: The outcome-reward channel: ``reward_scorers`` is the ordered ``func(completions, **columns)`` list,
        #: ``reward_scorer_names`` labels each column for the metric, ``reward_weights`` (``--orm_weights``)
        #: combines them, ``reward_fn`` is the single-callable fallback used only when no scorer list resolved.
        self.reward_scorers = resolved_scorers
        self.reward_scorer_names = scorer_names or [type(scorer).__name__ for scorer in resolved_scorers]
        self.reward_weights = reward_weights
        self.reward_fn = reward_fn
        #: Samples per ``forward_backward`` (``per_device_train_batch_size * dp_size``), so one mini-batch
        #: splits evenly across the slice_dp ranks.
        self.train_batch_size = max(1, train_batch_size)
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.rl_config = rl_config
        self.num_ppo_epochs = max(1, rl_config.num_ppo_epochs) if rl_config is not None else 1
        # gamma/lam are GAE *construction* hyperparameters (its __call__ is keyword-only for masks/
        # normalize), so they are bound here rather than passed per call.
        self._gae = GAEAdvantage(gamma=rl_config.gamma if rl_config is not None else 1.0,
                                 gae_lambda=rl_config.lam if rl_config is not None else 0.95)
        #: Driver-held reward component fed once per assembled batch and read back by :meth:`step_telemetry`.
        #: PPO folds only the weighted TOTAL (the per-token reward summed over the response -- KL penalty
        #: plus the outcome scalar), not the per-scorer columns, because its reward is per-token, not a
        #: per-completion outcome matrix.
        self._reward_metric = CompletionRewardMetric()
        #: Version spans of the last assembled batch, read by :meth:`step_telemetry`.
        self._last_batch_version_spans: List[int] = []

    # --- the per-sample score seam ------------------------------------------------------------------------

    def score_sample(self, sample: Any) -> None:
        """Score ONE rollout trajectory in place (the driver's per-sample score seam).

        Computes the raw outcome reward into ``sample.orm_reward`` (one float per resolved ORM scorer, the
        row of the ``[N, n_scorers]`` matrix a batch score would build) through the shared
        :func:`~swift.dev.recipe.common.scoring.score_samples` chokepoint. Returns ``None`` so the driver keeps
        this same (mutated) sample. The per-token KL reward and GAE need the critic / reference forwards,
        so they run in :meth:`assemble`, not here.
        """
        rewards = score_samples([sample], reward_scorers=self.reward_scorers, reward_fn=self.reward_fn)
        sample.orm_reward = rewards[0].tolist()
        return None

    # --- the assemble seam --------------------------------------------------------------------------------

    def assemble(self, records: List['BufferRecord']) -> TrainingBatch:
        """Assemble one per-sample pull into a :class:`TrainingBatch` with the GAE anchors computed ONCE.

        ``records`` are the driver's buffered trajectories for this pull (an exact multiple of
        ``train_batch_size * ga`` rows -- see :meth:`assembly_ready`). Each ``record.sample`` already
        carries its raw ``orm_reward`` from :meth:`score_sample`. This method does the whole-batch work
        the per-sample seam could not:

        1. rebuild the ``[N, n_scorers]`` reward matrix from the per-sample ``orm_reward`` rows and combine
           it into one weighted scalar per sample;
        2. drop samples with no response tokens, then run the two chunked whole-batch forwards (critic
           values + reference logps) over the kept features;
        3. build the per-token reward (``-kl_coef * (logp_t - ref_logp_t)`` plus the outcome scalar at the
           final token) and walk twinkle's ``GAEAdvantage`` backwards over it, whitening the advantages
           when ``whiten_rewards`` is set;
        4. feed the kept samples' scalar rewards + completion lengths into the driver-held
           ``CompletionRewardMetric`` so the step record surfaces ``train/total_reward``.

        Everything the ``num_ppo_epochs`` re-use (advantages, returns, old_values, old_logps) is computed
        here before any optimizer step, so the passes see the SAME anchors -- the definition of PPO's batch
        re-use.
        """
        cfg = self.rl_config
        samples = [record.sample for record in records]
        # Stash this batch's version spans for the per-step telemetry fold BEFORE training (one assemble
        # feeds several optimizer steps and every step of this batch reports the same batch-level span
        # aggregate).
        self._last_batch_version_spans = [int(getattr(sample, 'version_span', 0)) for sample in samples]
        rewards_per_func = torch.tensor([list(sample.orm_reward) for sample in samples], dtype=torch.float32)
        rm_scores = weighted_rewards(rewards_per_func, self.reward_weights).tolist()
        positions_all = [response_positions(sample.input_feature) for sample in samples]
        # Drop samples with no response tokens, then batch the two whole-rollout forwards (critic values +
        # reference logps) over the KEPT features -- one chunked ``forward_only`` each, not one per sample
        # (the per-sample form was 2N serial blocking round-trips that starved DP>1).
        kept = [(sample, positions_all[i], rm_scores[i]) for i, sample in enumerate(samples) if positions_all[i]]
        kept_features = [sample.input_feature for sample, _, _ in kept]
        values_all = self._batched_token_values(kept_features)
        ref_all = self._batched_ref_logps(kept_features)
        kept_samples: List[Any] = []
        advantages: List[List[float]] = []
        returns: List[List[float]] = []
        values: List[List[float]] = []
        old_logps: List[List[float]] = []
        sample_rewards: List[float] = []
        completion_lengths: List[int] = []
        for (sample, positions, rm), value_tokens, ref_tokens in zip(kept, values_all, ref_all):
            token_rewards = self._token_rewards(sample, ref_tokens, rm)
            adv, ret = self._gae(token_rewards, value_tokens)
            # GAE returns [1, T] tensors; flatten to the same per-token float lists the batch carries
            # (values/rewards), so the rollout anchors stay plain Python across num_ppo_epochs.
            kept_samples.append(sample)
            advantages.append(adv.squeeze(0).tolist())
            returns.append(ret.squeeze(0).tolist())
            values.append(value_tokens)
            old_logps.append(sample.old_logps)
            sample_rewards.append(float(sum(token_rewards)))
            completion_lengths.append(len(positions))
        if cfg is not None and cfg.whiten_rewards:
            self._whiten_advantages(advantages)
        # Reset per rollout so the metric reflects exactly the batch the upcoming optimizer steps train on.
        self._reward_metric.reset()
        self._reward_metric.accumulate(rewards={'total': sample_rewards}, completion_lengths=completion_lengths)
        return TrainingBatch(
            samples=kept_samples,
            advantages=advantages,
            old_logps=old_logps,
            values=values,
            returns=returns)

    def _batched_token_values(self, features: List[dict]) -> List[List[float]]:
        """The critic's per-response-token value ``V(s_t)`` for a batch (the GAE input + clip anchor).

        Batched over ``train_batch_size`` chunks: one ``forward_only`` per chunk, not one per sample, so
        DP>1 feeds every rank (a length-1 ``inputs`` would starve all but one) and a rollout is not N
        serial blocking round-trips. ``task='value'`` keeps the seq_cls head's PER-TOKEN output.
        """
        return batched_response_scores(
            self.value_model,
            features,
            train_batch_size=self.train_batch_size,
            output_key='logits',
            forward_kwargs={'return_logits': True, 'task': 'value'})

    def _batched_ref_logps(self, features: List[dict]) -> List[Optional[List[float]]]:
        """Per-response-token reference log-probs for the KL penalty, batched (disable_lora, or a frozen ref).

        ``allow_missing`` preserves the old per-sample contract: a reference forward with no ``logps`` means
        "no KL penalty" (``None``) rather than an error.
        """
        if self.reference == 'disable_lora':
            scorer, forward_kwargs = self.model, {'disable_lora': True}
        else:
            scorer, forward_kwargs = self.reference, {}
        return batched_response_scores(
            scorer,
            features,
            train_batch_size=self.train_batch_size,
            output_key='logps',
            forward_kwargs=forward_kwargs,
            allow_missing=True)

    def _token_rewards(self, sample: Any, ref_tokens: Optional[List[float]], rm_score: float) -> List[float]:
        """Per-token reward: ``-kl_coef * (logp_t - ref_logp_t)`` with the RM scalar at the final token."""
        kl_coef = self.rl_config.kl_coef if self.rl_config is not None else 0.0
        old = [float(x) for x in sample.old_logps]
        n = len(old)
        if n == 0:
            return []
        if ref_tokens is None:
            rewards = [0.0] * n
        else:
            rewards = [-kl_coef * (old[t] - ref_tokens[t]) for t in range(n)]
        rewards[-1] += rm_score
        return rewards

    @staticmethod
    def _whiten_advantages(advantages: List[List[float]]) -> None:
        """Standardise per-token advantages across the whole rollout in place (TRL's reward whitening).

        ``advantages`` is the batch's per-row advantage column (one per-token list per sample); each row is
        rebound in place so the :class:`~swift.dev.recipe.common.scoring.TrainingBatch` holding this list sees the
        whitened values.
        """
        import statistics
        flat = [a for row in advantages for a in row]
        if len(flat) < 2:
            return
        mean = statistics.fmean(flat)
        std = statistics.pstdev(flat) or 1.0
        for i, row in enumerate(advantages):
            advantages[i] = [(a - mean) / (std + 1e-8) for a in row]

    # --- the encoding half (recipe body calls these per mini-batch) --------------------------------------

    def forward_kwargs(self, mini_batch: TrainingBatch) -> Dict[str, Any]:
        """The policy's clipped-surrogate ``forward_backward`` kwargs for one mini-batch.

        Parallel lists over the mini-batch's rows (one entry per rollout sample), so slice_dp splits inputs
        and the advantage/old_logp columns consistently. PPO does not deep-copy ``inputs`` here (the
        scoring forwards never mutate a feature in place, unlike the routing-replay path), so the list is
        passed as-is. The critic's kwargs are built separately in :meth:`critic_kwargs` (a second model,
        stepped within the same micro-step).
        """
        return {
            'inputs': [sample.input_feature for sample in mini_batch.samples],
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'advantages': mini_batch.advantages,
            'old_logps': mini_batch.old_logps,
        }

    def critic_kwargs(self, mini_batch: TrainingBatch) -> Dict[str, Any]:
        """The critic's clipped-value-loss ``forward_backward`` kwargs (``task='value'`` per-token head)."""
        return {
            'inputs': [sample.input_feature for sample in mini_batch.samples],
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'task': 'value',
            'returns': mini_batch.returns,
            'old_values': mini_batch.values,
        }

    def num_replays(self) -> int:
        """One rollout is replayed ``num_ppo_epochs`` times (genuine PPO batch re-use of the same anchors)."""
        return self.num_ppo_epochs

    def plan_mini_batches(self, batch: TrainingBatch) -> List[TrainingBatch]:
        """Split one batch into full ``train_batch_size`` mini-batches, warning on a dropped tail.

        Raises when the batch cannot fill even one mini-batch, so the caller never silently runs a no-op
        training round.
        """
        mini_batches = split_mini_batches(batch, self.train_batch_size)
        if not mini_batches:
            raise RuntimeError(
                f'rollout produced {len(batch)} samples but train_batch_size={self.train_batch_size} '
                '(per_device_train_batch_size * dp_size), so no full mini-batch fits and no optimizer step '
                'is possible. Enlarge the prompt dataset or num_generations, or lower '
                'per_device_train_batch_size / the DP world size.')
        dropped = len(batch) - len(mini_batches) * self.train_batch_size
        if dropped:
            logger.warning('Dropping %d of %d rollout samples that do not fill a train_batch_size=%d mini-batch.',
                           dropped, len(batch), self.train_batch_size)
        return mini_batches

    # --- the streaming geometry + telemetry ---------------------------------------------------------------

    def assembly_ready(self, buffer: List['BufferRecord']) -> Optional[List['BufferRecord']]:
        """PPO pull rule: individual samples, an exact multiple of ``train_batch_size * ga`` rows.

        PPO scores each sample independently (per-token GAE against the critic), so there is no group to
        wait for -- the prefix of the buffer (completion order) up to the largest whole-unit multiple is
        pulled, the remainder stays buffered.
        """
        need_rows = self.train_batch_size * self.gradient_accumulation_steps
        if len(buffer) < need_rows:
            return None
        return buffer[:(len(buffer) // need_rows) * need_rows]

    def step_telemetry(self) -> dict:
        """This batch's version-span aggregate + the driver-held reward metric, folded per optimizer step.

        The span fields are emitted only when a stream actually stamped spans, so a synchronous run does not
        report meaningless zeros. ``CompletionRewardMetric.calculate`` returns an empty mapping when nothing
        was accumulated, so a step with no reward channel adds no keys. The driver-side counters
        (trainer-idle ratio, lifetime stream stats) are folded by the recipe body, not here.
        """
        extra: Dict[str, Any] = {}
        spans = self._last_batch_version_spans
        if spans:
            extra.update({
                'partial_ratio': sum(1 for span in spans if span > 0) / len(spans),
                'max_partial_span': max(spans),
                'version_span_mean': sum(spans) / len(spans),
            })
        extra.update(self._reward_metric.calculate())
        return extra
