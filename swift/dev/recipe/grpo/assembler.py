from __future__ import annotations

import copy
import json
import math
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

import torch
from twinkle.metric import CompletionRewardMetric
from twinkle_agentic.async_rl.streaming_driver import GroupDecision

from swift.dev.recipe.common.scoring import (
    TrainingBatch,
    accumulate_reward_metric,
    batched_response_scores,
    copy_inputs,
    encode_privileged_view,
    overlap_by_placement,
    response_positions,
    score_chunks,
    score_samples,
    split_mini_batches,
    weighted_rewards,
)
from swift.dev.rewards.advantage import compute_advantages
from swift.dev.rewards.reward import build_reward_weights, score_completions
from swift.dev.utils import get_logger

if TYPE_CHECKING:
    from twinkle_agentic.async_rl.streaming_driver import BufferRecord

    from swift.dev.config import RLConfig, RewardConfig
    from swift.dev.model import TrainableModel
    from swift.dev.recipe.common.tracking import RunTracker

logger = get_logger()


def compute_group_advantages(rewards: List[float], num_generations: int, scale: str = 'group') -> List[float]:
    """Group-relative advantages for a flat reward list (thin alias over `swift.dev.rewards.advantage`).

    Kept for the single-scalar-reward smoke path + existing tests. rewards are ordered so that each
    consecutive block of `num_generations` belongs to one prompt group; returns a flat list aligned
    with rewards.
    """
    return compute_advantages(rewards, num_generations, scale_rewards=scale)


def toy_length_reward(sample: Any) -> float:
    """Deterministic toy reward: normalized completion length. Reward quality is out of scope
    here — we only need a real, non-constant advantage signal so the loss is non-trivial.

    ``response_token_ids`` is a flat ``List[int]`` (twinkle accounts a trajectory in one array).
    """
    return float(len(getattr(sample, 'response_token_ids', None) or []))


def _grpo_log_line(record: dict, loss: float) -> str:
    """Console head signal: loss plus this rollout's mean total reward when one was computed.

    The reward is what actually shows RL converging -- the policy-gradient loss oscillates around zero by
    construction and hides the trend -- so the step line leads with ``reward`` when the reward metric produced
    one. Injected into ``StepRunner`` as its ``log_formatter``.
    """
    reward = record.get('train/total_reward')
    reward_str = f'  reward={reward:.4f}' if reward is not None else ''
    return f"step {record.get('step')}  loss={loss:.4f}{reward_str}"


class GRPOAssembler:
    """The GRPO score + assemble + encode component injected into the twinkle per-sample rollout driver.

    The driver (``twinkle_agentic.async_rl.streaming_driver.StreamingDriver``) owns the rollout mechanism --
    per-sample submit/poll/collect, version/staleness admission, publish cadence -- and knows nothing about
    GRPO. This class supplies the GRPO-specific callables the driver invokes, plus the encoding half the flat
    recipe body calls:

    * :meth:`score_sample` -- the per-sample **score seam**, run by the driver between collect and consume. It
      computes each trajectory's raw outcome reward (``sample.orm_reward``, one float per resolved ORM scorer)
      and, when a PRM channel is wired, its per-token process reward (``sample.prm_process``). It returns
      ``None`` so the driver keeps the in-place-mutated sample.
    * :meth:`assemble` -- the **assemble seam**, run by the driver once a whole-group pull is ready. It turns
      the scored samples into a :class:`~swift.dev.recipe.common.scoring.TrainingBatch`: group-relative advantage, the
      PRM/ORM convex fusion, the reference / teacher / policy-old logp forwards overlapped by placement, and the
      MoE routing-replay preparation.
    * :meth:`forward_kwargs` -- the **encoding half**, called by the recipe body per mini-batch to build the
      ``forward_backward`` kwargs.
    * the group-completion helpers (:meth:`group_reward_variance` / :meth:`remax_greedy_baseline` /
      :meth:`stash_remax_baseline`) that :class:`GroupCompletionHook` drives.

    CHORD uses a second homogeneous SFT ``forward_backward`` RPC for each GRPO micro-step. This assembler
    owns the expert-feature cycle and the warmup/cosine-decay mixing schedule, while the recipe keeps the
    two RPC calls adjacent so they share one optimizer update. The outcome-reward half (``_score`` /
    ``_score`` / ``_weighted_rewards`` / ``_accumulate_reward_metric``) and the response-logp primitive
    (``_response_logps``) are thin delegates over the shared :mod:`~swift.dev.recipe.common.scoring` functions, so
    GRPO is a flat recipe with no loop class or inheritance at all.
    """

    def __init__(self,
                 model: 'TrainableModel',
                 rollout: Any,
                 *,
                 template: Any = None,
                 reference: Any = None,
                 teacher: Any = None,
                 reward_scorers: Optional[List[Any]] = None,
                 reward_scorer_names: Optional[List[str]] = None,
                 reward_weights: Optional[List[float]] = None,
                 reward_fn: Callable[[Any], float] = toy_length_reward,
                 prm_scorer: Optional[Any] = None,
                 prm_scorers: Optional[List[Any]] = None,
                 prm_weights: Optional[List[float]] = None,
                 num_generations: int = 4,
                 train_batch_size: int = 1,
                 gradient_accumulation_steps: int = 1,
                 advantage_estimator: str = 'grpo',
                 scale_rewards: str = 'group',
                 rl_config: Optional['RLConfig'] = None,
                 reward_config: Optional['RewardConfig'] = None,
                 sampling_params: Optional[dict] = None,
                 output_dir: str = 'output',
                 tracker: Optional['RunTracker'] = None,
                 global_step_fn: Optional[Callable[[], int]] = None,
                 chord_features: Optional[List[dict]] = None,
                 chord_train_batch_size: int = 0):
        # Validate the outcome-reward roster up front: the recipe resolved the ``--orm`` channel's rules +
        # model scorers into ONE ordered list; this only checks the display names align one-to-one with the scorers.
        resolved_scorers = list(reward_scorers or [])
        scorer_names = list(reward_scorer_names or [])
        if scorer_names and len(scorer_names) != len(resolved_scorers):
            raise ValueError('reward_scorer_names must align one-to-one with reward_scorers.')

        self.model = model
        self.rollout = rollout
        self.template = template
        self.reference = reference
        self.teacher = teacher
        #: The outcome-reward channel: ``reward_scorers`` is the ordered ``func(completions, **columns)`` list,
        #: ``reward_scorer_names`` labels each column for the metric, ``reward_weights`` (``--orm_weights``)
        #: combines them, ``reward_fn`` is the single-callable fallback used only when no scorer list resolved.
        self.reward_scorers = resolved_scorers
        self.reward_scorer_names = scorer_names or [type(scorer).__name__ for scorer in resolved_scorers]
        self.reward_weights = reward_weights
        self.reward_fn = reward_fn
        #: Process-reward (PRM) channel: ``prm_scorer`` segments a response into reasoning steps and broadcasts
        #: each step's score onto its tokens; ``prm_scorers`` is the ordered ``--prm`` channel scoring each step,
        #: weighted by ``prm_weights``. Active only when a scorer AND at least one PRM channel scorer are wired.
        self.prm_scorer = prm_scorer
        self.prm_scorers = list(prm_scorers or [])
        self.prm_weights = prm_weights
        self._prm_active = bool(prm_scorer is not None and self.prm_scorers)
        self.num_generations = num_generations
        #: Samples per ``forward_backward`` (``per_device_train_batch_size * dp_size``), so one mini-batch splits
        #: evenly across the slice_dp ranks.
        self.train_batch_size = max(1, train_batch_size)
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.advantage_estimator = advantage_estimator
        self.scale_rewards = scale_rewards
        self.rl_config = rl_config
        #: The reward-channel Config the recipe co-parsed: the one surface the PRM fusion ratio
        #: (``prm_orm_mix``) is read from, kept apart from ``rl_config``'s RL-training knobs.
        self.reward_config = reward_config
        self.sampling_params = sampling_params
        self.output_dir = output_dir
        #: The recipe's RunTracker (for ``log_prompts``) and a zero-arg callable returning the current optimizer
        #: step, both injected so this component holds no training-loop state of its own.
        self.tracker = tracker
        self._global_step_fn = global_step_fn if global_step_fn is not None else (lambda: 0)
        self.chord_features = list(chord_features or [])
        self.chord_train_batch_size = int(chord_train_batch_size)
        if self.chord_features and self.chord_train_batch_size < 1:
            raise ValueError('chord_train_batch_size must be positive when CHORD expert features are configured.')
        self._chord_offset = 0
        #: Driver-held reward component fed once per assembled batch by :meth:`_accumulate_reward_metric` and
        #: read back by :meth:`step_telemetry`.
        self._reward_metric = CompletionRewardMetric()
        #: ReMax greedy baselines stashed per group_key by :class:`GroupCompletionHook` at group completion and
        #: consumed (read + popped) by :meth:`_estimator_inputs` at assemble time.
        self._remax_baselines: Dict[str, float] = {}
        #: Version spans of the last assembled batch, read by :meth:`step_telemetry`.
        self._last_batch_version_spans: List[int] = []

    # --- the per-sample score seam -------------------------------------------------------------------------

    def score_sample(self, sample: Any) -> None:
        """Score ONE rollout trajectory in place (the driver's per-sample score seam).

        Computes the raw outcome reward into ``sample.orm_reward`` (one float per resolved ORM scorer, the row of
        the ``[N, n_scorers]`` matrix :meth:`_score` would build for a batch) and, when a PRM channel is active,
        the per-token process reward into ``sample.prm_process``. Returns ``None`` so the driver keeps this same
        (mutated) sample. Group-relative advantage is NOT computed here -- it needs the whole group, so it runs
        in :meth:`assemble`.

        PRM is per-sample here (not per-group): the scorer segments this one response into reasoning steps and
        broadcasts each step's weighted score onto its tokens. A multi-turn rollout gets a
        :class:`~swift.dev.rewards.prm.MultiTurnContext` (see :meth:`_prm_turn_context`) so a step never spans
        two assistant turns and is scored against the real conversation history -- prior turns AND their tool
        observations -- rather than a flat splice of trainable tokens.
        """
        rewards = self._score([sample])
        sample.orm_reward = rewards[0].tolist()
        if self._prm_active:
            if getattr(sample, 'prompt', None) is None:
                raise ValueError('PRM scoring requires the sample prompt to be stamped by the driver collect '
                                 'step, got None.')
            turn_context = self._prm_turn_context(sample)
            sample.prm_process = self.prm_scorer(
                sample.prompt,
                sample.response_token_ids,
                sample.decoded,
                self._score_steps,
                self.template.decode,
                columns=getattr(sample, 'extra', None),
                turn_context=turn_context)
        return None

    def _prm_turn_context(self, sample: Any):
        """The :class:`~swift.dev.rewards.prm.MultiTurnContext` for a multi-turn sample, else ``None``.

        Mirrors :meth:`_teacher_messages`'s own multi-turn detection (``sample.messages`` is stamped only by
        the multi-turn rollout path) and reuses the same trainable-run/assistant-turn correspondence the
        teacher's privileged view relies on (:func:`swift.dev.rollout.multi_turn.turn_token_spans`), so a
        run-count/assistant-message-count mismatch fails loudly here exactly as it does there.
        """
        messages = getattr(sample, 'messages', None)
        if messages is None:
            return None
        from swift.dev.rewards.prm import MultiTurnContext
        from swift.dev.rollout.multi_turn import turn_token_spans

        turn_spans = turn_token_spans(sample.input_feature)
        assistant_message_indices = [
            index for index, message in enumerate(messages) if message.get('role') == 'assistant']
        if len(assistant_message_indices) != len(turn_spans):
            raise RuntimeError(
                f'multi-turn PRM scoring found {len(assistant_message_indices)} assistant turns in the '
                f'messages but {len(turn_spans)} trainable token runs; a run-splitting mismatch means an '
                'assistant turn produced no trainable token (an empty generation) or a completion_mask '
                'split one turn in two.')
        return MultiTurnContext(
            messages=messages, turn_spans=turn_spans, assistant_message_indices=assistant_message_indices)

    def _score_steps(self, rows: List[Dict[str, Any]]) -> List[float]:
        """Score a batch of PRM step rows -> one weighted process score per row.

        The PRM counterpart of :meth:`_score`: the scorer plugin builds each row as ``{'messages': <the real
        conversation prefix up to this step>, **columns}`` -- ``prompt + assistant(response[:end])`` for a
        single-turn rollout, or the prior turns (assistant AND tool observations) plus this turn's text up to
        the step for a multi-turn one (see :class:`~swift.dev.rewards.prm.MultiTurnContext`) -- and the one
        ordered ``--prm`` scorer list (:attr:`prm_scorers`) scores it: a rule PRM reads the last message's
        text as the completion, a frozen PRM model reads the whole ``messages`` row. The per-scorer scores are
        then combined with ``prm_weights``, the same weighted sum :meth:`_weighted_rewards` applies to outcome
        rewards. The frozen PRM runs on its OWN device group (not sliced across the policy ranks), so a step
        batch smaller than the policy's dp_size is fine.
        """
        completions = [row['messages'][-1]['content'] for row in rows]
        columns = {key: [row.get(key) for row in rows] for key in {k for row in rows for k in row}}
        scores = score_completions(completions, self.prm_scorers, columns)
        weights = build_reward_weights(self.prm_weights, scores.shape[1]).to(
            device=scores.device, dtype=scores.dtype)
        return (scores * weights.unsqueeze(0)).nansum(dim=1).tolist()

    # --- the outcome-reward half (thin delegates over the shared _scoring functions) -----------------------

    def _score(self, samples: List[Any]):
        """Thin delegate over :func:`~swift.dev.recipe.common.scoring.score_samples` (the shared reward matrix)."""
        return score_samples(samples, reward_scorers=self.reward_scorers, reward_fn=self.reward_fn)

    def _weighted_rewards(self, rewards_per_func):
        """Thin delegate over :func:`~swift.dev.recipe.common.scoring.weighted_rewards` (the shared weighted sum)."""
        return weighted_rewards(rewards_per_func, self.reward_weights)

    def _accumulate_reward_metric(self, samples: List[Any], rewards_per_func) -> None:
        """Thin delegate over :func:`~swift.dev.recipe.common.scoring.accumulate_reward_metric` (the shared fold)."""
        accumulate_reward_metric(
            self._reward_metric,
            samples,
            rewards_per_func,
            scorer_names=self.reward_scorer_names,
            reward_weights=self.reward_weights)

    def _response_logps(self, scorer: Any, features: List[dict]) -> List[List[float]]:
        """Score ``features`` with ``forward_only``, returning one response-token logp list per feature.

        A thin wrapper over :func:`~swift.dev.recipe.common.scoring.batched_response_scores` (one chunked ``forward_only``
        per ``train_batch_size`` block, DP>1-correct, right-padding aware): the shared "score a frozen model over
        the student's response tokens" primitive used for the reference / teacher logps.
        """
        forward_kwargs = None
        if self.chord_enabled and getattr(scorer, 'placement_key', None) is None:
            forward_kwargs = {'loss_key': 'grpo'}
        return batched_response_scores(
            scorer,
            features,
            train_batch_size=self.train_batch_size,
            output_key='logps',
            forward_kwargs=forward_kwargs)

    # --- the assemble seam ---------------------------------------------------------------------------------

    def assemble(self, records: List['BufferRecord']) -> TrainingBatch:
        """Assemble one whole-group pull into a trainable :class:`TrainingBatch` (the driver's assemble seam).

        ``records`` are the driver's buffered trajectories for this pull (group-contiguous, an exact multiple of
        ``train_batch_size * ga`` rows -- see :meth:`assembly_ready`). Each ``record.sample`` already carries its
        raw ``orm_reward`` (and ``prm_process`` when a PRM channel is active) from :meth:`score_sample`. This
        method does the group-level work the per-sample seam could not:

        1. rebuild the ``[N, n_scorers]`` reward matrix from the per-sample ``orm_reward`` rows;
        2. overlap the reference / teacher / policy-old logp forwards on their disjoint DeviceGroups;
        3. compute the group-relative advantage (``compute_advantages``), with the ReMax greedy baseline read back
           from the stash :class:`GroupCompletionHook` filled;
        4. fold the PRM process reward into a per-token advantage (convex mix) when active;
        5. prepare each row's MoE expert routing for the training forward's replay.
        """
        samples = [record.sample for record in records]
        group_keys = [record.group_key for record in records]
        rewards_per_func = torch.tensor([list(sample.orm_reward) for sample in samples], dtype=torch.float32)
        # Stash this batch's version spans for the per-step telemetry fold BEFORE training (one assemble feeds
        # several optimizer steps and every step of this batch reports the same batch-level span aggregate).
        self._last_batch_version_spans = [int(getattr(sample, 'version_span', 0)) for sample in samples]
        if self.tracker is not None:
            self.tracker.log_prompts(
                [getattr(sample, 'prompt', None) for sample in samples], samples, self._global_step_fn() + 1, 1)
        self._log_completions(samples, rewards_per_func)
        self._accumulate_reward_metric(samples, rewards_per_func)
        cfg = self.rl_config
        preserve_policy_logps = bool(cfg and (
            cfg.rollout_importance_sampling_mode or cfg.log_rollout_offpolicy_metrics))
        # ref / teacher / policy-old each forward on their own DeviceGroup; overlap the disjoint ones and serialize
        # any that share the policy actor (a disable_lora reference or teacher scores on the policy,
        # ``placement_key is None``). Only the SCHEDULE changes: each thunk is the identical scoring call, so the
        # numbers are bit-for-bit what the serial path produced.
        teacher_owner = self._teacher_owner(samples)
        tasks: List[tuple] = []
        slots: List[str] = []
        if self.reference is not None:
            tasks.append((getattr(self.reference, 'placement_key', None), lambda: self._reference_logps(samples)))
            slots.append('ref')
        if teacher_owner is not None:
            teacher_features = [self._teacher_feature(sample) for sample in samples]
            tasks.append((getattr(teacher_owner, 'placement_key', None),
                          lambda: self._response_logps(teacher_owner, teacher_features)))
            slots.append('teacher')
        if preserve_policy_logps:
            policy_features = [sample.input_feature for sample in samples]
            tasks.append((getattr(self.model, 'placement_key', None),
                          lambda: self._response_logps(self.model, policy_features)))
            slots.append('old')
        scored = dict(zip(slots, overlap_by_placement(tasks)))
        ref_logps = scored.get('ref')
        teacher_logps = scored.get('teacher')
        old_logps = scored.get('old', [sample.old_logps for sample in samples])
        kl_in_reward = bool(cfg and cfg.kl_in_reward and cfg.beta)
        if kl_in_reward and ref_logps is None:
            raise ValueError('kl_in_reward=True requires an active reference model.')
        kl_values = self._sequence_kl([sample.old_logps for sample in samples], ref_logps) if kl_in_reward else None
        advantages = compute_advantages(
            rewards_per_func,
            self.num_generations,
            reward_weights=self.reward_weights,
            advantage_estimator=self.advantage_estimator,
            scale_rewards=self.scale_rewards,
            kl_in_reward=kl_in_reward,
            beta=float(cfg.beta or 0.0) if cfg else 0.0,
            kl_values=kl_values,
            **self._estimator_inputs(samples, group_keys, cfg))
        # PRM process reward: fold the per-token process scores (computed per sample in score_sample) into a
        # per-token advantage, blending each token's process reward with the group-relative ORM advantage above by
        # prm_orm_mix. Replaces the scalar advantages with per-token lists, which _training_advantage passes through.
        if self._prm_active:
            advantages = self._apply_prm(samples, advantages)
        # MoE routing replay: establish each row's expert routing BEFORE the batch reaches the training update,
        # whose forward REPLAYS it. Runs AFTER the scoring forwards so those never carry the routed_experts column.
        self._prepare_routing_replay(samples)
        return TrainingBatch(
            samples=list(samples),
            advantages=list(advantages),
            old_logps=list(old_logps),
            rollout_logps=[sample.old_logps for sample in samples],
            ref_logps=ref_logps,
            teacher_logps=teacher_logps,
            sampling_masks=([sample.sampling_mask for sample in samples]
                            if cfg and cfg.enable_sampling_replay else None))

    def _apply_prm(self, samples: List[Any], orm_advantages: Sequence[float]) -> List[List[float]]:
        """Fold the PRM process reward and the ORM outcome advantage into one per-token advantage (convex mix).

        Each sample's ``prm_process`` (a per-token process reward in the response-token frame, computed in
        :meth:`score_sample`) is blended with its scalar group-relative ORM advantage as
        ``mix * process[i] + (1 - mix) * orm_advantage`` under ``RewardConfig.prm_orm_mix``, so the outcome and process
        signals drive EVERY response token in a fixed proportion. ``prm_orm_mix=0`` reproduces standard GRPO
        byte-for-byte; ``prm_orm_mix=1`` is pure process reward. Returns one per-token advantage list per sample,
        which :meth:`_training_advantage` passes through to ``forward_backward`` unchanged.
        """
        mix = float(getattr(self.reward_config, 'prm_orm_mix', 0.5))
        per_token: List[List[float]] = []
        for sample, orm_advantage in zip(samples, orm_advantages):
            process = getattr(sample, 'prm_process', None)
            if process is None:
                raise RuntimeError('PRM fusion requires a per-sample process reward, but sample.prm_process is '
                                   'None; the score seam (score_sample) must run before assemble.')
            orm = float(orm_advantage)
            per_token.append([mix * step + (1.0 - mix) * orm for step in process])
        return per_token

    def _estimator_inputs(self, samples: List[Any], group_keys: List[Any],
                          cfg: Optional['RLConfig']) -> Dict[str, Any]:
        """Extra ``compute_advantages`` kwargs the active ``advantage_estimator`` needs.

        Returns an empty dict for the estimators that read only the rewards and the group size (grpo / rloo /
        reinforce_plus_plus / reinforce_plus_plus_baseline / grpo_passk). The three that need a per-sample side
        input are wired here:

        - ``opo``: per-sample response token counts (its group baseline is length-weighted).
        - ``gpg``: the ``f_norm`` divisor from the config.
        - ``remax``: the per-group greedy-decoding baseline :class:`GroupCompletionHook` stashed at group
          completion, read back by ``group_key`` and popped so the stash never grows.
        """
        estimator = self.advantage_estimator
        if estimator == 'opo':
            return {
                'response_lengths': [len(self._response_positions(sample.input_feature)) for sample in samples]
            }
        if estimator == 'gpg':
            return {'f_norm': float(cfg.f_norm) if cfg is not None else 1.0}
        if estimator == 'remax':
            missing = [key for key in set(group_keys) if key not in self._remax_baselines]
            if missing:
                # Only reachable when dynamic_sample + remax exhausted the group's regeneration cap: the
                # driver accepted the last retry without re-firing the hook, so no baseline was stashed.
                # A silent 0.0 baseline would quietly disable ReMax centering for the whole batch.
                raise ValueError(
                    f"{len(missing)} remax group(s) have no stashed greedy baseline: the group-completion "
                    'hook could not fire on their final regeneration. Avoid dynamic_sample=True together with '
                    "advantage_estimator='remax', or raise --max_resample_times.")
            baselines = [self._remax_baselines[group_key] for group_key in group_keys]
            for group_key in set(group_keys):
                self._remax_baselines.pop(group_key, None)
            return {'reward_baselines': torch.tensor(baselines, dtype=torch.float32)}
        return {}

    def _log_completions(self, samples: List[Any], rewards_per_func) -> None:
        cfg = self.rl_config
        if not cfg or not cfg.log_completions:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, 'completions.jsonl')
        rewards = rewards_per_func.detach().float().cpu().tolist()
        step = self._global_step_fn() + 1
        with open(path, 'a', encoding='utf-8') as stream:
            for sample, per_func in zip(samples, rewards):
                stream.write(json.dumps({
                    'step': step,
                    'prompt': getattr(sample, 'prompt', None),
                    'completion': sample.decoded,
                    'rewards': per_func,
                }, ensure_ascii=False) + '\n')

    # --- the reference / teacher advantage-pipeline forwards -----------------------------------------------

    def _teacher_messages(self, sample: Any, teacher_prompt: str) -> tuple:
        is_multi_turn = getattr(sample, 'messages', None) is not None
        base = sample.messages if is_multi_turn else getattr(sample, 'prompt', None)
        if base is None:
            raise ValueError('teacher_prompt scoring requires the sample prompt to be stamped by the driver '
                             'collect step, got None.')
        messages = copy.deepcopy(base)
        for message in messages:
            if message.get('role') == 'user':
                message['content'] = teacher_prompt
                break
        response_ids = [int(token) for token in sample.response_token_ids]
        if not is_multi_turn:
            messages.append({'role': 'assistant', 'content': response_ids})
            return messages, response_ids

        from swift.dev.rollout.multi_turn import _messages_with_token_ids
        return _messages_with_token_ids(messages, sample.input_feature), response_ids

    def _teacher_feature(self, sample: Any) -> dict:
        """The input feature the teacher forward scores: the student's response under the teacher's view.

        This teacher -- like :meth:`_reference_logps`'s reference -- is a GRPO *advantage-pipeline* forward, NOT
        the distillation teacher. GRPO needs a second model's per-token logps for two of its OWN features: the KL
        penalty (the reference) and RLSD/SDAR teacher-signal advantage reweighting (``advantage_reweight='rlsd'``
        / ``sdar_loss_coef``). With no ``teacher_prompt`` column the teacher scores the student's own encoding;
        with one, :func:`encode_privileged_view` re-encodes the SAME response tokens under the teacher's privileged
        prompt, so the teacher grades the student's answer from a different (hint-augmented) view.
        """
        teacher_prompt = (getattr(sample, 'extra', None) or {}).get('teacher_prompt')
        if not teacher_prompt:
            return sample.input_feature
        messages, response_ids = self._teacher_messages(sample, teacher_prompt)
        return encode_privileged_view(self.template, messages, response_ids, getattr(sample, 'extra', None))

    def _reference_logps(self, samples: List[Any]) -> Optional[List[List[float]]]:
        if self.reference is None:
            return None
        return self._response_logps(self.reference, [sample.input_feature for sample in samples])

    def _teacher_owner(self, samples: List[Any]) -> Optional[Any]:
        """The teacher scoring owner for these samples, or None when no teacher forward is needed.

        Split out so :meth:`assemble` can read the owner's ``placement_key`` and overlap
        the teacher forward with the reference / policy-old forwards on disjoint DeviceGroups. A privileged
        ``teacher_prompt`` column with no explicit teacher scores on the policy actor itself (``self.model``).
        """
        has_privileged_prompt = any((sample.extra or {}).get('teacher_prompt') for sample in samples)
        needs_teacher = bool(self.teacher is not None or has_privileged_prompt)
        advanced = bool(self.rl_config and
                        (self.rl_config.advantage_reweight == 'rlsd' or self.rl_config.sdar_loss_coef > 0))
        if not needs_teacher:
            if advanced:
                raise ValueError('RLSD/SDAR requires teacher_model or a teacher_prompt dataset column.')
            return None
        return self.teacher if self.teacher is not None else self.model

    @staticmethod
    def _sequence_kl(old_logps: List[List[float]], ref_logps: List[List[float]]):
        values = []
        for old, ref in zip(old_logps, ref_logps):
            if len(old) != len(ref):
                raise RuntimeError(f'reference logps misaligned: old={len(old)}, ref={len(ref)}.')
            delta = torch.tensor(ref, dtype=torch.float32) - torch.tensor(old, dtype=torch.float32)
            values.append((torch.exp(delta) - delta - 1.0).mean() if delta.numel() else torch.tensor(0.0))
        return torch.stack(values)

    def _rlsd_lambda(self) -> float:
        cfg = self.rl_config
        if cfg is None:
            return 0.0
        step = self._global_step_fn()
        if cfg.rlsd_lambda_warmup_steps > 0 and step < cfg.rlsd_lambda_warmup_steps:
            return cfg.rlsd_lambda * step / cfg.rlsd_lambda_warmup_steps
        if cfg.rlsd_lambda_decay_steps > 0 and step >= cfg.rlsd_lambda_warmup_steps:
            progress = (step - cfg.rlsd_lambda_warmup_steps) / cfg.rlsd_lambda_decay_steps
            return cfg.rlsd_lambda * max(1.0 - progress, 0.0)
        return cfg.rlsd_lambda

    def _training_advantage(self, advantage: Any, old_logps: List[float], teacher_logps: Optional[List[float]]):
        if isinstance(advantage, (list, tuple)):
            # Already per-token -- the PRM process reward (see _apply_prm), in the response-token frame the loss
            # expects, so pass it through with no scalar expansion. PRM and the teacher-signal reweighting below
            # are mutually exclusive (both produce per-token advantages from different sources); validate rejects
            # enabling them together.
            return list(advantage)
        cfg = self.rl_config
        if teacher_logps is None or cfg is None:
            return advantage

        from twinkle.advantage.teacher_signal import apply_rlsd_reweight, expand_advantage_to_per_token

        mask = torch.ones((1, len(old_logps)), dtype=torch.bool)
        base = torch.tensor([advantage], dtype=torch.float32)
        teacher = torch.tensor([teacher_logps], dtype=torch.float32)
        old = torch.tensor([old_logps], dtype=torch.float32)
        if cfg.advantage_reweight == 'rlsd':
            return apply_rlsd_reweight(
                base,
                mask,
                teacher,
                old,
                lam=self._rlsd_lambda(),
                clip_range=cfg.rlsd_reweight_clip_range,
                negative_only=cfg.rlsd_negative_only)[0].tolist()
        if cfg.sdar_loss_coef > 0:
            return advantage
        return expand_advantage_to_per_token(
            base, mask, teacher, old, teacher_kl_coef=cfg.teacher_kl_coef)[0].tolist()

    @property
    def chord_enabled(self) -> bool:
        return bool(self.chord_features)

    def chord_mu(self) -> float:
        """Return CHORD's SFT mixing weight for the current optimizer step."""
        if not self.chord_enabled or self.rl_config is None:
            return 0.0
        cfg = self.rl_config
        warmup = int(cfg.chord_mu_warmup_steps or 0)
        decay = int(cfg.chord_mu_decay_steps or 0)
        peak = float(cfg.chord_mu_peak or 0.0)
        valley = float(cfg.chord_mu_valley or 0.0)
        global_step = self._global_step_fn()
        if warmup > 0 and global_step < warmup:
            return peak * global_step / warmup
        if decay == 0 or global_step >= warmup + decay:
            return valley
        progress = (global_step - warmup) / decay
        return valley + (peak - valley) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def sft_forward_kwargs(self) -> Dict[str, Any]:
        """Take the next global CHORD expert batch without GRPO-only replay columns."""
        if not self.chord_enabled:
            raise RuntimeError('CHORD SFT inputs were requested without chord_sft_dataset.')
        features = []
        for _ in range(self.chord_train_batch_size):
            features.append(copy.deepcopy(self.chord_features[self._chord_offset % len(self.chord_features)]))
            self._chord_offset += 1
        return {
            'inputs': features,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
        }

    # --- the encoding half (recipe body calls this per mini-batch) -----------------------------------------

    def forward_kwargs(self, mini_batch: TrainingBatch) -> Dict[str, Any]:
        """Assemble one mini-batch's parallel-list ``forward_backward`` kwargs.

        Every list is parallel to ``mini_batch.samples`` (one entry per rollout row) so slice_dp splits them all
        consistently across the DP ranks. ``ref_logps`` / ``teacher_logps`` are uniform across a rollout (both
        scoring helpers return None or a full list), so the column being None decides whether the key is passed at
        all. CHORD expert rows are intentionally absent from this mapping: the recipe sends them through a
        separate homogeneous SFT RPC from :meth:`sft_forward_kwargs`, so router and sampling replay remain GRPO-only.
        """
        cfg = self.rl_config
        teacher_logps = mini_batch.teacher_logps
        kwargs = {
            'inputs': copy_inputs([sample.input_feature for sample in mini_batch.samples]),
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'advantages': [
                self._training_advantage(
                    mini_batch.advantages[idx], mini_batch.old_logps[idx],
                    teacher_logps[idx] if teacher_logps is not None else None)
                for idx in range(len(mini_batch))
            ],
            'old_logps': list(mini_batch.old_logps),
            'rollout_logps': list(mini_batch.rollout_logps),
            'truncated': [
                bool(cfg and cfg.overlong_filter and getattr(sample, 'truncated', False))
                for sample in mini_batch.samples
            ],
        }
        if mini_batch.ref_logps is not None:
            kwargs['ref_logps'] = list(mini_batch.ref_logps)
        if teacher_logps is not None and cfg and cfg.sdar_loss_coef > 0:
            kwargs['teacher_logps'] = list(teacher_logps)
        if cfg and cfg.router_replay_mode != 'disabled':
            # The training forward self-judges REPLAY from the mode (twinkle resolves router_replay_mode
            # to the backend action; its backward auto-switches to REPLAY_BACKWARD).
            kwargs['router_replay_mode'] = cfg.router_replay_mode
        if cfg and cfg.enable_sampling_replay:
            kwargs['sampling_masks'] = list(mini_batch.sampling_masks)
        return kwargs

    def num_replays(self) -> int:
        """GRPO replays each collected rollout ``num_iterations`` times (at least once)."""
        return max(1, self.rl_config.num_iterations if self.rl_config else 1)

    def plan_mini_batches(self, batch: TrainingBatch) -> List[TrainingBatch]:
        """Split one batch into full ``train_batch_size`` mini-batches, warning on a dropped tail.

        Raises when the batch cannot fill even one mini-batch, so the caller never silently runs a no-op training
        round.
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

    # --- the streaming geometry + telemetry ----------------------------------------------------------------

    def assembly_ready(self, buffer: List['BufferRecord']) -> Optional[List['BufferRecord']]:
        """GRPO pull rule: whole groups, an exact multiple of ``train_batch_size * ga`` rows.

        A group is pullable only when ALL ``num_generations`` trajectories of one prompt are buffered
        (group-relative advantage is undefined on a partial group); siblings are admitted back-to-back under one
        policy, so a complete group always carries ONE version.

        The pull quantum is the smallest whole-group count whose rows fill whole optimizer steps --
        ``need_rows / gcd(num_generations, need_rows)`` groups, i.e. ``lcm(num_generations, need_rows)`` rows -- so
        every pull is an exact multiple of ``need_rows`` (no gradient-accumulation raggedness, and
        ``split_mini_batches`` never drops a tail). Complete groups are selected in first-completion order up to
        the largest whole-quantum count; the rest stay buffered.

        The pulled records are re-emitted GROUP-CONTIGUOUS (all siblings of one group adjacent, groups in
        selection order) because ``compute_advantages`` normalizes each consecutive block of ``num_generations`` as
        one prompt group by POSITION -- the buffer's completion order would otherwise silently mix two groups into
        one normalization block.
        """
        need_rows = self.train_batch_size * self.gradient_accumulation_steps
        counts: Dict[Any, int] = {}
        order: List[Any] = []
        for record in buffer:
            key = record.group_key
            if key not in counts:
                counts[key] = 0
                order.append(key)
            counts[key] += 1
        complete = [key for key in order if counts[key] >= self.num_generations]
        groups_per_pull = need_rows // math.gcd(self.num_generations, need_rows)
        n_pull = groups_per_pull * (len(complete) // groups_per_pull)
        if n_pull < 1:
            return None
        selected = complete[:n_pull]
        selected_set = set(selected)
        by_group: Dict[Any, List['BufferRecord']] = {}
        for record in buffer:
            if record.group_key in selected_set:
                by_group.setdefault(record.group_key, []).append(record)
        pulled: List['BufferRecord'] = []
        for key in selected:
            pulled.extend(by_group[key])
        return pulled

    def step_telemetry(self) -> dict:
        """This batch's version-span aggregate + the shared outcome-reward metric, folded per optimizer step.

        The span fields are emitted only when a stream actually stamped spans, so a synchronous run does not report
        meaningless zeros. ``CompletionRewardMetric.calculate`` returns an empty mapping when nothing was
        accumulated, so a step with no reward channel adds no keys. The driver-side counters (trainer-idle ratio,
        lifetime stream stats) are folded by the recipe body, not here.
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

    # --- the group-completion helpers (driven by GroupCompletionHook) --------------------------------------

    def group_reward_variance(self, samples: List[Any]) -> float:
        """The weighted-reward std of one group's samples (``dynamic_sample``'s zero-variance test).

        Pure computation over the already-scored ``orm_reward`` rows -- no generation -- so it is safe to call from
        the group-completion hook under colocate (the trainer device is not needed). A single-sample group has no
        variance and returns 0.0.
        """
        rewards = torch.tensor([list(sample.orm_reward) for sample in samples], dtype=torch.float32)
        weighted = self._weighted_rewards(rewards)
        if weighted.numel() <= 1:
            return 0.0
        return float(torch.std(weighted, unbiased=False))

    def remax_greedy_baseline(self, prompt: Any, extra: Optional[Dict[str, Any]]) -> float:
        """ReMax baseline: the greedy (argmax) completion's weighted reward for ONE prompt.

        ReMax centers each sampled completion's reward on the reward of a GREEDY decoding of the same prompt (a
        lower-variance baseline than the group mean, at the cost of one extra generation pass). The group-completion
        hook calls this per group. ``temperature=0`` argmax decoding, scored with the SAME reward functions / RM
        plugins as the rollout.

        Only valid under disaggregated placement: the pass needs the sampler to generate, which colocate's
        exclusive-device handover cannot safely do at group-completion time (the trainer holds the device). The
        driver builder refuses colocate + remax up front. The bare ``generate`` does NOT stamp ``sample.prompt``
        (only the driver's collect step does), so it is stamped here for shared reward scoring.
        """
        greedy_params = dict(self.sampling_params or {})
        greedy_params['temperature'] = 0.0  # argmax decoding
        greedy_params.pop('top_p', None)
        greedy_params.pop('top_k', None)
        greedy_samples = self.rollout.generate(
            [prompt], num_samples=1, sampling_params=greedy_params,
            prompt_extras=[extra] if extra else None)
        if len(greedy_samples) != 1:
            raise RuntimeError(f'greedy baseline rollout returned {len(greedy_samples)} samples, expected 1.')
        greedy_sample = greedy_samples[0]
        greedy_sample.prompt = prompt
        return float(self._weighted_rewards(self._score([greedy_sample]))[0])

    def stash_remax_baseline(self, group_key: str, value: float) -> None:
        """Record one group's greedy baseline; :meth:`_estimator_inputs` reads + pops it at assemble time."""
        self._remax_baselines[group_key] = float(value)

    # --- MoE routing replay --------------------------------------------------------------------------------

    def _prepare_routing_replay(self, samples: List[Any]) -> None:
        """Establish each sample's MoE expert routing per ``RLConfig.router_replay_mode``, in place.

        Routing replay keeps an MoE policy's importance ratio honest: the training forward must REPLAY the expert
        choices that actually produced each token rather than recompute them. The two modes differ only in WHERE the
        routing comes from, then feed the SAME REPLAY path (each row's ``encoded['routed_experts']`` + a
        ``router_replay_mode`` forward_backward kwarg the model layer self-judges into REPLAY):

        - ``'R3'``: the sampler exported the generating pass's routing. Require it -- a non-exporting engine leaves
          none, and silently recomputing would be wrong, so fail loudly and point at R2.
        - ``'R2'``: RECORD the training model's own routing for these tokens (under the just-synced generating
          weights, a faithful proxy for the generating pass vLLM did not export).
        """
        mode = self.rl_config.router_replay_mode if self.rl_config else 'disabled'
        if mode == 'disabled':
            return
        if mode == 'R3':
            missing = sum(1 for sample in samples if sample.encoded.get('routed_experts') is None)
            if missing:
                raise RuntimeError(
                    f'router_replay_mode="R3" replays the sampler-reported MoE routing, but {missing} of '
                    f'{len(samples)} rollout samples carry none: the sampling engine did not export '
                    'routed_experts (that needs a vLLM build + MoE checkpoint that report routing). Use '
                    'router_replay_mode="R2" to RECORD the routing on the training model instead, or '
                    '"disabled" to recompute it.')
            return
        if mode == 'R2':
            # R2 records the training model's routing regardless of what the sampler returned, so drop any sampler
            # routing first.
            for sample in samples:
                sample.encoded.pop('routed_experts', None)
            self._record_routing(samples)
            return
        raise ValueError(f'Unknown router_replay_mode {mode!r}; expected one of disabled/R2/R3.')

    def _record_routing(self, samples: List[Any]) -> None:
        """R2: RECORD the training model's own expert routing for the rollout tokens, in place into encoded.

        Mirrors :meth:`_response_logps`' chunking -- ``forward_only`` is a slice_dp method, so each chunk must hold
        >= dp_size rows -- but asks for ``router_replay_mode='R2'`` (the model layer self-judges the RECORD action
        on this capture pass) and reads back ``routed_experts`` instead of logps. It runs BEFORE any training update, so the routing is captured under the weights that just
        generated these tokens. The returned whole-sequence routing is sliced back to each row's own ``input_ids``
        length, so the REPLAY forward's ``align_routed_experts`` re-pads it exactly as it does the R3 sampler routing.
        """
        features = [sample.input_feature for sample in samples]
        rows: List[Any] = []
        for chunk in self._score_chunks(features):
            forward_kwargs = {'router_replay_mode': 'R2'}
            if self.chord_enabled:
                forward_kwargs['loss_key'] = 'grpo'
            out = self.model.forward_only(inputs=copy_inputs(chunk), **forward_kwargs)
            recorded = out.get('routed_experts') if isinstance(out, dict) else None
            if recorded is None:
                raise RuntimeError(
                    'router_replay_mode="R2" asked the training model to RECORD its MoE routing, but the '
                    'forward returned none. The policy must be an MoE model built with routing replay '
                    'enabled; a dense policy has no expert routing to record (use "disabled").')
            # A uniform-sequence RECORD returns one [n, seq, layers, topk] tensor; a variable-sequence one returns a
            # per-microbatch list of them. Flatten either to row-major order (submission order).
            for tensor in (recorded if isinstance(recorded, (list, tuple)) else [recorded]):
                tensor = torch.as_tensor(tensor).detach().cpu()
                for i in range(tensor.shape[0]):
                    rows.append(tensor[i])
        if len(rows) != len(samples):
            raise RuntimeError(f'R2 routing RECORD returned {len(rows)} rows for {len(samples)} samples; '
                               'per-row routing alignment is impossible.')
        for sample, routed in zip(samples, rows):
            length = len(sample.encoded['input_ids'])
            if routed.shape[0] < length:
                raise RuntimeError(f'R2 recorded routing covers {routed.shape[0]} tokens for a {length}-token '
                                   'sequence; the RECORD forward must route the whole sequence.')
            sample.encoded['routed_experts'] = routed[:length]
            # Replay only the causal rows that produce response logprobs (labels != -100) -- the same set
            # completion_mask encodes. R2's RECORD covers the whole sequence, so there is no short-by-one tail.
            sample.encoded['replay_mask'] = sample.encoded['completion_mask']

    def _score_chunks(self, features: List[dict]) -> List[List[dict]]:
        """Split ``features`` into ``train_batch_size`` chunks (over :func:`_scoring.score_chunks`).

        Kept for :meth:`_record_routing`, which chunks like a scoring pass but reads back ``routed_experts`` rather
        than per-token logps, so it drives the chunking itself.
        """
        return score_chunks(features, self.train_batch_size)

    @staticmethod
    def _response_positions(feature: dict) -> List[int]:
        return response_positions(feature)


class GroupCompletionHook:
    """The driver's group-completion seam: ``dynamic_sample`` regeneration + ``remax`` greedy baseline.

    Invoked by the driver once ALL ``num_generations`` trajectories of one prompt group are buffered (a组内
    barrier; groups still overlap in the stream). It returns a :class:`GroupDecision` telling the driver what to do
    with the complete group:

    - ``dynamic_sample``: if the group's weighted-reward variance is zero (every completion scored alike, so the
      group-relative advantage is degenerate), return ``REGENERATE`` -- the driver re-admits the same prompt under
      the same group_key, within ``max_group_regenerations``. The variance test is pure computation over the
      already-scored rewards, so it is safe under colocate.
    - ``remax``: generate + score ONE greedy baseline for the group, stash it on the assembler (read back at
      assemble time), and return ``AUGMENT`` so the driver keeps the group. The greedy pass needs the sampler, so
      the driver builder refuses colocate + remax up front.
    - otherwise ``ACCEPT``.
    """

    def __init__(self, assembler: GRPOAssembler, rl_config: Optional['RLConfig'] = None):
        self._assembler = assembler
        self._rl_config = rl_config

    def __call__(self, group_key: str, records: List['BufferRecord']) -> GroupDecision:
        samples = [record.sample for record in records]
        cfg = self._rl_config
        if cfg is not None and getattr(cfg, 'dynamic_sample', False):
            if self._assembler.group_reward_variance(samples) <= 0.0:
                return GroupDecision.REGENERATE
        if cfg is not None and cfg.advantage_estimator == 'remax':
            value = self._assembler.remax_greedy_baseline(
                getattr(samples[0], 'prompt', None), getattr(samples[0], 'extra', None))
            self._assembler.stash_remax_baseline(group_key, value)
            return GroupDecision.AUGMENT
        return GroupDecision.ACCEPT
