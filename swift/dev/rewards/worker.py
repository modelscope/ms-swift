# Copyright (c) ModelScope Contributors. All rights reserved.
# GRPO's algorithm halves (advantage fusion, teacher reweighting) on twinkle's ScoreWorker.
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from twinkle import remote_class, remote_function
from twinkle_agentic.async_rl.score_worker import ScoreWorker, _resolve


def _prm_plugin(plugin: Any):
    """Adapt swift's PRMScorer plugin to the base worker's four-argument protocol.

    The base passes the resolved SAMPLE (its own row view is forwarded as ``columns``); the plugin
    family historically takes (messages, token ids, decoded, score_steps, decode, columns,
    turn_context) and builds its own multi-turn context off the sample. Both shapes are swift's
    own; this thin shim keeps the base worker free of any plugin knowledge.
    """
    def _score(sample: Any, decode: Any, score_steps: Any, columns: Any = None) -> List[float]:
        messages = getattr(sample, 'messages', None)
        if messages is None:
            messages = list(getattr(sample, 'prompt', None) or [])
        return plugin(
            messages, sample.response_token_ids, sample.decoded, score_steps, decode,
            columns=columns, turn_context=_turn_context(sample))
    return _score


def _turn_context(sample: Any):
    """The MultiTurnContext for a multi-turn sample, else ``None`` (see rewards.prm)."""
    messages = getattr(sample, 'messages', None)
    if messages is None:
        return None
    from .prm import MultiTurnContext
    from twinkle_agentic.rollout import turn_token_spans

    turn_spans = turn_token_spans(sample.input_feature)
    assistant_indices = [i for i, m in enumerate(messages) if m.get('role') == 'assistant']
    if len(assistant_indices) != len(turn_spans):
        raise RuntimeError(
            f'multi-turn PRM scoring found {len(assistant_indices)} assistant turns but '
            f'{len(turn_spans)} trainable token runs; a run-splitting mismatch means an assistant '
            'turn produced no trainable token (an empty generation) or a completion_mask split '
            'one turn in two.')
    return MultiTurnContext(
        messages=messages, turn_spans=turn_spans, assistant_message_indices=assistant_indices)


def _messages_with_token_ids(messages: List[dict], response_ids: List[int],
                             encoded: Dict[str, Any]) -> List[dict]:
    """Splice each assistant turn's exact sampled tokens back into the messages.

    The teacher-distillation view re-encodes under a different prompt but must score the *same*
    tokens the policy produced, so each turn carries its ids verbatim rather than its text (re-
    encoding text can drift to different ids). ``encoded`` is the policy's flat feature: twinkle's
    ``turn_token_spans`` splits ``response_ids`` into one contiguous slice per assistant turn, and
    the newest slice maps to the newest assistant message.
    """
    from twinkle_agentic.rollout import turn_token_spans

    spans = turn_token_spans(encoded)
    turn_token_ids = [list(response_ids[start:end]) for start, end in spans]

    result = [dict(message) for message in messages]
    completion_index = 0
    for message in reversed(result):
        if message.get('role') != 'assistant':
            continue
        if completion_index >= len(turn_token_ids):
            break
        ids = turn_token_ids[-1 - completion_index]
        message['content'] = {'token_ids': ids, 'loss_scale': [1] * len(ids)}
        completion_index += 1
    if completion_index != len(turn_token_ids):
        raise RuntimeError(
            f'multi-turn messages contain {completion_index} generated assistant turns, but the '
            f'trajectory has {len(turn_token_ids)} trainable token runs. A run-splitting mismatch '
            'means an assistant turn produced no trainable token (an empty generation) or a '
            'completion_mask split one turn in two.')
    return result


@remote_class(execute='first')
class RewardWorker(ScoreWorker):
    """Score rollout groups where the data lives: rules, RM handles, judges and the PRM channel.

    Both scoring channels resolve refs, rebuild rows and run scorers in the inherited base-class
    orchestration (the PRM channel calls back into swift's PRMScorer plugin through the adapter
    shim); this subclass keeps only GRPO's training-half algorithms: the convex advantage fusion,
    the privileged teacher view and per-token advantage reweighting.
    """

    def __init__(self,
                 *,
                 orm_scorers: Sequence[Any],
                 prm_scorers: Sequence[Any] = (),
                 prm_step_scorer: Any = None,
                 prm_weights: Optional[Sequence[float]] = None,
                 template: Any = None):
        super().__init__(orm_scorers=orm_scorers, prm_scorers=prm_scorers,
                         prm_step_scorer=(prm_step_scorer if prm_step_scorer is None
                                          else _prm_plugin(prm_step_scorer)),
                         prm_weights=prm_weights, template=template)

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def gae(self, samples: List[Any], scored: Any, reward_config: Any, rl_config: Any
            ) -> Tuple[List[int], List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
        """PPO's GAE anchors from the scoring forwards: returns (kept indices, adv, returns, values, old).

        ``scored`` maps lane names to whole-batch ``forward_only`` handles (``'values'`` / ``'ref'``)
        -- resolved, key-read and row-aligned HERE, never in the driver. The weighted ORM scalar and
        old logps are read off the samples; a missing reference logp row means "no KL penalty"
        (reuses the old logps so every per-token KL term zeroes).
        """
        from swift.dev.utils.scoring import response_positions, response_token_scores
        from twinkle.advantage import GAEAdvantage

        features = [sample.encoded for sample in samples]
        kept = [i for i, sample in enumerate(samples) if response_positions(sample.encoded)]
        if not kept:
            raise RuntimeError('the pulled batch has no response tokens to train on.')
        values_all = response_token_scores(_resolve(scored['values']), features, output_key='logits')
        ref_all = response_token_scores(_resolve(scored['ref']), features, output_key='logps',
                                        allow_missing=True)
        kept_samples = [samples[i] for i in kept]
        matrix = torch.tensor([list(sample.orm_reward) for sample in kept_samples], dtype=torch.float32)
        if matrix.shape[1] == 0:
            rm_scores = [0.0] * len(kept_samples)
        else:
            from swift.dev.rewards.advantage import build_reward_weights

            weight = build_reward_weights(reward_config.orm_weights, matrix.shape[1])
            rm_scores = (matrix * weight.unsqueeze(0)).nansum(dim=1).tolist()
        old_logps = [[float(x) for x in sample.rollout_logps] for sample in kept_samples]
        # the frame rows are in whole-batch order; the kept samples pick theirs by index
        ref_rows = [[float(x) for x in ref_all[i]] if ref_all[i] is not None else old
                    for i, old in zip(kept, old_logps)]
        token_rewards = GAEAdvantage.build_token_rewards(
            rm_scores, [len(old) for old in old_logps],
            old_logps=old_logps, ref_logps=ref_rows, kl_coef=rl_config.kl_coef)
        values = [values_all[i] for i in kept]
        advantages, returns = GAEAdvantage(
            gamma=rl_config.gamma, gae_lambda=rl_config.lam,
            normalize=rl_config.whiten_rewards)(token_rewards, values)
        return kept, advantages.tolist(), returns.tolist(), values, old_logps

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def reward_metrics(self, samples: List[Any], scored: Any, reward_config: Any,
                       rl_config: Any) -> Dict[str, Any]:
        """The reward telemetry for exactly the batch the upcoming optimizer steps train on.

        Eager small dict (a companion of ``gae`` over the same inputs): total token reward and
        completion length per sample, folded the way ``CompletionRewardMetric`` consumes them.
        """
        from swift.dev.utils.scoring import response_positions, response_token_scores
        from twinkle.advantage import GAEAdvantage

        features = [sample.encoded for sample in samples]
        kept = [i for i, sample in enumerate(samples) if response_positions(sample.encoded)]
        ref_all = response_token_scores(_resolve(scored['ref']), features, output_key='logps',
                                        allow_missing=True)
        kept_samples = [samples[i] for i in kept]
        matrix = torch.tensor([list(sample.orm_reward) for sample in kept_samples], dtype=torch.float32)
        if matrix.shape[1] == 0:
            rm_scores = [0.0] * len(kept_samples)
        else:
            from swift.dev.rewards.advantage import build_reward_weights

            weight = build_reward_weights(reward_config.orm_weights, matrix.shape[1])
            rm_scores = (matrix * weight.unsqueeze(0)).nansum(dim=1).tolist()
        old_logps = [[float(x) for x in sample.rollout_logps] for sample in kept_samples]
        # the frame rows are in whole-batch order; the kept samples pick theirs by index
        ref_rows = [[float(x) for x in ref_all[i]] if ref_all[i] is not None else old
                    for i, old in zip(kept, old_logps)]
        token_rewards = GAEAdvantage.build_token_rewards(
            rm_scores, [len(old) for old in old_logps],
            old_logps=old_logps, ref_logps=ref_rows, kl_coef=rl_config.kl_coef)
        return {'total': [float(sum(row)) for row in token_rewards],
                'completion_lengths': [len(old) for old in old_logps]}

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def grpo_advantages(self, samples: List[Any], scored: Any, rl_config: Any,
                        reward_config: Any, global_step: int) -> Tuple[Any, ...]:
        """GRPO's advantage half from the scoring forwards: group-relative adv + PRM / teacher fusion.

        ``scored`` maps lane names to whole-batch ``forward_only`` handles (``'ref'`` / ``'teacher'`` /
        ``'old'``) -- resolved, key-read and row-aligned HERE. The ORM rewards / baselines / PRM
        process read off the samples; ``kl_in_reward`` needs the reference lane. PRM fusion and
        teacher-signal reweighting are mutually exclusive per-token sources and PRM wins (validate
        rejects enabling both). Returns ``(advantages, old_logps, ref_logps, teacher_logps)`` as one
        reference.
        """
        from swift.dev.rewards.advantage import compute_advantages
        from swift.dev.utils.scoring import response_token_scores

        features = [sample.encoded for sample in samples]
        rewards = [sample.orm_reward for sample in samples]
        baselines = ([sample.baseline for sample in samples]
                     if any(sample.baseline is not None for sample in samples) else None)
        prm_process = ([sample.prm_process for sample in samples]
                       if any(sample.prm_process is not None for sample in samples) else None)
        old_logps: List[Any] = [list(sample.rollout_logps) for sample in samples]
        preserve = bool(rl_config.rollout_importance_sampling_mode or rl_config.log_rollout_offpolicy_metrics)
        ref_logps = (response_token_scores(_resolve(scored['ref']), features)
                     if scored.get('ref') is not None else None)
        teacher_logps = (response_token_scores(_resolve(scored['teacher']), features)
                         if scored.get('teacher') is not None else None)
        if preserve:
            old_logps = response_token_scores(_resolve(scored['old']), features)
        kl_in_reward = bool(rl_config.kl_in_reward and rl_config.beta)
        if kl_in_reward and ref_logps is None:
            raise ValueError('kl_in_reward=True requires an active reference model.')
        kl_values = (torch.tensor(self.sequence_kl(old_logps, ref_logps), dtype=torch.float32)
                     if kl_in_reward else None)
        # None (an unparseable judge verdict) rides as nan, exactly as dev recorded it; a no-scorer
        # row (no --orm wired) stays [nan] -- a zero-column tensor cannot carry a group mean.
        reward_matrix = torch.tensor(
            [[float('nan') if value is None or value != value else float(value) for value in row]
             or [float('nan')] for row in rewards], dtype=torch.float32)
        estimator = rl_config.advantage_estimator
        advantages = compute_advantages(
            reward_matrix,
            rl_config.num_generations,
            reward_weights=reward_config.orm_weights,
            advantage_estimator=estimator,
            scale_rewards=rl_config.scale_rewards or 'group',
            kl_in_reward=kl_in_reward,
            beta=float(rl_config.beta or 0.0),
            kl_values=kl_values,
            response_lengths=([len(sample.response_token_ids) for sample in samples]
                              if estimator == 'opo' else None),
            f_norm=(float(rl_config.f_norm) if estimator == 'gpg' else 1.0),
            reward_baselines=(torch.tensor(baselines, dtype=torch.float32) if baselines is not None else None))
        # Fail-loudly on the dead-parameter trap: RLSD/SDAR consume the teacher signal per token, so
        # enabling either without a teacher lane (no --teacher_model, no teacher_prompt column, no
        # adapter-disabled self-teacher) would silently degrade the run to plain GRPO.
        if teacher_logps is None and (rl_config.advantage_reweight == 'rlsd' or rl_config.sdar_loss_coef > 0):
            raise ValueError(
                'advantage_reweight=rlsd / sdar_loss_coef>0 requires a teacher signal (a --teacher_model, a '
                'teacher_prompt column, or the adapter-disabled self-teacher); without one the reweighting '
                'would silently degrade to plain GRPO. Add a teacher, or drop the flag.')
        if prm_process is not None:
            advantages = self.fuse_prm(advantages, prm_process, float(reward_config.prm_orm_mix))
        elif teacher_logps is not None:
            # teacher-signal per-token reweighting; sdar keeps the scalar advantage (the loss reads
            # teacher_logps itself), rlsd / expand turn it per-token on the worker.
            if rl_config.advantage_reweight == 'rlsd':
                if rl_config.rlsd_lambda_warmup_steps > 0 and global_step < rl_config.rlsd_lambda_warmup_steps:
                    lam = rl_config.rlsd_lambda * global_step / rl_config.rlsd_lambda_warmup_steps
                elif (rl_config.rlsd_lambda_decay_steps > 0
                      and global_step >= rl_config.rlsd_lambda_warmup_steps):
                    progress = ((global_step - rl_config.rlsd_lambda_warmup_steps)
                                / rl_config.rlsd_lambda_decay_steps)
                    lam = rl_config.rlsd_lambda * max(1.0 - progress, 0.0)
                else:
                    lam = rl_config.rlsd_lambda
                advantages = self.reweight_advantages(
                    advantages, old_logps, teacher_logps, mode='rlsd', lam=lam,
                    clip_range=rl_config.rlsd_reweight_clip_range, negative_only=rl_config.rlsd_negative_only)
            elif rl_config.sdar_loss_coef <= 0:
                advantages = self.reweight_advantages(
                    advantages, old_logps, teacher_logps, mode='expand',
                    teacher_kl_coef=rl_config.teacher_kl_coef)
        return advantages, old_logps, ref_logps, teacher_logps

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def correction_metrics(self, samples: List[Any], scored: Any) -> Dict[str, float]:
        """Legacy ``rollout_correction/*`` diagnostics: how far the sampler is from the trainer.

        Eager small dict computed from the training-side forward (the ``'old'`` lane of ``scored``)
        and the sampler's own log-probs (read off the samples), BEFORE any optimizer step. ``kl``
        is the direct estimator of ``KL(pi_rollout || pi_training)``; a freshly synced rollout
        should sit near 0.
        """
        from swift.dev.utils.scoring import response_token_scores

        features = [sample.encoded for sample in samples]
        training_logps = response_token_scores(_resolve(scored['old']), features)
        rollout_logps = [sample.rollout_logps for sample in samples]
        train_rows: List[Any] = []
        rollout_rows: List[Any] = []
        for train, rollout in zip(training_logps, rollout_logps):
            if train is None or rollout is None:
                continue
            length = min(len(train), len(rollout))
            if length == 0:
                continue
            train_rows.append(torch.tensor(list(train[:length]), dtype=torch.float32))
            rollout_rows.append(torch.tensor(list(rollout[:length]), dtype=torch.float32))
        if not train_rows:
            return {}
        train = torch.cat(train_rows)
        rollout = torch.cat(rollout_rows)
        log_ratio = train - rollout
        metrics: Dict[str, float] = {
            'kl': float((-log_ratio).mean()),
            'k3_kl': float((torch.exp(torch.clamp(log_ratio, -20.0, 20.0)) - torch.clamp(log_ratio, -20.0, 20.0)
                            - 1.0).clamp(-10.0, 10.0).mean()),
            'chi2_token': float(torch.exp(torch.clamp(log_ratio, -20.0, 20.0)).square().mean() - 1.0),
        }
        train_seq_mean = torch.stack([row.mean() for row in train_rows])
        rollout_seq_mean = torch.stack([row.mean() for row in rollout_rows])
        metrics['training_log_ppl'] = float(-train_seq_mean.mean())
        metrics['rollout_log_ppl'] = float(-rollout_seq_mean.mean())
        metrics['training_ppl'] = float(torch.exp(-train_seq_mean).mean())
        metrics['rollout_ppl'] = float(torch.exp(-rollout_seq_mean).mean())
        log_ppl_diff = rollout_seq_mean - train_seq_mean
        metrics['log_ppl_diff'] = float(log_ppl_diff.mean())
        metrics['log_ppl_abs_diff'] = float(log_ppl_diff.abs().mean())
        metrics['log_ppl_diff_max'] = float(log_ppl_diff.max())
        metrics['log_ppl_diff_min'] = float(log_ppl_diff.min())
        metrics['ppl_ratio'] = float(torch.exp(log_ppl_diff).mean())
        # Sequence-level chi2: rho_geo = exp(mean over the sequence of log(training/rollout)).
        rho_geo = torch.exp(torch.clamp(train_seq_mean - rollout_seq_mean, -20.0, 20.0))
        metrics['chi2_seq'] = float(rho_geo.square().mean() - 1.0)
        return metrics

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def fuse_prm(self, orm_advantages: List[float], prm_process: Any, mix: float) -> List[List[float]]:
        """Convex-blend the per-token process reward with the scalar ORM advantage (per token).

        ``advantage[i] = mix * process[i] + (1 - mix) * orm_advantage`` on every response token --
        ``mix=0`` is standard GRPO byte-for-byte, ``mix=1`` pure PRM. The output is token-scale, so
        it stays a ref and is handed to ``forward_backward`` as the ``advantages`` kwarg.
        """
        process = _resolve(prm_process)
        fused: List[List[float]] = []
        for per_token, orm in zip(process, orm_advantages):
            if per_token is None:
                raise RuntimeError('PRM fusion requires a per-sample process reward; the '
                                   'score_group_prm call must run before fuse_prm.')
            fused.append([mix * step + (1.0 - mix) * float(orm) for step in per_token])
        return fused

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def privileged_features(self, samples: List[Any]) -> List[Any]:
        """Re-encode each response under its privileged ``teacher_prompt`` column -> feature refs.

        Without a ``teacher_prompt`` column the sample's own input feature is returned unchanged;
        with one, the SAME response tokens are re-encoded from the privileged view (the first user
        turn carries the hint), so the teacher grades the student's answer hint-augmented.
        """
        from swift.dev.utils.scoring import encode_privileged_feature

        features = []
        for sample in samples:
            sample = _resolve(sample)
            teacher_prompt = (getattr(sample, 'extra', None) or {}).get('teacher_prompt')
            if not teacher_prompt:
                features.append(sample.input_feature)
                continue
            extra = getattr(sample, 'extra', None) or {}
            response_ids = [int(t) for t in sample.response_token_ids]
            messages = getattr(sample, 'messages', None)
            if messages is None:
                # single-turn: prompt + the assistant turn carrying the sampled ids verbatim
                prompt = getattr(sample, 'prompt', None)
                if prompt is None:
                    raise ValueError('teacher_prompt scoring requires the sample prompt to be stamped '
                                     'by the driver collect step, got None.')
                features.append(
                    encode_privileged_feature(self.template, list(prompt), teacher_prompt, response_ids, extra))
            else:
                # multi-turn: the hint rides the first user turn; every assistant turn is pinned to
                # its exact sampled ids.
                from swift.dev.utils.scoring import encode_privileged_view

                pinned = [dict(message) for message in messages]
                for message in pinned:
                    if message.get('role') == 'user':
                        message['content'] = teacher_prompt
                        break
                else:
                    raise ValueError('teacher_prompt requires a user turn in the prompt to carry the '
                                     'privileged context.')
                features.append(encode_privileged_view(
                    self.template, _messages_with_token_ids(pinned, response_ids, sample.input_feature),
                    response_ids, extra))
        return features

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def reweight_advantages(self,
                            advantages: List[float],
                            old_logps: Any,
                            teacher_logps: Any,
                            *,
                            mode: str = 'expand',
                            lam: float = 0.0,
                            clip_range: float = 5.0,
                            negative_only: bool = False,
                            teacher_kl_coef: float = 0.0) -> List[List[float]]:
        """Turn scalar advantages into per-token ones via the teacher signal (RLSD / SDAR / expand).

        Delegates to ``twinkle.advantage.teacher_signal`` on the resolved logp lists; the per-token
        output is token-scale and stays a ref (the ``advantages`` kwarg of ``forward_backward``).
        """
        from twinkle.advantage.teacher_signal import apply_rlsd_reweight, expand_advantage_to_per_token

        old = _resolve(old_logps)
        teacher = _resolve(teacher_logps)
        mask = torch.zeros((len(advantages), max((len(row) for row in old), default=1)), dtype=torch.bool)
        for index, row in enumerate(old):
            mask[index, :len(row)] = True
        # The twinkle teacher-signal estimators take the per-sequence base advantage as [B] (1-D) and
        # broadcast it to [B, T] themselves (advantages.unsqueeze(1).expand_as(mask)); building [B, 1]
        # here would make unsqueeze(1) a [B, 1, 1] tensor that cannot expand_as the [B, T] mask.
        base = torch.tensor(list(advantages), dtype=torch.float32)
        if mode == 'rlsd':
            out = apply_rlsd_reweight(base, mask, torch.tensor(teacher, dtype=torch.float32),
                                      torch.tensor(old, dtype=torch.float32), lam=lam,
                                      clip_range=clip_range, negative_only=negative_only)
        else:
            out = expand_advantage_to_per_token(base, mask, torch.tensor(teacher, dtype=torch.float32),
                                                torch.tensor(old, dtype=torch.float32),
                                                teacher_kl_coef=teacher_kl_coef)
        return out.tolist()

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def sequence_kl(self, old_logps: Any, ref_logps: Any) -> List[float]:
        """Per-sample sequence KL ``mean(exp(d) - d - 1)`` over resolved logp lists -> ``[N]`` floats."""
        old = _resolve(old_logps)
        ref = _resolve(ref_logps)
        values = []
        for prev, reference in zip(old, ref):
            if len(prev) != len(reference):
                raise RuntimeError(f'reference logps misaligned: old={len(prev)}, ref={len(reference)}.')
            delta = torch.tensor(reference, dtype=torch.float32) - torch.tensor(prev, dtype=torch.float32)
            values.append(float((torch.exp(delta) - delta - 1.0).mean()) if delta.numel() else 0.0)
        return values
