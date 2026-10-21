# Copyright (c) ModelScope Contributors. All rights reserved.
# GRPO's algorithm halves (advantage fusion, teacher reweighting) on twinkle's ScoreWorker.
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
from twinkle import remote_class, remote_function
from twinkle_agentic.async_rl.score_worker import ScoreWorker


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

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def response_indices(self, samples):
        """返回有响应 token 的样本索引，encoded 保留在 worker。"""
        from swift.dev.utils.scoring import response_positions

        indices = [index for index, sample in enumerate(samples) if response_positions(sample.encoded)]
        if not indices:
            raise RuntimeError('the pulled batch has no response tokens to train on.')
        return indices

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def has_teacher_prompt(self, samples):
        """仅返回特权视角是否存在，不把样本 extra 传回 driver。"""
        return any((sample.extra or {}).get('teacher_prompt') for sample in samples)

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def response_scores(self, tasks):
        """多路响应前向：同 placement 串行、不同 placement 线程重叠，返回按任务序的分数。

        每个任务是 ``(placement_key, scorer, features, scoring_kwargs)``，其中 ``scoring_kwargs``
        直接透传给 ``batched_response_scores``（``train_batch_size`` / ``output_key`` /
        ``forward_kwargs`` / ``allow_missing``）。优势方法只消费这里的输出引用，不再接模型句柄。
        """
        from swift.dev.utils.scoring import batched_response_scores
        from twinkle_agentic.async_rl.score_worker import _run_grouped

        return _run_grouped([
            (placement, lambda scorer=scorer, features=features, kwargs=dict(kwargs):
             batched_response_scores(scorer, features, **kwargs))
            for placement, scorer, features, kwargs in tasks])

    @staticmethod
    def _token_rewards(samples, ref_logps, rl_config, reward_config):
        """含 KL 惩罚的逐 token 奖励（GAE 与批内 reward 指标共用同一构造）。"""
        from swift.dev.utils.scoring import build_reward_weights
        from twinkle.advantage import GAEAdvantage

        matrix = torch.tensor([list(sample.orm_reward) for sample in samples], dtype=torch.float32)
        weights = build_reward_weights(reward_config.orm_weights, matrix.shape[1])
        rewards = (matrix * weights.unsqueeze(0)).nansum(dim=1).tolist()
        old = [list(sample.rollout_logps) for sample in samples]
        refs = [list(ref) if ref is not None else prev for ref, prev in zip(ref_logps, old)]
        token_rewards = GAEAdvantage.build_token_rewards(
            rewards, [len(row) for row in old], old_logps=old, ref_logps=refs, kl_coef=rl_config.kl_coef)
        return token_rewards, old

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def gae(self, samples, values, ref_logps, rl_config, reward_config):
        """在已筛选样本及其数值信号上计算 token 级 GAE，返回纯训练信号引用。"""
        from twinkle.advantage import GAEAdvantage

        if len(values) != len(samples) or len(ref_logps) != len(samples):
            raise ValueError('values、ref_logps 必须与已筛选样本按行对齐。')
        token_rewards, old = self._token_rewards(samples, ref_logps, rl_config, reward_config)
        advantages, returns = GAEAdvantage(gamma=rl_config.gamma, gae_lambda=rl_config.lam,
                                           normalize=rl_config.whiten_rewards)(token_rewards, values)
        return {'advantages': advantages.tolist(), 'returns': returns.tolist(), 'old_logps': old}

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def reward_metrics(self, samples, ref_logps, rl_config, reward_config):
        """批内 reward 总和与 completion 长度（小控制量，直接返回真实 dict）。"""
        token_rewards, old = self._token_rewards(samples, ref_logps, rl_config, reward_config)
        return {'rewards': {'total': [float(sum(row)) for row in token_rewards]},
                'completion_lengths': [len(row) for row in old]}

    @staticmethod
    def _rollout_correction_metrics(training_logps, rollout_logps):
        """保持 legacy 的逐 token 与逐序列 off-policy 诊断公式。"""
        train_rows, rollout_rows = [], []
        for train, rollout in zip(training_logps, rollout_logps):
            if train is None or rollout is None:
                continue
            length = min(len(train), len(rollout))
            if length:
                train_rows.append(torch.tensor(train[:length], dtype=torch.float32))
                rollout_rows.append(torch.tensor(rollout[:length], dtype=torch.float32))
        if not train_rows:
            return {}
        ratio = torch.cat(train_rows) - torch.cat(rollout_rows)
        clipped = torch.clamp(ratio, -20.0, 20.0)
        train_mean = torch.stack([row.mean() for row in train_rows])
        rollout_mean = torch.stack([row.mean() for row in rollout_rows])
        diff = rollout_mean - train_mean
        return {'kl': float((-ratio).mean()),
                'k3_kl': float((torch.exp(clipped) - clipped - 1.0).clamp(-10.0, 10.0).mean()),
                'chi2_token': float(torch.exp(clipped).square().mean() - 1.0),
                'training_log_ppl': float(-train_mean.mean()), 'rollout_log_ppl': float(-rollout_mean.mean()),
                'training_ppl': float(torch.exp(-train_mean).mean()),
                'rollout_ppl': float(torch.exp(-rollout_mean).mean()),
                'log_ppl_diff': float(diff.mean()), 'log_ppl_abs_diff': float(diff.abs().mean()),
                'log_ppl_diff_max': float(diff.max()), 'log_ppl_diff_min': float(diff.min()),
                'ppl_ratio': float(torch.exp(diff).mean()),
                'chi2_seq': float(torch.exp(torch.clamp(-diff, -20.0, 20.0)).square().mean() - 1.0)}

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def grpo_advantages(self, samples, rl_config, reward_config, global_step, prm_orm_mix,
                        *, old_logps=None, ref_logps=None, teacher_logps=None):
        """消费数值信号引用，完成优势估计及 PRM/teacher 融合，返回纯训练信号。"""
        from swift.dev.rewards.advantage import compute_advantages

        rewards = [sample.orm_reward for sample in samples]
        baselines = [sample.baseline for sample in samples]
        if all(value is None for value in baselines):
            baselines = None
        processes = [sample.prm_process for sample in samples]
        if all(value is None for value in processes):
            processes = None
        old = [list(sample.rollout_logps) for sample in samples] if old_logps is None else old_logps
        if len(old) != len(samples):
            raise ValueError('old_logps 必须与样本按行对齐。')
        kl_in_reward = bool(rl_config.kl_in_reward and rl_config.beta)
        if kl_in_reward and ref_logps is None:
            raise ValueError('kl_in_reward=True requires reference logps.')
        kl = torch.tensor(self.sequence_kl(old, ref_logps), dtype=torch.float32) if kl_in_reward else None
        matrix = torch.tensor([[float('nan') if value is None or value != value else float(value)
                                for value in row] or [float('nan')] for row in rewards], dtype=torch.float32)
        estimator = rl_config.advantage_estimator
        advantages = compute_advantages(
            matrix, rl_config.num_generations, reward_weights=reward_config.orm_weights,
            advantage_estimator=estimator, scale_rewards=rl_config.scale_rewards or 'group',
            kl_in_reward=kl_in_reward, beta=float(rl_config.beta or 0.0), kl_values=kl,
            response_lengths=[len(s.response_token_ids) for s in samples] if estimator == 'opo' else None,
            f_norm=float(rl_config.f_norm) if estimator == 'gpg' else 1.0,
            reward_baselines=torch.tensor(baselines, dtype=torch.float32) if baselines is not None else None)
        if processes is not None:
            advantages = self.fuse_prm(advantages, processes, float(prm_orm_mix))
        elif teacher_logps is not None:
            if rl_config.advantage_reweight == 'rlsd':
                lam = rl_config.rlsd_lambda
                if rl_config.rlsd_lambda_warmup_steps > 0 and global_step < rl_config.rlsd_lambda_warmup_steps:
                    lam *= global_step / rl_config.rlsd_lambda_warmup_steps
                elif rl_config.rlsd_lambda_decay_steps > 0 and global_step >= rl_config.rlsd_lambda_warmup_steps:
                    lam *= max(1.0 - (global_step - rl_config.rlsd_lambda_warmup_steps)
                               / rl_config.rlsd_lambda_decay_steps, 0.0)
                advantages = self.reweight_advantages(
                    advantages, old, teacher_logps, mode='rlsd', lam=lam,
                    clip_range=rl_config.rlsd_reweight_clip_range, negative_only=rl_config.rlsd_negative_only)
            elif rl_config.sdar_loss_coef <= 0:
                advantages = self.reweight_advantages(advantages, old, teacher_logps, mode='expand',
                                                      teacher_kl_coef=rl_config.teacher_kl_coef)
        return {'advantages': advantages, 'old_logps': old}

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def correction_metrics(self, old_logps, samples):
        """重算 training logps 与 rollout logps 的 off-policy 诊断（小控制量，直接返回真实 dict）。"""
        return self._rollout_correction_metrics(old_logps, [s.rollout_logps for s in samples])

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def prepare_routing_replay(self, model, samples, mode, train_batch_size):
        """校验 R3 或记录 R2，返回带 routing replay 的 feature 引用。"""
        from swift.dev.utils.scoring import copy_inputs, score_chunks
        features = [sample.encoded for sample in samples]
        if mode == 'R3':
            missing = sum(feature.get('routed_experts') is None for feature in features)
            if missing:
                raise RuntimeError(f'R3 缺少 {missing} 条样本的 routed_experts；使用 R2 记录或 disabled。')
            return features
        if mode != 'R2':
            raise ValueError(f'Unknown router_replay_mode {mode!r}; expected R2/R3.')
        rows = []
        for chunk in score_chunks(features, train_batch_size):
            out = model.forward_only(inputs=copy_inputs(chunk), router_replay_mode='R2')
            recorded = out.get('routed_experts') if isinstance(out, dict) else None
            if recorded is None:
                raise RuntimeError('R2 未返回 routed_experts；请使用启用 routing replay 的 MoE 模型。')
            for tensor in recorded if isinstance(recorded, (list, tuple)) else [recorded]:
                rows.extend(tensor[index] for index in range(tensor.shape[0]))
        if len(rows) != len(features):
            raise RuntimeError('R2 routing 行数与样本数不一致。')
        for feature, routed in zip(features, rows):
            length = len(feature['input_ids'])
            if routed.shape[0] < length:
                raise RuntimeError('R2 routing 未覆盖整个输入序列。')
            feature['routed_experts'] = routed[:length]
            feature['replay_mask'] = feature['completion_mask']
        return features

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def fuse_prm(self, orm_advantages: List[float], prm_process: Any, mix: float) -> List[List[float]]:
        """Convex-blend the per-token process reward with the scalar ORM advantage (per token).

        ``advantage[i] = mix * process[i] + (1 - mix) * orm_advantage`` on every response token --
        ``mix=0`` is standard GRPO byte-for-byte, ``mix=1`` pure PRM. The output is token-scale, so
        it stays a ref and is handed to ``forward_backward`` as the ``advantages`` kwarg.
        """
        process = prm_process
        if len(process) != len(orm_advantages):
            raise ValueError('PRM 和 ORM 必须按样本等长对齐。')
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

        old = old_logps
        teacher = teacher_logps
        if len(old) != len(advantages) or len(teacher) != len(old):
            raise ValueError('advantages、old_logps 和 teacher_logps 必须按样本等长对齐。')
        if any(len(left) != len(right) for left, right in zip(old, teacher)):
            raise ValueError('teacher_logps 必须与每条响应的 old_logps 等长。')
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
        old = old_logps
        ref = ref_logps
        if len(old) != len(ref):
            raise ValueError('reference logps 必须按样本等长对齐。')
        values = []
        for prev, reference in zip(old, ref):
            if len(prev) != len(reference):
                raise RuntimeError(f'reference logps misaligned: old={len(prev)}, ref={len(reference)}.')
            delta = torch.tensor(reference, dtype=torch.float32) - torch.tensor(prev, dtype=torch.float32)
            values.append(float((torch.exp(delta) - delta - 1.0).mean()) if delta.numel() else 0.0)
        return values
