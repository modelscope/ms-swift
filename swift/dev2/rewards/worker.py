# Copyright (c) ModelScope Contributors. All rights reserved.
# GRPO's algorithm halves (advantage fusion, teacher reweighting) on twinkle's ScoreWorker --
# both scoring channels (ORM rows, PRM process rewards) are the base class's protocol now.
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import torch
from twinkle import remote_class, remote_function
from twinkle_agentic.async_rl.score_worker import ScoreWorker, _resolve, _score_rows


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
                raise RuntimeError('PRM fusion requires a per-sample process reward; the score_prm '
                                   'call must run before fuse_prm.')
            fused.append([mix * step + (1.0 - mix) * float(orm) for step in per_token])
        return fused

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def privileged_features(self, samples: List[Any]) -> List[Any]:
        """Re-encode each response under its privileged ``teacher_prompt`` column -> feature refs.

        Without a ``teacher_prompt`` column the sample's own input feature is returned unchanged;
        with one, the SAME response tokens are re-encoded from the teacher's privileged view, so the
        teacher grades the student's answer from a hint-augmented prompt (RLSD/SDAR).
        """
        # TODO(dev2): bridge until the recipe/common batch migrates encode_privileged_view
        from swift.dev.recipe.common.scoring import encode_privileged_view

        features = []
        for sample in samples:
            sample = _resolve(sample)
            teacher_prompt = (getattr(sample, 'extra', None) or {}).get('teacher_prompt')
            if not teacher_prompt:
                features.append(sample.input_feature)
            else:
                extra = getattr(sample, 'extra', None) or {}
                features.append(
                    encode_privileged_view(
                        self.template,
                        _score_rows([sample])[0]['messages'],
                        [int(t) for t in sample.response_token_ids], extra))
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
        base = torch.tensor([[a] for a in advantages], dtype=torch.float32)
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
