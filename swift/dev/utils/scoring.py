# Copyright (c) ModelScope Contributors. All rights reserved.
# Backend-agnostic scoring pieces: the privileged teacher view, reward folds, calibration scores.
from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Sequence

import torch
from twinkle import remote_class, remote_function
from twinkle.data_format import ground_truth
from twinkle.signals import response_positions

_MULTIMODAL_KEYS = ('images', 'videos', 'audios', 'tools', 'objects', 'chat_template_kwargs')


def collect(result: Any) -> Any:
    """Resolve a maybe-lazy remote result to its value.

    A method declared ``lazy_collect=True`` returns a pending handle under ray (``.result()``
    materialises it, raising the worker's exception here) and the plain value when it ran in this
    process. Callers that consume the VALUE, or need a worker-side failure to fail THIS step, route
    through here so they are correct under both.
    """
    return result.result() if hasattr(result, 'result') else result


def copy_inputs(features: Sequence[Any]) -> List[Any]:
    """Defensive-copy a forward's inputs: a forward in this process may pad / annotate them in place.

    Remotely the copies are of references, so the cost is small.
    """
    return [copy.deepcopy(feature) for feature in features]


def encode_privileged_view(
    template: Any,
    messages: List[dict],
    response_ids: List[int],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Encode a privileged teacher view (``messages`` carrying the shared ``response_ids``) and assert alignment.

    Shared by GRPO's RLSD/SDAR teacher view and OPSD's, which differ only in HOW ``messages`` is built
    (multi-turn reconstruction vs. a single privileged user turn) but not in the encode-and-check tail.
    Encoded in ``train`` mode with ``add_eos=False`` so the assistant turn is exactly ``response_ids`` (no
    extra end-of-sequence token), then checked: the encoded feature must expose the SAME number of trainable
    response positions as ``response_ids``. If a template or tokenizer difference changes that count, teacher
    and student would no longer score identical tokens and the alignment would be silently wrong -- so it is
    a hard error, not a truncation.
    """
    if template is None:
        raise ValueError('teacher_prompt requires a template to build the privileged teacher view.')
    extra = extra or {}
    row: Dict[str, Any] = {'messages': messages, 'add_eos': False}
    for key in _MULTIMODAL_KEYS:
        if extra.get(key) is not None:
            row[key] = extra[key]
    teacher_template = copy.copy(template)
    teacher_template.set_mode('train')
    feature = teacher_template.encode(row)
    if len(response_positions(feature)) != len(response_ids):
        raise RuntimeError('teacher_prompt encoding changed the response-token count; teacher and student must '
                           'share a tokenizer and score exactly the same response tokens.')
    return feature


def encode_privileged_feature(template: Any, prompt_messages: List[dict], teacher_prompt: Any,
                              response_ids: List[int], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Encode one single-turn privileged teacher view (RLSD / OPSD share the exact contract).

    Replaces the FIRST user turn's content with the privileged ``teacher_prompt``, appends the
    assistant turn carrying the student's response token ids verbatim, then delegates the
    encode-and-align-check tail to :func:`encode_privileged_view` -- teacher and student must score
    exactly the same response tokens.
    """
    messages = [dict(message) for message in prompt_messages]
    if messages and messages[-1].get('role') == 'assistant':
        messages = messages[:-1]
    for message in messages:
        if message.get('role') == 'user':
            message['content'] = teacher_prompt
            break
    else:
        raise ValueError('teacher_prompt requires a user turn in the prompt to carry the privileged context.')
    messages.append({'role': 'assistant', 'content': list(response_ids)})
    return encode_privileged_view(template, messages, response_ids, extra)


def _messages_with_token_ids(messages: List[dict], response_ids: List[int],
                             encoded: Dict[str, Any]) -> List[dict]:
    """Splice each assistant turn's exact sampled tokens back into the messages.

    The teacher view re-encodes under a different prompt but must score the *same* tokens the
    policy produced, so each turn carries its ids verbatim rather than its text (re-encoding text
    can drift to different ids). ``encoded`` is the policy's flat feature: twinkle's
    ``turn_token_spans`` splits ``response_ids`` into one contiguous slice per assistant turn, and
    the newest slice maps to the newest assistant message.
    """
    from twinkle_agentic.rollout import turn_token_spans

    turn_token_ids = [list(response_ids[start:end]) for start, end in turn_token_spans(encoded)]
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
class PrivilegedView:
    """Re-encode each response under its privileged ``teacher_prompt`` column, for the teacher forward.

    A trajectory without a ``teacher_prompt`` is returned unchanged; with one, the SAME response
    tokens are re-encoded from the privileged view (the first user turn carries the hint), so the
    teacher grades the policy's answer hint-augmented. Single-turn rows go through
    :func:`encode_privileged_feature`, multi-turn rows pin every assistant turn to its sampled ids.
    """

    def __init__(self, *, template: Any):
        self.template = template

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def __call__(self, trajs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        features = []
        for traj in trajs:
            teacher_prompt = ground_truth(traj, 'teacher_prompt') or traj.get('teacher_prompt')
            if not teacher_prompt:
                features.append(traj)
                continue
            response_ids = [int(t) for t in traj['response_token_ids']]
            messages = list(traj.get('messages') or [])
            assistant = [i for i, message in enumerate(messages) if message.get('role') == 'assistant']
            if len(assistant) == 1:
                features.append(encode_privileged_feature(
                    self.template, messages[:assistant[0]], teacher_prompt, response_ids, traj))
                continue
            pinned = [dict(message) for message in messages]
            for message in pinned:
                if message.get('role') == 'user':
                    message['content'] = teacher_prompt
                    break
            else:
                raise ValueError('teacher_prompt requires a user turn in the prompt to carry the '
                                 'privileged context.')
            features.append(encode_privileged_view(
                self.template, _messages_with_token_ids(pinned, response_ids, traj), response_ids, traj))
        return features


def build_reward_weights(reward_weights: Optional[Sequence[float]], n_funcs: int) -> torch.Tensor:
    """Build/validate the per-function weight vector (``None`` -> all-ones).

    Public because the advantage API needs the same validation, so the "weights must match the number
    of reward functions" rule has one implementation.

    Raises:
        ValueError: ``len(reward_weights) != n_funcs``.
    """
    if reward_weights is None:
        return torch.ones(n_funcs, dtype=torch.float32)
    if len(reward_weights) != n_funcs:
        raise ValueError(f'reward_weights length {len(reward_weights)} != number of reward funcs {n_funcs}.')
    return torch.tensor(list(reward_weights), dtype=torch.float32)


def weighted_rewards(rewards_per_func: Any, reward_weights: Optional[List[float]]):
    """Combine the ``[N, n_scorers]`` matrix into one ``[N]`` weighted-sum reward per sample.

    The weights are ``reward_weights`` (``--orm_weights``), defaulting to all-ones so an unweighted channel is
    a plain sum; ``nansum`` keeps a judge's parse-failure ``nan`` from poisoning the total.
    """
    weight_tensor = build_reward_weights(reward_weights, rewards_per_func.shape[1]).to(
        device=rewards_per_func.device, dtype=rewards_per_func.dtype)
    return (rewards_per_func * weight_tensor.unsqueeze(0)).nansum(dim=1)


def accumulate_reward_metric(metric: Any,
                             samples: List[Any],
                             rewards_per_func: Any,
                             *,
                             scorer_names: List[str],
                             reward_weights: Optional[List[float]]) -> None:
    """Feed one batch's rewards + completion lengths into a CompletionRewardMetric.

    Reset first, so the metric reflects exactly the batch the upcoming optimizer step(s) train on. Per-scorer
    columns keep their resolved names; a single-callable fallback resolves no names, so only the weighted
    ``total`` is reported there.
    """
    metric.reset()
    names = scorer_names
    rewards: Dict[str, List[float]] = {}
    for col in range(rewards_per_func.shape[1]):
        if col < len(names):
            rewards[names[col]] = [float(v) for v in rewards_per_func[:, col].tolist()]
    rewards['total'] = [float(v) for v in weighted_rewards(rewards_per_func, reward_weights).tolist()]
    completion_lengths = [sample.token_lengths[1] if sample.token_lengths else 0 for sample in samples]
    metric.accumulate(rewards=rewards, completion_lengths=completion_lengths)


def brier_per_question(probs: "torch.Tensor", target: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    """Per-question Brier score over real option slots (row-wise; the batch form averages these)."""
    diff2 = (probs - target).pow(2) * mask
    n_real = mask.sum(dim=-1).clamp(min=1).to(diff2.dtype)
    return diff2.sum(dim=-1) / n_real


def brier_score(probs: "torch.Tensor", target: "torch.Tensor", mask: "torch.Tensor") -> float:
    """Mean Brier score over real option slots (lower is better)."""
    return float(brier_per_question(probs, target, mask).mean())


def nll_per_question(
    logits: "torch.Tensor", target: "torch.Tensor", mask: "torch.Tensor", temperature: float
) -> "torch.Tensor":
    """Per-question NLL (cross-entropy) of ``softmax(logits / temperature)`` vs target, masked."""
    scaled = logits.float() / max(temperature, 1e-8)
    scaled = scaled.masked_fill(~mask, float("-inf"))
    log_probs = torch.log_softmax(scaled, dim=-1)
    log_probs = torch.nan_to_num(log_probs, nan=-1e4)
    loss = -(target * log_probs).sum(dim=-1)
    n_real = mask.sum(dim=-1).clamp(min=1).to(loss.dtype)
    return loss / n_real


def nll_score(logits: "torch.Tensor", target: "torch.Tensor", mask: "torch.Tensor", temperature: float) -> float:
    """Mean NLL (cross-entropy) of ``softmax(logits / temperature)`` vs target, masked."""
    return float(nll_per_question(logits, target, mask, temperature).mean())


def ece_score(confidences: List[float], correct: List[bool], n_bins: int = 15) -> float:
    """Expected Calibration Error via uniform binning on [0, 1] (lower is better; ~0 = calibrated)."""
    import numpy as np

    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray([1.0 if c else 0.0 for c in correct], dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(conf, bin_edges) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(conf)
    for b in range(n_bins):
        m = bin_idx == b
        cnt = int(m.sum())
        if cnt > 0:
            ece += (cnt / n) * abs(float(conf[m].mean()) - float(corr[m].mean()))
    return float(ece)

