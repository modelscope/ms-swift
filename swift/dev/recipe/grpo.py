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

Weight-sync is delegated to the rollout object, not owned by the loop: when the rollout exposes
``sync_weights`` (+ optional ``finish_generate``) -- as ``run_grpo``'s ``SamplerRollout`` does over
twinkle's ``CheckpointEngineManager`` -- the loop pushes the trained policy into the sampler BEFORE
each rollout, so the behaviour policy tracks the trained one (correct GRPO). The local
``RolloutEngine`` smoke path has no such hook, so the loop simply skips it and stays a pipeline smoke
on the INITIAL weights.

Advanced paths are explicit in this loop: frozen-reference KL, DAPO dynamic sampling,
RLSD/SDAR teacher scoring, CHORD auxiliary SFT, and periodic reference synchronization.
"""
from __future__ import annotations
import copy
import math
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

import json

from swift.dev.advantage import compute_advantages
from swift.dev.reward import (
    build_reward_weights,
    compute_reward_model_scores,
    compute_rewards_per_func,
    get_reward_funcs,
)
from swift.dev.utils import get_logger

if TYPE_CHECKING:
    from swift.dev.config import LoggingConfig, RLHFConfig
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


class _RemoteGRPOTeacher:
    """Sampled-token log-prob teacher backed by one or more vLLM ``/infer/`` servers."""

    def __init__(self, server_spec: str, *, client_factory=None):
        from swift.rlhf_trainers.gkd_helpers import parse_teacher_model_server

        self.configs = parse_teacher_model_server(server_spec)
        if client_factory is None:
            from swift.rlhf_trainers.vllm_client import VLLMInferClient

            def client_factory(url):
                return VLLMInferClient(base_urls=[url])
        self.clients = [client_factory(config.url) for config in self.configs]

    def _routing(self, samples: List[Any], tag_key: str) -> Dict[int, List[int]]:
        if len(self.configs) == 1:
            return {0: list(range(len(samples)))}
        tag_to_teacher = {tag: index for index, config in enumerate(self.configs) for tag in config.tags}
        routing = {index: [] for index in range(len(self.configs))}
        for sample_index, sample in enumerate(samples):
            value = (getattr(sample, 'extra', None) or {}).get(tag_key)
            tag = str(value) if value is not None else None
            teacher_index = tag_to_teacher.get(tag)
            if teacher_index is None:
                raise ValueError(f'GRPO sample[{sample_index}] tag {tag!r} from {tag_key!r} matches no teacher.')
            routing[teacher_index].append(sample_index)
        return routing

    @staticmethod
    def _response_ids(sample: Any) -> List[int]:
        return [int(token) for turn in sample.response_token_ids for token in turn]

    def _requests(self, samples: List[Any], prompts: List[List[dict]], template: Any) -> List[Any]:
        from swift.infer_engine.protocol import RolloutInferRequest
        from swift.rlhf_trainers.utils import get_response_prefix_ids, replace_assistant_response_with_ids

        request_fields = ('images', 'audios', 'videos', 'tools', 'objects', 'chat_template_kwargs')
        requests = []
        for sample in samples:
            try:
                prompt_index = int(sample.prompt_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f'Remote GRPO teacher requires a numeric prompt_id, got {sample.prompt_id!r}.') from exc
            messages = copy.deepcopy(prompts[prompt_index])
            extra = getattr(sample, 'extra', None) or {}
            teacher_prompt = extra.get('teacher_prompt')
            if teacher_prompt:
                for message in reversed(messages):
                    if message.get('role') == 'user':
                        message['content'] = teacher_prompt
                        break
            messages.append({'role': 'assistant', 'content': None})
            response_ids = self._response_ids(sample)
            if not response_ids:
                raise ValueError('Remote GRPO teacher cannot score an empty completion.')
            chat_template_kwargs = extra.get('chat_template_kwargs') or {}
            prefix_ids = (get_response_prefix_ids(
                template, sample_enable_thinking=chat_template_kwargs.get('enable_thinking'))
                          if template is not None else None)
            messages = replace_assistant_response_with_ids(
                messages, response_ids, non_thinking_prefix_ids=prefix_ids)
            kwargs = {name: copy.deepcopy(extra[name]) for name in request_fields if extra.get(name) is not None}
            requests.append(RolloutInferRequest(messages=messages, **kwargs))
        return requests

    def score(self, samples: List[Any], prompts: List[List[dict]], template: Any, tag_key: str) -> List[List[float]]:
        import torch

        from swift.infer_engine import RequestConfig
        from swift.rlhf_trainers.gkd_helpers import assemble_teacher_completion_logprobs
        from swift.rlhf_trainers.utils import parse_prompt_logprobs

        requests = self._requests(samples, prompts, template)
        parsed: List[Any] = [None] * len(samples)
        request_config = RequestConfig(prompt_logprobs=0, max_tokens=1, temperature=0.0)
        for teacher_index, sample_indices in self._routing(samples, tag_key).items():
            if not sample_indices:
                continue
            subset = [requests[index] for index in sample_indices]
            responses = self.clients[teacher_index].infer(subset, request_config=request_config, use_tqdm=False)
            if len(responses) != len(subset):
                raise RuntimeError(f'Teacher server returned {len(responses)} responses for {len(subset)} requests.')
            for sample_index, response in zip(sample_indices, responses):
                parsed[sample_index] = parse_prompt_logprobs(response, topk=0)
        if any(item is None for item in parsed):
            raise RuntimeError('Teacher server routing left one or more GRPO samples unscored.')

        response_ids = [self._response_ids(sample) for sample in samples]
        max_length = max(len(ids) for ids in response_ids)
        completion_mask = torch.zeros((len(samples), max_length), dtype=torch.bool)
        for index, ids in enumerate(response_ids):
            completion_mask[index, :len(ids)] = True
        teacher_output = assemble_teacher_completion_logprobs(
            parsed, completion_mask, torch.device('cpu'), response_token_ids=response_ids)
        return [teacher_output.topk_logprobs[index, :len(ids), 0].tolist()
                for index, ids in enumerate(response_ids)]


class GRPOLoop:
    """Online GRPO loop with reference, teacher, DAPO, RLSD/SDAR and CHORD support."""

    def __init__(self,
                 model: TrainableModel,
                 rollout_engine: Any,
                 prompts: List[List[dict]],
                 *,
                 prompt_extras: Optional[List[Dict[str, Any]]] = None,
                 reference: Any = None,
                 teacher: Any = None,
                 template: Any = None,
                 teacher_tag_key: str = 'dataset',
                 chord_features: Optional[List[dict]] = None,
                 num_generations: int = 4,
                 reward_funcs: Optional[List[Any]] = None,
                 reward_model_plugins: Optional[List[Callable]] = None,
                 reward_model_names: Optional[List[str]] = None,
                 reward_weights: Optional[List[float]] = None,
                 advantage_estimator: str = 'grpo',
                 scale_rewards: str = 'group',
                 rlhf_config: Optional['RLHFConfig'] = None,
                 reward_fn: Callable[[Any], float] = toy_length_reward,
                 max_steps: int = 3,
                 gradient_accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 sampling_params: Optional[dict] = None,
                 logging_config: Optional['LoggingConfig'] = None,
                 output_dir: str = 'output',
                 save_steps: Optional[int] = None,
                 no_save_optim: bool = False,
                 no_save_rng: bool = False,
                 safe_serialization: bool = True,
                 max_shard_size: str = '5GB',
                 save_total_limit: Optional[int] = None,
                 manual_gc: bool = False,
                 manual_gc_steps: int = 0):
        self.model = model
        self.rollout = rollout_engine
        self.prompts = prompts
        self.prompt_extras = ([{} for _ in prompts] if prompt_extras is None else prompt_extras)
        if len(self.prompt_extras) != len(prompts):
            raise ValueError('prompt_extras must contain exactly one mapping per prompt.')
        self.reference = reference
        self.teacher = teacher
        self.template = template
        self.teacher_tag_key = teacher_tag_key
        self.chord_features = chord_features or []
        self._chord_offset = 0
        self.num_generations = num_generations
        self.advantage_estimator = advantage_estimator
        self.scale_rewards = scale_rewards
        self.reward_weights = reward_weights
        self.rlhf_config = rlhf_config
        self.reward_funcs, self.reward_func_names = (
            get_reward_funcs(reward_funcs, rlhf_config) if reward_funcs else ([], []))
        self.reward_model_plugins = list(reward_model_plugins or [])
        model_names = list(reward_model_names or [])
        if model_names and len(model_names) != len(self.reward_model_plugins):
            raise ValueError('reward_model_names must align one-to-one with reward_model_plugins.')
        self.reward_func_names.extend(model_names or [type(plugin).__name__ for plugin in self.reward_model_plugins])
        if rlhf_config is not None and rlhf_config.use_gym_env:
            self.reward_func_names.append('gym')
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.sampling_params = sampling_params
        self.logging_config = logging_config
        from swift.dev.recipe.tracking import RunTracker
        self.tracker = RunTracker(logging_config, output_dir)
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.no_save_optim = no_save_optim
        self.no_save_rng = no_save_rng
        self.safe_serialization = safe_serialization
        self.max_shard_size = max_shard_size
        self.save_total_limit = save_total_limit
        self.manual_gc = manual_gc
        self.manual_gc_steps = manual_gc_steps
        if self.manual_gc_steps < 0:
            raise ValueError('manual_gc_steps must be >= 0.')
        self.global_step = 0
        self.micro_step = 0
        self.history: list = []

    def _active_group(self):
        return self.model.optimizer_group[self.model._get_default_group()]

    def _generate(self, prompt_indices: Sequence[int]) -> List[Any]:
        prompts = [self.prompts[i] for i in prompt_indices]
        extras = [self.prompt_extras[i] for i in prompt_indices]
        if hasattr(self.rollout, 'sync_weights'):
            self.rollout.sync_weights()
        try:
            samples = self.rollout.generate(
                prompts,
                num_samples=self.num_generations,
                sampling_params=self.sampling_params,
                prompt_extras=extras)
        finally:
            if hasattr(self.rollout, 'finish_generate'):
                self.rollout.finish_generate()
        expected = len(prompt_indices) * self.num_generations
        if len(samples) != expected:
            raise RuntimeError(f'rollout returned {len(samples)} samples, expected {expected} '
                               f'({len(prompt_indices)} prompts * {self.num_generations} generations).')
        for local_idx, prompt_idx in enumerate(prompt_indices):
            for sample in samples[local_idx * self.num_generations:(local_idx + 1) * self.num_generations]:
                sample.prompt_id = str(prompt_idx)
        return samples

    def _reward_rows(self, samples: List[Any]) -> List[Dict[str, Any]]:
        """Rebuild complete conversations for RM plugins and retain dataset-side reward columns."""
        rows = []
        for sample in samples:
            messages = getattr(sample, 'messages', None)
            if messages is None:
                try:
                    prompt_index = int(sample.prompt_id)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f'reward model scoring requires a numeric prompt_id, got {sample.prompt_id!r}.') from exc
                messages = copy.deepcopy(self.prompts[prompt_index])
                messages.append({'role': 'assistant', 'content': sample.decoded})
            row = copy.deepcopy(getattr(sample, 'extra', None) or {})
            row['messages'] = copy.deepcopy(messages)
            rows.append(row)
        return rows

    def _score(self, samples: List[Any]):
        import torch

        reward_parts = []
        if self.reward_funcs:
            completions = [sample.decoded for sample in samples]
            column_keys = {key for sample in samples for key in (getattr(sample, 'extra', None) or {})}
            columns = {
                key: [(getattr(sample, 'extra', None) or {}).get(key) for sample in samples]
                for key in column_keys
            }
            reward_parts.append(compute_rewards_per_func(completions, self.reward_funcs, columns))
        if self.reward_model_plugins:
            reward_parts.append(compute_reward_model_scores(self._reward_rows(samples), self.reward_model_plugins))
        if getattr(getattr(self, 'rlhf_config', None), 'use_gym_env', False):
            gym_rewards = []
            for index, sample in enumerate(samples):
                rollout_infos = getattr(sample, 'rollout_infos', None) or {}
                if 'total_reward' not in rollout_infos:
                    raise RuntimeError(
                        f'gym rollout sample[{index}] has no total_reward; the configured scheduler must expose it.')
                gym_rewards.append([float(rollout_infos['total_reward'])])
            reward_parts.append(torch.tensor(gym_rewards, dtype=torch.float32))
        if reward_parts:
            return torch.cat(reward_parts, dim=1)
        return torch.tensor([[self.reward_fn(sample)] for sample in samples], dtype=torch.float32)

    def _weighted_rewards(self, rewards_per_func):
        import torch

        weight_tensor = build_reward_weights(self.reward_weights, rewards_per_func.shape[1]).to(
            device=rewards_per_func.device, dtype=rewards_per_func.dtype)
        return (rewards_per_func * weight_tensor.unsqueeze(0)).nansum(dim=1)

    def _dynamic_rollout(self):
        """Replace zero-variance prompt groups, falling back atomically if retries are exhausted."""
        import torch

        prompt_indices = list(range(len(self.prompts)))
        original_samples = self._generate(prompt_indices)
        original_rewards = self._score(original_samples)
        if not (self.rlhf_config and self.rlhf_config.dynamic_sample):
            return original_samples, original_rewards

        accepted: Dict[int, tuple] = {}
        pending = prompt_indices
        samples = original_samples
        rewards = original_rewards
        max_retries = self.rlhf_config.max_resample_times
        for attempt in range(max_retries + 1):
            weighted = self._weighted_rewards(rewards)
            next_pending = []
            for local_idx, prompt_idx in enumerate(pending):
                start = local_idx * self.num_generations
                end = start + self.num_generations
                group_rewards = weighted[start:end]
                if group_rewards.numel() > 1 and torch.std(group_rewards, unbiased=False) > 0:
                    accepted[prompt_idx] = (samples[start:end], rewards[start:end])
                else:
                    next_pending.append(prompt_idx)
            if not next_pending:
                ordered_samples = []
                ordered_rewards = []
                for prompt_idx in prompt_indices:
                    group_samples, group_rewards = accepted[prompt_idx]
                    ordered_samples.extend(group_samples)
                    ordered_rewards.append(group_rewards)
                return ordered_samples, torch.cat(ordered_rewards, dim=0)
            if attempt == max_retries:
                break
            pending = next_pending
            samples = self._generate(pending)
            rewards = self._score(samples)

        logger.warning('Dynamic sampling still has zero-variance groups after %s retries; using the original batch.',
                       max_retries)
        return original_samples, original_rewards

    @staticmethod
    def _response_positions(feature: dict) -> List[int]:
        labels = feature.get('labels')
        if labels is None:
            return []
        return [idx for idx, label in enumerate(labels) if int(label) != -100]

    def _response_logps(self, owner: Any, feature: dict, *, disable_lora: bool = False) -> List[float]:
        import torch

        forward_owner = self.model if owner == 'disable_lora' else owner
        kwargs = {'disable_lora': True} if owner == 'disable_lora' or disable_lora else {}
        out = forward_owner.forward_only(inputs=[copy.deepcopy(feature)], **kwargs)
        logps = out.get('logps') if isinstance(out, dict) else None
        if logps is None:
            raise RuntimeError('reference/teacher forward returned no logps.')
        values = torch.as_tensor(logps).detach().float().cpu().reshape(-1)
        positions = self._response_positions(feature)
        if values.numel() == len(positions):
            return values.tolist()
        labels = feature.get('labels')
        feature_length = len(labels) if labels is not None else 0
        if values.numel() < feature_length:
            raise RuntimeError(f'forward logps have {values.numel()} values for a {feature_length}-token feature; '
                               'response-token alignment is impossible.')
        return values[positions].tolist()

    def _teacher_messages(self, sample: Any, prompt_idx: int, teacher_prompt: str) -> tuple[List[dict], List[int]]:
        is_multi_turn = getattr(sample, 'messages', None) is not None
        messages = copy.deepcopy(sample.messages if is_multi_turn else self.prompts[prompt_idx])
        for message in messages:
            if message.get('role') == 'user':
                message['content'] = teacher_prompt
                break
        response_ids = [token for turn in sample.response_token_ids for token in turn]
        if not is_multi_turn:
            messages.append({'role': 'assistant', 'content': response_ids})
            return messages, response_ids

        from swift.dev.rollout.multi_turn import _messages_with_token_ids
        response_loss_mask = getattr(sample, 'response_loss_mask', None)
        if not response_loss_mask:
            response_loss_mask = [[1] * len(turn) for turn in sample.response_token_ids]
        return _messages_with_token_ids(messages, sample.response_token_ids, response_loss_mask), response_ids

    def _teacher_feature(self, sample: Any) -> dict:
        teacher_prompt = (getattr(sample, 'extra', None) or {}).get('teacher_prompt')
        if not teacher_prompt:
            return sample.input_feature
        if self.template is None:
            raise ValueError('teacher_prompt requires a template to build the privileged teacher view.')
        try:
            prompt_idx = int(sample.prompt_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'teacher_prompt requires a numeric prompt_id, got {sample.prompt_id!r}.') from exc
        messages, response_ids = self._teacher_messages(sample, prompt_idx, teacher_prompt)
        row = {'messages': messages, 'add_eos': False}
        for key in ('images', 'videos', 'audios', 'tools', 'objects', 'chat_template_kwargs'):
            if key in sample.extra:
                row[key] = sample.extra[key]
        teacher_template = copy.copy(self.template)
        teacher_template.set_mode('train')
        feature = teacher_template.encode(row)
        if len(self._response_positions(feature)) != len(response_ids):
            raise RuntimeError('teacher_prompt encoding changed the sampled response-token count; teacher and policy '
                               'must share a tokenizer and score exactly the same response tokens.')
        return feature

    def _reference_logps(self, samples: List[Any]) -> Optional[List[List[float]]]:
        if self.reference is None:
            return None
        return [self._response_logps(self.reference, sample.input_feature) for sample in samples]

    def _teacher_logps(self, samples: List[Any]) -> Optional[List[List[float]]]:
        has_privileged_prompt = any((sample.extra or {}).get('teacher_prompt') for sample in samples)
        needs_teacher = bool(self.teacher is not None or has_privileged_prompt)
        advanced = bool(self.rlhf_config and
                        (self.rlhf_config.advantage_reweight == 'rlsd' or self.rlhf_config.sdar_loss_coef > 0))
        if not needs_teacher:
            if advanced:
                raise ValueError('RLSD/SDAR requires teacher_model or a teacher_prompt dataset column.')
            return None
        if isinstance(self.teacher, _RemoteGRPOTeacher):
            return self.teacher.score(samples, self.prompts, self.template, self.teacher_tag_key)
        owner = self.teacher if self.teacher is not None else self.model
        return [self._response_logps(owner, self._teacher_feature(sample)) for sample in samples]

    @staticmethod
    def _sequence_kl(old_logps: List[List[float]], ref_logps: List[List[float]]):
        import torch

        values = []
        for old, ref in zip(old_logps, ref_logps):
            if len(old) != len(ref):
                raise RuntimeError(f'reference logps misaligned: old={len(old)}, ref={len(ref)}.')
            delta = torch.tensor(ref, dtype=torch.float32) - torch.tensor(old, dtype=torch.float32)
            values.append((torch.exp(delta) - delta - 1.0).mean() if delta.numel() else torch.tensor(0.0))
        return torch.stack(values)

    def _log_completions(self, samples: List[Any], rewards_per_func) -> None:
        cfg = self.rlhf_config
        if not cfg or not cfg.log_completions:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, 'completions.jsonl')
        rewards = rewards_per_func.detach().float().cpu().tolist()
        with open(path, 'a', encoding='utf-8') as stream:
            for sample, per_func in zip(samples, rewards):
                prompt_index = int(sample.prompt_id)
                stream.write(json.dumps({
                    'step': self.global_step + 1,
                    'prompt': self.prompts[prompt_index],
                    'completion': sample.decoded,
                    'rewards': per_func,
                }, ensure_ascii=False) + '\n')

    def _rollout_step(self):
        samples, rewards_per_func = self._dynamic_rollout()
        self.tracker.log_prompts(self.prompts, samples, self.global_step + 1, self.num_generations)
        self._log_completions(samples, rewards_per_func)
        ref_logps = self._reference_logps(samples)
        teacher_logps = self._teacher_logps(samples)
        cfg = self.rlhf_config
        preserve_policy_logps = bool(cfg and (
            cfg.rollout_importance_sampling_mode or cfg.log_rollout_offpolicy_metrics))
        old_logps = ([self._response_logps(self.model, sample.input_feature) for sample in samples]
                     if preserve_policy_logps else [sample.old_logps for sample in samples])
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
            kl_values=kl_values)
        return [{
            'sample': sample,
            'advantage': advantage,
            'old_logps': old_logps[idx],
            'rollout_logps': sample.old_logps,
            'ref_logps': ref_logps[idx] if ref_logps is not None else None,
            'teacher_logps': teacher_logps[idx] if teacher_logps is not None else None,
        } for idx, (sample, advantage) in enumerate(zip(samples, advantages))]

    def _rlsd_lambda(self) -> float:
        cfg = self.rlhf_config
        if cfg is None:
            return 0.0
        step = self.global_step
        if cfg.rlsd_lambda_warmup_steps > 0 and step < cfg.rlsd_lambda_warmup_steps:
            return cfg.rlsd_lambda * step / cfg.rlsd_lambda_warmup_steps
        if cfg.rlsd_lambda_decay_steps > 0 and step >= cfg.rlsd_lambda_warmup_steps:
            progress = (step - cfg.rlsd_lambda_warmup_steps) / cfg.rlsd_lambda_decay_steps
            return cfg.rlsd_lambda * max(1.0 - progress, 0.0)
        return cfg.rlsd_lambda

    def _training_advantage(self, advantage: float, old_logps: List[float], teacher_logps: Optional[List[float]]):
        cfg = self.rlhf_config
        if teacher_logps is None or cfg is None:
            return advantage
        import torch

        from swift.rl_core.advantage import apply_rlsd_reweight, expand_advantage_to_per_token

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

    def _chord_mu(self) -> float:
        cfg = self.rlhf_config
        if cfg is None or not self.chord_features:
            return 0.0
        warmup = int(cfg.chord_mu_warmup_steps or 0)
        decay = int(cfg.chord_mu_decay_steps or 0)
        peak = float(cfg.chord_mu_peak or 0.0)
        valley = float(cfg.chord_mu_valley or 0.0)
        if warmup > 0 and self.global_step < warmup:
            return peak * self.global_step / warmup
        if decay == 0 or self.global_step >= warmup + decay:
            return valley
        progress = (self.global_step - warmup) / decay
        return valley + (peak - valley) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _next_chord_features(self, mu: float) -> List[dict]:
        if mu <= 0 or not self.chord_features:
            return []
        count = int(self.rlhf_config.chord_sft_per_device_train_batch_size or 1)
        result = []
        for _ in range(count):
            result.append(copy.deepcopy(self.chord_features[self._chord_offset % len(self.chord_features)]))
            self._chord_offset += 1
        return result

    def _sync_reference(self) -> None:
        cfg = self.rlhf_config
        if not cfg or not cfg.sync_ref_model or self.global_step % cfg.ref_model_sync_steps:
            return
        if self.reference is None or self.reference == 'disable_lora':
            raise RuntimeError('reference synchronization requires a distinct mutable reference model.')
        if hasattr(self.reference, 'mix_from_policy'):
            self.reference.mix_from_policy(self.model, cfg.ref_model_mixup_alpha)
            return
        import torch

        state = self.model.get_state_dict()
        raw_reference = self.reference.strategy.unwrap_model(self.reference.model)
        reference_params = dict(raw_reference.named_parameters())
        overlap = set(state) & set(reference_params)
        if not overlap:
            raise RuntimeError('policy and reference have no matching parameter names for synchronization.')
        alpha = cfg.ref_model_mixup_alpha
        with torch.no_grad():
            for name in overlap:
                target = reference_params[name]
                source = state[name].detach().to(device=target.device, dtype=target.dtype)
                target.mul_(1.0 - alpha).add_(source, alpha=alpha)

    def fit(self) -> list:  # noqa: C901
        """Train for ``max_steps`` optimizer steps, reusing each rollout ``num_iterations`` times."""
        from swift.dev.recipe.train_loop import collect_manual_gc, finish_manual_gc, start_manual_gc

        ga = self.gradient_accumulation_steps
        group = self._active_group()
        manual_gc = getattr(self, 'manual_gc', False)
        manual_gc_steps = getattr(self, 'manual_gc_steps', 0)
        gc_was_enabled = start_manual_gc(manual_gc)
        try:
            while self.global_step < self.max_steps:
                batch = self._rollout_step()
                iterations = max(1, self.rlhf_config.num_iterations if self.rlhf_config else 1)
                for _ in range(iterations):
                    for item in batch:
                        if self.global_step >= self.max_steps:
                            break
                        sample = item['sample']
                        self.micro_step += 1
                        chord_mu = self._chord_mu()
                        chord = self._next_chord_features(chord_mu)
                        kwargs = {
                            'inputs': [copy.deepcopy(sample.input_feature), *chord],
                            'gradient_accumulation_steps': ga,
                            'advantages': [self._training_advantage(
                                item['advantage'], item['old_logps'], item['teacher_logps'])],
                            'old_logps': [item['old_logps']],
                            'rollout_logps': [item['rollout_logps']],
                            'truncated': [bool(self.rlhf_config and self.rlhf_config.overlong_filter
                                               and getattr(sample, 'truncated', False))],
                            'chord_count': len(chord),
                            'chord_mu': chord_mu,
                            'chord_phi': bool(self.rlhf_config and self.rlhf_config.chord_enable_phi_function),
                        }
                        if item['ref_logps'] is not None:
                            kwargs['ref_logps'] = [item['ref_logps']]
                        if (item['teacher_logps'] is not None and self.rlhf_config
                                and self.rlhf_config.sdar_loss_coef > 0):
                            kwargs['teacher_logps'] = [item['teacher_logps']]
                        self.model.forward_backward(**kwargs)
                        is_boundary = group.do_grad_sync(ga)
                        self.model.clip_grad_and_step(max_grad_norm=self.max_grad_norm, gradient_accumulation_steps=ga)
                        if is_boundary:
                            self.global_step += 1
                            collect_manual_gc(manual_gc, manual_gc_steps, self.global_step)
                            self._sync_reference()
                            metrics = group.calculate_metrics(True)
                            loss = float(metrics['loss']) if metrics.get('loss') is not None else float('nan')
                            record_data = {'step': self.global_step, 'loss': loss}
                            if metrics.get('loss_entropy') is not None:
                                record_data['entropy'] = float(metrics['loss_entropy'])
                            if metrics.get('loss_rollout_log_ratio') is not None:
                                record_data['rollout_log_ratio'] = float(metrics['loss_rollout_log_ratio'])
                            record = self.tracker.log(record_data, self.global_step)
                            self.history.append(record)
                            if self.tracker.should_log(self.global_step):
                                logger.info(f'step {self.global_step}  loss={record["loss"]:.4f}')
                            save_steps = getattr(self, 'save_steps', None)
                            if save_steps and self.global_step % save_steps == 0:
                                self.save(f'checkpoint-{self.global_step}')
            return self.history
        finally:
            finish_manual_gc(gc_was_enabled)
            self.tracker.close()

    def save(self, name: str = 'checkpoint-final') -> str:
        """Persist the policy and its optimizer/RNG state."""
        from swift.dev.recipe.train_loop import save_training_checkpoint

        return save_training_checkpoint(
            self.model,
            name,
            output_dir=self.output_dir,
            consumed_train_samples=self.global_step,
            no_save_optim=self.no_save_optim,
            no_save_rng=self.no_save_rng,
            safe_serialization=self.safe_serialization,
            max_shard_size=self.max_shard_size,
            save_total_limit=self.save_total_limit)

    def resume(self, state: dict) -> None:
        """Resume optimizer phase and completed rollout-step count."""
        self.micro_step = int(state['cur_step'])
        self.global_step = int(state.get('consumed_train_samples', 0))
