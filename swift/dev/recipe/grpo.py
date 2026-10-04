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

Weight-sync is delegated to the rollout object, not owned by the loop: every rollout exposes
``sync_weights`` / ``finish_generate``, so the loop calls them unconditionally around each rollout.
``run_grpo``'s ``SyncableRollout`` implements them over twinkle's ``CheckpointEngineManager``, pushing
the trained policy into the sampler BEFORE each rollout so the behaviour policy tracks the trained one
(correct GRPO). The base ``RolloutEngine`` (the smoke path) implements them as a warn-once no-op, so a
loop wired to it says out loud that it stays on the INITIAL weights instead of skipping silently (W14).

Advanced paths are explicit in this loop: frozen-reference KL, DAPO dynamic sampling,
RLSD/SDAR teacher scoring, CHORD auxiliary SFT, and periodic reference synchronization.
"""
from __future__ import annotations
import copy
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

import json

from swift.dev.advantage import compute_advantages
from swift.dev.recipe._teacher import DisableAdapterTeacher, encode_privileged_view, response_positions
from swift.dev.recipe.train_loop import PromptBatchScheduler, TrainLoop
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

    ``response_token_ids`` is a flat ``List[int]`` (twinkle accounts a trajectory in one array).
    """
    return float(len(getattr(sample, 'response_token_ids', None) or []))


def split_mini_batches(items: List[Any], mini_batch_size: int) -> List[List[Any]]:
    """Split a rollout into full ``mini_batch_size`` chunks, dropping an undersized tail.

    Every ``forward_backward`` must carry exactly ``mini_batch_size`` rows so slice_dp hands each DP rank
    the same ``per_device_train_batch_size`` count; a partial tail would give ranks uneven batches (and,
    below dp_size rows, no data at all). Dropping it keeps each optimizer step uniform -- the same
    exact-global-batch invariant Megatron enforces -- at the cost of a few rollout samples per round.
    """
    return [items[i:i + mini_batch_size] for i in range(0, len(items) - mini_batch_size + 1, mini_batch_size)]


@dataclass
class RolloutBatch:
    """One rollout step's per-row training payload, held column-wise with a construction-time length check.

    Each field is a list parallel to ``samples`` (one entry per rollout row). Replacing the old per-sample
    dict plus the hand-assembled parallel lists in ``_mini_batch_kwargs`` makes an off-by-one between any two
    columns impossible to build silently: ``__post_init__`` rejects unequal lengths up front. ``ref_logps`` /
    ``teacher_logps`` are all-or-none across a rollout (both scoring helpers return ``None`` or a full list),
    so they are ``Optional`` columns rather than per-row optionals. RFT -- which trains the kept completions
    with plain cross-entropy -- builds a samples-only batch and leaves the RL columns ``None``.

    Slicing returns a ``RolloutBatch`` (every present column sliced consistently), so ``split_mini_batches``
    -- which only uses ``len`` and ``items[i:j]`` -- splits a batch into mini-batches unchanged.
    """
    samples: List[Any]
    advantages: Optional[List[Any]] = None
    old_logps: Optional[List[List[float]]] = None
    rollout_logps: Optional[List[List[float]]] = None
    ref_logps: Optional[List[List[float]]] = None
    teacher_logps: Optional[List[List[float]]] = None
    #: Per-row sampler support sets (twinkle ``SamplingMask``), present only under ``enable_sampling_replay``;
    #: assembled into the top-level ``sampling_masks`` forward_backward kwarg (NOT into ``encoded``).
    sampling_masks: Optional[List[Any]] = None

    #: Per-row columns that must align with ``samples`` when present (unannotated, so not a dataclass field).
    _COLUMNS = ('advantages', 'old_logps', 'rollout_logps', 'ref_logps', 'teacher_logps', 'sampling_masks')

    def __post_init__(self) -> None:
        rows = len(self.samples)
        for name in self._COLUMNS:
            column = getattr(self, name)
            if column is not None and len(column) != rows:
                raise ValueError(f'RolloutBatch column {name!r} has {len(column)} rows but samples has {rows}; '
                                 'per-row rollout fields must align exactly.')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, key: slice) -> 'RolloutBatch':
        if not isinstance(key, slice):
            raise TypeError(f'RolloutBatch supports slice indexing only, got {type(key).__name__}.')

        def sliced(column):
            return None if column is None else column[key]

        return RolloutBatch(
            samples=self.samples[key],
            advantages=sliced(self.advantages),
            old_logps=sliced(self.old_logps),
            rollout_logps=sliced(self.rollout_logps),
            ref_logps=sliced(self.ref_logps),
            teacher_logps=sliced(self.teacher_logps),
            sampling_masks=sliced(self.sampling_masks))


class GRPOLoop(TrainLoop):
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
                 chord_features: Optional[List[dict]] = None,
                 num_generations: int = 4,
                 reward_funcs: Optional[List[Any]] = None,
                 reward_model_plugins: Optional[List[Callable]] = None,
                 reward_model_names: Optional[List[str]] = None,
                 reward_weights: Optional[List[float]] = None,
                 prm_scorer: Optional[Any] = None,
                 prm_funcs: Optional[List[Any]] = None,
                 prm_model_plugins: Optional[List[Callable]] = None,
                 prm_weights: Optional[List[float]] = None,
                 advantage_estimator: str = 'grpo',
                 scale_rewards: str = 'group',
                 rlhf_config: Optional['RLHFConfig'] = None,
                 reward_fn: Callable[[Any], float] = toy_length_reward,
                 max_steps: int = 3,
                 gradient_accumulation_steps: int = 1,
                 train_batch_size: int = 1,
                 generation_batch_size: Optional[int] = None,
                 num_train_epochs: float = 1.0,
                 seed: int = 42,
                 max_grad_norm: float = 1.0,
                 sampling_params: Optional[dict] = None,
                 async_generate: bool = False,
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
        # Resolve + validate the rollout inputs and reward functions BEFORE super().__init__, so a
        # misconfiguration still fails before the tracker initialises its reporters (a RunTracker side
        # effect). get_reward_funcs may build reward-model scorers, so it stays on this pre-tracker side.
        resolved_prompt_extras = [{} for _ in prompts] if prompt_extras is None else prompt_extras
        if len(resolved_prompt_extras) != len(prompts):
            raise ValueError('prompt_extras must contain exactly one mapping per prompt.')
        resolved_reward_funcs, resolved_reward_func_names = (
            get_reward_funcs(reward_funcs, rlhf_config) if reward_funcs else ([], []))
        resolved_reward_plugins = list(reward_model_plugins or [])
        model_names = list(reward_model_names or [])
        if model_names and len(model_names) != len(resolved_reward_plugins):
            raise ValueError('reward_model_names must align one-to-one with reward_model_plugins.')
        resolved_reward_func_names.extend(
            model_names or [type(plugin).__name__ for plugin in resolved_reward_plugins])

        super().__init__(
            model,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            logging_config=logging_config,
            output_dir=output_dir,
            save_steps=save_steps,
            no_save_optim=no_save_optim,
            no_save_rng=no_save_rng,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
            save_total_limit=save_total_limit,
            manual_gc=manual_gc,
            manual_gc_steps=manual_gc_steps)
        self.rollout = rollout_engine
        self.prompts = prompts
        self.prompt_extras = resolved_prompt_extras
        self.reference = reference
        self.teacher = teacher
        self.template = template
        self.chord_features = chord_features or []
        self._chord_offset = 0
        self.num_generations = num_generations
        self.advantage_estimator = advantage_estimator
        self.scale_rewards = scale_rewards
        self.reward_weights = reward_weights
        self.rlhf_config = rlhf_config
        self.reward_funcs = resolved_reward_funcs
        self.reward_func_names = resolved_reward_func_names
        self.reward_model_plugins = resolved_reward_plugins
        self.reward_fn = reward_fn
        #: Process-reward (PRM) channel: ``prm_scorer`` segments a response into reasoning steps and
        #: broadcasts each step's score onto its tokens (see :mod:`swift.dev.rewards.prm`); ``prm_funcs``
        #: (rule PRMs) and ``prm_model_plugins`` (frozen PRM models) score each step, weighted by
        #: ``prm_weights``. Active only when a scorer AND at least one PRM channel are wired together -- a
        #: half-configured pair produces nothing, so validate rejects it rather than this silently no-op'ing.
        self.prm_scorer = prm_scorer
        self.prm_funcs = list(prm_funcs or [])
        self.prm_model_plugins = list(prm_model_plugins or [])
        self.prm_weights = prm_weights
        self._prm_active = bool(prm_scorer is not None and (self.prm_funcs or self.prm_model_plugins))
        #: Samples per ``forward_backward``: ``per_device_train_batch_size * dp_size``, computed by the
        #: recipe (which owns the distributed/train configs) so one mini-batch splits evenly across the
        #: slice_dp ranks -- each rank receives exactly ``per_device_train_batch_size`` rows. Feeding fewer
        #: than dp_size rows is what raised "Batch too small" under the old one-sample-per-call loop.
        self.train_batch_size = max(1, train_batch_size)
        self.sampling_params = sampling_params
        #: Overlap the sampler generation of the next rollout batch with this step's scoring + training
        #: (1-batch lookahead, staleness <= 1). Driver-side double buffer over the SAME GRPOLoop -- NOT a
        #: separate async runtime -- so every feature wired here (PRM / routing replay / sampling replay /
        #: CHORD / teacher / IS correction) applies to the async path unchanged. validate gates it to
        #: disaggregated + a rollout importance-sampling mode (staleness is always > 0, so the off-policy
        #: correction is mandatory) and rejects it with dynamic_sample / multi-turn (both need synchronous,
        #: adaptive re-generation of the just-collected batch).
        self.async_generate = bool(async_generate)
        #: Prompt-set iterator: one batch per rollout, exhausted after ``num_train_epochs`` passes. Built
        #: here (not in the recipe) so every on-policy subclass shares one epoch/shard/seed policy; the
        #: recipe only derives the matching ``max_steps`` LR horizon via ``rollout_step_budget``.
        self._prompt_batches = PromptBatchScheduler(
            len(prompts),
            generation_batch_size=generation_batch_size,
            num_train_epochs=num_train_epochs,
            seed=seed)

    def _prompt_payload(self, prompt_indices: Sequence[int]):
        """The ``(prompts, prompt_extras)`` rows for one rollout batch, by global prompt position."""
        prompts = [self.prompts[i] for i in prompt_indices]
        extras = [self.prompt_extras[i] for i in prompt_indices]
        return prompts, extras

    def _finalize_samples(self, samples: List[Any], prompt_indices: Sequence[int]) -> List[Any]:
        """Check the rollout count and stamp each sample with its group's global prompt id.

        Shared by the blocking :meth:`_generate` and the async :meth:`_collect_generation` so both paths
        enforce the same ``num_generations``-per-prompt contract and group id (which drives the
        group-relative advantage).
        """
        expected = len(prompt_indices) * self.num_generations
        if len(samples) != expected:
            raise RuntimeError(f'rollout returned {len(samples)} samples, expected {expected} '
                               f'({len(prompt_indices)} prompts * {self.num_generations} generations).')
        for local_idx, prompt_idx in enumerate(prompt_indices):
            for sample in samples[local_idx * self.num_generations:(local_idx + 1) * self.num_generations]:
                sample.prompt_id = str(prompt_idx)
        return samples

    def _generate(self, prompt_indices: Sequence[int]) -> List[Any]:
        """Blocking rollout: push the policy, sample to completion, hand the device back (sync path)."""
        prompts, extras = self._prompt_payload(prompt_indices)
        self.rollout.sync_weights()
        try:
            samples = self.rollout.generate(
                prompts,
                num_samples=self.num_generations,
                sampling_params=self.sampling_params,
                prompt_extras=extras)
        finally:
            self.rollout.finish_generate()
        return self._finalize_samples(samples, prompt_indices)

    def _submit_generation(self, prompt_indices: Sequence[int]) -> Any:
        """Async admit half: push the current policy into the idle sampler and schedule this batch WITHOUT
        blocking, returning a handle for :meth:`_collect_generation`.

        ``sync_weights`` runs here, not in collect: pushing weights rewrites the sampler's live tensors and
        needs it quiescent, and the caller only submits right after a collect (no generation in flight). The
        batch is thus generated by the policy as of THIS call -- one version behind the policy that will
        train it under a 1-batch lookahead (staleness 1), which the rollout importance sampling validate
        forces on for async corrects.
        """
        prompts, extras = self._prompt_payload(prompt_indices)
        self.rollout.sync_weights()
        return self.rollout.submit_generate(
            prompts,
            num_samples=self.num_generations,
            sampling_params=self.sampling_params,
            prompt_extras=extras)

    def _collect_generation(self, handle: Any, prompt_indices: Sequence[int]) -> List[Any]:
        """Async collect half: block until the admitted generation finishes, then finalize its samples."""
        try:
            samples = self.rollout.collect_generate(handle)
        finally:
            self.rollout.finish_generate()
        return self._finalize_samples(samples, prompt_indices)

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
        if reward_parts:
            return torch.cat(reward_parts, dim=1)
        return torch.tensor([[self.reward_fn(sample)] for sample in samples], dtype=torch.float32)

    def _weighted_rewards(self, rewards_per_func):
        import torch

        weight_tensor = build_reward_weights(self.reward_weights, rewards_per_func.shape[1]).to(
            device=rewards_per_func.device, dtype=rewards_per_func.dtype)
        return (rewards_per_func * weight_tensor.unsqueeze(0)).nansum(dim=1)

    def _score_steps(self, rows: List[Dict[str, Any]]) -> List[float]:
        """Score a batch of PRM step rows -> one weighted process score per row.

        The PRM counterpart of :meth:`_score`: the scorer plugin builds each row as ``{'messages': prompt +
        assistant(the response prefix up to one step), **columns}``, and every PRM channel scores it -- rule
        PRMs read the prefix text as the completion, frozen PRM models read the whole row -- then the
        per-channel scores are combined with ``prm_weights``, the same weighted sum :meth:`_weighted_rewards`
        applies to outcome rewards. The frozen PRM runs on its OWN device group (not sliced across the policy
        ranks), so a step batch smaller than the policy's dp_size is fine here.
        """
        import torch

        parts = []
        if self.prm_funcs:
            completions = [row['messages'][-1]['content'] for row in rows]
            column_keys = {key for row in rows for key in row if key != 'messages'}
            columns = {key: [row.get(key) for row in rows] for key in column_keys}
            parts.append(compute_rewards_per_func(completions, self.prm_funcs, columns))
        if self.prm_model_plugins:
            parts.append(compute_reward_model_scores(rows, self.prm_model_plugins))
        scores = torch.cat(parts, dim=1)
        weights = build_reward_weights(self.prm_weights, scores.shape[1]).to(
            device=scores.device, dtype=scores.dtype)
        return (scores * weights.unsqueeze(0)).nansum(dim=1).tolist()

    def _apply_prm(self, samples: List[Any], orm_advantages: Sequence[float]) -> List[List[float]]:
        """Fold the PRM process reward into a per-token advantage (segmented placement).

        The scorer broadcasts each step's weighted score onto that step's response tokens -- a per-token
        *process* reward in the response-token frame (length ``len(response_token_ids)``, the same frame
        ``old_logps`` and :meth:`_training_advantage`'s expansion use). The outcome (ORM) group-relative
        advantage is then placed on the LAST response token only, so process and terminal rewards occupy
        disjoint token segments and no token is driven by both. Returns one per-token advantage list per
        sample, which :meth:`_training_advantage` passes through to ``forward_backward`` unchanged.
        """
        decode = self.template.decode
        per_token: List[List[float]] = []
        for sample, orm_advantage in zip(samples, orm_advantages):
            if getattr(sample, 'messages', None) is not None:
                raise ValueError('PRM process reward does not support multi-turn rollouts: a step prefix '
                                 'would span several assistant turns, so per-step scoring is undefined. Drop '
                                 '--prm or disable multi-turn (--max_turns).')
            try:
                prompt_index = int(sample.prompt_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'PRM scoring requires a numeric prompt_id, got {sample.prompt_id!r}.') from exc
            process = self.prm_scorer(
                self.prompts[prompt_index],
                sample.response_token_ids,
                sample.decoded,
                self._score_steps,
                decode,
                columns=getattr(sample, 'extra', None))
            advantage = list(process)
            if advantage:
                advantage[-1] = float(orm_advantage)  # segmented placement: terminal ORM on the last token
            per_token.append(advantage)
        return per_token

    def _dynamic_rollout(self, prompt_indices: Sequence[int]):
        """Replace zero-variance prompt groups, falling back atomically if retries are exhausted."""
        import torch

        prompt_indices = list(prompt_indices)
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
        return response_positions(feature)

    def _response_logps(self, scorer: Any, features: List[dict]) -> List[List[float]]:
        """Score ``features`` with ``forward_only``, returning one response-token logp list per feature.

        Batching -- one ``forward_only`` per ``train_batch_size`` chunk, not one call per feature -- is
        what makes DP>1 work: ``forward_only`` is a slice_dp method, so a length-1 ``inputs`` leaves all
        but one rank with no data ("Batch too small"). Chunks also bound the no-grad scoring memory (the
        materialised logits) the same way design B bounds a training mini-batch.

        ``forward_only`` returns full-sequence, right-padded ``logps`` of shape ``[n, seq_len]`` in
        submission order (``collect_tensor_dict`` pads+stacks the per-rank shards), so each row is
        indexed by that feature's own response positions -- right-padding never shifts them.
        """
        import torch

        results: List[List[float]] = []
        for chunk in self._score_chunks(features):
            out = scorer.forward_only(inputs=[copy.deepcopy(feature) for feature in chunk])
            logps = out.get('logps') if isinstance(out, dict) else None
            if logps is None:
                raise RuntimeError('reference/teacher forward returned no logps.')
            values = torch.as_tensor(logps).detach().float().cpu()
            if values.dim() == 1:
                values = values.unsqueeze(0)
            for row, feature in zip(values, chunk):
                results.append(self._row_response_logps(row.reshape(-1), feature))
        return results

    def _score_chunks(self, features: List[dict]) -> List[List[dict]]:
        """Split ``features`` into ``train_batch_size`` chunks, folding a ragged tail into the last one.

        Every chunk must hold >= dp_size rows or slice_dp starves a rank. ``train_batch_size`` is a
        multiple of dp_size, so full chunks qualify, and folding the remainder into the last full chunk
        keeps it >= dp_size too. A rollout smaller than one chunk yields a single short chunk that
        ``forward_only`` rejects loudly -- the same "rollout too small" condition training guards.
        """
        chunks = split_mini_batches(features, self.train_batch_size)
        tail = features[len(chunks) * self.train_batch_size:]
        if tail:
            chunks[-1] = chunks[-1] + tail if chunks else tail
        return chunks

    def _row_response_logps(self, row: Any, feature: dict) -> List[float]:
        """Extract one feature's response-token logps from its (possibly padded) full-sequence row."""
        positions = self._response_positions(feature)
        if row.numel() == len(positions):
            return row.tolist()  # response-only form: nothing to strip
        labels = feature.get('labels')
        feature_length = len(labels) if labels is not None else 0
        if row.numel() < feature_length:
            raise RuntimeError(f'forward logps have {row.numel()} values for a {feature_length}-token feature; '
                               'response-token alignment is impossible.')
        return row[positions].tolist()

    def _prepare_routing_replay(self, samples: List[Any]) -> None:
        """Establish each sample's MoE expert routing per ``RLHFConfig.router_replay_mode``, in place.

        Routing replay keeps an MoE policy's importance ratio honest: the training forward must REPLAY the
        expert choices that actually produced each token rather than recompute them (recomputed routing under
        drifted weights silently corrupts the ratio). The two modes differ only in WHERE the routing comes
        from, then feed the SAME REPLAY path (each row's ``encoded['routed_experts']`` + a
        ``router_replay_action='replay_forward'`` forward_backward kwarg):

        - ``'R3'``: the sampler exported the generating pass's routing (``samples_from_responses`` carried it
          into ``encoded``). Require it -- a non-exporting engine leaves none, and silently recomputing would
          be wrong, so fail loudly and point at R2.
        - ``'R2'``: RECORD the training model's own routing for these tokens (under the just-synced
          generating weights, so it is a faithful proxy for the generating pass vLLM did not export).
        """
        mode = self.rlhf_config.router_replay_mode if self.rlhf_config else 'disabled'
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
            # R2 records the training model's routing regardless of what the sampler returned (legacy R2
            # semantics: set RECORD, forward, capture), so drop any sampler routing first.
            for sample in samples:
                sample.encoded.pop('routed_experts', None)
            self._record_routing(samples)
            return
        raise ValueError(f'Unknown router_replay_mode {mode!r}; expected one of disabled/R2/R3.')

    def _record_routing(self, samples: List[Any]) -> None:
        """R2: RECORD the training model's own expert routing for the rollout tokens, in place into encoded.

        Mirrors :meth:`_response_logps`' chunking -- ``forward_only`` is a slice_dp method, so each chunk must
        hold >= dp_size rows -- but asks for ``router_replay_action='record'`` and reads back
        ``routed_experts`` instead of logps. It runs BEFORE any training update, so the routing is captured
        under the weights that just generated these tokens. The returned whole-sequence routing is sliced
        back to each row's own ``input_ids`` length, so the REPLAY forward's ``align_routed_experts`` re-pads
        it exactly as it does the R3 sampler routing (a padded RECORD row re-fed whole would misalign).
        """
        import torch

        features = [sample.input_feature for sample in samples]
        rows: List[Any] = []
        for chunk in self._score_chunks(features):
            out = self.model.forward_only(
                inputs=[copy.deepcopy(feature) for feature in chunk], router_replay_action='record')
            recorded = out.get('routed_experts') if isinstance(out, dict) else None
            if recorded is None:
                raise RuntimeError(
                    'router_replay_mode="R2" asked the training model to RECORD its MoE routing, but the '
                    'forward returned none. The policy must be an MoE model built with routing replay '
                    'enabled; a dense policy has no expert routing to record (use "disabled").')
            # A uniform-sequence RECORD returns one [n, seq, layers, topk] tensor; a variable-sequence one
            # returns a per-microbatch list of them. Flatten either to row-major order (submission order).
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

    def _teacher_messages(self, sample: Any, prompt_idx: int, teacher_prompt: str) -> tuple[List[dict], List[int]]:
        is_multi_turn = getattr(sample, 'messages', None) is not None
        messages = copy.deepcopy(sample.messages if is_multi_turn else self.prompts[prompt_idx])
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
        teacher_prompt = (getattr(sample, 'extra', None) or {}).get('teacher_prompt')
        if not teacher_prompt:
            return sample.input_feature
        try:
            prompt_idx = int(sample.prompt_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'teacher_prompt requires a numeric prompt_id, got {sample.prompt_id!r}.') from exc
        messages, response_ids = self._teacher_messages(sample, prompt_idx, teacher_prompt)
        return encode_privileged_view(self.template, messages, response_ids, getattr(sample, 'extra', None))

    def _reference_logps(self, samples: List[Any]) -> Optional[List[List[float]]]:
        if self.reference is None:
            return None
        return self._response_logps(self.reference, [sample.input_feature for sample in samples])

    def _teacher_logps(self, samples: List[Any]) -> Optional[List[List[float]]]:
        has_privileged_prompt = any((sample.extra or {}).get('teacher_prompt') for sample in samples)
        needs_teacher = bool(self.teacher is not None or has_privileged_prompt)
        advanced = bool(self.rlhf_config and
                        (self.rlhf_config.advantage_reweight == 'rlsd' or self.rlhf_config.sdar_loss_coef > 0))
        if not needs_teacher:
            if advanced:
                raise ValueError('RLSD/SDAR requires teacher_model or a teacher_prompt dataset column.')
            return None
        owner = self.teacher if self.teacher is not None else self.model
        return self._response_logps(owner, [self._teacher_feature(sample) for sample in samples])

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

    def _rollout_step(self, prompt_indices: Sequence[int]):
        """Synchronous rollout: generate this batch, then score/assemble it into a trainable batch."""
        samples, rewards_per_func = self._dynamic_rollout(prompt_indices)
        return self._assemble_rollout_batch(samples, rewards_per_func)

    def _assemble_rollout_batch(self, samples: List[Any], rewards_per_func) -> 'RolloutBatch':
        """Score an already-generated rollout into a trainable ``RolloutBatch`` (generation excluded).

        Split from :meth:`_rollout_step` so the async path overlaps ONLY the sampler generation with the
        previous step's training, then runs this scoring / advantage / PRM / routing-replay assembly (all on
        the training / reference / teacher models, never the sampler) on the collected samples. The
        synchronous path reaches it through :meth:`_rollout_step` unchanged.
        """
        self.tracker.log_prompts(self.prompts, samples, self.global_step + 1, self.num_generations)
        self._log_completions(samples, rewards_per_func)
        ref_logps = self._reference_logps(samples)
        teacher_logps = self._teacher_logps(samples)
        cfg = self.rlhf_config
        preserve_policy_logps = bool(cfg and (
            cfg.rollout_importance_sampling_mode or cfg.log_rollout_offpolicy_metrics))
        old_logps = (self._response_logps(self.model, [sample.input_feature for sample in samples])
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
        # PRM process reward: fold the per-step scores into a per-token advantage (segmented placement -- the
        # group-relative ORM advantage above lands on the last response token, the process reward on the rest).
        # Replaces the scalar advantages with per-token lists, which _training_advantage passes through.
        if self._prm_active:
            advantages = self._apply_prm(samples, advantages)
        # MoE routing replay: establish each row's expert routing BEFORE the batch reaches the training
        # update, whose forward REPLAYS it. R3 consumes the sampler-exported routing (already in encoded);
        # R2 records the training model's own under the just-synced generating weights. Runs AFTER the
        # scoring forwards so those never carry the extra routed_experts column.
        self._prepare_routing_replay(samples)
        return RolloutBatch(
            samples=list(samples),
            advantages=list(advantages),
            old_logps=list(old_logps),
            rollout_logps=[sample.old_logps for sample in samples],
            ref_logps=ref_logps,
            teacher_logps=teacher_logps,
            sampling_masks=([sample.sampling_mask for sample in samples]
                            if cfg and cfg.enable_sampling_replay else None))

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

    def _training_advantage(self, advantage: Any, old_logps: List[float], teacher_logps: Optional[List[float]]):
        if isinstance(advantage, (list, tuple)):
            # Already per-token -- the PRM process reward with segmented terminal placement (see _apply_prm),
            # in the response-token frame the loss expects, so pass it through with no scalar expansion. PRM
            # and the teacher-signal reweighting below are mutually exclusive (both produce per-token
            # advantages from different sources); validate rejects enabling them together.
            return list(advantage)
        cfg = self.rlhf_config
        if teacher_logps is None or cfg is None:
            return advantage
        import torch

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
        if self.reference is None or isinstance(self.reference, DisableAdapterTeacher):
            raise RuntimeError('reference synchronization requires a distinct mutable reference model.')
        reference = self.reference.model
        if hasattr(reference, 'mix_from_policy'):
            reference.mix_from_policy(self.model, cfg.ref_model_mixup_alpha)
            return
        import torch

        state = self.model.get_state_dict()
        raw_reference = reference.strategy.unwrap_model(reference.model)
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

    def _mini_batch_kwargs(self, mini_batch: 'RolloutBatch') -> Dict[str, Any]:
        """Assemble one mini-batch's parallel-list ``forward_backward`` kwargs (design B).

        Every list is parallel to ``mini_batch.samples`` (one entry per rollout row) so slice_dp splits them
        all consistently across the DP ranks. CHORD auxiliary SFT rows are appended to ``inputs`` and their
        count carried in ``chord_count`` (a broadcast scalar the loss uses to split RL rows from SFT rows);
        that split is only correct at dp_size==1, so run_grpo rejects CHORD with dp>1 (RL_PLAN Bug#6).
        ``ref_logps`` / ``teacher_logps`` are uniform across a rollout (both scoring helpers return None or
        a full list), so the column being None decides whether the key is passed at all.
        """
        cfg = self.rlhf_config
        chord_mu = self._chord_mu()
        chord = self._next_chord_features(chord_mu)
        teacher_logps = mini_batch.teacher_logps
        kwargs = {
            'inputs': [*[copy.deepcopy(sample.input_feature) for sample in mini_batch.samples], *chord],
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
            'chord_count': len(chord),
            'chord_mu': chord_mu,
            'chord_phi': bool(cfg and cfg.chord_enable_phi_function),
        }
        if mini_batch.ref_logps is not None:
            kwargs['ref_logps'] = list(mini_batch.ref_logps)
        if teacher_logps is not None and cfg and cfg.sdar_loss_coef > 0:
            kwargs['teacher_logps'] = list(teacher_logps)
        if cfg and cfg.router_replay_mode != 'disabled':
            # The training forward REPLAYS each row's routing (established in _rollout_step); twinkle's
            # backward auto-switches to REPLAY_BACKWARD. CHORD SFT rows carry no routing, so routing replay
            # + CHORD is rejected up front in validate (they cannot share one forward).
            kwargs['router_replay_action'] = 'replay_forward'
        if cfg and cfg.enable_sampling_replay:
            # Per-row sampler support sets, parallel to ``inputs``. CHORD rows are never sampled, so validate
            # rejects CHORD + sampling replay -- ``chord`` is empty here and the masks align with the rows.
            kwargs['sampling_masks'] = list(mini_batch.sampling_masks)
        return kwargs

    def _plan_mini_batches(self, batch: 'RolloutBatch') -> List['RolloutBatch']:
        """Split one rollout into full ``train_batch_size`` mini-batches, warning on a dropped tail.

        Structure-agnostic (only ``len`` + slicing): both GRPOLoop and RFTLoop pass a ``RolloutBatch``, whose
        slice indexing yields ``RolloutBatch`` mini-batches. Raises when the rollout cannot fill even one
        mini-batch, so the caller never silently runs a no-op training round.
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

    def _pre_metric_step(self) -> None:
        """Sync the (mutable) reference model toward the policy before reading this step's metrics."""
        self._sync_reference()

    def _extra_step_metrics(self, metrics: dict) -> dict:
        """Policy entropy + rollout/policy log-prob ratio, when the GRPO loss exposes them."""
        extra: dict = {}
        if metrics.get('loss_entropy') is not None:
            extra['entropy'] = float(metrics['loss_entropy'])
        if metrics.get('loss_rollout_log_ratio') is not None:
            extra['rollout_log_ratio'] = float(metrics['loss_rollout_log_ratio'])
        return extra

    def _should_log(self) -> bool:
        """GRPO logs on the tracker's own cadence, with no logging_steps fallback."""
        return self.tracker.should_log(self.global_step)

    def _train_rollout_batch(self, batch: 'RolloutBatch') -> None:
        """Run this rollout's optimizer steps: split into mini-batches, replay ``num_iterations`` times.

        Design B (faithful mini-batch GRPO): a rollout is split into ``train_batch_size``-row mini-batches,
        one ``forward_backward`` runs per mini-batch, ``gradient_accumulation_steps`` mini-batches make one
        optimizer step, and the whole rollout is replayed ``num_iterations`` times. This mirrors SFTLoop's
        batched feeding and is what makes DP>1 work: each mini-batch carries >= dp_size rows, so slice_dp
        gives every rank data. Shared by the synchronous and the async (overlapped) drivers.
        """
        mini_batches = self._plan_mini_batches(batch)
        iterations = max(1, self.rlhf_config.num_iterations if self.rlhf_config else 1)
        for _ in range(iterations):
            for mini_batch in mini_batches:
                if self._reached_max():
                    break
                self._run_micro_step(self._mini_batch_kwargs(mini_batch))

    def _run_sync(self) -> None:
        """Synchronous driver: generate, then train, one rollout batch at a time (no overlap)."""
        for prompt_indices in self._prompt_batches:
            if self._reached_max():
                break
            self._train_rollout_batch(self._rollout_step(prompt_indices))

    def _consume_async_samples(self, samples: List[Any]) -> None:
        """Score an already-collected async batch and run its optimizer steps (the driver's consume half)."""
        rewards = self._score(samples)
        self._train_rollout_batch(self._assemble_rollout_batch(samples, rewards))

    def _run_async(self) -> None:
        """Overlapped driver (1-batch lookahead): generate batch ``N+1`` on the sampler while training ``N``.

        A thin wiring of this loop's rollout callbacks into the shared :func:`overlap_rollout_batches`
        double buffer (see ``train_loop`` for the control flow and the staleness<=1 argument). ``submit``
        syncs weights at the sampler's idle point and admits a batch; ``collect`` blocks on it; ``consume``
        scores and trains it; ``cancel`` drops an in-flight batch when ``max_steps`` is reached. The
        lookahead batch is produced by the policy one version behind the one that trains it (staleness 1),
        which the mandatory rollout importance sampling corrects.
        """
        from swift.dev.recipe.train_loop import overlap_rollout_batches

        overlap_rollout_batches(
            prompt_batches=self._prompt_batches,
            submit=self._submit_generation,
            collect=self._collect_generation,
            cancel=self.rollout.cancel_generate,
            consume=self._consume_async_samples,
            reached_max=self._reached_max)

    def fit(self) -> list:
        """Train over the prompt set for ``num_train_epochs`` passes, reusing each rollout ``num_iterations``.

        The prompt-set scheduler yields one generation batch per rollout and is exhausted after
        ``num_train_epochs`` full passes (B1: an epoch is now a real dataset pass, not an unbounded
        whole-set regeneration bounded only by ``max_steps``); an explicit ``max_steps`` still caps the
        optimizer-step count early. ``async_generate`` selects the overlapped driver (:meth:`_run_async`),
        otherwise the synchronous one (:meth:`_run_sync`); both drive the same per-rollout training
        (:meth:`_train_rollout_batch`).
        """
        from swift.dev.recipe.train_loop import finish_manual_gc, start_manual_gc

        gc_was_enabled = start_manual_gc(self.manual_gc)
        try:
            if self.async_generate:
                self._run_async()
            else:
                self._run_sync()
            return self.history
        finally:
            finish_manual_gc(gc_was_enabled)
            self.tracker.close()
