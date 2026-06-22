# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import concurrent.futures
import copy
import os
import torch
import uuid
from typing import Any, Dict, List, Optional, Tuple

from swift.dataset import RowPreprocessor
from swift.infer_engine.protocol import RolloutOutput
from swift.rl_core.advantage import compute_advantages, compute_reward_metrics
from swift.rl_core.data import GRPOBatch, GRPOSample
from swift.rl_core.grpo_algorithm import compute_std_for_dynamic_sampling, score_completions
from swift.rlhf_trainers.utils import (encode_sample, get_non_thinking_prefix_ids, make_reward_weights,
                                       resolve_reward_funcs)
from swift.rollout import MultiTurnScheduler, invoke_async_hook, multi_turns, run_multi_turn
from swift.utils import get_logger, remove_response
from .base_trainer import BaseRayTrainer
from .driver_utils import extract_iteration

logger = get_logger()


class GRPOTrainer(BaseRayTrainer):
    """Driver-side GRPO trainer."""

    def _prepare_state(self) -> None:
        super()._prepare_state()
        args = self.args

        self.num_generations = args.num_generations
        self.advantage_estimator = args.advantage_estimator
        self.scale_rewards = args.scale_rewards
        self.kl_in_reward = args.kl_in_reward

        self._prepare_rewards()
        self._prepare_multi_turn()

        # Ray supports router replay only in R3 (rollout records routed_experts, the driver
        # collates them into the train micro-batch). R2 records during the policy logps
        # forward, which — with driver-side collation — would not flow back into the train
        # batch; reject it explicitly (mirrors pipeline.py's R3-only rollout wiring).
        router_mode = getattr(args, 'router_replay_mode', 'disabled')
        if router_mode not in ('disabled', 'R3'):
            raise ValueError(f"Ray Megatron GRPO supports router_replay_mode in {{'disabled', 'R3'}}, "
                             f'got {router_mode!r}. Use R3 (rollout-recorded routing) for the Ray pipeline.')

        # DAPO dynamic_sample + truncation_strategy='delete' resampling (driver-side).
        self.dynamic_sample = getattr(args, 'dynamic_sample', False)
        self.max_resample_times = getattr(args, 'max_resample_times', 3)
        self.truncation_strategy = args.truncation_strategy
        self._max_resample_rounds = getattr(args, 'max_resample_times', 10)
        self._needs_resample_iterator = self.dynamic_sample or self.truncation_strategy == 'delete'

    def _prepare_multi_turn(self) -> None:
        """Configure driver-side multi-turn scheduler (Mode A only).

        Mode B (server-side scheduler) is intentionally not enabled here because
        :class:`VllmServer.launch_server` does not yet wrap the engine via
        ``get_rollout_engine_type`` — the server-side scheduler plumbing is a
        separate cross-process change.  When that lands, set
        ``self._enable_server_multi_turn`` from a new
        ``RolloutReplica.get_engine_type()`` passthrough.
        """
        args = self.args
        self._multi_turn_scheduler: Optional[MultiTurnScheduler] = None
        self._max_turns: Optional[int] = getattr(args, 'max_turns', None)
        self._enable_server_multi_turn = False

        scheduler_cfg = getattr(args, 'multi_turn_scheduler', None)
        if not scheduler_cfg:
            return
        if isinstance(scheduler_cfg, str):
            if scheduler_cfg not in multi_turns:
                raise ValueError(f'Unknown multi_turn_scheduler: {scheduler_cfg!r}; '
                                 f'available: {list(multi_turns)}')
            scheduler_kwargs = {'max_turns': self._max_turns}
            gym_env = getattr(args, 'gym_env', None)
            if gym_env is not None:
                scheduler_kwargs['gym_env'] = gym_env
            self._multi_turn_scheduler = multi_turns[scheduler_cfg](**scheduler_kwargs)
        else:
            assert isinstance(scheduler_cfg, MultiTurnScheduler)
            self._multi_turn_scheduler = scheduler_cfg

    def _prepare_rewards(self):
        args = self.args
        reward_funcs_cfg = (args.reward_funcs or []).copy()
        if not isinstance(reward_funcs_cfg, list):
            reward_funcs_cfg = [reward_funcs_cfg]

        self.reward_funcs, self.reward_func_names = resolve_reward_funcs(reward_funcs_cfg, args=args)

        # use_gym_env: gym total_reward is appended as an extra reward column so it can
        # blend with reward_funcs via reward_weights.  When reward_funcs is empty, it becomes
        # the single reward source.
        self.use_gym_env = bool(getattr(args, 'use_gym_env', False))
        if self.use_gym_env:
            self.reward_func_names.append('gym_reward')

        self.reward_weights = make_reward_weights(args.reward_weights, len(self.reward_func_names), self.device)
        self.reward_model_plugins = [None] * len(self.reward_funcs)

        if not self.reward_funcs and not self.use_gym_env:
            raise ValueError('GRPOTrainer: no reward functions configured '
                             '(or pass use_gym_env: true to use the env-provided total_reward)')

    def _get_request_config(self):
        """Build a RequestConfig for rollout generation."""
        from swift.infer_engine.protocol import RequestConfig
        args = self.args
        return RequestConfig(
            n=1,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop=args.stop_words or None,
            return_details=True,
            logprobs=True,
        )

    def _train_loop(self, tg, train_iters, iteration):
        ckpt = self.ckpt_manager
        merge_and_sync = not self.args.vllm_enable_lora
        spg = self._steps_per_generation

        while iteration < train_iters:
            ckpt.sync_weights(merge_and_sync=merge_and_sync)

            with self._generation_context(tg, ckpt):
                prompt_batch = next(self._data_iter)
                if self.truncation_strategy == 'delete':
                    prompt_batch = self._resample_failed_prompts(prompt_batch)
                rollout_batch = self.expand_for_generation(prompt_batch)
                completions = self._generate(rollout_batch)
                rollout_with_outputs = self._postprocess_rollout(rollout_batch, completions)
                rewards_per_func = self.score_completions(rollout_with_outputs)
                # DAPO dynamic sampling: drop zero-variance (std==0) prompt groups and
                # resample fresh prompts (the rollout engine is still awake in this context).
                if self.dynamic_sample:
                    rollout_with_outputs, rewards_per_func = self._dynamic_sampling(rollout_with_outputs,
                                                                                    rewards_per_func)

            self._maybe_log_completions(
                rollout_with_outputs, rewards=rewards_per_func.sum(dim=1).tolist(), gen_step=iteration)

            n_samples = len(rollout_with_outputs)
            chunk_size = n_samples // spg

            all_chunks = []  # per spg step: (dispatch, flat_grpo_batches)
            for step_idx in range(spg):
                chunk_start = step_idx * chunk_size
                chunk_end = chunk_start + chunk_size
                chunk_rollout = rollout_with_outputs[chunk_start:chunk_end]
                chunk_samples = self.encode_rollout_batch(chunk_rollout)

                dispatch, grpo_batches = self._collate_for_workers(tg, chunk_samples)
                logps_rows = tg.compute_logps(dispatch)
                self._scatter_logps(grpo_batches, logps_rows, 'old_per_token_logps')
                if self.beta != 0.0:
                    ref_rows = tg.compute_ref_logps(dispatch)
                    self._scatter_logps(grpo_batches, ref_rows, 'ref_per_token_logps')
                all_chunks.append((dispatch, grpo_batches))

            for step_idx in range(spg):
                if iteration >= train_iters:
                    break
                dispatch, grpo_batches = all_chunks[step_idx]
                chunk_start = step_idx * chunk_size
                chunk_end = chunk_start + chunk_size
                chunk_rewards_pf = rewards_per_func[chunk_start:chunk_end]

                kl_values = self._compute_kl_from_batches(grpo_batches) if self.beta != 0.0 else None

                chunk_advantages, rewards = self.compute_advantages(chunk_rewards_pf, kl_values=kl_values)
                self._scatter_advantages(grpo_batches, chunk_advantages)

                results = tg.train_step(
                    dispatch, extra_metrics=self._build_grpo_log_metrics(rewards, chunk_advantages, chunk_rewards_pf))
                iteration = extract_iteration(results)

        return iteration

    @staticmethod
    def _scatter_logps(grpo_batches: List[GRPOBatch], rows: List[Dict[str, torch.Tensor]], key: str) -> None:
        """Stack the flat per-sample logps rows (dp_flat, sample order) back onto each
        micro-batch's GRPOBatch as ``[B, T]`` — the same carrier non-Ray Megatron uses.

        ``completion_mask`` is NOT touched here: it was built by the driver collate and the
        worker only forwards logps, so the existing ``gb.completion_mask`` is already correct.
        """
        src_key = 'per_token_logps' if key == 'old_per_token_logps' else 'ref_per_token_logps'
        pos = 0
        for gb in grpo_batches:
            b = gb.completion_mask.shape[0]
            chunk = rows[pos:pos + b]
            pos += b
            setattr(gb, key, torch.stack([r[src_key] for r in chunk], dim=0))
        assert pos == len(rows), f'_scatter_logps: consumed {pos} rows but got {len(rows)}'

    @staticmethod
    def _scatter_advantages(grpo_batches: List[GRPOBatch], advantages: torch.Tensor) -> None:
        """Write the [N] advantages (sample order) onto each micro-batch's GRPOBatch."""
        pos = 0
        for gb in grpo_batches:
            b = gb.completion_mask.shape[0]
            gb.advantages = advantages[pos:pos + b]
            pos += b
        assert pos == advantages.shape[0], f'_scatter_advantages: wrote {pos} but got {advantages.shape[0]}'

    def _build_grpo_log_metrics(self, rewards, advantages, rewards_per_func) -> Dict[str, float]:
        """Driver-computed GRPO metrics (reward / reward_std / adv_nonzero / per-func),
        injected into the worker megatron on_log so all logging is unified there."""
        reward_metrics = compute_reward_metrics(
            rewards=rewards,
            rewards_per_func=rewards_per_func,
            reward_func_names=self.reward_func_names,
            num_generations=self.num_generations,
            scale_rewards=self.scale_rewards,
        )
        metrics = {
            'reward': reward_metrics.reward_mean,
            'reward_std': reward_metrics.reward_std,
            'frac_reward_zero_std': reward_metrics.frac_reward_zero_std,
            'adv_nonzero': (advantages.abs() > 1e-8).float().mean().item(),
        }
        # Flatten per-function metrics into scalar values the worker can inject.
        for name in self.reward_func_names:
            metrics[name] = reward_metrics.per_func_mean[name]
            metrics[f'rewards/{name}/std'] = reward_metrics.per_func_std[name]
        return metrics

    def _generate(self, samples: List[GRPOSample]) -> List[RolloutOutput]:
        """Run a prompt batch through rollout replicas.

        Returns ``List[RolloutOutput]`` (one per request). For Mode A
        (driver-side multi-turn) the per-turn ``response_token_ids`` and
        ``response_loss_mask`` are accumulated inside each ``RolloutOutput``.
        """
        request_config = self._get_request_config()
        # Convert samples to RolloutInferRequest at the engine boundary.
        requests = [s.to_infer_request() for s in samples]

        if self._multi_turn_scheduler is not None and not self._enable_server_multi_turn:
            # Mode A: driver-side trainer loop. run_multi_turn mutates `messages`
            # in place on RolloutInferRequest objects.
            invoke_async_hook(self._multi_turn_scheduler.on_trajectory_start(requests))
            first_turn = [
                RolloutOutput(response=resp) for resp in self._distribute_to_replicas(requests, request_config)
            ]
            return run_multi_turn(
                requests=requests,
                first_turn_outputs=first_turn,
                scheduler=self._multi_turn_scheduler,
                rollout_fn=lambda reqs, cfg:
                [RolloutOutput(response=resp) for resp in self._distribute_to_replicas(reqs, cfg)],
                request_config=request_config,
                max_turns=self._max_turns,
            )

        # Mode B (server-side multi-turn, currently disabled) + single-turn share this path.
        completions = self._distribute_to_replicas(requests, request_config)
        assert len(completions) == len(requests)
        return [RolloutOutput(response=resp) for resp in completions]

    def _postprocess_rollout(self, samples: List[GRPOSample], outputs: List[RolloutOutput]) -> List[GRPOSample]:
        if not outputs:
            return list(samples)
        if len(outputs) != len(samples):
            raise RuntimeError(f'GRPOTrainer: rollout produced {len(outputs)} completions '
                               f'for {len(samples)} samples; shapes mismatch.')
        results = []
        for sample, output in zip(samples, outputs):
            if output is None:
                results.append(sample)
                continue
            sample = copy.deepcopy(sample)
            sample.apply_rollout_output(rollout_output=output)
            results.append(sample)
        return results

    def expand_for_generation(
        self,
        prompt_batch: List[Dict[str, Any]],
    ) -> List[GRPOSample]:
        num_gen = self.num_generations
        samples: List[GRPOSample] = []
        for item in prompt_batch:
            base = GRPOSample.from_row(item)
            if base.messages:
                remove_response(base.messages)
            base.request_id = uuid.uuid4().hex
            samples.append(base)
            for _ in range(num_gen - 1):
                dup = copy.deepcopy(base)
                dup.request_id = uuid.uuid4().hex
                samples.append(dup)
        return samples

    def score_completions(
        self,
        samples: List[GRPOSample],
    ) -> torch.Tensor:
        """Score completions using the backend-agnostic shared helper.

        The driver-side Ray trainer already sees the global prompt/completion
        batch, so no distributed gather is performed here.
        """
        return score_completions(
            samples,
            reward_funcs=self.reward_funcs,
            reward_model_plugins=self.reward_model_plugins,
            use_gym_env=self.use_gym_env,
            device=self.device,
        )

    def compute_advantages(
        self,
        rewards_per_func: torch.Tensor,
        kl_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (advantages, rewards) shaped [N] where N = B * num_gen.

        The driver already holds every completion of each group, so no gather is
        needed before calling the pure advantage function.
        """
        return compute_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=self.reward_weights,
            num_generations=self.num_generations,
            advantage_estimator=self.advantage_estimator,
            scale_rewards=self.scale_rewards,
            kl_in_reward=self.kl_in_reward,
            beta=self.beta,
            kl_values=kl_values,
        )

    def _dynamic_sampling(
        self,
        samples: List[GRPOSample],
        rewards_per_func: torch.Tensor,
    ) -> Tuple[List[GRPOSample], torch.Tensor]:
        num_gen = self.num_generations
        target = len(samples)
        valid_samples: List[GRPOSample] = []
        valid_rewards: List[torch.Tensor] = []
        cur_samples, cur_rewards = samples, rewards_per_func

        for resample_count in range(self.max_resample_times + 1):
            grouped_std = compute_std_for_dynamic_sampling(
                cur_rewards,
                self.reward_weights,
                num_gen,
            )
            keep_mask = grouped_std > 0
            for i in range(len(cur_samples)):
                if keep_mask[i]:
                    valid_samples.append(cur_samples[i])
                    valid_rewards.append(cur_rewards[i])
            logger.info('dynamic_sample round %d: kept %d/%d (std>0), accumulated %d/%d', resample_count,
                        int(keep_mask.sum().item()), len(cur_samples), len(valid_samples), target)
            if len(valid_samples) >= target or resample_count >= self.max_resample_times:
                break
            prompt_batch = next(self._resample_iter)
            if self.truncation_strategy == 'delete':
                prompt_batch = self._resample_failed_prompts(prompt_batch)
            cur_samples = self.expand_for_generation(prompt_batch)
            comp = self._generate(cur_samples)
            cur_samples = self._postprocess_rollout(cur_samples, comp)
            cur_rewards = self.score_completions(cur_samples)

        if len(valid_samples) >= target:
            return valid_samples[:target], torch.stack(valid_rewards[:target])
        logger.warning('dynamic_sample: only %d/%d std>0 samples after %d retries; using original batch.',
                       len(valid_samples), target, self.max_resample_times)
        return samples, rewards_per_func

    def _batch_encode_parallel(self, infer_requests: List[Dict[str, Any]], strict: bool):
        max_workers = max(min(32, os.cpu_count() or 1, len(infer_requests)), 1)
        encoded: List[Dict[str, Any]] = []
        errors: List[Tuple[int, Exception]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(self.template.encode, req, return_length=True) for req in infer_requests]
            concurrent.futures.wait(futures)
            for i, fut in enumerate(futures):
                try:
                    encoded.append(fut.result())
                except Exception as e:  # pragma: no cover
                    if strict:
                        raise
                    errors.append((i, e))
        return encoded, errors

    def encode_rollout_batch(
        self,
        samples: List[GRPOSample],
    ) -> List[GRPOSample]:
        """Encode each sample in place and return the samples.

        This is the driver → worker boundary: the same ``GRPOSample`` objects
        cross the RPC (``tg.compute_logps`` / ``tg.train_step``) — the worker
        feeds them straight to ``collate_to_grpo_micro_batch`` (the shared collate
        used by HF / Megatron). Uses the shared ``encode_sample`` helper so bug
        fixes to loss_mask / non_thinking_prefix propagate across all backends.
        """
        non_thinking_prefix_ids = get_non_thinking_prefix_ids(self.template)
        for sample in samples:
            encoded = encode_sample(sample, self.template, non_thinking_prefix_ids=non_thinking_prefix_ids)
            encoded.pop('_extra_kwargs', None)
            sample.encoded = encoded
        return samples

    def _compute_kl_from_batches(self, grpo_batches: List[GRPOBatch]) -> Optional[torch.Tensor]:
        """Per-sample KL = sum_t (old_lp - ref_lp) * completion_mask, in sample order.

        Reads the [B, T] logps/mask off each micro-batch GRPOBatch (the unified logps
        carrier), so the driver-side DAPO ``kl_in_reward`` penalty matches non-Ray.
        """
        if not (self.kl_in_reward and self.beta != 0.0):
            return None
        kl_values = []
        for gb in grpo_batches:
            old_lp, ref_lp, mask = gb.old_per_token_logps, gb.ref_per_token_logps, gb.completion_mask
            if old_lp is None or ref_lp is None or mask is None:
                return None
            old_lp = old_lp.to(self.device)
            ref_lp = ref_lp.to(self.device)
            mask = mask.to(self.device)
            width = min(old_lp.shape[-1], ref_lp.shape[-1], mask.shape[-1])
            per_token_kl = (old_lp[..., :width] - ref_lp[..., :width]) * mask[..., :width].to(old_lp.dtype)
            kl_values.append(per_token_kl.sum(dim=-1))  # [B] per-sample
        if not kl_values:
            return None
        return torch.cat(kl_values, dim=0)
