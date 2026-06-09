# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import os
import torch
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from swift.dataset import RowPreprocessor
from swift.infer_engine.protocol import RolloutInferRequest, RolloutOutput
from swift.rlhf_trainers.utils import compute_grpo_advantages, make_reward_weights, resolve_reward_funcs
from swift.rollout import MultiTurnScheduler, invoke_async_hook, multi_turns, run_multi_turn
from swift.utils import get_logger
from .base_trainer import BaseRayTrainer
from .driver_utils import extract_iteration

logger = get_logger()


class GRPOTrainer(BaseRayTrainer):
    """Driver-side GRPO trainer."""

    def _prepare_state(self) -> None:
        assert hasattr(self, '_data_info'), 'call set_data_info() before train()'
        info = self._data_info
        args = info['_driver_args']
        template = info['template']
        self.args = args
        self.template = template
        self.device = torch.device('cpu')

        self.num_generations = args.num_generations
        self.global_batch_size = int(args.global_batch_size)
        self.temperature = args.temperature
        self.advantage_estimator = args.advantage_estimator
        self.scale_rewards = args.scale_rewards
        self.beta = args.beta
        self.kl_in_reward = args.kl_in_reward

        generation_batch_size = getattr(args, 'generation_batch_size', None)
        steps_per_generation = getattr(args, 'steps_per_generation', None)
        if generation_batch_size is not None:
            self._steps_per_generation = generation_batch_size // self.global_batch_size
        elif steps_per_generation is not None:
            self._steps_per_generation = int(steps_per_generation)
        else:
            self._steps_per_generation = 1

        self._padding_to = info.get('_padding_to')
        self._prepare_rewards()
        self._prepare_multi_turn()

        # DAPO dynamic_sample + truncation_strategy='delete' resampling (driver-side).
        self.dynamic_sample = getattr(args, 'dynamic_sample', False)
        self.max_resample_times = getattr(args, 'max_resample_times', 3)
        self.truncation_strategy = getattr(args, 'truncation_strategy', None)
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

            all_chunk_samples = []
            for step_idx in range(spg):
                chunk_start = step_idx * chunk_size
                chunk_end = chunk_start + chunk_size
                chunk_rollout = rollout_with_outputs[chunk_start:chunk_end]
                chunk_samples = self.encode_rollout_batch(chunk_rollout)

                logps_results = tg.compute_logps(chunk_samples)
                for sample, result in zip(chunk_samples, logps_results):
                    sample['old_per_token_logps'] = result['per_token_logps']
                    sample['completion_mask'] = result['completion_mask']
                    if result.get('routed_experts') is not None:
                        sample['routed_experts'] = result['routed_experts']

                if self.beta != 0.0:
                    ref_results = tg.compute_ref_logps(chunk_samples)
                    for sample, result in zip(chunk_samples, ref_results):
                        sample['ref_per_token_logps'] = result['ref_per_token_logps']

                all_chunk_samples.append(chunk_samples)

            for step_idx in range(spg):
                if iteration >= train_iters:
                    break
                chunk_samples = all_chunk_samples[step_idx]
                chunk_start = step_idx * chunk_size
                chunk_end = chunk_start + chunk_size
                chunk_rewards_pf = rewards_per_func[chunk_start:chunk_end]

                if self.beta != 0.0:
                    kl_values = self._compute_kl_from_samples(chunk_samples)
                else:
                    kl_values = None

                chunk_advantages, rewards = self.compute_advantages(chunk_rewards_pf, kl_values=kl_values)
                for i, sample in enumerate(chunk_samples):
                    sample['advantage'] = float(chunk_advantages[i].item())
                for sample in chunk_samples:
                    sample.pop('completion_mask', None)

                results = tg.train_step(
                    chunk_samples,
                    extra_metrics=self._build_grpo_log_metrics(rewards, chunk_advantages, chunk_rewards_pf))
                iteration = extract_iteration(results)

        return iteration

    def _build_grpo_log_metrics(self, rewards, advantages, rewards_per_func) -> Dict[str, float]:
        """Driver-computed GRPO metrics (reward / reward_std / adv_nonzero / per-func),
        injected into the worker megatron on_log so all logging is unified there."""
        K = self.num_generations
        grouped = rewards.view(-1, K)
        metrics = {
            'reward': rewards.mean().item(),
            'reward_std': grouped.std(dim=1).mean().item(),
            'adv_nonzero': (advantages.abs() > 1e-8).float().mean().item(),
        }
        for i in range(rewards_per_func.shape[1]):
            val = rewards_per_func[:, i].nanmean().item()
            if val == val:  # skip NaN (a reward func produced no finite value this step)
                metrics[self.reward_func_names[i]] = val
        return metrics

    def _generate(self, expanded_batch) -> List[RolloutOutput]:
        """Run a prompt batch through rollout replicas.

        Returns ``List[RolloutOutput]`` (one per request). For Mode A
        (driver-side multi-turn) the per-turn ``response_token_ids`` and
        ``response_loss_mask`` are accumulated inside each ``RolloutOutput``.
        """
        request_config = self._get_request_config()
        prompt_batch = list(expanded_batch)

        if self._multi_turn_scheduler is not None and not self._enable_server_multi_turn:
            # Mode A: driver-side trainer loop. Convert dict prompts to
            # RolloutInferRequest so the loop can mutate `messages` in place.
            requests = self._inputs_to_requests(prompt_batch)
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
        completions = self._distribute_to_replicas(prompt_batch, request_config)
        assert len(completions) == len(prompt_batch)
        return [RolloutOutput(response=resp) for resp in completions]

    def _inputs_to_requests(self, inputs: Sequence[Dict[str, Any]]) -> List[RolloutInferRequest]:
        """Convert driver-side prompt dicts to ``RolloutInferRequest`` objects."""
        from dacite import from_dict

        REQUEST_METADATA_FIELDS = ('messages', 'images', 'audios', 'videos', 'tools', 'objects', 'uuid')
        requests: List[RolloutInferRequest] = []
        for data in inputs:
            if isinstance(data, RolloutInferRequest):
                requests.append(data)
                continue
            payload = {key: data[key] for key in REQUEST_METADATA_FIELDS if key in data and data[key] is not None}
            if 'uuid' not in payload:
                payload['uuid'] = data.get('request_id') or uuid.uuid4().hex
            requests.append(from_dict(RolloutInferRequest, payload))
        return requests

    def _postprocess_rollout(self, rollout_batch, outputs: List[RolloutOutput]):
        """Merge ``RolloutOutput`` data back into each sample.

        Mirrors ``MegatronRolloutMixin._postprocess_rollout_outputs`` — including
        per-turn ``response_token_ids`` / ``response_loss_mask`` handling and the
        multimodal pass-through from ``rollout_infos``.
        """
        if not outputs:
            return list(rollout_batch)

        if len(outputs) != len(rollout_batch):
            raise RuntimeError(f'GRPOTrainer: rollout produced {len(outputs)} completions '
                               f'for {len(rollout_batch)} samples; shapes mismatch.')

        from swift.utils import remove_response

        merged = []
        for inp, output in zip(rollout_batch, outputs):
            item = dict(inp)
            if output is None:
                merged.append(item)
                continue
            response = output.response
            choice = response.choices[0]

            if output.messages:
                item['messages'] = output.messages
            else:
                messages = copy.deepcopy(item.get('messages') or [])
                remove_response(messages)
                messages.append({'role': 'assistant', 'content': choice.message.content or ''})
                item['messages'] = messages

            if output.response_token_ids:
                item['response_token_ids'] = output.response_token_ids
                if output.response_loss_mask:
                    item['response_loss_mask'] = output.response_loss_mask
            else:
                item['response_token_ids'] = choice.token_ids or []

            if output.rollout_infos:
                item['rollout_infos'] = output.rollout_infos
                # Multimodal pass-through: schedulers / envs may inject observation
                # images / videos / audios mid-trajectory.
                for key in ('images', 'videos', 'audios'):
                    if key in output.rollout_infos:
                        item[key] = output.rollout_infos[key]

            item['finish_reason'] = choice.finish_reason or 'stop'
            item['is_truncated'] = item['finish_reason'] == 'length'
            item['add_eos'] = False

            if output.rollout_logprobs:
                item['rollout_logprobs'] = output.rollout_logprobs
            elif choice.logprobs and 'content' in choice.logprobs:
                item['rollout_logprobs'] = [[lp['logprob'] for lp in choice.logprobs['content']]]

            if getattr(choice, 'routed_experts', None) is not None:
                item['routed_experts'] = choice.routed_experts
            merged.append(item)
        return merged

    def expand_for_generation(
        self,
        prompt_batch: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Repeat each prompt ``num_generations`` times before rollout.

        Each item is passed directly to ``VllmEngine.infer_async`` which
        handles encoding internally.
        """
        from swift.utils import remove_response

        num_gen = self.num_generations
        expanded = []
        for item in prompt_batch:
            copy0 = copy.deepcopy(item)
            if 'messages' in copy0:
                remove_response(copy0['messages'])
            expanded.append(copy0)
            for _ in range(num_gen - 1):
                dup = copy.deepcopy(copy0)
                expanded.append(dup)
        return expanded

    def score_completions(
        self,
        rollout_batch: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        device = self.device

        # Gym path: pull `total_reward` from the multi-turn scheduler's rollout_infos and
        # append it as an extra column so reward_weights can blend it with reward_funcs.
        if self.use_gym_env:
            gym_reward = torch.tensor([inp['rollout_infos']['total_reward'] for inp in rollout_batch],
                                      dtype=torch.float32,
                                      device=device).unsqueeze(1)
            if not self.reward_funcs:
                return gym_reward
            func_rewards = self._compute_reward_funcs(rollout_batch)
            return torch.cat([func_rewards, gym_reward], dim=1)

        return self._compute_reward_funcs(rollout_batch)

    def _compute_reward_funcs(
        self,
        rollout_batch: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        device = self.device
        rewards_per_func = torch.zeros((len(rollout_batch), len(self.reward_funcs)), device=device)
        completions = [inp['messages'][-1]['content'] for inp in rollout_batch]
        reward_kwargs: Dict[str, Any] = {}
        reward_kwargs.update(RowPreprocessor.rows_to_batched(list(rollout_batch)))

        def _as_tensor(values):
            return torch.tensor([torch.nan if v is None else v for v in values], dtype=torch.float32, device=device)

        def _is_async(func):
            return asyncio.iscoroutinefunction(func) or asyncio.iscoroutinefunction(getattr(func, '__call__', None))

        sync_indices, async_indices = [], []
        for i, func in enumerate(self.reward_funcs):
            (async_indices if _is_async(func) else sync_indices).append(i)

        for i in sync_indices:
            rewards_per_func[:, i] = _as_tensor(self.reward_funcs[i](completions, **reward_kwargs))

        if async_indices:

            async def _run_all():
                return await asyncio.gather(
                    *[self.reward_funcs[i](completions, **reward_kwargs) for i in async_indices])

            for idx, out in zip(async_indices, asyncio.run(_run_all())):
                rewards_per_func[:, idx] = _as_tensor(out)

        if torch.isnan(rewards_per_func).all(dim=1).any():
            logger.warning('GRPOTrainer: some rows have NaN rewards for every reward function.')

        return rewards_per_func

    def compute_advantages(
        self,
        rewards_per_func: torch.Tensor,
        kl_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (advantages, rewards) shaped [N] where N = B * num_gen."""
        return compute_grpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=self.reward_weights,
            num_generations=self.num_generations,
            advantage_estimator=self.advantage_estimator,
            scale_rewards=self.scale_rewards,
            kl_in_reward=self.kl_in_reward,
            beta=self.beta,
            kl_values=kl_values,
        )

    def _encode_check(self, item) -> None:
        """Detect over-length / encode failures for a GRPO prompt. The response is
        removed first (GRPO encodes prompt-only at rollout time). Raises on failure."""
        from swift.utils import remove_response
        probe = dict(item)
        if probe.get('messages'):
            probe['messages'] = [m.copy() for m in probe['messages']]
            remove_response(probe['messages'])
        self.template.encode(probe)

    def _dynamic_sampling(
        self,
        rollout_with_outputs: List[Dict[str, Any]],
        rewards_per_func: torch.Tensor,
    ) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
        num_gen = self.num_generations
        target = len(rollout_with_outputs)
        valid_samples: List[Dict[str, Any]] = []
        valid_rewards: List[torch.Tensor] = []
        cur_rollout, cur_rewards = rollout_with_outputs, rewards_per_func

        for resample_count in range(self.max_resample_times + 1):
            rewards = (cur_rewards * self.reward_weights.unsqueeze(0)).nansum(dim=1)
            if num_gen > 1:
                grouped_std = rewards.view(-1, num_gen).std(dim=1).repeat_interleave(num_gen)
            else:
                grouped_std = torch.zeros_like(rewards)
            keep_mask = grouped_std > 0
            for i in range(len(cur_rollout)):
                if keep_mask[i]:
                    valid_samples.append(cur_rollout[i])
                    valid_rewards.append(cur_rewards[i])
            logger.info('dynamic_sample round %d: kept %d/%d (std>0), accumulated %d/%d', resample_count,
                        int(keep_mask.sum().item()), len(cur_rollout), len(valid_samples), target)
            if len(valid_samples) >= target or resample_count >= self.max_resample_times:
                break
            prompt_batch = next(self._resample_iter)
            if self.truncation_strategy == 'delete':
                prompt_batch = self._resample_failed_prompts(prompt_batch)
            rb = self.expand_for_generation(prompt_batch)
            comp = self._generate(rb)
            cur_rollout = self._postprocess_rollout(rb, comp)
            cur_rewards = self.score_completions(cur_rollout)

        if len(valid_samples) >= target:
            return valid_samples[:target], torch.stack(valid_rewards[:target])
        logger.warning('dynamic_sample: only %d/%d std>0 samples after %d retries; using original batch.',
                       len(valid_samples), target, self.max_resample_times)
        return rollout_with_outputs, rewards_per_func

    def _batch_encode_parallel(self, infer_requests: Sequence[Dict[str, Any]], strict: bool):
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
        rollout_batch: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Encode rollout samples and keep them as per-sample payloads."""
        from swift.rlhf_trainers.utils import replace_assistant_response_with_ids

        rollout_for_encode: List[Dict[str, Any]] = []
        for data in rollout_batch:
            item = dict(data)
            if 'messages' in item and item['messages'] is not None:
                item['messages'] = [m.copy() for m in item['messages']]
            if 'response_token_ids' in item and item['response_token_ids']:
                loss_mask = None
                if 'response_loss_mask' in item and item['response_loss_mask']:
                    loss_mask = item['response_loss_mask']
                item['messages'] = replace_assistant_response_with_ids(item['messages'], item['response_token_ids'],
                                                                       loss_mask)
            rollout_for_encode.append(item)

        encoded_list, error_list = self._batch_encode_parallel(rollout_for_encode, strict=True)
        if error_list:
            raise RuntimeError(f'GRPOTrainer: batch encode failed with errors={error_list}')
        samples: List[Dict[str, Any]] = []
        for encoded, rollout in zip(encoded_list, rollout_batch):
            encoded.pop('_extra_kwargs', None)
            samples.append({
                'encoded': encoded,
                'is_truncated': bool(rollout.get('is_truncated', False)),
                'rollout_logprobs': rollout.get('rollout_logprobs'),
            })
            if rollout.get('routed_experts') is not None:
                samples[-1]['routed_experts'] = rollout.get('routed_experts')
        return samples

    def _compute_kl_from_samples(self, samples: Sequence[Dict[str, Any]]) -> Optional[torch.Tensor]:
        if not (self.kl_in_reward and self.beta != 0.0):
            return None
        kl_values = []
        for sample in samples:
            old_lp = sample.get('old_per_token_logps')
            ref_lp = sample.get('ref_per_token_logps')
            mask = sample.get('completion_mask')
            if old_lp is None or ref_lp is None or mask is None:
                return None
            old_lp = old_lp.to(self.device)
            ref_lp = ref_lp.to(self.device)
            mask = mask.to(self.device)
            width = min(old_lp.shape[-1], ref_lp.shape[-1], mask.shape[-1])
            kl_values.append(((old_lp[:width] - ref_lp[:width]) * mask[:width].to(old_lp.dtype)).sum())
        if not kl_values:
            return None
        return torch.stack(kl_values)
