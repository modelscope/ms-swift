# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import os
import ray
import torch
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

from swift.dataset import RowPreprocessor
from swift.rlhf_trainers.utils import (compute_grpo_advantages, create_cyclic_iterator, make_reward_weights,
                                       resolve_reward_funcs)
from swift.utils import get_logger
from .driver_utils import compute_iter_params, extract_iteration, extract_train_metrics

logger = get_logger()


class GRPOTrainer:
    """Driver-side GRPO trainer.

    Constructed by ``MegatronRayPipeline._create_trainer`` with the
    WorkerGroups and rollout replicas.  Heavy state (args / template /
    reward funcs) is bootstrapped lazily in ``_prepare_state`` on the
    first call to ``train()``.
    """

    def __init__(
        self,
        worker_groups: Dict[str, Any],
        rollout_replicas: List[Any],
        weight_sync_mode: str = 'nccl',
        sleep_level: int = 1,
    ):
        self.worker_groups = worker_groups
        self.rollout_replicas = rollout_replicas
        self._weight_sync_mode = weight_sync_mode
        self._sleep_level = sleep_level

    def set_data_info(self, data_info: Dict[str, Any]) -> None:
        self._data_info = data_info

    @property
    def train_group(self):
        return self.worker_groups['train']

    def _distribute_to_replicas(self, batch, params):
        """Split *batch* across all replicas and gather results in parallel."""
        n = len(self.rollout_replicas)
        chunk_size = (len(batch) + n - 1) // n
        refs = []
        for i, replica in enumerate(self.rollout_replicas):
            shard = batch[i * chunk_size:(i + 1) * chunk_size]
            if not shard:
                continue
            refs.append(replica.generate(shard, params))
        parts = ray.get(refs)
        result = []
        for p in parts:
            result.extend(p)
        return result

    @property
    def is_colocated_rollout(self) -> bool:
        """True if rollout and train share the same placement group."""
        from .rollout.replica import RolloutMode
        if not self.rollout_replicas:
            return False
        return self.rollout_replicas[0].mode == RolloutMode.HYBRID

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

    def _prepare_rewards(self):
        args = self.args
        reward_funcs_cfg = (args.reward_funcs or []).copy()
        if not isinstance(reward_funcs_cfg, list):
            reward_funcs_cfg = [reward_funcs_cfg]

        self.reward_funcs, self.reward_func_names = resolve_reward_funcs(reward_funcs_cfg, args=args)
        self.reward_weights = make_reward_weights(args.reward_weights, len(self.reward_funcs), self.device)

        if not self.reward_funcs:
            raise ValueError('GRPOTrainer: no reward functions configured')

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

    def train(self) -> Any:
        self._prepare_state()

        tg = self.train_group
        self._build_dataloader(tg)

        args_override = compute_iter_params(self._data_info, tg.dp_size)
        meta = tg.setup(args_override)
        train_iters = meta['train_iters']
        iteration = meta['iteration']

        try:
            iteration = self._train_loop(tg, train_iters, iteration)
        finally:
            results = tg.finalize()

        return results

    @property
    def ckpt_manager(self):
        if not hasattr(self, '_ckpt_manager'):
            from .checkpoint_engine import CheckpointEngineManager
            tg = self.train_group

            self._ckpt_manager = CheckpointEngineManager(
                train_actors=tg.workers,
                rollout_replicas=self.rollout_replicas,
                weight_sync_mode=self._weight_sync_mode,
                is_colocated=self.is_colocated_rollout,
                sleep_level=self._sleep_level,
                train_group=tg,
            )
        return self._ckpt_manager

    @contextmanager
    def _generation_context(self, tg, ckpt):
        """Offload train model + wake vLLM for generation, reload afterwards."""
        offload_model = getattr(self.args, 'offload_model', False)
        offload_optimizer = getattr(self.args, 'offload_optimizer', False)
        enable_offload = offload_model or offload_optimizer or self.is_colocated_rollout

        if enable_offload:
            tg.offload_to_cpu()
        if self.is_colocated_rollout:
            ckpt.wake_up_rollout(tags=['kv_cache'])

        try:
            yield
        finally:
            tg.finalize_generation()
            if self.is_colocated_rollout:
                ckpt.sleep_rollout()
            if enable_offload:
                tg.reload_to_gpu()

    def _train_loop(self, tg, train_iters, iteration):
        ckpt = self.ckpt_manager
        merge_and_sync = not self.args.vllm_enable_lora
        spg = self._steps_per_generation

        while iteration < train_iters:
            ckpt.sync_weights(merge_and_sync=merge_and_sync)

            with self._generation_context(tg, ckpt):
                prompt_batch = next(self._data_iter)
                rollout_batch = self.expand_for_generation(prompt_batch)
                completions = self._generate(rollout_batch)
                rollout_with_outputs = self._postprocess_rollout(rollout_batch, completions)

            rewards_per_func = self.score_completions(rollout_with_outputs)

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

                results = tg.train_step(chunk_samples)
                iteration = extract_iteration(results)
                train_m = extract_train_metrics(results)

                self._log_iteration(iteration, train_iters, rewards, chunk_advantages, chunk_rewards_pf, train_m)

        return iteration

    def _log_iteration(self, iteration, train_iters, rewards, advantages, rewards_per_func, train_m):
        mean_reward = rewards.mean().item()
        nonzero_adv = (advantages.abs() > 1e-8).float().mean().item()
        K = self.num_generations
        grouped = rewards.view(-1, K)
        group_std = grouped.std(dim=1).mean().item()
        per_func_parts = [
            f'{self.reward_func_names[i]}={rewards_per_func[:, i].nanmean().item():.4f}'
            for i in range(rewards_per_func.shape[1])
        ]
        per_func_str = '  '.join(per_func_parts)
        core_keys = ('loss', 'grad_norm', 'lr', 'kl')
        core_parts = [f'{k}={train_m[k]:.6f}' for k in core_keys if k in train_m]
        extra_parts = [f'{k}={v:.6f}' for k, v in train_m.items() if k not in core_keys]
        train_str = '  '.join(core_parts + extra_parts)
        logger.info('iter %d/%d  reward=%.4f  group_std=%.4f  adv_nonzero=%.1f%%  %s  %s', iteration, train_iters,
                    mean_reward, group_std, nonzero_adv * 100, per_func_str, train_str)

    def _build_dataloader(self, tg):
        info = self._data_info
        dataset = info['train_dataset']
        num_gen = int(info.get('num_generations', 1) or 1)
        spg = self._steps_per_generation
        prompts_per_generation = max(info['global_batch_size'] * spg // max(num_gen, 1), 1)
        self._dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=prompts_per_generation,
            shuffle=True,
            collate_fn=info['data_collator'],
            drop_last=True,
        )
        self._data_iter = create_cyclic_iterator(self._dataloader)
        logger.info('GRPO driver dataloader: dataset=%d, prompts_per_generation=%d, num_gen=%d, spg=%d', len(dataset),
                    prompts_per_generation, num_gen, spg)

    def _generate(self, expanded_batch):
        """Run a prompt batch through rollout replicas."""
        return self._distribute_to_replicas(list(expanded_batch), self._get_request_config())

    def _postprocess_rollout(self, rollout_batch, outputs):
        """Merge ChatCompletionResponse outputs back into each sample.

        Mirrors ``MegatronGRPOTrainer._postprocess_rollout_outputs``.
        """
        if not outputs:
            return list(rollout_batch)

        if len(outputs) != len(rollout_batch):
            raise RuntimeError(f'GRPOTrainer: rollout produced {len(outputs)} completions '
                               f'for {len(rollout_batch)} samples; shapes mismatch.')

        from swift.utils import remove_response

        merged = []
        for inp, response in zip(rollout_batch, outputs):
            item = dict(inp)
            if response is None:
                merged.append(item)
                continue
            choice = response.choices[0]
            messages = copy.deepcopy(item.get('messages') or [])
            remove_response(messages)
            messages.append({'role': 'assistant', 'content': choice.message.content or ''})
            item['messages'] = messages
            item['response_token_ids'] = choice.token_ids or []
            item['finish_reason'] = choice.finish_reason or 'stop'
            item['is_truncated'] = item['finish_reason'] == 'length'
            item['add_eos'] = False
            if choice.logprobs and 'content' in choice.logprobs:
                item['rollout_logprobs'] = [[lp['logprob'] for lp in choice.logprobs['content']]]
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
