# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import concurrent.futures
import os
import ray
import torch
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

from swift.dataset import RowPreprocessor
from swift.rlhf_trainers.utils import compute_grpo_advantages, make_reward_weights, resolve_reward_funcs
from swift.utils import get_logger, to_device
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
                rollout_actors=[r.primary for r in self.rollout_replicas],
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

            if self.beta == 0.0:
                advantages, rewards = self.compute_advantages(rewards_per_func, kl_values=None)

            n_samples = len(rollout_with_outputs)
            chunk_size = n_samples // spg
            for step_idx in range(spg):
                if iteration >= train_iters:
                    break
                chunk_start = step_idx * chunk_size
                chunk_end = chunk_start + chunk_size
                chunk_rollout = rollout_with_outputs[chunk_start:chunk_end]
                chunk_rewards_pf = rewards_per_func[chunk_start:chunk_end]

                batch = self.encode_rollout_batch(chunk_rollout)

                logps_results = tg.compute_logps(batch)
                self._merge_logps(batch, logps_results, 'per_token_logps', 'old_per_token_logps')

                if self.beta != 0.0:
                    ref_results = tg.compute_ref_logps(batch)
                    self._merge_logps(batch, ref_results, 'ref_per_token_logps', 'ref_per_token_logps')
                    kl_values = self._compute_kl(batch)
                    chunk_advantages, rewards = self.compute_advantages(chunk_rewards_pf, kl_values=kl_values)
                else:
                    chunk_advantages = advantages[chunk_start:chunk_end]

                batch['advantages'] = chunk_advantages.to(self.device)

                results = tg.train_step(batch)
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
        if group_std < 1e-6 and iteration <= 5:
            for pi in range(min(grouped.shape[0], 4)):
                logger.info('  prompt[%d] rewards=%s', pi, grouped[pi].tolist())

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
        self._data_iter = self._cyclic_iter(self._dataloader)
        logger.info('GRPO driver dataloader: dataset=%d, prompts_per_generation=%d, num_gen=%d, spg=%d', len(dataset),
                    prompts_per_generation, num_gen, spg)

    @staticmethod
    def _cyclic_iter(dataloader):
        while True:
            for batch in dataloader:
                yield batch

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

        import copy

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
            copy0 = {k: v for k, v in item.items()}
            if 'messages' in copy0:
                copy0['messages'] = [m.copy() for m in copy0['messages']]
                remove_response(copy0['messages'])
            expanded.append(copy0)
            for _ in range(num_gen - 1):
                dup = {k: v for k, v in copy0.items()}
                if 'messages' in dup:
                    dup['messages'] = [m.copy() for m in dup['messages']]
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
    ) -> Dict[str, Any]:
        """Encode all rollout samples into a single global batch dict.

        The driver passes one dict to each WorkerGroup RPC; DP sharding
        is handled by the dispatch layer and micro-batch splitting is
        done inside each worker (``_split_micro_batches``).
        """
        from swift.rlhf_trainers.utils import build_completion_mask_and_seq_lengths, build_rollout_logps

        encoded_list, error_list = self._batch_encode_parallel(rollout_batch, strict=True)
        if error_list:
            raise RuntimeError(f'GRPOTrainer: batch encode failed with errors={error_list}')
        for item in encoded_list:
            item.pop('_extra_kwargs', None)

        template = self.template
        encoded_batch = to_device(template.data_collator(encoded_list, padding_to=self._padding_to), self.device)

        labels = encoded_batch['labels']
        batch_size = len(rollout_batch)

        completion_mask, seq_lengths, _ = build_completion_mask_and_seq_lengths(
            labels,
            batch_size,
            padding_free=template.padding_free,
            encoded_batch=encoded_batch,
            device=self.device,
        )

        encoded_batch.update({
            'completion_mask':
            completion_mask,
            'truncated_mask':
            torch.tensor([b.get('is_truncated', False) for b in rollout_batch], dtype=torch.bool, device=self.device),
            'num_samples':
            batch_size,
            'seq_lengths':
            seq_lengths,
            'rollout_per_token_logps':
            build_rollout_logps(rollout_batch, completion_mask, self.device),
        })
        return encoded_batch

    @staticmethod
    def _merge_logps(batch: Dict[str, Any], results: List, src_key: str, dst_key: str) -> None:
        """Concatenate dp_flat logps results and store in batch."""
        parts = [r[src_key] for r in results if r and isinstance(r, dict) and src_key in r]
        if parts:
            batch[dst_key] = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

    def _compute_kl(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Compute per-sample KL divergence between old and ref logps."""
        if not (self.kl_in_reward and self.beta != 0.0):
            return None
        old_lp = batch.get('old_per_token_logps')
        ref_lp = batch.get('ref_per_token_logps')
        mask = batch.get('completion_mask')
        if old_lp is None or ref_lp is None or mask is None:
            return None
        return ((old_lp - ref_lp) * mask).sum(-1)
