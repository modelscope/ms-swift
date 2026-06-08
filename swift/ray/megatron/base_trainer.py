# Copyright (c) ModelScope Contributors. All rights reserved.
"""Base class for Ray-based Megatron trainers (driver-side)."""
from __future__ import annotations

import os
import ray
import torch
from contextlib import contextmanager
from typing import Any, Dict, List

from swift.rlhf_trainers.utils import create_cyclic_iterator
from swift.utils import JsonlWriter, get_logger
from .driver_utils import compute_iter_params

logger = get_logger()


class BaseRayTrainer:
    """Shared driver-side logic for Ray Megatron trainers.

    Subclasses implement ``_prepare_state`` and ``_train_loop``.
    """

    def __init__(
        self,
        worker_groups: Dict[str, Any],
        rollout_replicas: List[Any],
        weight_sync_mode: str = 'nccl',
        sleep_level: int = 1,
        teacher_replicas: List[Any] = None,
    ):
        self.worker_groups = worker_groups
        self.rollout_replicas = rollout_replicas
        self._weight_sync_mode = weight_sync_mode
        self._sleep_level = sleep_level
        self.teacher_replicas = teacher_replicas or []

    def set_data_info(self, data_info: Dict[str, Any]) -> None:
        self._data_info = data_info

    @property
    def train_group(self):
        return self.worker_groups['train']

    @property
    def is_colocated_rollout(self) -> bool:
        from .rollout.replica import RolloutMode
        if not self.rollout_replicas:
            return False
        return self.rollout_replicas[0].mode == RolloutMode.HYBRID

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

    def _distribute_to_replicas(self, batch, params):
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

    @contextmanager
    def _generation_context(self, tg, ckpt):
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
        logger.info('%s driver dataloader: dataset=%d, prompts_per_generation=%d, num_gen=%d, spg=%d',
                    type(self).__name__, len(dataset), prompts_per_generation, num_gen, spg)

    def _build_resample_iterator(self) -> None:
        """Independent cyclic prompt iterator (different shuffle order) used to replace
        encode-failed prompts (truncation_strategy='delete') and to refill DAPO
        dynamic_sample std=0 groups (driver-side)."""
        info = self._data_info
        dataset = info['train_dataset']
        num_gen = int(info.get('num_generations', 1) or 1)
        spg = self._steps_per_generation
        prompts_per_generation = max(info['global_batch_size'] * spg // max(num_gen, 1), 1)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=prompts_per_generation,
            shuffle=True,
            collate_fn=info['data_collator'],
            drop_last=True,
        )
        self._resample_iter = create_cyclic_iterator(loader)

    def _encode_check(self, item) -> None:
        """Try to encode a prompt to detect over-length / encode failures. Raises on
        failure. Subclasses override to match their encode semantics."""
        self.template.encode(item)

    def _resample_failed_prompts(self, prompts):
        """Replace prompts whose encode() fails (e.g. exceeds max_length) with fresh prompts
        from the resample iterator. Mirrors non-ray resample_encode_failed_inputs, but caps the
        TOTAL encode attempts (fail-fast): a systematic failure (e.g. max_length too small, so
        every prompt is over-length) raises quickly instead of churning through the iterator,
        and an empty batch from the iterator breaks the loop instead of spinning forever."""
        required = len(prompts)
        max_rounds = getattr(self, '_max_resample_rounds', 10)
        max_attempts = required * (max_rounds + 1)  # total encode budget (== max_rounds resamples)
        valid, pending = [], list(prompts)
        attempts = n_dropped = 0
        while len(valid) < required and attempts < max_attempts:
            if not pending:
                batch = list(next(self._resample_iter))
                if not batch:  # guard: an empty batch would otherwise spin forever
                    break
                pending.extend(batch)
            item = pending.pop(0)
            attempts += 1
            try:
                self._encode_check(item)
                valid.append(item)
            except Exception:
                n_dropped += 1
        if len(valid) < required:
            raise RuntimeError(f'resample(truncation=delete): only collected {len(valid)}/{required} valid prompts '
                               f'after {attempts} encode attempts ({n_dropped} failed); '
                               f'increase max_length or change truncation_strategy.')
        return valid[:required]

    def _prepare_state(self) -> None:
        raise NotImplementedError

    def _train_loop(self, tg, train_iters, iteration) -> int:
        raise NotImplementedError

    def train(self) -> Any:
        self._prepare_state()
        tg = self.train_group
        self._build_dataloader(tg)
        if getattr(self, '_needs_resample_iterator', False):
            self._build_resample_iterator()

        args_override = compute_iter_params(self._data_info, tg.dp_size)
        meta = tg.setup(args_override)
        train_iters = meta['train_iters']
        iteration = meta['iteration']

        try:
            iteration = self._train_loop(tg, train_iters, iteration)
        finally:
            results = tg.finalize()
        return results

    def _maybe_log_completions(self, rollout_with_outputs, rewards=None, gen_step=None) -> None:
        """Driver-side ``log_completions``: dump prompt/completion (+reward) to
        ``output_dir/completions.jsonl``. No-op unless ``args.log_completions`` is set.
        Completions live on the driver (rollout side), so this is the right place to log them
        (worker on_log handles scalar metrics)."""
        args = self.args
        if not getattr(args, 'log_completions', False) or not rollout_with_outputs:
            return

        if getattr(self, '_completions_writer', None) is None:
            self._completions_writer = JsonlWriter(os.path.join(args.output_dir, 'completions.jsonl'))
        table = []
        for i, item in enumerate(rollout_with_outputs):
            msgs = item.get('messages') or []
            has_resp = bool(msgs) and msgs[-1].get('role') == 'assistant'
            completion = msgs[-1].get('content') or '' if has_resp else ''
            prompt_msgs = msgs[:-1] if has_resp else msgs
            prompt = ''.join(f"{m.get('role')}: {m.get('content')}\n" for m in prompt_msgs)
            row = {'gen_step': gen_step, 'prompt': prompt, 'completion': completion}
            if rewards is not None and i < len(rewards):
                row['reward'] = float(rewards[i])
            table.append(row)
        self._completions_writer.append(table)
