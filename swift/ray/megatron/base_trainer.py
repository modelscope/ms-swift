# Copyright (c) ModelScope Contributors. All rights reserved.
"""Base class for Ray-based Megatron trainers (driver-side)."""
from __future__ import annotations

import os
import ray
import torch
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

from swift.rl_core.data import GRPOBatch, OnPolicySample
from swift.rl_core.resample import resample_encode_failed_inputs
from swift.rlhf_trainers.utils import create_cyclic_iterator
from swift.template.base import Template
from swift.utils import JsonlWriter, get_logger
from .driver_utils import compute_iter_params
from .worker_group import DPDispatchedDict

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

    def _build_dataloader(self):
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

    def _resample_failed_prompts(self, prompts: List[dict], strip_response: bool = True) -> List[dict]:
        """Replace prompts whose encode fails with fresh ones from the resample iterator.
        Shares the backend-agnostic loop with HF / Megatron (see ``rl_core.resample``)."""
        return resample_encode_failed_inputs(
            self.template,
            self._resample_iter,
            prompts,
            max_resample_rounds=getattr(self, '_max_resample_rounds', 10),
            strip_response=strip_response,
        )

    def _collate_for_workers(self, tg, samples: List[OnPolicySample],
                             **collate_kwargs) -> Tuple['DPDispatchedDict', List['GRPOBatch']]:
        """Driver-side collate: ``List[OnPolicySample]`` -> ``({dp_rank: [model_inputs]}, flat_grpo_batches)``.

        The driver owns the whole global batch, so it does the (pure-CPU)
        ``template.data_collator`` itself — mirroring the non-Ray Megatron path
        where each rank encodes then collates its own micro-batches. The worker
        receives the collated micro-batches directly (dispatch='dp') and only runs
        the rank-local ``prepare_batch`` (PP/CP slice) + forward.

        Layout: split the global batch into ``dp_size`` contiguous shards, each into
        ``micro_batch_size`` micro-batches, collate each via the shared
        ``collate_to_grpo_micro_batch``. The second return value is the per-micro-batch
        ``GRPOBatch`` list IN SAMPLE ORDER (dp_rank major, then micro-batch): the same
        objects referenced inside the dispatch dict, so the caller fills old/ref logps +
        advantages on them and they reach ``train_step`` via the dispatch dict.
        """
        from swift.rlhf_trainers.utils import collate_to_grpo_micro_batch

        dp_size = tg.dp_size
        mbs = int(self.args.micro_batch_size)
        n = len(samples)
        if n % dp_size != 0:
            raise ValueError(f'_collate_for_workers: batch size {n} not divisible by dp_size {dp_size}.')
        shard_size = n // dp_size

        dispatch = DPDispatchedDict()
        flat_grpo_batches: List['GRPOBatch'] = []
        for dp_rank in range(dp_size):
            shard = samples[dp_rank * shard_size:(dp_rank + 1) * shard_size]
            micro_batches = []
            for i in range(0, len(shard), mbs):
                chunk = shard[i:i + mbs]
                model_inputs, grpo_batch = collate_to_grpo_micro_batch(
                    chunk,
                    self.template,
                    device=self.device,
                    padding_to=self._padding_to,
                    router_replay_mode=getattr(self.args, 'router_replay_mode', 'disabled'),
                    **collate_kwargs,
                )
                model_inputs['grpo_batch'] = grpo_batch
                micro_batches.append(model_inputs)
                flat_grpo_batches.append(grpo_batch)
            dispatch[dp_rank] = micro_batches
        return dispatch, flat_grpo_batches

    def _prepare_state(self) -> None:
        """Shared ``_prepare_state`` prefix for all Ray trainers.

        Resolves the fields every driver needs (args / template / device /
        global_batch_size / temperature / beta / steps_per_generation /
        padding_to) from ``_data_info``. Subclasses call ``super()._prepare_state()``
        first, then set algorithm-specific state (advantage / dynamic_sample for
        GRPO; lmbda / teacher for GKD).
        """
        assert hasattr(self, '_data_info'), 'call set_data_info() before train()'
        info = self._data_info
        args = info['_driver_args']
        self.args = args
        self.template: Template = info['template']
        self.device = torch.device('cpu')

        self.global_batch_size = int(args.global_batch_size)
        self.temperature = args.temperature
        self.beta = args.beta

        # steps_per_generation>1: one generation feeds spg training steps.
        gen_bs = getattr(args, 'generation_batch_size', None)
        spg = getattr(args, 'steps_per_generation', None)
        if gen_bs is not None:
            self._steps_per_generation = max(int(gen_bs) // self.global_batch_size, 1)
        elif spg is not None:
            self._steps_per_generation = int(spg)
        else:
            self._steps_per_generation = 1

        self._padding_to = info.get('_padding_to')

    def _train_loop(self, tg, train_iters, iteration) -> int:
        raise NotImplementedError

    def train(self) -> Any:
        self._prepare_state()
        tg = self.train_group
        self._build_dataloader()
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

    def _maybe_log_completions(self, rollout_with_outputs: List[OnPolicySample], rewards=None, gen_step=None) -> None:
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
            msgs = item.messages
            has_resp = bool(msgs) and msgs[-1].get('role') == 'assistant'
            completion = self._decode_log_content(msgs[-1].get('content')) if has_resp else ''
            prompt_msgs = msgs[:-1] if has_resp else msgs
            row = {'gen_step': gen_step, 'prompt': self._format_log_prompt(prompt_msgs), 'completion': completion}
            if rewards is not None and i < len(rewards):
                row['reward'] = float(rewards[i])
            table.append(row)
        self._completions_writer.append(table)

    def _format_log_prompt(self, prompt_msgs) -> str:
        """Render the prompt as the model actually sees it (chat template applied),
        matching the non-Ray ``_apply_chat_template_to_messages_list`` so completions.jsonl
        is consistent across backends. Falls back to a plain role/content join if encode fails
        (e.g. multimodal placeholders the driver template can't re-encode standalone)."""
        from swift.template import TemplateInputs
        try:
            template_inputs = TemplateInputs.from_dict({'messages': [dict(m) for m in prompt_msgs]})
            res = self.template.encode(template_inputs)
            return self.template.safe_decode(res['input_ids'])
        except Exception:
            return ''.join(f"{m.get('role')}: {m.get('content')}\n" for m in prompt_msgs)

    def _decode_log_content(self, content) -> str:
        """Decode an assistant message content for logging (mirrors non-Ray)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return self.template.safe_decode(content)
        if isinstance(content, dict) and 'input_ids' in content:
            return self.template.safe_decode(content['input_ids'])
        return str(content) if content is not None else ''
