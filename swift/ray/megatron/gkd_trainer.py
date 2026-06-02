# Copyright (c) ModelScope Contributors. All rights reserved.
"""Driver-side GKD trainer for Ray-based Megatron training."""
from __future__ import annotations

import copy
import torch
from typing import Any, Dict, List, Optional, Sequence

from swift.infer_engine.protocol import RolloutOutput
from swift.utils import get_logger
from .base_trainer import BaseRayTrainer
from .driver_utils import extract_iteration, extract_train_metrics

logger = get_logger()


class GKDTrainer(BaseRayTrainer):
    """Driver-side GKD trainer: student generation + teacher distillation."""

    def _prepare_state(self) -> None:
        assert hasattr(self, '_data_info'), 'call set_data_info() before train()'
        info = self._data_info
        args = info['_driver_args']
        template = info['template']
        self.args = args
        self.template = template
        self.device = torch.device('cpu')

        self.global_batch_size = int(args.global_batch_size)
        self.temperature = getattr(args, 'temperature', 1.0)
        self.beta = getattr(args, 'beta', 0.5)
        self.sft_alpha = getattr(args, 'sft_alpha', 0.0)

        self._steps_per_generation = 1
        self._padding_to = info.get('_padding_to')
        self._teacher_model_dir = getattr(args, 'teacher_model', None) or getattr(args, 'teacher_model_dir', None)
        self._teacher_model_server = getattr(args, 'teacher_model_server', None)

    def _train_loop(self, tg, train_iters, iteration):
        ckpt = self.ckpt_manager

        # Initialize colocated teacher if configured
        if self._teacher_model_dir and not self._teacher_model_server:
            tg.execute('init_teacher_model', self._teacher_model_dir)
            logger.info('Colocated teacher model initialized from %s', self._teacher_model_dir)

        while iteration < train_iters:
            ckpt.sync_weights(merge_and_sync=True)

            with self._generation_context(tg, ckpt):
                prompt_batch = next(self._data_iter)
                rollout_batch = self._expand_for_generation(prompt_batch)
                completions = self._generate(rollout_batch)
                rollout_with_outputs = self._postprocess_rollout(rollout_batch, completions)

            samples = self._encode_rollout_batch(rollout_with_outputs)

            # Teacher forward: attach teacher_output to each sample
            teacher_outputs = tg.compute_teacher_logits(samples)
            for sample, t_out in zip(samples, teacher_outputs):
                sample['teacher_output'] = t_out

            results = tg.train_step(samples)
            iteration = extract_iteration(results)
            train_m = extract_train_metrics(results)
            self._log_iteration(iteration, train_iters, train_m)

        return iteration

    def _log_iteration(self, iteration, train_iters, train_m):
        core_keys = ('loss', 'jsd_loss', 'sft_loss', 'grad_norm', 'lr')
        core_parts = [f'{k}={train_m[k]:.6f}' for k in core_keys if k in train_m]
        extra_parts = [f'{k}={v:.6f}' for k, v in train_m.items() if k not in core_keys]
        train_str = '  '.join(core_parts + extra_parts)
        logger.info('iter %d/%d  %s', iteration, train_iters, train_str)

    def _expand_for_generation(self, prompt_batch):
        from swift.utils import remove_response
        expanded = []
        for item in prompt_batch:
            item_copy = copy.deepcopy(item)
            if 'messages' in item_copy:
                remove_response(item_copy['messages'])
            expanded.append(item_copy)
        return expanded

    def _generate(self, batch) -> List[RolloutOutput]:
        from swift.infer_engine.protocol import RequestConfig
        args = self.args
        request_config = RequestConfig(
            n=1,
            max_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=getattr(args, 'top_p', 1.0),
            top_k=getattr(args, 'top_k', -1),
            stop=getattr(args, 'stop_words', None),
            return_details=True,
        )
        completions = self._distribute_to_replicas(list(batch), request_config)
        return [RolloutOutput(response=resp) for resp in completions]

    def _postprocess_rollout(self, rollout_batch, outputs):
        from swift.utils import remove_response
        merged = []
        for inp, output in zip(rollout_batch, outputs):
            item = dict(inp)
            if output is None:
                merged.append(item)
                continue
            response = output.response
            choice = response.choices[0]
            messages = copy.deepcopy(item.get('messages') or [])
            remove_response(messages)
            messages.append({'role': 'assistant', 'content': choice.message.content or ''})
            item['messages'] = messages
            item['response_token_ids'] = choice.token_ids or []
            item['finish_reason'] = choice.finish_reason or 'stop'
            item['is_truncated'] = item['finish_reason'] == 'length'
            item['add_eos'] = False
            merged.append(item)
        return merged

    def _encode_rollout_batch(self, rollout_batch):
        """Encode rollout samples for the training workers."""
        template = self.template
        samples = []
        for item in rollout_batch:
            encoded = template.encode(item)
            if encoded is None:
                continue
            samples.append({'encoded': encoded})
        return samples
