# Copyright (c) ModelScope Contributors. All rights reserved.
import collections

import torch

from swift.utils import get_logger
from .base import TrainerCallback

logger = get_logger()


class DynamicMixingCallback(TrainerCallback):
    """Callback that dynamically adjusts data sampling weights based on per-domain loss."""

    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.update_steps = args.dynamic_mix_update_steps
        self.temperature = args.dynamic_mix_temperature
        self.warmup_steps = args.dynamic_mix_warmup_steps
        self._sampler = None
        self._domain_names = None
        self._loss_buffer = collections.defaultdict(list)
        self._last_update_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        # Reuse the already-created training dataloader instead of re-calling the getter
        dataloader = getattr(self.trainer, 'train_dataloader', None)
        if dataloader is None:
            logger.warning('DynamicMixingCallback: train_dataloader not found, dynamic mixing disabled.')
            return
        sampler = getattr(dataloader, 'batch_sampler', None)
        from swift.dataloader import DynamicMixBatchSampler
        # Unwrap SkipBatchSampler if present
        if hasattr(sampler, 'batch_sampler'):
            sampler = sampler.batch_sampler
        if not isinstance(sampler, DynamicMixBatchSampler):
            logger.warning('DynamicMixingCallback: sampler is not '
                           'DynamicMixBatchSampler, dynamic mixing disabled.')
            return
        self._sampler = sampler
        self._domain_names = sampler.domain_names
        domain_sizes = {n: len(sampler.domain_indices[n]) for n in self._domain_names}
        logger.info(f'Dynamic mixing initialized. Domains: {domain_sizes}')
        logger.info(f'Initial probabilities: {sampler.probabilities}')

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._sampler is None or logs is None:
            return
        # Capture loss_{channel} values from logs
        for name in self._domain_names:
            # channel=None samples have log key "loss_None"
            log_key = f'loss_{name}'
            if log_key in logs and logs[log_key] is not None:
                self._loss_buffer[name].append(logs[log_key])

        # Check if it's time to update weights
        if (state.global_step >= self.warmup_steps
                and state.global_step - self._last_update_step >= self.update_steps):
            self._update_probabilities(state.global_step)

    def _update_probabilities(self, global_step):
        domain_losses = {}
        for name in self._domain_names:
            values = self._loss_buffer.get(name, [])
            if values:
                domain_losses[name] = sum(values) / len(values)

        if not domain_losses:
            logger.info(f'Step {global_step}: no channel loss data yet, '
                        'skipping dynamic mix update.')
            return

        # Use global mean for domains without loss data
        mean_loss = sum(domain_losses.values()) / len(domain_losses)
        for name in self._domain_names:
            if name not in domain_losses:
                domain_losses[name] = mean_loss

        # softmax(loss / T)
        loss_tensor = torch.tensor([domain_losses[n] for n in self._domain_names])
        probs = torch.softmax(loss_tensor / self.temperature, dim=0)
        probs_dict = {name: probs[i].item() for i, name in enumerate(self._domain_names)}

        self._sampler.set_probabilities(probs_dict)
        self._loss_buffer.clear()
        self._last_update_step = global_step

        # Log new weights to metrics (will appear in tensorboard/wandb)
        for name, prob in probs_dict.items():
            self.trainer.custom_metrics['train'][f'mix_prob_{name}'].update(torch.tensor([prob]))

        logger.info(f'Step {global_step}: updated mix probabilities: {probs_dict}')
