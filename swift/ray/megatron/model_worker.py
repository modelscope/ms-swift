# Copyright (c) ModelScope Contributors. All rights reserved.
"""Single-model worker abstraction for Ray-based Megatron training.

MegatronModelWorker wraps one Megatron model (inference-only).
TrainableModelWorker extends it with training capabilities via _LifecycleTrainer.
"""
from __future__ import annotations

import torch
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence

from swift.utils import gc_collect, get_current_device, get_logger, to_device

logger = get_logger()


class MegatronModelWorker:
    """Wraps a single Megatron model with inference / offload interfaces.

    Two creation paths:
    - ``__init__(args, models)``: wraps models already created in the current
      process (e.g. ref models created by MegatronRLHFTrainer.prepare_model).
    - ``from_pretrained(args, model_dir)``: independently loads a new model
      from disk (e.g. colocated teacher with different weights).
    """

    def __init__(self, args, models, bridge=None):
        self.args = args
        self.models = models
        self.bridge = bridge

    @classmethod
    def from_pretrained(cls, args, model_dir):
        """Load an inference-only model (ref / teacher) from disk."""
        from transformers import AutoConfig

        from swift.megatron.model import get_mcore_model

        hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        models = get_mcore_model(args, hf_config)
        for m in models:
            if not args.use_cpu_initialization:
                m.cuda(torch.cuda.current_device())
            m.requires_grad_(False)
            m.eval()
        models[0].config.bridge.load_weights(models, model_dir)
        return cls(args, models, bridge=models[0].config.bridge)

    def compute_per_token_logps(self, data_iterator, temperature=1.0, enable_routing_replay=False):
        from swift.megatron.trainers.utils import compute_per_token_logps_fn
        return compute_per_token_logps_fn(
            self.models[0],
            self.args,
            data_iterator,
            temperature=temperature,
            enable_routing_replay=enable_routing_replay)

    def offload_to_cpu(self):
        from swift.megatron.trainers.utils import offload_megatron_model_to_cpu
        offload_megatron_model_to_cpu(self.models)
        gc_collect()

    def reload_to_gpu(self, load_grad=False):
        from swift.megatron.trainers.utils import load_megatron_model_to_gpu
        load_megatron_model_to_gpu(self.models, load_grad=load_grad)

    @contextmanager
    def loaded_context(self, load_grad=False):
        """Temporarily load model to GPU, offload on exit.

        No-op if offloading is not configured. Use this to bracket
        inference on an offloaded model (e.g. teacher forward).
        """
        if not getattr(self.args, 'offload_teacher_model', False):
            yield
            return
        self.reload_to_gpu(load_grad=load_grad)
        try:
            yield
        finally:
            self.offload_to_cpu()


class TrainableModelWorker(MegatronModelWorker):
    """Trainable model wrapping a _LifecycleTrainer with optimizer / training step.

    Note: the lifecycle_trainer dependency on MegatronRLHFTrainer is a
    transitional design. Future refactoring should extract the needed
    capabilities (optimizer, model wrapping, ref model context) into
    standalone components so the ray module no longer depends on the
    non-ray trainer hierarchy.
    """

    def __init__(self, args, lifecycle_trainer):
        self._trainer = lifecycle_trainer
        super().__init__(args, lifecycle_trainer.wrapped_models, lifecycle_trainer.bridge)

    @property
    def trainer(self):
        return self._trainer

    @property
    def unwrapped_models(self):
        return self._trainer.unwrapped_models

    def set_forward_step(self, fn):
        self._trainer.forward_step = fn

    def run_train_step(self, data_iterator):
        self._trainer.run_train_step(data_iterator, None)

    def null_ref_context(self):
        return self._trainer.null_ref_context()

    def offload_to_cpu(self):
        from swift.megatron.trainers.utils import offload_megatron_model_to_cpu, offload_megatron_optimizer
        offload_megatron_model_to_cpu(self._trainer.wrapped_models)
        if getattr(self._trainer, 'optimizer', None) and self.args.offload_optimizer:
            offload_megatron_optimizer(self._trainer.optimizer)
        gc_collect()

    def reload_to_gpu(self, load_grad=True):
        from swift.megatron.trainers.utils import load_megatron_model_to_gpu, load_megatron_optimizer
        load_megatron_model_to_gpu(self._trainer.wrapped_models)
        if getattr(self._trainer, 'optimizer', None) and self.args.offload_optimizer:
            load_megatron_optimizer(self._trainer.optimizer)
