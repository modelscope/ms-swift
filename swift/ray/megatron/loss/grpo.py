# Copyright (c) ModelScope Contributors. All rights reserved.
"""GRPO loss for Ray-based Megatron training.

``GRPOLoss`` reuses ``MegatronGRPOTrainer.forward_step`` and
``loss_func`` via a dummy trainer instance (created with ``__new__``
to skip heavy ``__init__`` side-effects like vLLM / reward setup).
This is an internal implementation detail for code reuse --
users writing custom losses do NOT need to understand or replicate
this pattern; they simply subclass ``Loss`` and implement
``forward_step`` / ``loss_func``.
"""
from __future__ import annotations

from typing import Any, Dict

from .base import Loss


class GRPOLoss(Loss):
    """GRPO loss registered in the pipeline registry.

    Builds a minimal ``MegatronGRPOTrainer`` stub that only holds
    algorithm parameters (beta, epsilon, …) and reuses its
    ``forward_step`` / ``loss_func`` without duplicating code.

    To define a custom loss, subclass ``Loss``, override
    ``forward_step`` / ``loss_func``, and pass the dotted path to
    ``register_ray_trainer(..., loss='your.module.YourLoss')``.
    """

    def __init__(self, args):
        self._dummy = self._create_dummy_trainer(args)

    @staticmethod
    def _create_dummy_trainer(args):
        """Create a minimal MegatronGRPOTrainer for loss computation only.

        Skips the heavy __init__ side-effects (vLLM, reward, model init)
        by using __new__ and manually initialising only the fields that
        ``forward_step`` / ``loss_func`` actually read.

        ``_init_grpo_params`` is the single source of truth for every algorithm
        parameter the loss path needs (including the OPD-RL ``_has_teacher`` /
        ``teacher_kl_coef`` flags), so loss-only deps must be added there rather
        than to ``__init__`` -- which this stub deliberately bypasses.
        """
        import torch

        from swift.megatron.trainers.grpo_trainer import MegatronGRPOTrainer
        from swift.utils import is_last_rank
        cls = MegatronGRPOTrainer
        dummy = cls.__new__(cls)
        dummy.args = args
        dummy._init_grpo_params()
        dummy._prepare_metrics()
        dummy.log_rollout_offpolicy_metrics = args.log_rollout_offpolicy_metrics
        dummy.disable_rollout_importance_sampling = False
        dummy.enable_routing_replay = args.router_replay_mode != 'disabled'
        dummy.micro_batch_size = args.micro_batch_size
        dummy.temperature = args.temperature
        dummy.is_main_process = is_last_rank()
        dummy.process_index = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        dummy.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        dummy._step = 0
        dummy.max_completion_length = args.max_completion_length

        class _AlwaysTraining:
            training = True

        dummy.unwrapped_models = [_AlwaysTraining()]
        return dummy

    def forward_step(self, data_iterator, model):
        from swift.megatron.trainers.grpo_trainer import MegatronGRPOTrainer
        cls = MegatronGRPOTrainer
        return cls.forward_step(self._dummy, data_iterator, model)

    def loss_func(self, output_tensor, *, data: Dict[str, Any]):
        return self._dummy.loss_func(output_tensor, data=data)
