# Copyright (c) ModelScope Contributors. All rights reserved.
"""Megatron Trainer subclasses for Ray remote-model scenarios.

These run inside Ray workers.  They inherit the algorithm-specific
loss logic from the standard Megatron trainers but skip local
ref / teacher / rollout model creation — that part is handled
by the outer RayTrainer (driver layer).
"""
from functools import partial

from swift.megatron.trainers.base import BaseMegatronTrainer
from swift.megatron.trainers.dpo_trainer import MegatronDPOTrainer


class RayMegatronDPOTrainer(MegatronDPOTrainer):
    """DPO trainer for Ray remote-ref mode.

    Differences from ``MegatronDPOTrainer``:
      - No local ref model (``prepare_model`` / ``_load_checkpoint`` skip it)
      - ``forward_step`` only runs policy model
      - ``ref_logps`` is passed in as data from the driver
    """

    def prepare_model(self):
        BaseMegatronTrainer.prepare_model(self)
        self.ref_models = []

    def _load_checkpoint(self):
        BaseMegatronTrainer._load_checkpoint(self)

    def forward_step(self, data_iterator, model):
        unwrapped_model = model.module.module
        vp_stage = unwrapped_model.vp_stage
        data = self.get_batch(data_iterator, vp_stage)
        data.pop('loss_scale', None)
        ref_logps = data.pop('ref_logps', None)

        output_tensor = model(**data)
        return output_tensor, partial(
            self.loss_func,
            labels=data.get('labels'),
            packed_seq_params=data.get('packed_seq_params'),
            ref_logps=ref_logps)
