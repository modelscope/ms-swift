# Copyright (c) Alibaba, Inc. and its affiliates.
import types

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class DeepspeedElasticCallBack(TrainerCallback):

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """

        if args.deepspeed and args.elastic:
            from deepspeed.elasticity import compute_elastic_config
            from deepspeed.git_version_info import version as __version__
            args.deepspeed['checkpoint'] = {'load_universal': True}
            if 'elasticity' not in args.deepspeed:
                args.deepspeed['elasticity'] = {
                    'ignore_non_elastic_batch_info': True,
                    'enabled': True,
                    'max_train_batch_size': 8,
                    'micro_batch_sizes': [2],
                    'min_gpus': 1,
                    'max_gpus': 4,
                    'min_time': 20,
                    'version': 0.1
                }
                world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
                final_batch_size, valid_gpus, micro_batch_size = compute_elastic_config(
                    ds_config=args.deepspeed,
                    target_deepspeed_version=__version__,
                    world_size=world_size,
                )
                denom = micro_batch_size * world_size
                gradient_accu_steps = max(1, final_batch_size // denom)
                args.per_device_train_batch_size = micro_batch_size
                args.gradient_accumulation_steps = gradient_accu_steps
                state.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
