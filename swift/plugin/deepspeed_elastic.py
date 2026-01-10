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
        # with self.template.forward_context(self.model, inputs),get_act_offloading_ctx_manager(model):
        #

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
                args.deepspeed['checkpoint'] = {'load_universal': True}
                final_batch_size, valid_gpus, micro_batch_size = compute_elastic_config(
                    ds_config=args.deepspeed,
                    target_deepspeed_version=__version__,
                    world_size=dist.get_world_size(),
                )
                gradient_accu_steps = final_batch_size // (micro_batch_size * dist.get_world_size())
                args.per_device_train_batch_size = micro_batch_size
                args.gradient_accumulation_steps = gradient_accu_steps
                state.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
