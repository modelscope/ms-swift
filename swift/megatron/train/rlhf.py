# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Union

import torch
from megatron.core import mpu
from megatron.training import get_args, get_model, get_timers, training
from megatron.training.checkpointing import load_checkpoint
from megatron.training.utils import unwrap_model

from swift.utils import get_logger
from ..argument import MegatronRLHFArguments
from .sft import MegatronSft
from .utils import get_batch

logger = get_logger()


class MegatronRLHF(MegatronSft):
    args_class = MegatronRLHFArguments
    args: args_class

    def _patch_setup_model_and_optimizer(self):
        origin_setup_model_and_optimizer = training.setup_model_and_optimizer

        def setup_model_and_optimizer(model_provider_func, model_type, *_args, **kwargs):
            args = get_args()
            ref_model = get_model(model_provider_func, model_type)
            if args.ref_load is not None:
                args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                    ref_model, None, None, load_arg='ref_load')
            args.ref_model = ref_model
            return origin_setup_model_and_optimizer(model_provider_func, model_type, *_args, **kwargs)

        training.setup_model_and_optimizer = setup_model_and_optimizer

    def ref_forward(self, data_iterator):
        args = get_args()
        ref_model = unwrap_model(args.ref_model[0])
        timers = get_timers()
        timers('batch-ref-generator', log_level=2).start()
        with self.stimer(bdata=True):
            data = get_batch(data_iterator)
        timers('batch-ref-generator').stop()

        ref_forward_step = InferenceForwardStep(ref_model)
        ref_logits = ref_forward_step(**data)

        if mpu.is_pipeline_last_stage():
            data['ref_logits'] = ref_logits.detach()
        else:
            data['ref_logits'] = None
        return data

    def train_step(self, forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config):
        args = get_args()
        num_iters_per_step = args.global_batch_size // (args.micro_batch_size * mpu.get_data_parallel_world_size())
        res = []
        for i in range(num_iters_per_step):
            with torch.no_grad():
                res.append(self.ref_forward(data_iterator))
        super().train_step(self, forward_step_func, iter(res), model, optimizer, opt_param_scheduler, config)

    def run(self):
        self._patch_setup_model_and_optimizer()
        super().run()


def megatron_rlhf_main(args: Union[List[str], MegatronRLHFArguments, None] = None):
    return MegatronRLHF(args).main()
