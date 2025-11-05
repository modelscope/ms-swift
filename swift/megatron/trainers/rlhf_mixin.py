# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import contextmanager

from megatron.core import mpu
from megatron.training import get_args, get_model
from megatron.training.checkpointing import load_checkpoint
from megatron.training.utils import unwrap_model
from torch.distributed.nn import all_reduce
from transformers.utils import ContextManagers

from swift.utils import get_logger
from .base import BaseMegatronTrainer

logger = get_logger()


class MegatronRLHFTrainer(BaseMegatronTrainer):

    def setup_model_and_optimizer(self, model_provider_func, model_type, *_args, **kwargs):
        args = get_args()
        if args.train_type == 'full' and args.rlhf_type != 'rm':
            ref_models = get_model(model_provider_func, model_type, wrap_with_ddp=False)
            args.ref_model = args.ref_model or args.model
            for m in ref_models:
                m = unwrap_model(m)
                if args.load_safetensors:
                    self.bridge.load_weights(m, args.ref_model)
                m.requires_grad_(False).eval()
            if args.ref_load is None:
                args.ref_load = args.load
            if args.ref_load:
                args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
                    ref_models, None, None, load_arg='ref_load')
            self.ref_models = ref_models
        return super().setup_model_and_optimizer(model_provider_func, model_type, *_args, **kwargs)

    @contextmanager
    def null_ref_context(self):
        args = get_args()
        contexts = []
        if args.train_type == 'full':
            ref_models = self.ref_models
        else:
            if args.ref_adapter_load is None:
                for m in self.peft_models:
                    contexts.append(m.disable_adapter())
            ref_models = self.unwrapped_models
        with ContextManagers(contexts):
            if args.ref_adapter_load:
                for m in self.peft_models:
                    m.set_adapter('ref_adapter')
            yield ref_models
            if args.ref_adapter_load:
                for m in self.peft_models:
                    m.set_adapter('default')

    def get_logps(self, output_tensor, labels, packed_seq_params, num_samples=None):
        args = get_args()
        per_token_logps = -output_tensor
        loss_mask = labels != -100
        per_token_logps = per_token_logps * loss_mask
        if num_samples is None:
            num_samples = packed_seq_params.num_samples * 2
        cu_seqlens = packed_seq_params.cu_seqlens_q[:num_samples + 1] // args.context_parallel_size
        all_logps = per_token_logps.new_zeros((num_samples, ))
        for i in range(num_samples):
            start, end = cu_seqlens[i], cu_seqlens[i + 1]
            all_logps[i] = per_token_logps[:, start:end].sum()
        if args.context_parallel_size > 1:
            all_logps = all_reduce(all_logps, group=mpu.get_context_parallel_group())
        return all_logps
