# Copyright (c) ModelScope Contributors. All rights reserved.
from contextlib import contextmanager
from megatron.core import mpu
from torch.distributed.nn import all_reduce
from transformers.utils import ContextManagers

from swift.megatron.model import get_mcore_model
from swift.megatron.utils import load_mcore_checkpoint
from swift.rlhf_trainers.utils import identity_data_collator
from swift.utils import get_current_device, get_logger, safe_snapshot_download
from .base import BaseMegatronTrainer

logger = get_logger()


class MegatronRLHFTrainer(BaseMegatronTrainer):

    def _load_checkpoint(self):
        args = self.args
        if args.mcore_ref_model is not None:
            load_mcore_checkpoint(args, self.ref_models, load_arg='mcore_ref_model')
        if args.mcore_ref_adapter is not None:
            load_mcore_checkpoint(args, self.wrapped_models, load_arg='mcore_ref_adapter')
        super()._load_checkpoint()

    def prepare_model(self):
        super().prepare_model()
        args = self.args
        self.ref_models = []
        if args.tuner_type == 'full' and args.rlhf_type not in ['rm', 'gkd']:
            self.ref_models = get_mcore_model(args, self.template.config)
        for ref_model in self.ref_models:
            if not args.use_cpu_initialization:
                ref_model.to(get_current_device())
            ref_model.requires_grad_(False)
            ref_model.eval()
        if self.ref_models and args.mcore_ref_model is None:
            ref_model_id_or_path = args.ref_model or args.model
            ref_model_dir = safe_snapshot_download(ref_model_id_or_path, use_hf=args.use_hf, hub_token=args.hub_token)
            self.bridge.load_weights(self.ref_models, ref_model_dir)
        if args.tuner_type == 'lora' and args.ref_adapters and args.mcore_ref_adapter is None:
            assert len(args.ref_adapters) == 1, 'Currently only support one adapter.'
            self.bridge.load_weights(
                self.ref_models, args.ref_adapters[0], peft_format=True, adapter_name='ref_adapter')

    def _get_data_collator(self):
        if self.args.rlhf_type in ('grpo', 'gkd'):
            return identity_data_collator
        return super()._get_data_collator()

    @contextmanager
    def null_ref_context(self):
        args = self.args
        contexts = []
        has_ref_adapter = bool(args.mcore_ref_adapter or args.ref_adapters)
        if args.tuner_type == 'full':
            ref_models = self.ref_models
        else:
            if not has_ref_adapter:
                for m in self.peft_models:
                    contexts.append(m.disable_adapter())
            ref_models = self.unwrapped_models
        with ContextManagers(contexts):
            if has_ref_adapter:
                for m in self.peft_models:
                    m.set_adapter('ref_adapter')
            yield ref_models
            if has_ref_adapter:
                for m in self.peft_models:
                    m.set_adapter('default')

    def get_logps(self, output_tensor, labels, packed_seq_params, num_samples, per_token=False):
        args = self.args
        per_token_logps = -output_tensor
        loss_mask = labels != -100
        per_token_logps = per_token_logps * loss_mask
        if per_token:
            if args.context_parallel_size > 1:
                from .utils import _postprocess_packed_tensor_cp
                per_token_logps = _postprocess_packed_tensor_cp(args.context_parallel_size, per_token_logps,
                                                                packed_seq_params, num_samples
                                                                or packed_seq_params.num_samples)
            return per_token_logps

        if args.padding_free:
            cu_seqlens = packed_seq_params.cu_seqlens_q[:num_samples + 1] // args.context_parallel_size
            all_logps = per_token_logps.new_zeros((num_samples, ))
            for i in range(num_samples):
                start, end = cu_seqlens[i], cu_seqlens[i + 1]
                all_logps[i] = per_token_logps[:, start:end].sum()
        else:
            all_logps = per_token_logps.sum(-1)
        if args.context_parallel_size > 1:
            all_logps = all_reduce(all_logps, group=mpu.get_context_parallel_group())
        return all_logps

    def compute_per_token_logps(self, model, data_iterator, no_grad=True, temperature=1.0):
        from .utils import compute_per_token_logps_fn
        return compute_per_token_logps_fn(
            model,
            self.args,
            data_iterator,
            temperature=temperature,
            no_grad=no_grad,
            enable_routing_replay=self.enable_routing_replay)
