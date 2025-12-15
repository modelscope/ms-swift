# Copyright (c) Alibaba, Inc. and its affiliates.
from contextlib import contextmanager

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.training import get_args, get_model
from megatron.training.checkpointing import load_checkpoint
from megatron.training.utils import unwrap_model
from torch.distributed.nn import all_gather, all_reduce
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
        has_ref_adapter = bool(args.ref_adapter_load or args.ref_adapters)
        if args.train_type == 'full':
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

    def get_logps(self, output_tensor, labels, packed_seq_params, num_samples=None, per_token=False):
        args = get_args()
        per_token_logps = -output_tensor
        loss_mask = labels != -100
        per_token_logps = per_token_logps * loss_mask
        if per_token:
            # In CP mode, all_gather and reconstruct full sequence
            if args.context_parallel_size > 1:
                per_token_logps = self._postprocess_packed_tensor_cp(per_token_logps, packed_seq_params, num_samples
                                                                     or packed_seq_params.num_samples)
            return per_token_logps

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

    def _postprocess_packed_tensor_cp(self, tensor, packed_seq_params, num_samples):
        """
        Generic method: In CP mode, all_gather and reconstruct full tensor sequences.
        Works for both logps (float) and masks (bool/int).

        Args:
            tensor: [1, packed_len/cp_size] - CP-split tensor (any dtype)
            packed_seq_params: PackedSeqParams object
            num_samples: Number of samples in the batch

        Returns:
            output_full: [1, packed_len] - Full sequence tensor
        """
        args = get_args()
        cp_size = args.context_parallel_size
        cp_rank = mpu.get_context_parallel_rank()

        # All-gather across CP ranks
        output_list = [torch.empty_like(tensor) for _ in range(cp_size)]
        torch.distributed.all_gather(output_list, tensor.contiguous(), group=mpu.get_context_parallel_group())
        output_list[cp_rank] = tensor

        # Reconstruct full sequence
        # Shape: [1, packed_len/cp_size] -> [1, packed_len]
        cu_seqlens_full = packed_seq_params.cu_seqlens_q
        cu_seqlens_cp = cu_seqlens_full // cp_size

        # Calculate total packed length
        total_packed_len = cu_seqlens_full[num_samples].item()
        output_full = tensor.new_zeros(1, total_packed_len)

        # Reconstruct each sequence
        for i in range(num_samples):
            start_full = cu_seqlens_full[i].item()
            end_full = cu_seqlens_full[i + 1].item()
            seq_len = end_full - start_full

            # Length of each chunk after CP split
            chunk_len = seq_len // cp_size
            half_chunk = chunk_len // 2

            # Concatenate from each CP rank's output (load-balanced split)
            for j in range(cp_size):
                o = output_list[j][0]
                start_cp = cu_seqlens_cp[i].item()

                # Get two half chunks (CP's load-balanced split)
                o0 = o[start_cp:start_cp + half_chunk]
                o1 = o[start_cp + half_chunk:start_cp + chunk_len]

                # Place back to full sequence
                output_full[0, start_full + j * half_chunk:start_full + (j + 1) * half_chunk] = o0
                output_full[0, end_full - (j + 1) * half_chunk:end_full - j * half_chunk] = o1

        return output_full
