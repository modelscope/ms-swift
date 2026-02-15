# Copyright (c) ModelScope Contributors. All rights reserved.
from contextlib import contextmanager

import torch
from megatron.core import mpu
from torch.distributed.nn import all_reduce
from transformers.utils import ContextManagers

from swift.megatron.model import get_mcore_model
from swift.megatron.utils import load_mcore_checkpoint
from swift.rlhf_trainers.utils import identity_data_collator
from swift.utils import get_logger
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
            ref_model.requires_grad_(False)
            ref_model.eval()
        if self.ref_models and args.mcore_ref_model is None:
            ref_model_id_or_path = args.ref_model or args.model
            self.bridge.load_weights(self.ref_models, ref_model_id_or_path)
        if args.tuner_type == 'lora' and args.ref_adapters and args.mcore_ref_adapter is None:
            assert len(args.ref_adapters) == 1, 'Currently only support one adapter.'
            self.bridge.load_weights(
                self.ref_models, args.ref_adapters[0], is_peft_format=True, adapter_name='ref_adapter')

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
            # In CP mode, all_gather and reconstruct full sequence
            if args.context_parallel_size > 1:
                per_token_logps = self._postprocess_packed_tensor_cp(per_token_logps, packed_seq_params, num_samples
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

    def _postprocess_packed_tensor_cp(self, tensor, packed_seq_params, num_samples):
        """
        Generic method: In CP mode, all_gather and reconstruct full tensor sequences.
        Works for both logps (float) and masks (bool/int).

        Args:
            tensor: [1, packed_len/cp_size] in padding_free mode, or [batch_size, seq_len/cp_size] otherwise
            packed_seq_params: PackedSeqParams object (None in non-padding_free mode)
            num_samples: Number of samples in the batch

        Returns:
            output_full: [1, packed_len] in padding_free mode, or [batch_size, seq_len] otherwise
        """
        args = self.args
        cp_size = args.context_parallel_size
        cp_rank = mpu.get_context_parallel_rank()

        # All-gather across CP ranks
        output_list = [torch.empty_like(tensor) for _ in range(cp_size)]
        torch.distributed.all_gather(output_list, tensor.contiguous(), group=mpu.get_context_parallel_group())
        output_list[cp_rank] = tensor

        if packed_seq_params is not None:
            # padding_free mode: [1, packed_len/cp_size] -> [1, packed_len]
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
        else:
            # non-padding_free mode: [batch_size, seq_len/cp_size] -> [batch_size, seq_len]
            # Each CP rank has chunks split with load-balanced pattern (2*cp_size chunks)
            batch_size = tensor.shape[0]
            seq_len_per_cp = tensor.shape[1]
            full_seq_len = seq_len_per_cp * cp_size

            output_full = tensor.new_zeros(batch_size, full_seq_len)

            # Each CP rank j holds chunks j and (2*cp_size - j - 1) from the original 2*cp_size split
            # Reconstruct the full sequence by placing chunks back in correct positions
            chunk_len = full_seq_len // (2 * cp_size)

            for j in range(cp_size):
                o = output_list[j]  # [batch_size, seq_len_per_cp]
                # This rank holds 2 chunks: chunk j and chunk (2*cp_size - j - 1)
                half_len = seq_len_per_cp // 2
                o0 = o[:, :half_len]  # First half -> chunk j
                o1 = o[:, half_len:]  # Second half -> chunk (2*cp_size - j - 1)

                # Place chunk j at position j * chunk_len
                output_full[:, j * chunk_len:(j + 1) * chunk_len] = o0
                # Place chunk (2*cp_size - j - 1) at position (2*cp_size - j - 1) * chunk_len
                reverse_idx = 2 * cp_size - j - 1
                output_full[:, reverse_idx * chunk_len:(reverse_idx + 1) * chunk_len] = o1

        return output_full
