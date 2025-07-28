# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from functools import partial
from types import MethodType

import torch
import torch.nn.functional as F
from packaging import version

from swift.llm import get_llm_model
from .base import CommonSequenceParallel
from .utils import (SequenceParallelDispatcher, SequenceParallelSampler, _get_per_token_logps_and_entropies_grpo,
                    _get_train_sampler_grpo, _prepare_inputs, _prepare_inputs_grpo, get_common_dataloader,
                    get_per_token_logps, loss_scale_sp_func, old_policy_grpo, setup_compute_acc,
                    split_by_mini_batches_grpo)

RING_ATTN_GROUP = None


def set_ring_attn_group(group):
    global RING_ATTN_GROUP
    RING_ATTN_GROUP = group


def get_ring_attn_group():
    return RING_ATTN_GROUP


def reset_ring_attn_position_ids(start, end, packed_seq_lens):
    """
    Calculate position ids for packed_seq_ids[start:end].
    For example, if the packed_seq_lens is [3, 2, 4, 1], start=2, end=8,
    the position ids will be [2, 0, 1, 0, 1, 2].

    Args:
        start: the start position
        end: the end position
        packed_seq_lens: the sequence lengths of packed sequences
    """
    position_ids = torch.zeros((1, end - start), dtype=torch.long, device=torch.cuda.current_device())
    offset = 0
    for seqlen in packed_seq_lens:
        seq_start = max(offset, start)
        seq_end = min(offset + seqlen, end)
        if seq_start < seq_end:
            position_ids[0, seq_start - start:seq_end - start] = torch.arange(seq_start - offset, seq_end - offset)

        offset += seqlen
        if offset >= end:
            break
    return position_ids


def update_ring_attn_params(packed_seq_lens, total_seq_len):
    """
    Calculate the cu_seqlens for the current forward pass and pass the value to
    the substituted ring_flash_attn.
    """
    assert RING_ATTN_GROUP is not None
    cu_seqlens = torch.cumsum(
        torch.tensor(packed_seq_lens, device=torch.cuda.current_device(), dtype=torch.int32),
        dim=-1,
        dtype=torch.int32,
    )
    cu_seqlens = F.pad(F.pad(cu_seqlens, (1, 0), value=0), (0, 1), value=total_seq_len)

    from ring_flash_attn import update_ring_flash_attn_params

    update_ring_flash_attn_params(cu_seqlens, RING_ATTN_GROUP)


def infer_packed_seq_lens_from_position_ids(position_ids):
    """
    Infer packed sequence lengths from position_ids.

    For packed sequences, position_ids are flattened. For example:
    - Original sequence lengths: [2, 3]
    - Position_ids: [0, 1, 0, 1, 2]

    Args:
        position_ids: torch.Tensor of shape (batch_size, seq_len) or (seq_len,)

    Returns:
        List of sequence lengths for each packed sequence
    """
    if position_ids.dim() == 2:
        # Handle batch dimension - assume batch_size=1 for packed sequences
        position_ids = position_ids.squeeze(0)

    position_ids = position_ids.cpu().tolist()
    packed_seq_lens = []
    current_seq_len = 0

    for i, pos_id in enumerate(position_ids):
        current_seq_len += 1
        # When position_id resets to 0 (except for the first token), it indicates a new sequence
        if i > 0 and pos_id == 0:
            packed_seq_lens.append(current_seq_len - 1)
            current_seq_len = 1

    # Add the last sequence length
    if current_seq_len > 0:
        packed_seq_lens.append(current_seq_len)

    return packed_seq_lens


class RingAttention(CommonSequenceParallel):

    def __init__(self):
        """Initialize RingAttention sequence parallel implementation."""
        super().__init__()

    def init_sequence_parallel(self, size):
        """Initialize ring attention sequence parallel with given size.

        Args:
            size: The sequence parallel world size
        """
        if self._inited:
            return
        self._inited = True

        self.sp_world_size = size
        self._init_device_mesh()

        # Set global ring attention group using the sequence dimension
        ring_attn_group = self.device_mesh['sequence'].get_group()
        set_ring_attn_group(ring_attn_group)

        # Import and setup ring flash attention
        ring_head_stride = int(os.environ.get('RING_HEAD_STRIDE', 2))
        try:
            from ring_flash_attn import substitute_hf_flash_attn
            # Substitute HuggingFace flash attention with ring attention
            substitute_hf_flash_attn(ring_attn_group, ring_head_stride)
        except ImportError:
            raise ImportError('ring-flash-attn is required for RingAttention. '
                              'Please install it with: pip install ring-flash-attn')

    def prepare_model(self, model, tokenizer):
        """Prepare the model for ring attention sequence parallel.

        Args:
            model: The model to prepare
            tokenizer: The tokenizer to use
        """

        def pre_forward_hook(_self, args, kwargs):
            """Hook to process inputs before forward pass for ring attention."""
            input_ids = kwargs.get('input_ids', None)
            inputs_embeds = kwargs.get('inputs_embeds', None)
            position_ids = kwargs.get('position_ids', None)
            attention_mask = kwargs.get('attention_mask', None)

            # packed_seq_lens is calculated in data_collator when padding_free/packing mode is enabled
            packed_seq_lens = kwargs.get('packed_seq_lens', None)

            if packed_seq_lens is None:
                packed_seq_lens = infer_packed_seq_lens_from_position_ids(position_ids)

            # Get embed_tokens for padding
            if hasattr(_self, 'language_model'):
                embed_tokens = getattr(_self.language_model, 'embed_tokens', None)
            else:
                embed_tokens = getattr(_self, 'embed_tokens', None)

            input_ids, inputs_embeds, _, position_ids, attention_mask, _ = self.pad_and_split_inputs(
                input_ids, inputs_embeds, None, position_ids, attention_mask, None, embed_tokens=embed_tokens)

            kwargs['input_ids'] = input_ids
            kwargs['inputs_embeds'] = inputs_embeds
            kwargs['position_ids'] = position_ids
            kwargs['attention_mask'] = attention_mask

            total_seq_len = position_ids.numel() * self.sp_world_size

            update_ring_attn_params(packed_seq_lens, total_seq_len)

            return args, kwargs

        # Get the base model to register the hook
        llm_model = get_llm_model(model)

        if hasattr(llm_model, 'thinker'):
            base_model = llm_model.thinker.model
        else:
            base_model = llm_model.model

        # Register the pre-forward hook
        base_model.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)

        # Store model dtype and tokenizer
        self.model_dtype = next(model.parameters()).dtype
        self.tokenizer = tokenizer

    def get_dataloader(self, trainer, dataset, batch_size, skip_batches: int = 0):
        return get_common_dataloader(
            self,
            trainer,
            dataset,
            batch_size,
            SequenceParallelSampler,
            SequenceParallelDispatcher,
            skip_batches=skip_batches)

    def prepare_trainer(self, trainer):
        """Prepare trainer for ring attention sequence parallel.

        Args:
            trainer: The trainer to prepare
        """
        if trainer.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')

        trainer.ring_attention = self

        if trainer.__class__.__name__ == 'Seq2SeqTrainer':
            trainer._origin_prepare_inputs = trainer._prepare_inputs
            trainer._prepare_inputs = MethodType(partial(_prepare_inputs, sp_instance=self), trainer)
            trainer.compute_loss_func = partial(loss_scale_sp_func, sp_instance=self)

        elif trainer.__class__.__name__ == 'DPOTrainer':
            trainer._origin_prepare_inputs = trainer._prepare_inputs
            trainer._prepare_inputs = MethodType(partial(_prepare_inputs, sp_instance=self), trainer)
            trainer.get_per_token_logps = partial(get_per_token_logps, sp_instance=self)

        elif trainer.__class__.__name__ == 'GRPOTrainer':
            try:
                import trl
                assert version.parse(trl.__version__) >= version.parse('0.18.0')
            except (ImportError, AssertionError):
                raise ImportError('trl>=0.18.0 is required for GRPOTrainer with ring attention. '
                                  'Please install it with: pip install trl>=0.18.0')

            trainer.ring_attention = self
            trainer.args.gradient_accumulation_steps = trainer.args.gradient_accumulation_steps * self.sp_world_size
            trainer.old_policy = MethodType(partial(old_policy_grpo, sp_instance=self), trainer)
            trainer._get_train_sampler = MethodType(partial(_get_train_sampler_grpo, sp_instance=self), trainer)
            trainer._prepare_inputs = MethodType(partial(_prepare_inputs_grpo, sp_instance=self), trainer)
            trainer._get_per_token_logps_and_entropies = MethodType(
                partial(_get_per_token_logps_and_entropies_grpo, sp_instance=self), trainer)
            trainer.split_by_mini_batches = MethodType(partial(split_by_mini_batches_grpo, sp_instance=self), trainer)

        setup_compute_acc(self)
