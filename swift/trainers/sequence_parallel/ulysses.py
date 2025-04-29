from typing import Optional, Any

import torch
import torch.distributed as dist
from torch import Tensor

from swift.trainers.sequence_parallel.base import SequenceParallel
from swift.utils import get_dist_setting


class Ulysses(SequenceParallel):

    def __init__(self):
        self.group = None
        self.sequence_parallel_size = 1

    @staticmethod
    def assert_ulysses_runtime_condition():
        from swift.utils import is_deepspeed_available
        assert is_deepspeed_available(), 'Please install deepspeed to use ulysses'
        assert dist.is_initialized(), 'pack_to_max_length is only available with distributed training.'

    def init_sequence_parallel(self, size):
        self.sequence_parallel_size = size
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        num_sequence_parallel_groups: int = world_size // size

        for i in range(num_sequence_parallel_groups):
            ranks = list(range(i * size,
                               (i + 1) * size))
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                self.group = group

    def prepare_model(self, model):
        _sequence_parallel_size = self.sequence_parallel_size

        from functools import partial
        from deepspeed.sequence.layer import DistributedAttention
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'] = ALL_ATTENTION_FUNCTIONS['flash_attention_2']
        ALL_ATTENTION_FUNCTIONS['sdpa_origin'] = ALL_ATTENTION_FUNCTIONS['sdpa']

        def local_flash_attn(module: torch.nn.Module,
                             query_states,
                             key_states,
                             value_states,
                             *args, dist_attn, **kwargs):
            if dist_attn.local_attn is None:
                def _attention(*args, **kwargs):
                    return ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'](module, *args, **kwargs)[0]

                dist_attn.local_attn = _attention

            return dist_attn(query_states,
                             key_states,
                             value_states, 0, None, *args, **kwargs), None

        def local_sdpa_attn(module: torch.nn.Module,
                            query_states,
                            key_states,
                            value_states,
                            *args, dist_attn, **kwargs):
            if dist_attn.local_attn is None:
                def _attention(*args, **kwargs):
                    return ALL_ATTENTION_FUNCTIONS['sdpa_origin'](module, *args, **kwargs)[0]

                dist_attn.local_attn = _attention

            return dist_attn(query_states,
                             key_states,
                             value_states, 0, None, *args, **kwargs), None

        ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = partial(local_flash_attn, dist_attn=
        DistributedAttention(None, self.group))
        ALL_ATTENTION_FUNCTIONS['sdpa'] = partial(local_sdpa_attn, dist_attn=
        DistributedAttention(None, self.group))

    @staticmethod
    def _pad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
        shape = list(x.shape)
        shape[dim] = padding_size
        pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=dim)

    @staticmethod
    def _unpad_tensor(x: Tensor, dim: int, padding_size: int) -> Tensor:
        slc = [slice(None)] * len(x.shape)
        slc[dim] = slice(0, -padding_size)
        return x[slc]

    def all_gather_tensor(self, local_tensor: Tensor, group: Optional[dist.ProcessGroup] = None, async_op: bool = False):
        group = self.group if group is None else group
        sp_world_size = dist.get_world_size(group=group)
        output_shape = list(local_tensor.shape)
        output_shape[0] = output_shape[0] * sp_world_size
        output = torch.empty(output_shape, dtype=local_tensor.dtype, device=local_tensor.device)
        dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
        return output

    def slice_input_tensor(self, x: Tensor, dim: int, padding: bool = True, group: dist.ProcessGroup = None) -> Tensor:
        group = self.group if group is None else group
        sp_world_size = dist.get_world_size(group)
        sp_rank = self.group
        dim_size = x.size(dim)
        # pad before slice
        if padding and dim_size % sp_world_size:
            padding_size = sp_world_size - (dim_size % sp_world_size)
            x = self._pad_tensor(x, dim, padding_size)
        # slice the input tensor
        parts = x.size(dim) // sp_world_size
        slc = [slice(None)] * len(x.shape)
        slc[dim] = slice(sp_rank * parts, (sp_rank + 1) * parts)
        return x[slc].contiguous()

    def pad_and_split_inputs(self, tokenizer, input_ids, labels, position_ids, attention_mask, loss_scale):
        from xtuner.parallel.sequence import (pad_for_sequence_parallel, split_for_sequence_parallel,
                                              get_sequence_parallel_group)
        input_ids = pad_for_sequence_parallel(input_ids, padding_value=tokenizer.pad_token_id, dim=-1)
        labels = pad_for_sequence_parallel(labels, padding_value=-100, dim=-1)
        position_ids = pad_for_sequence_parallel(position_ids, padding_value=0, dim=-1)
        attention_mask = pad_for_sequence_parallel(attention_mask, padding_value=0, dim=-1)

        sp_group = get_sequence_parallel_group()
        input_ids = split_for_sequence_parallel(input_ids, dim=1, sp_group=sp_group)
        labels = split_for_sequence_parallel(labels, dim=1, sp_group=sp_group)
        position_ids = split_for_sequence_parallel(position_ids, dim=1, sp_group=sp_group)
        attention_mask = split_for_sequence_parallel(attention_mask, dim=-1, sp_group=sp_group)
        if loss_scale is not None:
            loss_scale = pad_for_sequence_parallel(loss_scale, padding_value=0., dim=-1)
            loss_scale = split_for_sequence_parallel(loss_scale, dim=1, sp_group=sp_group)

        return input_ids, labels, position_ids, attention_mask, loss_scale

    def reduce_outputs(self, loss, labels):
        from xtuner.parallel.sequence import (reduce_sequence_parallel_loss, get_sequence_parallel_group)
        # reduce loss for logging correctly
        num_tokens = (labels != -100).sum()
        return reduce_sequence_parallel_loss(loss, num_tokens, get_sequence_parallel_group())

    def world_size(self):
        return self.sequence_parallel_size

    def get_dataloader(self, trainer):
        pass