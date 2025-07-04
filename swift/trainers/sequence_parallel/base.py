import abc
from abc import abstractmethod

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from swift.utils import get_device, get_dist_setting


class SequenceParallel(abc.ABC):
    """Base abstract class for sequence parallel implementations"""

    @abstractmethod
    def init_sequence_parallel(self, size):
        """Initialize sequence parallel with given size"""
        pass

    @abstractmethod
    def prepare_model(self, model, tokenizer):
        """Prepare model for sequence parallel training"""
        pass

    @abstractmethod
    def pad_and_split_inputs(self,
                             input_ids,
                             input_embeds,
                             labels,
                             position_ids,
                             attention_mask,
                             loss_scale,
                             embed_tokens=None):
        """Pad and split inputs for sequence parallel training"""
        pass

    @abstractmethod
    def reduce_outputs(self, loss, labels):
        """Reduce outputs for sequence parallel training"""
        pass

    @property
    @abstractmethod
    def sp_group(self):
        """Return the sequence parallel group"""
        pass

    @abstractmethod
    def world_size(self):
        """Return the sequence parallel world size"""
        pass

    @abstractmethod
    def prepare_trainer(self, trainer):
        """Prepare trainer for sequence parallel training"""
        pass

    @abstractmethod
    def get_dataloader(self, trainer, dataset, batch_size):
        """Get dataloader for sequence parallel training"""
        pass


class CommonSequenceParallel(SequenceParallel):
    """Common base class for Ulysses and RingAttention implementations"""

    def __init__(self):
        self.sp_world_size = None
        self.dp_world_size = None
        self.model_dtype = None
        self.tokenizer = None
        self.device_mesh = None
        self._inited = False

    def _pad_sp(self, tensor, padding_value, dim=-1):
        """Pad tensor for sequence parallel"""
        length = tensor.shape[dim]
        if length % self.sp_world_size == 0:
            return tensor

        pad_num = self.sp_world_size - (length % self.sp_world_size)
        if not isinstance(padding_value, torch.Tensor):
            # ids
            pad_shape = ((*tensor.shape[:dim], pad_num, *tensor.shape[dim + 1:]) if dim != -1 else
                         (*tensor.shape[:dim], pad_num))
            pad = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, pad], dim=dim)
        else:
            # For embeddings
            tensor = torch.cat([tensor, padding_value.unsqueeze(0).repeat(tensor.shape[0], pad_num, 1)], dim=dim)
        return tensor

    def _split_sp(self, input, dim: int):
        """Split tensor for sequence parallel"""
        if self.sp_world_size == 1:
            return input

        rank = self.sp_rank
        dim_size = input.size(dim)
        assert dim_size % self.sp_world_size == 0, (f'The dimension to split ({dim_size}) is not a multiple of '
                                                    f'world size ({self.sp_world_size}), cannot split tensor evenly')

        tensor_list = torch.split(input, dim_size // self.sp_world_size, dim=dim)
        output = tensor_list[rank].contiguous()

        return output

    def pad_and_split_inputs(self,
                             input_ids,
                             input_embeds,
                             labels,
                             position_ids,
                             attention_mask,
                             loss_scale,
                             embed_tokens=None):
        """Common implementation for padding and splitting inputs"""
        tokenizer = self.tokenizer
        if input_ids is not None:
            input_ids = self._pad_sp(input_ids, padding_value=tokenizer.pad_token_id, dim=-1)
        if input_embeds is not None:
            pad_emb = torch.zeros(
                (1, embed_tokens.weight.shape[-1])).to(embed_tokens.weight.device).to(embed_tokens.weight.dtype)
            input_embeds = self._pad_sp(input_embeds, padding_value=pad_emb, dim=1)
        batch_size = input_ids.shape[
            0] if input_ids is not None else input_embeds.shape[0] if input_embeds is not None else 1
        if position_ids is not None:
            position_ids = self._pad_sp(position_ids, padding_value=0, dim=-1)
        if (input_ids is not None or input_embeds is not None) and batch_size > 1:
            inputs = input_ids if input_ids is not None else input_embeds
            attn_shape = inputs.shape[1]  # The sequence length
            if attention_mask is None:
                attention_mask = torch.ones_like(position_ids)
            attention_mask = self._pad_sp(attention_mask, padding_value=0, dim=-1)
            cache_position = torch.arange(0, attn_shape, device=inputs.device)
            # pad attention mask to 4d to avoid calculation errors
            if hasattr(self, 'causal_mask_func') and self.causal_mask_func is not None:
                attention_mask = self.causal_mask_func(attention_mask, inputs.to(self.model_dtype), cache_position,
                                                       None, None)
        if input_ids is not None:
            input_ids = self._split_sp(input_ids, dim=1)
        if input_embeds is not None:
            input_embeds = self._split_sp(input_embeds, dim=1)
        if position_ids is not None:
            position_ids = self._split_sp(position_ids, dim=-1)
        if labels is not None:
            labels = self._pad_sp(labels, padding_value=-100, dim=-1)
            labels = torch.roll(labels, shifts=-1, dims=-1)
            labels = self._split_sp(labels, dim=-1)

        if loss_scale is not None:
            loss_scale = self._pad_sp(loss_scale, padding_value=0., dim=-1)
            loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1)
            loss_scale = self._split_sp(loss_scale, dim=-1)

        return input_ids, input_embeds, labels, position_ids, attention_mask, loss_scale

    def reduce_outputs(self, loss, labels):
        """Default implementation for reducing outputs"""
        return loss

    def world_size(self):
        """Return the sequence parallel world size"""
        return self.sp_world_size

    def _init_device_mesh(self):
        """Initialize device mesh for sequence parallel"""
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        self.dp_world_size = world_size // self.sp_world_size

        # Create device mesh: (dp_world_size, sp_world_size)
        self.device_mesh = init_device_mesh(
            get_device().split(':')[0],
            mesh_shape=(self.dp_world_size, self.sp_world_size),
            mesh_dim_names=['data', 'sequence'])

    @property
    def sp_group(self):
        """Return the sequence parallel group"""
        return self.device_mesh['sequence'].get_group() if self.device_mesh else None

    @property
    def sp_rank(self):
        """Return the sequence parallel rank"""
        return dist.get_rank(self.device_mesh['sequence'].get_group()) if self.device_mesh else 0

    @property
    def dp_group(self):
        """Return the data parallel group"""
        return self.device_mesh['data'].get_group() if self.device_mesh else None

    @property
    def dp_rank(self):
        """Return the data parallel rank"""
        return dist.get_rank(self.device_mesh['data'].get_group()) if self.device_mesh else 0
