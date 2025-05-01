from typing import Optional, Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch import Tensor
from torch.utils.data import DataLoader
from transformers.trainer_utils import seed_worker

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
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size // size, size), mesh_dim_names=["data", "sequence"]
        )

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
                def _attention(query, key, value, *args, **kwargs):
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    print('before attention query', query.shape, flush=True)
                    print('before attention key', key.shape, flush=True)
                    print('before attention value', value.shape, flush=True)
                    return ALL_ATTENTION_FUNCTIONS['sdpa_origin'](module, query, key, value, *args, **kwargs)[0]

                dist_attn.local_attn = _attention
            # print(query_states.shape)
            print('before transpose query', query_states.shape, flush=True)
            print('before transpose key', key_states.shape, flush=True)
            print('before transpose value', value_states.shape, flush=True)
            return dist_attn(query_states.transpose(1,2),
                             key_states.transpose(1,2),
                             value_states.transpose(1,2), 0, None, *args, **kwargs), None

        ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = partial(local_flash_attn, dist_attn=
        DistributedAttention(None, self.group, 2, 1))
        ALL_ATTENTION_FUNCTIONS['sdpa'] = partial(local_sdpa_attn, dist_attn=
        DistributedAttention(None, self.group, 2, 1))

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

    def pad_for_sequence_parallel(self, tensor, padding_value, dim=-1):
        length = tensor.shape[dim]
        seq_parallel_world_size = self.world_size()
        if length % seq_parallel_world_size == 0:
            return tensor

        pad_num = seq_parallel_world_size - (length % seq_parallel_world_size)
        pad_shape = (
            (*tensor.shape[:dim], pad_num, *tensor.shape[dim + 1 :])
            if dim != -1
            else (*tensor.shape[:dim], pad_num)
        )
        pad = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, pad], dim=dim)
        return tensor

    def pad_for_sequence_parallel(self, tensor, padding_value, dim=-1):
        length = tensor.shape[dim]
        seq_parallel_world_size = self.world_size()
        if length % seq_parallel_world_size == 0:
            return tensor

        pad_num = seq_parallel_world_size - (length % seq_parallel_world_size)
        pad_shape = (
            (*tensor.shape[:dim], pad_num, *tensor.shape[dim + 1:])
            if dim != -1
            else (*tensor.shape[:dim], pad_num)
        )
        pad = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, pad], dim=dim)
        return tensor

    def split_for_sequence_parallel(self, input, dim: int, sp_group: dist.ProcessGroup):
        """Splits the input tensor along a given dimension for sequence parallel.

        Args:
            input: The input tensor to be split.
            dim: The dimension along which the tensor should be split.
            sp_group: The sequence parallel process group.

        Returns:
            The split tensor corresponding to the current rank's chunk.
        """
        world_size = dist.get_world_size(sp_group)
        if world_size == 1:
            return input

        rank = dist.get_rank(sp_group)
        dim_size = input.size(dim)
        assert dim_size % world_size == 0, (
            f'The dimension to split ({dim_size}) is not a multiple of '
            f'world size ({world_size}), cannot split tensor evenly')

        tensor_list = torch.split(input, dim_size // world_size, dim=dim)
        output = tensor_list[rank].contiguous()

        return output

    def pad_and_split_inputs(self, tokenizer, input_ids, labels, position_ids, attention_mask, loss_scale):
        input_ids = self.pad_for_sequence_parallel(input_ids, padding_value=tokenizer.pad_token_id, dim=-1)
        labels = self.pad_for_sequence_parallel(labels, padding_value=-100, dim=-1)
        position_ids = self.pad_for_sequence_parallel(position_ids, padding_value=0, dim=-1)
        attention_mask = self.pad_for_sequence_parallel(attention_mask, padding_value=0, dim=-1)

        sp_group = self.group
        input_ids = self.split_for_sequence_parallel(input_ids, dim=1, sp_group=sp_group)
        labels = self.split_for_sequence_parallel(labels, dim=1, sp_group=sp_group)
        position_ids = self.split_for_sequence_parallel(position_ids, dim=1, sp_group=sp_group)
        attention_mask = self.split_for_sequence_parallel(attention_mask, dim=-1, sp_group=sp_group)
        if loss_scale is not None:
            loss_scale = self.pad_for_sequence_parallel(loss_scale, padding_value=0., dim=-1)
            loss_scale = self.split_for_sequence_parallel(loss_scale, dim=1, sp_group=sp_group)

        return input_ids, labels, position_ids, attention_mask, loss_scale

    def reduce_outputs(self, loss, labels):
        class _ReduceLoss(torch.autograd.Function):

            @staticmethod
            def forward(ctx, mean_loss, loss_scale, process_group):
                ctx.mode = process_group
                if loss_scale == 0:
                    # convert nan to 0 just for logging
                    mean_loss = torch.nan_to_num(mean_loss)
                loss_sum = mean_loss * loss_scale
                dist.all_reduce(loss_sum, group=process_group)
                dist.all_reduce(loss_scale, group=process_group)
                loss = loss_sum / loss_scale
                return loss

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None, None

        def reduce_sequence_parallel_loss(mean_loss,
                                          loss_scale,
                                          sp_group: dist.ProcessGroup = None):
            if dist.get_world_size(sp_group) == 1:
                return mean_loss
            if sp_group is None:
                # avoid bc breaking
                sp_group = self.group
            return _ReduceLoss.apply(mean_loss, loss_scale, sp_group)
        # reduce loss for logging correctly
        num_tokens = (labels != -100).sum()
        return reduce_sequence_parallel_loss(loss, num_tokens, self.group)

    def world_size(self):
        return self.sequence_parallel_size

    def get_dataloader(self, trainer):
        # modified from HFTrainer.get_train_dataloader
        # RandomSampler -> SequenceParallelSampler
        if trainer.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')

        train_dataset = trainer.train_dataset
        data_collator = trainer.data_collator
        import datasets
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = trainer._remove_unused_columns(train_dataset, description='training')
        else:
            data_collator = trainer._get_collator_with_removed_columns(data_collator, description='training')

        dataloader_params = {
            'batch_size': trainer._train_batch_size,
            'collate_fn': data_collator,
            'num_workers': trainer.args.dataloader_num_workers,
            'pin_memory': trainer.args.dataloader_pin_memory,
            'persistent_workers': trainer.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            from mmengine.dataset import DefaultSampler
            ulysses = self
            class SequenceParallelSampler(DefaultSampler):

                def __init__(self,
                             dataset,
                             shuffle: bool = True,
                             seed=None,
                             round_up: bool = True) -> None:
                    rank = ulysses.device_mesh['data'].get_coordinate()[0]
                    world_size = ulysses.device_mesh['data'].size()
                    self.rank = rank
                    self.world_size = world_size

                    self.dataset = dataset
                    self.shuffle = shuffle
                    assert seed is not None
                    self.seed = seed
                    self.epoch = 0
                    self.round_up = round_up

                    if self.round_up:
                        import math
                        self.num_samples = math.ceil(len(self.dataset) / world_size)
                        self.total_size = self.num_samples * self.world_size
                    else:
                        import math
                        self.num_samples = math.ceil(
                            (len(self.dataset) - rank) / world_size)
                        self.total_size = len(self.dataset)

            dataloader_params['sampler'] = SequenceParallelSampler(train_dataset, seed=1024)
            dataloader_params['drop_last'] = trainer.args.dataloader_drop_last
            dataloader_params['worker_init_fn'] = seed_worker

        return DataLoader(train_dataset, **dataloader_params)
