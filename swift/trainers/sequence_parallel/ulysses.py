from typing import Optional, Any, Tuple

import torch
from functools import partial
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch import Tensor
from torch.utils.data import DataLoader
from transformers.trainer_utils import seed_worker

from swift.trainers.sequence_parallel.base import SequenceParallel
from swift.utils import get_dist_setting


class ReduceLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, loss, labels, process_group):
        import torch.distributed as dist
        ctx.process_group = process_group
        ctx.shapes = labels.shape[0]
        world_size = dist.get_world_size(group=process_group)
        output = torch.empty((loss.shape[0] * world_size), dtype=loss.dtype, device=loss.device)
        dist.all_gather_into_tensor(output, loss, group=process_group)
        labels_output = torch.empty((labels.shape[0] * world_size), dtype=labels.dtype, device=labels.device)
        dist.all_gather_into_tensor(labels_output, labels, group=process_group)
        return output, labels_output

    @staticmethod
    def backward(ctx, grad_output, _):
        import torch.distributed as dist
        grad_output = grad_output * dist.get_world_size(group=ctx.process_group)
        return grad_output.split(ctx.shapes, dim=0)[dist.get_rank(ctx.process_group)].contiguous(), None, None


def loss_scale_sp_func(outputs, labels, loss_scale=None, num_items_in_batch=None, process_group=None) -> torch.Tensor:
    logits = outputs.logits
    device = logits.device
    masks = labels != -100
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.flatten().to(device)
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits, labels)

    if loss_scale is not None:
        loss_scale = loss_scale.flatten().to(logits.device)
        loss = (loss_scale * loss)
    loss, labels = ReduceLoss.apply(loss, labels, process_group)
    loss = loss[labels != -100].sum()/(labels != -100).sum()
    return loss


def _generate_layout_params(scatter_idx, batch_dim_idx, seq_world_size, input):
    """
    This function generates the parameters required for `permute` and `reshape` operations,
    which are used to process data before and after `all2all` communication.
    """
    if batch_dim_idx == 0:
        if scatter_idx < 2:
            bs, global_seq_len, num_local_head, head_dim = input.shape
            pre_all2all_inp_shape = [bs, seq_world_size, global_seq_len // seq_world_size, num_local_head, head_dim]
            pre_all2all_permute_idx = (1, 0, 2, 3, 4)

            post_all2all_permute_idx = (1, 2, 0, 3, 4)
            post_all2all_res_shape = [bs, global_seq_len // seq_world_size, seq_world_size * num_local_head, head_dim]
        else:
            bs, local_seq_len, num_total_head, head_dim = input.shape
            assert num_total_head % seq_world_size == 0, f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            pre_all2all_inp_shape = [bs, local_seq_len, seq_world_size, num_total_head // seq_world_size, head_dim]
            pre_all2all_permute_idx = (2, 0, 1, 3, 4)

            post_all2all_permute_idx = (1, 0, 2, 3, 4)
            post_all2all_res_shape = [bs, seq_world_size * local_seq_len, num_total_head // seq_world_size, head_dim]
    else:
        if scatter_idx < 2:
            global_seq_len, bs, num_local_head, head_dim = input.shape
            pre_all2all_inp_shape = [seq_world_size, global_seq_len // seq_world_size, bs, num_local_head, head_dim]
            pre_all2all_permute_idx = None

            post_all2all_permute_idx = (1, 2, 0, 3, 4)
            post_all2all_res_shape = [bs, seq_world_size * global_seq_len, num_local_head // seq_world_size, head_dim]
        else:
            local_seq_len, bs, num_total_head, head_dim = input.shape
            assert num_total_head % seq_world_size == 0, f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            pre_all2all_inp_shape = [local_seq_len, bs, seq_world_size, num_total_head // seq_world_size, head_dim]
            pre_all2all_permute_idx = (2, 0, 1, 3, 4)
            post_all2all_permute_idx = None
            post_all2all_res_shape = [local_seq_len * seq_world_size, bs, num_total_head // seq_world_size, head_dim]

    return pre_all2all_permute_idx, pre_all2all_inp_shape, post_all2all_permute_idx, post_all2all_res_shape


def post_all2all(permute_idx, res_shape):
    """
    Post-processing function for `all2all` communication.
    """

    def post_func(input):
        if permute_idx is not None:
            input = input.permute(permute_idx).contiguous()
        output = input.reshape(res_shape).contiguous()

        return output

    return post_func


def pre_all2all_fun(permute_idx, inp_shape, input):
    """
    Pre-processing function for `all2all` communication.
    """
    input_t = input.reshape(inp_shape).contiguous()
    if permute_idx is not None:
        input_t = input_t.permute(permute_idx).contiguous()
    return input_t


def single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, async_op=False, handle=None, type=None):
    seq_world_size = dist.get_world_size(group)
    num_heads = input.shape[2]
    if (num_heads % seq_world_size != 0 and not scatter_idx < 2):
        raise
    pre_all2all_permute_idx, pre_all2all_inp_shape, post_all2all_permute_idx, post_all2all_res_shape = _generate_layout_params(
        scatter_idx, batch_dim_idx, seq_world_size, input)

    input_t = pre_all2all_fun(pre_all2all_permute_idx, pre_all2all_inp_shape, input)

    post_all2all_fun = post_all2all(post_all2all_permute_idx, post_all2all_res_shape)
    output = torch.empty_like(input_t)
    work = dist.all_to_all_single(output, input_t, group=group, async_op=async_op)

    if async_op:
        if type in ('dq', 'dk'):
            handle[type + '_work'] = work
            handle[type + '_grad'] = output
            handle[type + '_post_all2all_func'] = post_all2all_fun
            return output.view(post_all2all_res_shape)

    res = post_all2all_fun(output)
    return res


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any,
                group: dist.ProcessGroup,
                input: Tensor,
                scatter_idx: int,
                gather_idx: int,
                batch_dim_idx: int,
                stream=None,
                handle=None,
                type=None,
                is_fwd=True) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.stream = stream
        ctx.handle = handle
        ctx.type = type
        ctx.batch_dim_idx = batch_dim_idx
        res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, False)
        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:

        return (None,
                _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.batch_dim_idx,
                                   ctx.stream, ctx.handle, ctx.type, False), None, None, None, None, None, None, None)


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
        class DistributedAttention(torch.nn.Module):
            def __init__(
                    self,
                    local_attention,
                    sequence_process_group: dist.ProcessGroup,
                    scatter_idx: int = 2,
                    gather_idx: int = 0,
            ) -> None:

                super(DistributedAttention, self).__init__()
                self.local_attn = local_attention
                self.spg = sequence_process_group
                self.scatter_idx = scatter_idx
                self.gather_idx = gather_idx
                self.overlap_handles = None
                self.sp_stream = None
                self.causal_mask_func = None

            def forward(self,
                        query: Tensor,
                        key: Tensor,
                        value: Tensor,
                        attention_mask: Tensor,
                        batch_dim_idx: int,
                        *args: Any,
                        **kwargs) -> Tensor:
                # from deepspeed.sequence.layer import _SeqAllToAll
                query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx, batch_dim_idx,
                                                 None,
                                                 self.overlap_handles, 'q')
                key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx, batch_dim_idx, None,
                                               self.overlap_handles, 'k')
                value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx, batch_dim_idx,
                                                 None,
                                                 self.overlap_handles, 'v')
                context_layer = self.local_attn(query_layer, key_layer, value_layer, attention_mask, *args, **kwargs)

                output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx, batch_dim_idx,
                                            self.sp_stream, self.overlap_handles, 'o')
                return output

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'] = ALL_ATTENTION_FUNCTIONS['flash_attention_2']
        ALL_ATTENTION_FUNCTIONS['sdpa_origin'] = ALL_ATTENTION_FUNCTIONS['sdpa']

        def local_flash_attn(module: torch.nn.Module,
                             query_states,
                             key_states,
                             value_states,
                             attention_mask,
                             *args, dist_attn, **kwargs):
            if dist_attn.local_attn is None:
                def _attention(query, key, value, *args, **kwargs):
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    return ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'](module, query, key, value, *args, **kwargs)[0]

                dist_attn.local_attn = _attention

            return dist_attn(query_states.transpose(1,2),
                             key_states.transpose(1,2),
                             value_states.transpose(1,2), attention_mask, 0, *args, **kwargs), None

        def local_sdpa_attn(module: torch.nn.Module,
                            query_states,
                            key_states,
                            value_states,
                            attention_mask,
                            *args, dist_attn, **kwargs):
            if dist_attn.local_attn is None:
                def _attention(query, key, value, *args, **kwargs):
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    return ALL_ATTENTION_FUNCTIONS['sdpa_origin'](module, query, key, value, *args, **kwargs)[0]

                dist_attn.local_attn = _attention
            return dist_attn(query_states.transpose(1,2),
                             key_states.transpose(1,2),
                             value_states.transpose(1,2), attention_mask, 0, *args, **kwargs), None

        ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = partial(local_flash_attn, dist_attn=
        DistributedAttention(None, self.group, 2, 1))
        ALL_ATTENTION_FUNCTIONS['sdpa'] = partial(local_sdpa_attn, dist_attn=
        DistributedAttention(None, self.group, 2, 1))

        self.causal_mask_func = model.model.model._update_causal_mask
        self.model_dtype = next(model.parameters()).dtype

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
        labels[:, 0] = -100
        position_ids = self.pad_for_sequence_parallel(position_ids, padding_value=0, dim=-1)
        attention_mask = self.pad_for_sequence_parallel(attention_mask, padding_value=0, dim=-1)

        sp_group = self.group
        cache_position = torch.arange(
            0, input_ids.shape[1], device=input_ids.device
        )
        attention_mask = self.causal_mask_func(attention_mask, input_ids.to(self.model_dtype), cache_position, None, None)
        input_ids = self.split_for_sequence_parallel(input_ids, dim=1, sp_group=sp_group)
        labels = torch.roll(labels, shifts=-1, dims=1)
        labels = self.split_for_sequence_parallel(labels, dim=1, sp_group=sp_group)
        position_ids = self.split_for_sequence_parallel(position_ids, dim=1, sp_group=sp_group)

        if loss_scale is not None:
            loss_scale = self.pad_for_sequence_parallel(loss_scale, padding_value=0., dim=-1)
            loss_scale = torch.roll(loss_scale, shifts=-1, dims=1)
            loss_scale = self.split_for_sequence_parallel(loss_scale, dim=1, sp_group=sp_group)

        return input_ids, labels, position_ids, attention_mask, loss_scale

    def reduce_outputs(self, loss, labels):
        return loss

    def world_size(self):
        return self.sequence_parallel_size

    @property
    def is_last_sequence(self):
        return dist.get_rank(self.device_mesh['sequence'].get_group()) == self.world_size() - 1

    def get_dataloader(self, trainer):
        if trainer.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')

        from swift.plugin.loss import loss_scale_sp_func
        trainer.compute_loss_func = partial(loss_scale_sp_func, process_group=self.device_mesh['sequence'].get_group())
        train_dataset = trainer.train_dataset
        data_collator = trainer.data_collator

        def _compute_acc(trainer, outputs, labels) -> None:
            args = trainer.args
            acc_steps = args.acc_steps
            preds = outputs.logits.argmax(dim=-1)
            if trainer.state.global_step % acc_steps == 0:
                import torch.distributed as dist
                world_size = dist.get_world_size(group=self.device_mesh['sequence'].get_group())
                preds_output = torch.empty((preds.shape[0] * world_size, preds.shape[1]), dtype=preds.dtype, device=preds.device)
                dist.all_gather_into_tensor(preds_output, preds, group=self.device_mesh['sequence'].get_group())
                preds_output = torch.cat(preds_output.split(preds.shape[0], dim=0), dim=1)
                labels_output = torch.empty((labels.shape[0] * world_size, labels.shape[1]), dtype=labels.dtype, device=labels.device)
                dist.all_gather_into_tensor(labels_output, labels, group=self.device_mesh['sequence'].get_group())
                labels_output = torch.cat(labels_output.split(labels.shape[0], dim=0), dim=1)
                labels_output = torch.roll(labels_output, shifts=1, dims=1)
                from swift.plugin import MeanMetric, compute_acc, extra_tuners
                metrics = compute_acc(
                    preds_output, labels_output, acc_strategy=args.acc_strategy, is_encoder_decoder=trainer.template.is_encoder_decoder)
                for k, v in metrics.items():
                    if k not in trainer._custom_metrics:
                        trainer._custom_metrics[k] = MeanMetric(nan_value=None)
                    trainer._custom_metrics[k].update(v)

        from types import MethodType
        trainer._compute_acc = MethodType(_compute_acc, trainer)

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
