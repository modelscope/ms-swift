import math
from functools import partial
from types import MethodType
from typing import Any, Iterator, Optional, Sized, Tuple
import inspect
import torch
import torch.distributed as dist
from torch import Tensor
from contextlib import contextmanager
from torch.distributed.device_mesh import init_device_mesh
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Sampler
from transformers.modeling_flash_attention_utils import is_flash_attn_available
from transformers.trainer_utils import seed_worker

from swift.trainers.sequence_parallel.base import SequenceParallel
from swift.utils import get_device, get_dist_setting


class GatherLoss(torch.autograd.Function):
    """Gather loss from sequence group"""

    @staticmethod
    def forward(ctx, loss, labels, process_group):
        """
        Args:
            loss: loss tensor after splitting
            labels: labels tensor after splitting
            process_group: the sequence parallel group
        """

        ctx.process_group = process_group
        ctx.shapes = labels.shape[0]  # flatten
        world_size = dist.get_world_size(group=process_group)  # the sp world size
        output = torch.empty((loss.shape[0] * world_size), dtype=loss.dtype, device=loss.device)
        # gather all from sp group
        dist.all_gather_into_tensor(output, loss, group=process_group)
        labels_output = torch.empty((labels.shape[0] * world_size), dtype=labels.dtype, device=labels.device)
        dist.all_gather_into_tensor(labels_output, labels, group=process_group)
        return output, labels_output

    @staticmethod
    def backward(ctx, *grad_output):
        _grad = grad_output[0] * dist.get_world_size(group=ctx.process_group)
        return _grad.split(ctx.shapes, dim=0)[dist.get_rank(ctx.process_group)].contiguous(), None, None


def loss_scale_sp_func(outputs, labels, loss_scale=None, num_items_in_batch=None, process_group=None) -> torch.Tensor:
    logits = outputs.logits
    device = logits.device
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.flatten().to(device)
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none')
    # flatten loss
    loss = loss_fct(logits, labels)

    if loss_scale is not None:
        loss_scale = loss_scale.flatten().to(loss.device)
        loss = (loss_scale * loss)
    loss, labels = GatherLoss.apply(loss, labels, process_group)
    loss = loss[labels != -100].sum() / (labels != -100).sum()
    return loss


class UlyssesSampler(Sampler):

    # Code borrwoed from mmengine
    def __init__(self, ulysses, dataset, shuffle: bool = True, seed=None, round_up: bool = True) -> None:
        self.ulysses = ulysses
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
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        if self.round_up:
            indices = (indices * int(self.total_size / len(indices) + 1))[:self.total_size]

        indices = indices[self.rank:self.total_size:self.world_size]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


# Code borrowed from deepspeed, here is why:
# 1. Reduce the dependency
# 2. The original code is complex
def _generate_layout_params(scatter_idx, seq_world_size, input):
    if scatter_idx < 2:
        bs, global_seq_len, num_local_head, head_dim = input.shape
        pre_all2all_inp_shape = [bs, seq_world_size, global_seq_len // seq_world_size, num_local_head, head_dim]
        pre_all2all_permute_idx = (1, 0, 2, 3, 4)

        post_all2all_permute_idx = (1, 2, 0, 3, 4)
        post_all2all_res_shape = [bs, global_seq_len // seq_world_size, seq_world_size * num_local_head, head_dim]
    else:
        bs, local_seq_len, num_total_head, head_dim = input.shape
        assert num_total_head % seq_world_size == 0, (f'Number of heads ({num_total_head}) must be divisible '
                                                      f'by the sequence parallel size ({seq_world_size})!')
        pre_all2all_inp_shape = [bs, local_seq_len, seq_world_size, num_total_head // seq_world_size, head_dim]
        pre_all2all_permute_idx = (2, 0, 1, 3, 4)

        post_all2all_permute_idx = (1, 0, 2, 3, 4)
        post_all2all_res_shape = [bs, seq_world_size * local_seq_len, num_total_head // seq_world_size, head_dim]

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


def single_all_to_all(input, scatter_idx, gather_idx, group, **kwargs):
    seq_world_size = dist.get_world_size(group)
    num_heads = input.shape[2]
    if num_heads % seq_world_size != 0 and not scatter_idx < 2:
        raise NotImplementedError
    pre_all2all_permute_idx, pre_all2all_inp_shape, post_all2all_permute_idx, post_all2all_res_shape = (
        _generate_layout_params(scatter_idx, seq_world_size, input))

    input_t = pre_all2all_fun(pre_all2all_permute_idx, pre_all2all_inp_shape, input)

    post_all2all_fun = post_all2all(post_all2all_permute_idx, post_all2all_res_shape)
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    res = post_all2all_fun(output)
    return res


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx: Any,
            group: dist.ProcessGroup,
            input: Tensor,
            scatter_idx: int,
            gather_idx: int,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        res = single_all_to_all(input, scatter_idx, gather_idx, group)
        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


class DistributedAttention(torch.nn.Module):

    def __init__(
            self,
            local_attention,
            sequence_process_group: dist.ProcessGroup,
            scatter_idx: int = 2,
            gather_idx: int = 1,
    ) -> None:
        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.causal_mask_func = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_mask: Tensor, *args: Any,
                **kwargs) -> Tensor:
        query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx)
        value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx)
        position_ids = kwargs.pop('position_ids', None)
        if position_ids is not None:
            position_ids_output = torch.empty(
                (position_ids.shape[0] * dist.get_world_size(self.spg), position_ids.shape[1]),
                dtype=position_ids.dtype, device=position_ids.device)
            dist.all_gather_into_tensor(position_ids_output, position_ids, group=self.spg)
            position_ids = torch.cat(position_ids_output.split(1, dim=0), dim=1)
        context_layer = self.local_attn(query_layer, key_layer, value_layer, attention_mask, *args,
                                        position_ids=position_ids, **kwargs)
        output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)
        return output


class Ulysses(SequenceParallel):

    def __init__(self):
        self.model_dtype = None
        self.causal_mask_func = None
        self.device_mesh = None
        self._llm_attn = False

    def init_sequence_parallel(self, size):
        self.sp_world_size = size
        rank, local_rank, world_size, local_world_size = get_dist_setting()
        self.dp_world_size = world_size // size
        self.device_mesh = init_device_mesh(
            get_device().split(':')[0], mesh_shape=(world_size // size, size), mesh_dim_names=['data', 'sequence'])

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'] = ALL_ATTENTION_FUNCTIONS['flash_attention_2']
        ALL_ATTENTION_FUNCTIONS['sdpa_origin'] = ALL_ATTENTION_FUNCTIONS['sdpa']

        def local_flash_attn(module: torch.nn.Module, query_states, key_states, value_states, attention_mask, *args,
                             dist_attn, **kwargs):
            if dist_attn.local_attn is None:
                def _attention(query, key, value, *args, **kwargs):
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    return ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'](module, query, key, value, *args,
                                                                               **kwargs)[0]

                dist_attn.local_attn = _attention

            return dist_attn(
                query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attention_mask,
                *args, **kwargs), None

        def local_sdpa_attn(module: torch.nn.Module, query_states, key_states, value_states, attention_mask, *args,
                            dist_attn, **kwargs):
            if dist_attn.local_attn is None:
                def _attention(query, key, value, *args, **kwargs):
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    return ALL_ATTENTION_FUNCTIONS['sdpa_origin'](module, query, key, value, *args, **kwargs)[0]

                dist_attn.local_attn = _attention
            return dist_attn(
                query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attention_mask,
                *args, **kwargs), None

        ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = partial(
            local_flash_attn, dist_attn=DistributedAttention(None, self.sp_group))
        ALL_ATTENTION_FUNCTIONS['sdpa'] = partial(local_sdpa_attn, dist_attn=DistributedAttention(None, self.sp_group))
        
        if is_flash_attn_available():
            from transformers import modeling_flash_attention_utils
            from transformers.modeling_flash_attention_utils import _flash_attention_forward
            _distributed_flash_attention = DistributedAttention(_flash_attention_forward, self.sp_group)
            def flash_attention_forward(query_states: torch.Tensor, key_states: torch.Tensor,
                                        value_states: torch.Tensor,
                                        attention_mask: Optional[torch.Tensor],
                                        q_len,
                                        *args, **kwargs):
                if self._llm_attn:
                    return _distributed_flash_attention(query_states, key_states, value_states, attention_mask, q_len*self.sp_world_size, *args, **kwargs)
                else:
                    return _flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len, *args, **kwargs)
            modeling_flash_attention_utils._flash_attention_forward = flash_attention_forward

        from torch.nn import functional
        from torch.nn.functional import scaled_dot_product_attention
        def _scaled_dot_product_attention_wrap(query_states,
            key_states,
            value_states,
            attn_mask, *args, position_ids, **kwargs):
            return scaled_dot_product_attention(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attn_mask, *args, **kwargs)
        _distributed_attention = DistributedAttention(_scaled_dot_product_attention_wrap, self.sp_group)
        def _scaled_dot_product_attention(query_states,
            key_states,
            value_states,
            attn_mask, *args, **kwargs):
            if self._llm_attn:
                if attn_mask.shape[2] != attn_mask.shape[3]:
                    attn_mask = self.attention_mask
                return _distributed_attention(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attn_mask, *args, **kwargs)
            else:
                return scaled_dot_product_attention(query_states, key_states, value_states, attn_mask, *args, **kwargs)

        functional.scaled_dot_product_attention = _scaled_dot_product_attention

    @contextmanager
    def sp_context(self):
        self._llm_attn = True
        yield
        self._llm_attn = False


    def prepare_model(self, model, tokenizer, split_in_forward):
        self.split_in_forward = split_in_forward
        def forward(_self, **kwargs):
            inputs_embeds = kwargs['inputs_embeds']
            position_ids = kwargs['position_ids']
            attention_mask = kwargs['attention_mask']
            _, inputs_embeds, _, position_ids, attention_mask, _ = self.pad_and_split_inputs(
                tokenizer, None, inputs_embeds, None,
                position_ids, attention_mask, None, embed_tokens=_self.embed_tokens)
            kwargs['inputs_embeds'] = inputs_embeds
            kwargs['position_ids'] = position_ids
            kwargs['attention_mask'] = attention_mask
            with self.sp_context():
                return _self.forward_origin(**kwargs)

        if hasattr(model.model, '_update_causal_mask'):
            self.causal_mask_func = model.model._update_causal_mask
            if self.split_in_forward:
                model.model.forward_origin = model.model.forward
                model.model.forward = MethodType(forward, model.model)
        else:
            self.causal_mask_func = model.model.model._update_causal_mask
            if self.split_in_forward:
                model.model.model.forward_origin = model.model.model.forward
                model.model.model.forward = MethodType(forward, model.model.model)
        self.model_dtype = next(model.parameters()).dtype

    def _pad_sp(self, tensor, padding_value, dim=-1):
        # code borrowed from xtuner
        length = tensor.shape[dim]
        if length % self.sp_world_size == 0:
            return tensor

        pad_num = self.sp_world_size - (length % self.sp_world_size)
        if not isinstance(padding_value, torch.Tensor):
            pad_shape = ((*tensor.shape[:dim], pad_num, *tensor.shape[dim + 1:]) if dim != -1 else
                         (*tensor.shape[:dim], pad_num))
            pad = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, pad], dim=dim)
        else:
            tensor = torch.cat([tensor, padding_value.unsqueeze(0).repeat(tensor.shape[0], pad_num, 1)], dim=dim)
        return tensor

    def world_size(self):
        return self.sp_world_size

    def _split_sp(self, input, dim: int, sp_group: dist.ProcessGroup):
        # code borrowed from xtuner
        if self.sp_world_size == 1:
            return input

        rank = dist.get_rank(sp_group)
        dim_size = input.size(dim)
        assert dim_size % self.sp_world_size == 0, (f'The dimension to split ({dim_size}) is not a multiple of '
                                                    f'world size ({self.sp_world_size}), cannot split tensor evenly')

        tensor_list = torch.split(input, dim_size // self.sp_world_size, dim=dim)
        output = tensor_list[rank].contiguous()

        return output

    def pad_and_split_inputs(self, tokenizer, input_ids, input_embeds,
                             labels, position_ids, attention_mask, loss_scale,
                             embed_tokens=None):
        sp_group = self.sp_group
        split_inputs = False
        if (input_ids is not None and not self.split_in_forward) or input_embeds is not None:
            # Whether split the model inputs
            # cannot split input_ids when multi modal
            split_inputs = True
        if input_ids is not None and split_inputs:
            input_ids = self._pad_sp(input_ids, padding_value=tokenizer.pad_token_id, dim=-1)
        if input_embeds is not None:
            pad_emb = embed_tokens(torch.tensor(tokenizer.pad_token_id).to(embed_tokens.weight.device)).unsqueeze(0)
            input_embeds = self._pad_sp(input_embeds, padding_value=pad_emb, dim=1)
        if position_ids is not None and split_inputs:
            position_ids = self._pad_sp(position_ids, padding_value=0, dim=-1)
        if split_inputs:
            inputs = input_ids if input_ids is not None else input_embeds
            attn_shape = inputs.shape[1]
            if attention_mask is None:
                attention_mask = torch.ones_like(position_ids)
            attention_mask = self._pad_sp(attention_mask, padding_value=0, dim=-1)
            cache_position = torch.arange(0, attn_shape, device=inputs.device)
            # pad attention mask to 4d to avoid calculation errors
            attention_mask = self.causal_mask_func(attention_mask, inputs.to(self.model_dtype), cache_position, None,
                                                   None)
            self.attention_mask = attention_mask
        if input_ids is not None and split_inputs:
            input_ids = self._split_sp(input_ids, dim=1, sp_group=sp_group)
        if input_embeds is not None:
            input_embeds = self._split_sp(input_embeds, dim=1, sp_group=sp_group)
        if position_ids is not None and split_inputs:
            position_ids = self._split_sp(position_ids, dim=-1, sp_group=sp_group)
        if labels is not None:
            labels = self._pad_sp(labels, padding_value=-100, dim=-1)
            labels[:, 0] = -100  # make the last invalid, so we do not need to cut the loss of last token
            labels = torch.roll(labels, shifts=-1, dims=1)
            labels = self._split_sp(labels, dim=1, sp_group=sp_group)

        if loss_scale is not None:
            loss_scale = self._pad_sp(loss_scale, padding_value=0., dim=-1)
            loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1)
            loss_scale = self._split_sp(loss_scale, dim=-1, sp_group=sp_group)

        return input_ids, input_embeds, labels, position_ids, attention_mask, loss_scale

    def reduce_outputs(self, loss, labels):
        return loss

    @property
    def sp_rank(self):
        return dist.get_rank(self.device_mesh['sequence'])

    @property
    def dp_rank(self):
        return dist.get_rank(self.device_mesh['data'])

    @property
    def sp_group(self):
        return self.device_mesh['sequence'].get_group()

    @property
    def dp_group(self):
        return self.device_mesh['data'].get_group()

    @property
    def is_last_sequence(self):
        return dist.get_rank(self.sp_rank) == self.sp_world_size - 1

    def get_dataloader(self, trainer):
        if trainer.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')

        trainer.compute_loss_func = partial(loss_scale_sp_func, process_group=self.sp_group)
        train_dataset = trainer.train_dataset
        data_collator = trainer.data_collator

        def _compute_acc(trainer, outputs, labels) -> None:
            args = trainer.args
            acc_steps = args.acc_steps
            preds = outputs.logits.argmax(dim=-1)
            if trainer.state.global_step % acc_steps == 0:
                # Gather preds and labels across the sp group
                preds_output = torch.empty((preds.shape[0] * self.sp_world_size, preds.shape[1]),
                                           dtype=preds.dtype,
                                           device=preds.device)
                dist.all_gather_into_tensor(preds_output, preds, group=self.sp_group)
                preds_output = torch.cat(preds_output.split(preds.shape[0], dim=0), dim=1)
                labels_output = torch.empty((labels.shape[0] * self.sp_world_size, labels.shape[1]),
                                            dtype=labels.dtype,
                                            device=labels.device)
                dist.all_gather_into_tensor(labels_output, labels, group=self.sp_group)
                labels_output = torch.cat(labels_output.split(labels.shape[0], dim=0), dim=1)
                # roll back to fit compute_acc
                labels_output = torch.roll(labels_output, shifts=1, dims=1)
                from swift.plugin import MeanMetric, compute_acc
                metrics = compute_acc(
                    preds_output,
                    labels_output,
                    acc_strategy=args.acc_strategy,
                    is_encoder_decoder=trainer.template.is_encoder_decoder)
                for k, v in metrics.items():
                    if k not in trainer._custom_metrics:
                        trainer._custom_metrics[k] = MeanMetric(nan_value=None)
                    trainer._custom_metrics[k].update(v)

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
            dataloader_params['sampler'] = UlyssesSampler(self, train_dataset, seed=42)
            dataloader_params['drop_last'] = trainer.args.dataloader_drop_last
            dataloader_params['worker_init_fn'] = seed_worker

        return DataLoader(train_dataset, **dataloader_params)
