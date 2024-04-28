# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
import sys
import types
from collections import OrderedDict
from typing import List, Optional, Tuple

import einops
import safetensors
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, trainer
from transformers.modeling_utils import unwrap_model

from swift.utils import get_logger, torchacc_trim_graph, use_torchacc

logger = get_logger()


# DataLoader
def get_bucket_sizes(max_length: int) -> List[int]:
    return [max_length // 4 * (i + 1) for i in range(4)]


def _get_closet_bucket(bucket_sizes, data_length):
    """Select the one from bucket_sizes that is closest in distance to
    data_length. This is required for TorchAcc.
    """
    cloest_length = sys.maxsize
    for b in bucket_sizes:
        if b == data_length or ((b < cloest_length) and (b > data_length)):
            cloest_length = b

    if cloest_length == sys.maxsize:
        bucket_sizes.append(data_length)
        cloest_length = data_length

    return cloest_length


def pad_and_split_batch(padding_to, input_ids, attention_mask, labels,
                        loss_scale, max_length, tokenizer, rank, world_size):
    if padding_to is None:
        longest_len = input_ids.shape[-1]
        bucket_sizes = get_bucket_sizes(max_length)
        bucket_data_length = _get_closet_bucket(bucket_sizes, longest_len)
        padding_length = bucket_data_length - input_ids.shape[1]
        input_ids = F.pad(input_ids, (0, padding_length), 'constant',
                          tokenizer.pad_token_id)
        attention_mask = F.pad(attention_mask, (0, padding_length), 'constant',
                               0)
        if loss_scale:
            loss_scale = F.pad(loss_scale, (0, padding_length), 'constant', 0.)
        labels = F.pad(labels, (0, padding_length), 'constant', -100)

    # manully split the batch to different DP rank.
    batch_size = input_ids.shape[0] // world_size
    if batch_size > 0:
        start = rank * batch_size
        end = (rank + 1) * batch_size
        input_ids = input_ids[start:end, :]
        attention_mask = attention_mask[start:end, :]
        labels = labels[start:end, :]
        if loss_scale:
            loss_scale = loss_scale[start:end, :]
    return input_ids, attention_mask, labels, loss_scale


def ta_train_dataloader(train_dataset, data_collator, sampler, args,
                        batch_size):
    # patch skip_first_batches for customized dataloader.
    def acc_skip_first_batches(dataloader, num_batches=0):
        from accelerate.data_loader import SkipBatchSampler
        batch_sampler = SkipBatchSampler(
            dataloader._loader.batch_sampler, skip_batches=num_batches)
        dataset = dataloader.dataset
        dataloader_params = {
            'collate_fn': data_collator,
            'num_workers': args.dataloader_num_workers,
            'pin_memory': args.dataloader_pin_memory,
            'persistent_workers': args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params['batch_sampler'] = batch_sampler
            dataloader_params['worker_init_fn'] = trainer.seed_worker

        return ta.AsyncLoader(
            DataLoader(dataset, **dataloader_params), args.device)

    trainer.skip_first_batches = acc_skip_first_batches

    # dataloader for TorchAcc.
    import torchacc as ta

    dataloader_params = {
        'batch_size': batch_size,
        'collate_fn': data_collator,
        'num_workers': args.dataloader_num_workers,
        'pin_memory': args.dataloader_pin_memory,
        'persistent_workers': args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params['sampler'] = sampler
        dataloader_params['drop_last'] = args.dataloader_drop_last
        dataloader_params['worker_init_fn'] = trainer.seed_worker

    return ta.AsyncLoader(
        DataLoader(train_dataset, **dataloader_params), args.device)


def ta_eval_dataloader(eval_dataset, data_collator, sampler, args):
    import torchacc as ta

    dataloader_params = {
        'batch_size': args.eval_batch_size,
        'collate_fn': data_collator,
        'num_workers': args.dataloader_num_workers,
        'pin_memory': args.dataloader_pin_memory,
        'persistent_workers': args.dataloader_persistent_workers,
    }

    if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
        dataloader_params['sampler'] = sampler
        dataloader_params['drop_last'] = args.dataloader_drop_last

    return ta.AsyncLoader(
        DataLoader(eval_dataset, **dataloader_params), args.device)


def ta_test_dataloader(test_dataset, data_collator, sampler, args):
    import torchacc as ta

    dataloader_params = {
        'batch_size': args.eval_batch_size,
        'collate_fn': data_collator,
        'num_workers': args.dataloader_num_workers,
        'pin_memory': args.dataloader_pin_memory,
        'persistent_workers': args.dataloader_persistent_workers,
    }

    if not isinstance(test_dataset, torch.utils.data.IterableDataset):
        dataloader_params['sampler'] = sampler
        dataloader_params['drop_last'] = args.dataloader_drop_last

    # We use the same batch_size as for eval.
    return ta.AsyncLoader(
        DataLoader(test_dataset, **dataloader_params), args.device)


# Save/load checkpoint
def consolidate_checkpoint(resume_from_checkpoint, model_name='adapter_model'):
    """ Consolidate the sharded TorchAcc checkpoints into a single model checkpoint.
    """
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.fsdp import consolidate_sharded_state_dicts

    if model_name not in ('adapter_model', 'model'):
        logger.error('Only support PeftModel and PreTrainedModel.')
        return

    model_dir = os.path.join(resume_from_checkpoint, '0')
    is_pretrained_model = False
    if os.path.exists(os.path.join(model_dir, f'{model_name}.safetensors')):
        use_safetensors = True
    elif os.path.exists(os.path.join(model_dir, f'{model_name}.bin')):
        use_safetensors = False
    elif os.path.exists(os.path.join(model_dir, 'pytorch_model.bin')):
        # PreTrainedModel use 'pytorch_model.bin' and 'model.safetensors'
        use_safetensors = False
        is_pretrained_model = True
    else:
        logger.error('Cannot find checkpoint.')

    state_dict_list = []
    if xm.is_master_ordinal(local=False) and use_safetensors:
        from safetensors.torch import load_file, save_file
        for rank in range(xm.xrt_world_size()):
            shard_dir = os.path.join(resume_from_checkpoint, f'{rank}')
            filename = os.path.join(shard_dir, f'{model_name}.safetensors')
            state_dict = load_file(filename, device='cpu')
            state_dict = OrderedDict(('_fsdp_wrapped_module.' + k, v)
                                     for k, v in state_dict.items())
            state_dict_list.append(state_dict)
        shard_metadata = torch.load(
            os.path.join(model_dir, 'shard_meta.pth'), map_location='cpu')
    elif xm.is_master_ordinal(local=False):
        for rank in range(xm.xrt_world_size()):
            shard_dir = os.path.join(resume_from_checkpoint, f'{rank}')
            if not is_pretrained_model:
                filename = os.path.join(shard_dir, f'{model_name}.bin')
            else:
                filename = os.path.join(shard_dir, 'pytorch_model.bin')
            state_dict = torch.load(filename, map_location='cpu')
            state_dict = OrderedDict(('_fsdp_wrapped_module.' + k, v)
                                     for k, v in state_dict.items())
            state_dict_list.append(state_dict)
        shard_metadata = torch.load(
            os.path.join(model_dir, 'shard_meta.pth'), map_location='cpu')

    if xm.is_master_ordinal(local=False):
        full_state_dict = consolidate_sharded_state_dicts(
            state_dict_list, shard_metadata)
        # peft will prepend "default." prefix automatically, so we remove the
        # "default." prefix to prevent the duplication of the prefix.
        full_state_dict = OrderedDict(
            (k.replace('default.', ''), v) for k, v in full_state_dict.items())
        torch.save(full_state_dict,
                   os.path.join(resume_from_checkpoint, f'{model_name}.bin'))
        if model_name == 'adapter_model':
            config_path = os.path.join(resume_from_checkpoint,
                                       'adapter_config.json')
            old_config_path = os.path.join(model_dir, 'adapter_config.json')
            os.system(f'cp {old_config_path} {config_path}')
    xm.rendezvous('ckpt_consolidation')


def ta_save_optimizer_and_scheduler(optimizer, lr_scheduler, output_dir):
    import torch_xla.core.xla_model as xm
    xm.rendezvous('saving_optimizer_states')
    torch.save(optimizer.state_dict(),
               os.path.join(output_dir, f'optimizer_{xm.get_ordinal()}.pt'))
    torch.save(lr_scheduler.state_dict(),
               os.path.join(output_dir, f'scheduler_{xm.get_ordinal()}.pt'))


def ta_load_optimizer_and_scheduler(optimizer, lr_scheduler, checkpoint,
                                    device):
    import torch_xla.core.xla_model as xm
    optimizer_state = torch.load(
        os.path.join(checkpoint, f'optimizer_{xm.get_ordinal()}.pt'),
        map_location='cpu')
    lr_scheduler_state = torch.load(
        os.path.join(checkpoint, f'scheduler_{xm.get_ordinal()}.pt'),
        map_location='cpu')
    xm.send_cpu_data_to_device(optimizer_state, device)
    xm.send_cpu_data_to_device(lr_scheduler_state, device)

    optimizer.load_state_dict(optimizer_state)
    lr_scheduler.load_state_dict(lr_scheduler_state)
    return optimizer, lr_scheduler


def save_ta_checkpoint(self_model, tokenizer, args, output_dir):
    import torch_xla.core.xla_model as xm

    if xm.is_master_ordinal(local=False):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))

    model = self_model._get_underlay_model().module.module

    supported_classes = (PreTrainedModel, PeftModel)
    save_safetensors = args.save_safetensors
    # Save a trained model and configuration using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    xm.rendezvous('saving_checkpoint')
    out_dir = os.path.join(output_dir, f'{xm.get_ordinal()}')
    if not isinstance(model, supported_classes):
        state_dict = model.state_dict()
        _unwrap_model = unwrap_model(model)
        if isinstance(_unwrap_model, supported_classes):
            _unwrap_model.save_pretrained(
                out_dir, safe_serialization=save_safetensors)
        else:
            logger.info(
                'Trainer.model is not a `PreTrainedModel`, only saving its state dict.'
            )
            if save_safetensors:
                safetensors.torch.save_file(
                    state_dict, os.path.join(out_dir, 'model.safetensors'))
            else:
                torch.save(state_dict,
                           os.path.join(out_dir, 'pytorch_model.bin'))
    else:
        model.save_pretrained(out_dir, safe_serialization=save_safetensors)
    # save shard_metadata for consolidation.
    shard_meta = self_model._get_underlay_model().get_shard_metadata()
    xm.save(shard_meta, os.path.join(out_dir, 'shard_meta.pth'))
    xm.rendezvous('saving_checkpoint_done')

    if tokenizer is not None and args.should_save:
        tokenizer.save_pretrained(
            output_dir,
            is_main_process=xm.is_master_ordinal(local=False),
            save_function=xm.save)


def ta_trim_graph():
    if use_torchacc() and torchacc_trim_graph():
        import torchacc as ta
        ta.mark_step()


def patch_acc_model(model, args):
    if not args.use_flash_attn:
        return model
    if args.model_type.startswith('qwen'):
        import torchacc as ta
        model = ta.patch_qwen_model(model)
    elif args.model_type.startswith('baichuan'):
        model = patch_baichuan_model(model)
    elif args.model_type.startswith('llama') or args.model_type.startswith(
            'yi'):
        model = patch_llama_model(model)
    elif args.model_type.startswith('chatglm'):
        model = patah_chatglm_model(model)
    return model


def patch_llama_model(model):

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`):
                The position indices of the tokens corresponding to the query and key tensors. For example, this can be
                used to pass offsetted position ids when working with a KV-cache.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k.
                For example, note that cos[position_ids] and sin[position_ids]
                have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k.
                Similarly, if q and k have the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def llama_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        from torchacc.ops import flash_attn_varlen_xla

        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states).view(bsz, q_len, self.num_heads,
                                            self.head_dim).transpose(1, 2))
        key_states = (
            self.k_proj(hidden_states).view(bsz, q_len,
                                            self.num_key_value_heads,
                                            self.head_dim).transpose(1, 2))
        value_states = (
            self.v_proj(hidden_states).view(bsz, q_len,
                                            self.num_key_value_heads,
                                            self.head_dim).transpose(1, 2))

        kv_seq_len = key_states.shape[-2]
        assert past_key_value is None, 'past_key_value is not supported'

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)
        assert not output_attentions, 'output_attentions is not supported'

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        # See https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attention.py
        # if attention_mask is not None:
        #     value_states = value_states * attention_mask.unsqueeze(1).unsqueeze(-1)
        q = einops.rearrange(query_states, 'b h s ... -> (b s) h ...')
        k = einops.rearrange(key_states, 'b h s ... -> (b s) h ...')
        v = einops.rearrange(value_states, 'b h s ... -> (b s) h ...')
        max_s = q_len
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len,
            step=q_len,
            dtype=torch.int32,
            device=q.device)
        output = flash_attn_varlen_xla(
            q,
            k,
            v,
            cu_q_lens,
            cu_q_lens,
            max_s,
            max_s,
            0.0,
            softmax_scale=None,
            causal=True)
        output = einops.rearrange(output, '(b s) ... -> b s ...', b=bsz)

        return self.o_proj(einops.rearrange(
            output, 'b s h d -> b s (h d)')), None, past_key_value

    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(llama_attn_forward,
                                                   layer.self_attn)

    return model


def patah_chatglm_model(model):

    def apply_rotary_pos_emb(x: torch.Tensor,
                             rope_cache: torch.Tensor) -> torch.Tensor:
        # x: [sq, b, np, hn]
        sq, _, np, _ = x.size(0), x.size(1), x.size(2), x.size(3)
        rot_dim = rope_cache.shape[-2] * 2
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        # truncate to support variable sizes
        rope_cache = rope_cache[:sq]
        xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
        rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out2 = x_out2.flatten(3)
        return torch.cat((x_out2, x_pass), dim=-1)

    def chatglm_attn_forward(self,
                             hidden_states,
                             attention_mask,
                             rotary_pos_emb,
                             kv_cache=None,
                             use_cache=True):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition
                    * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition
                    * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition
                    * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(query_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head))
            key_layer = key_layer.view(key_layer.size()[:-1] + (
                self.num_multi_query_groups_per_partition,
                self.hidden_size_per_attention_head))
            value_layer = value_layer.view(value_layer.size()[:-1] + (
                self.num_multi_query_groups_per_partition,
                self.hidden_size_per_attention_head))
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer,
             value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=0)
            value_layer = torch.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition
                // self.num_multi_query_groups_per_partition, -1)
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:2] + (self.num_attention_heads_per_partition,
                                        self.hidden_size_per_attention_head))
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition
                // self.num_multi_query_groups_per_partition, -1)
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2]
                + (self.num_attention_heads_per_partition,
                   self.hidden_size_per_attention_head))

        # ==================================
        # core attention computation
        # ==================================

        from torchacc.ops import flash_attn_varlen_qkvpacked_xla
        query_layer, key_layer, value_layer = [
            k.permute(1, 2, 0, 3)
            for k in [query_layer, key_layer, value_layer]
        ]
        bsz, _, q_len, _ = query_layer.size()
        qkv = torch.stack([query_layer, key_layer, value_layer], dim=2)
        qkv = qkv.transpose(1, 3)
        qkv = einops.rearrange(qkv, 'b s ... -> (b s) ...')
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len,
            step=q_len,
            dtype=torch.int32,
            device=qkv.device)
        context_layer = flash_attn_varlen_qkvpacked_xla(
            qkv, cu_q_lens, q_len, 0.0, None, True, False)
        context_layer = einops.rearrange(
            context_layer, '(b s) ... -> b s ...', b=bsz)
        context_layer = context_layer.permute(1, 0, 2, 3)
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.core_attention.hidden_size_per_partition, )
        context_layer = context_layer.reshape(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output = self.dense(context_layer)

        return output, kv_cache

    def torchacc_swiglu(x):
        x = torch.chunk(x, 2, dim=-1)
        return F.silu(x[0]).to(x[0].dtype) * x[1]

    # patch attention
    for layer in model.transformer.encoder.layers:
        layer.self_attention.forward = types.MethodType(
            chatglm_attn_forward, layer.self_attention)
        layer.mlp.activation_func = torchacc_swiglu

    return model


def patch_baichuan_model(model):

    def baichuan_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        proj = self.W_pack(hidden_states)
        proj = (
            proj.unflatten(-1, (3, self.hidden_size)).unsqueeze(0).transpose(
                0, -2).squeeze(-2))
        query_states = (
            proj[0].view(bsz, q_len, self.num_heads,
                         self.head_dim).transpose(1, 2))
        key_states = (
            proj[1].view(bsz, q_len, self.num_heads,
                         self.head_dim).transpose(1, 2))
        value_states = (
            proj[2].view(bsz, q_len, self.num_heads,
                         self.head_dim).transpose(1, 2))

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        from torchacc.ops import flash_attn_varlen_xla
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        q, k, v = [
            einops.rearrange(x, 'b s ... -> (b s) ...')
            for x in [query_states, key_states, value_states]
        ]
        cu_q_lens = torch.arange(
            0, (bsz + 1) * q_len,
            step=q_len,
            dtype=torch.int32,
            device=q.device)
        output = flash_attn_varlen_xla(
            q,
            k,
            v,
            cu_q_lens,
            cu_q_lens,
            q_len,
            q_len,
            0.0,
            softmax_scale=None,
            causal=True)
        output = einops.rearrange(output, '(b s) ... -> b s ...', b=bsz)
        output = self.o_proj(einops.rearrange(output, 'b s h d -> b s (h d)'))
        return output, None, past_key_value

    for layer in model.base_model.layers:
        layer.self_attn.forward = types.MethodType(baichuan_attn_forward,
                                                   layer.self_attn)

    return model
