# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import _get_extra_te_kwargs
from megatron.core.models.huggingface import HuggingFaceModule as _HuggingFaceModule
from megatron.core.tensor_parallel import (gather_from_sequence_parallel_region,
                                           reduce_scatter_to_sequence_parallel_region)
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig

from swift.model import ModelType
from swift.template import Template
from swift.utils import get_env_args
from ..constant import MegatronModelType
from ..gpts.qwen3_next import Qwen3NextBridge, Qwen3NextLoader
from ..register import MegatronModelMeta, register_megatron_model
from .utils import HuggingFaceModule

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet as _Qwen3_5MoeGatedDeltaNet
except ImportError:
    _Qwen3_5MoeGatedDeltaNet = object


class Qwen3_5MoeGatedDeltaNet(_HuggingFaceModule, _Qwen3_5MoeGatedDeltaNet):

    def __init__(self, config: TransformerConfig, submodules: SelfAttentionSubmodules, layer_number: int, **kwargs):
        assert _Qwen3_5MoeGatedDeltaNet is not object, 'please update the `transformers` version.'
        _Qwen3_5MoeGatedDeltaNet.__init__(self, config, layer_number)
        self.config = config
        self.pg_collection = kwargs.get('pg_collection')
        extra_kwargs = _get_extra_te_kwargs(config)
        self.to(dtype=extra_kwargs['params_dtype'], device=extra_kwargs['device'])

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        args = self.config.args
        cp_size = getattr(args, 'context_parallel_size', 1)
        cp_group = None
        if cp_size > 1:
            cp_group = (
                self.pg_collection.cp
                if self.pg_collection is not None and hasattr(self.pg_collection, 'cp')
                else parallel_state.get_context_parallel_group()
            )

        if args.sequence_parallel and args.tensor_model_parallel_size > 1:
            hidden_states = gather_from_sequence_parallel_region(hidden_states)
        if cp_size > 1:
            hidden_states = gather_from_sequence_parallel_region(hidden_states, group=cp_group)
        seq_len = hidden_states.shape[0]
        packed_seq_params = kwargs.get('packed_seq_params')
        thd_format = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        # Note: for packed inputs, we do not perform padding_free unpadding.
        # Doing so would allow different sequences to see each other; for efficiency we keep this implementation.
        # CP with thd_format requires global cu_seqlens; for now we support CP only in non-thd case.
        if thd_format and not args.packing:
            if cp_size > 1:
                raise NotImplementedError(
                    'Qwen3_5MoeGatedDeltaNet: context parallel with thd_format (packed sequences) is not yet supported.'
                )
            new_hidden_states = hidden_states.new_zeros(
                (packed_seq_params.num_samples, packed_seq_params.max_seqlen_q.item(), hidden_states.shape[-1]))
            attention_mask = hidden_states.new_zeros(
                (packed_seq_params.num_samples, packed_seq_params.max_seqlen_q.item()), dtype=torch.bool)
            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            for i in range(packed_seq_params.num_samples):
                start, end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
                attention_mask[i, :end - start] = True
                new_hidden_states[i, :end - start] = hidden_states[start:end, 0]
            hidden_states = new_hidden_states
        else:
            hidden_states = hidden_states.transpose(0, 1)
            attention_mask = kwargs.get('attention_mask')
            if attention_mask is not None:
                if cp_size > 1:
                    # Gather attention_mask along seq dim (dim 2): (batch, 1, sq, sk) -> full (batch, 1, seq, sk)
                    attn_for_gather = attention_mask.permute(2, 0, 1, 3).contiguous()
                    attn_gathered = gather_from_sequence_parallel_region(attn_for_gather, group=cp_group)
                    attention_mask = attn_gathered.permute(1, 2, 0, 3)
                attention_mask = (~attention_mask).sum(dim=(1, 2)) > 0
        res = super().forward(hidden_states=hidden_states, attention_mask=attention_mask)
        if thd_format and not args.packing:
            res = res[attention_mask][:, None]
            res = torch.concat([res, res.new_zeros(seq_len - res.shape[0], 1, res.shape[2])])
        else:
            res = res.transpose(0, 1).contiguous()
        if cp_size > 1:
            res = reduce_scatter_to_sequence_parallel_region(res, group=cp_group) / cp_size
        if args.sequence_parallel and args.tensor_model_parallel_size > 1:
            # Quick fix for dropout issue, awaiting ms-swift 4.0 refactor
            res = reduce_scatter_to_sequence_parallel_region(res) / args.tensor_model_parallel_size
        return res, None


class Qwen3_5Vit(HuggingFaceModule):
    module_mapping = {'model.visual': 'visual'}
    _vision_tower = ['visual']
    _aligner = ['visual.merger']

    def __init__(self, config):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextModel
        super().__init__(config, [Qwen3_5TextModel, Qwen3_5MoeTextModel])

    def get_inputs_embeds(self, inputs_embeds, **kwargs):
        return Template._get_inputs_embeds_hf(inputs_embeds, kwargs, self.visual, self.processor, self.hf_config)


class Qwen3_5Bridge(Qwen3NextBridge):
    hf_layers_prefix = 'model.language_model.layers'
    hf_embed_key = 'model.language_model.embed_tokens.weight'
    hf_final_layernorm_key = 'model.language_model.norm.weight'


class Qwen3_5Loader(Qwen3NextLoader):
    gated_delta_net = Qwen3_5MoeGatedDeltaNet


use_mcore_gdn = get_env_args('SWIFT_USE_MCORE_GDN', bool, False)

if not use_mcore_gdn:
    register_megatron_model(
        MegatronModelMeta(
            MegatronModelType.qwen3_5,
            [
                ModelType.qwen3_5,
                ModelType.qwen3_5_moe,
            ],
            bridge_cls=Qwen3_5Bridge,
            visual_cls=Qwen3_5Vit,
            loader=Qwen3_5Loader,
        ))
