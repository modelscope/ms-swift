from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import torch.nn.functional as F
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.transformer import TransformerConfig

from swift.megatron.utils import convert_hf_config
from swift.utils import get_logger, json_parse_to_dict

logger = get_logger()


@dataclass
class MegatronModelConfig(TransformerConfig):
    hf_model_type: Optional[str] = None
    llm_model_type: Optional[str] = None
    padded_vocab_size: Optional[int] = None
    rope_scaling: Optional[Union[dict, str]] = None

    # model
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_query_groups: Optional[int] = None
    softmax_type: Literal['vanilla', 'off-by-one', 'learnable'] = 'vanilla'
    window_size: Optional[str] = None
    window_attn_skip_freq: Optional[str] = None
    max_position_embeddings: Optional[int] = None

    position_embedding_type: Literal['learned_absolute', 'rope', 'mrope', 'none'] = 'rope',
    rotary_base: int = 10000
    rotary_percent: float = 1.
    rotary_interleaved: bool = False
    original_max_position_embeddings: Optional[int] = None
    partial_rotary_factor: Optional[float] = None
    mrope_section: Optional[List[int]] = None
    # qwen3_vl, qwen3_omni
    mrope_interleaved: bool = False

    normalization: Literal['LayerNorm', 'RMSNorm'] = 'RMSNorm'
    layernorm_epsilon: float = 1e-5
    swiglu: bool = True
    quick_geglu: bool = False
    activation_func_clamp_value: Optional[float] = None
    glu_linear_offset: float = 0.
    untie_embeddings_and_output_weights: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    attention_dropout: float = 0.
    hidden_dropout: float = 0.
    kv_channels: Optional[int] = None
    qk_layernorm: bool = False
    qk_l2_norm: Optional[bool] = None
    no_rope_freq: Optional[int] = None
    moe_apply_probs_on_input: Optional[bool] = None

    # moe
    num_experts: Optional[int] = None
    moe_layer_freq: str = 1
    moe_ffn_hidden_size: Optional[int] = None
    moe_shared_expert_intermediate_size: Optional[int] = None

    moe_router_topk: int = 2
    moe_router_num_groups: Optional[int] = None
    moe_router_group_topk: Optional[int] = None
    moe_router_pre_softmax: bool = False
    moe_router_dtype: Literal['none', 'fp32', 'fp64'] = 'fp32'
    moe_router_score_function: Literal['sigmoid', 'softmax'] = 'softmax'
    moe_router_bias_update_rate: Optional[float] = None
    moe_router_enable_expert_bias: bool = False
    moe_router_topk_scaling_factor: Optional[float] = None
    moe_router_load_balancing_type: Literal['aux_loss', 'seq_aux_loss', 'global_aux_loss', 'sinkhorn',
                                            'none'] = 'aux_loss'
    use_shared_expert_gate: bool = False

    # mla
    multi_latent_attention: bool = False
    q_lora_rank: Optional[int] = None
    kv_lora_rank: int = 32
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128

    # qwen3_next
    linear_num_value_heads: Optional[int] = None
    linear_num_key_heads: Optional[int] = None
    linear_key_head_dim: Optional[int] = None
    linear_value_head_dim: Optional[int] = None
    linear_conv_kernel_dim: Optional[int] = None
    layer_types: Optional[List[str]] = None

    # apply_layernorm_1p: bool = False  # TODO

    def __post_init__(self):
        if self.moe_router_dtype.lower() == 'none':
            self.moe_router_dtype = None
        if self.num_experts is not None:
            if self.moe_ffn_hidden_size is None:
                self.moe_ffn_hidden_size = self.ffn_hidden_size
        if self.rope_scaling is not None:
            self.rope_scaling = json_parse_to_dict(self.rope_scaling)
            if 'type' in self.rope_scaling and 'rope_type' not in self.rope_scaling:
                self.rope_scaling['rope_type'] = self.rope_scaling['type']
        super().__post_init__()
        self.variable_seq_lengths = True


def create_mcore_model_config(args, hf_config):
    # Translate args to core transformer configuration
    kw_args = convert_hf_config(hf_config)
    kw_args['persist_layer_norm'] = True
    # TODO: apply_layernorm_1p
    kw_args['layernorm_zero_centered_gamma'] = args.apply_layernorm_1p
    kw_args['deallocate_pipeline_outputs'] = True
    kw_args['pipeline_dtype'] = args.torch_dtype
    kw_args['batch_p2p_comm'] = True
    kw_args['num_moe_experts'] = args.num_experts
    kw_args['rotary_interleaved'] = args.rotary_interleaved
    kw_args['num_layers_in_first_pipeline_stage'] = args.decoder_first_pipeline_num_layers
    kw_args['num_layers_in_last_pipeline_stage'] = args.decoder_last_pipeline_num_layers
    kw_args['fp8_param'] = args.fp8_param_gather
    if args.swiglu:
        kw_args['activation_func'] = F.silu
        kw_args['gated_linear_unit'] = True
        kw_args['bias_activation_fusion'] = args.bias_swiglu_fusion
    else:
        kw_args['bias_activation_fusion'] = args.bias_gelu_fusion
    if args.quick_geglu:
        assert not args.swiglu
        kw_args['gated_linear_unit'] = True
        kw_args['activation_func'] = quick_gelu
    kw_args['cp_comm_type'] = 'p2p'
    kw_args['inference_sampling_seed'] = args.seed
    kw_args['variable_seq_lengths'] = True
    config = MegatronModelConfig(**kw_args)
    config.args = config
    return config
