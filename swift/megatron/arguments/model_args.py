from typing import Optional, Literal, List
from dataclasses import dataclass


from swift.utils import get_logger
from transformers.utils import is_torch_npu_available


MAX_NPU_EXPERTS_PER_EP = 128


logger = get_logger()


@dataclass
class MegatronModelArguments:
    hf_model_type: Optional[str] = None
    llm_model_type: Optional[str] = None
    padded_vocab_size: Optional[int] = None
    # model
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    group_query_attention: bool = False
    num_query_groups: Optional[int] = None
    softmax_type: Literal['vanilla', 'off-by-one', 'learnable'] = 'vanilla'
    window_size: Optional[str] = None
    window_attn_skip_freq: Optional[str] = None
    max_position_embeddings: Optional[int] = None

    position_embedding_type: Optional[Literal['learned_absolute', 'rope', 'mrope', 'relative', 'none']] = None
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
    moe_router_load_balancing_type: Literal['aux_loss', 'seq_aux_loss', 'global_aux_loss', 'sinkhorn', 'none'] = 'aux_loss'
    use_shared_expert_gate: bool = False

    # mla
    multi_latent_attention: bool = False
    q_lora_rank: Optional[int] = None
    kv_lora_rank: int = 32
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int= 64
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
        if self.num_query_groups is not None and self.num_query_groups > 1:
            self.group_query_attention = True
        self._init_moe()

    def _init_moe(self):
        if self.moe_router_dtype.lower() == 'none':
            self.moe_router_dtype = None
        if self.moe_shared_expert_intermediate_size == 0:
            self.moe_shared_expert_intermediate_size = None
        if self.num_experts is not None:
            if self.moe_ffn_hidden_size is None:
                self.moe_ffn_hidden_size = self.ffn_hidden_size
            # TODO: remove
            if is_torch_npu_available() and self.num_experts > MAX_NPU_EXPERTS_PER_EP:
                required_ep = (self.num_experts + MAX_NPU_EXPERTS_PER_EP - 1) // MAX_NPU_EXPERTS_PER_EP
                if self.expert_model_parallel_size < required_ep:
                    logger.warning(f'{">"*20} WARNING {"<"*20}\n'
                                   f'MindSpeed on NPU supports up to {MAX_NPU_EXPERTS_PER_EP} experts per EP group. '
                                   f'num_experts={self.num_experts}, '
                                   f'expert_model_parallel_size={self.expert_model_parallel_size}. '
                                   f'Please set expert_model_parallel_size (EP) to {required_ep} '
                                   f'(num_experts / {MAX_NPU_EXPERTS_PER_EP}) or higher.')
