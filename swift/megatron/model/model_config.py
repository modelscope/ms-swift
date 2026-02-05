from dataclasses import dataclass, fields
from typing import List, Literal, Optional, Union

import torch
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.transformer import TransformerConfig
from transformers.utils import is_torch_npu_available

from swift.megatron.utils import convert_hf_config
from swift.utils import get_logger, json_parse_to_dict

logger = get_logger()


@dataclass
class MegatronModelConfig(TransformerConfig):
    """
    During Megatron training, multiple models may be created. This class is used to
    distinguish the configurations of different models.
    """
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

    position_embedding_type: Literal['learned_absolute', 'rope', 'mrope', 'none'] = 'rope'
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
    num_moe_experts: Optional[int] = None
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

    layernorm_zero_centered_gamma: bool = False

    # Override
    persist_layer_norm: bool = True
    deallocate_pipeline_outputs: bool = True
    batch_p2p_comm: bool = True
    cp_comm_type: str = 'p2p'

    def __post_init__(self):
        if self.moe_router_dtype.lower() == 'none':
            self.moe_router_dtype = None
        if self.num_moe_experts is not None:
            if self.moe_ffn_hidden_size is None:
                self.moe_ffn_hidden_size = self.ffn_hidden_size
        if self.rope_scaling is not None:
            self.rope_scaling = json_parse_to_dict(self.rope_scaling)
            if 'type' in self.rope_scaling and 'rope_type' not in self.rope_scaling:
                self.rope_scaling['rope_type'] = self.rope_scaling['type']

        if self.swiglu:
            self.activation_func = F.silu
            self.gated_linear_unit = True
        if self.quick_geglu:
            assert not self.swiglu
            self.gated_linear_unit = True
            self.activation_func = quick_gelu
        super().__post_init__()
        self._check_npu()
        self.variable_seq_lengths = True

    def _check_npu(self):
        MAX_NPU_EXPERTS_PER_EP = 128
        num_experts = self.num_moe_experts
        expert_model_parallel_size = mpu.get_expert_model_parallel_world_size()
        if is_torch_npu_available() and num_experts > MAX_NPU_EXPERTS_PER_EP:
            required_ep = (num_experts + MAX_NPU_EXPERTS_PER_EP - 1) // MAX_NPU_EXPERTS_PER_EP
            if expert_model_parallel_size < required_ep:
                logger.warning(f'{">" * 20} WARNING {"<" * 20}\n'
                               f'MindSpeed on NPU supports up to {MAX_NPU_EXPERTS_PER_EP} experts per EP group. '
                               f'num_experts={num_experts}, '
                               f'expert_model_parallel_size={expert_model_parallel_size}. '
                               f'Please set expert_model_parallel_size (EP) to {required_ep} '
                               f'(num_experts / {MAX_NPU_EXPERTS_PER_EP}) or higher.')


def create_mcore_model_config(args, hf_config):
    # Translate args to core transformer configuration
    kwargs = convert_hf_config(hf_config)
    for f in fields(MegatronModelConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)

    if args.task_type == 'seq_cls':
        args.problem_type = args.problem_type or getattr(hf_config, 'problem_type', None)
        logger.info(f'args.problem_type: {args.problem_type}')

    kwargs['pipeline_dtype'] = args.torch_dtype
    kwargs['num_layers_in_first_pipeline_stage'] = args.decoder_first_pipeline_num_layers
    kwargs['num_layers_in_last_pipeline_stage'] = args.decoder_last_pipeline_num_layers
    kwargs['fp8_param'] = args.fp8_param_gather
    kwargs['inference_sampling_seed'] = args.seed
    swiglu = kwargs.get('swiglu', True)
    kwargs['bias_activation_fusion'] = args.bias_swiglu_fusion if swiglu else args.bias_gelu_fusion
    config = MegatronModelConfig(**kwargs)
    config.hf_config = hf_config
    config.args = args
    return config
