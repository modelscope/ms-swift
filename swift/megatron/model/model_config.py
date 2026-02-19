# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Literal, Optional, Union

import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.transformer import TransformerConfig
from transformers.utils import is_torch_npu_available

from swift.utils import get_logger, json_parse_to_dict

logger = get_logger()


# code borrowed from NVIDIA/Megatron-LM
def _eval_pattern(pattern):
    """ Validate and evaluate a string containing a Python list expression """
    assert isinstance(pattern, str)

    # validate input, only allow comma, digits, [, ], (, ), +, and *
    if bool(re.compile(r'[^,\d\[\]\(\)\+\*]').search(pattern)):
        raise ValueError(f'Invalid pattern: {pattern}')

    return eval(pattern)


# code borrowed from NVIDIA/Megatron-LM
def no_rope_freq_type(x):
    """ Controls which layers to skip performing Rotary Position Embedding.
    - An integer N: Represents a 1:N ratio, meaning RoPE is skipped every N-1 layers.
    - A string "N": Same as above, but provided as a string
    - A string containing a Python list expression that defines a custom pattern, e.g.:
      "([0]*3+[1]*1)*3" evaluates to [0,0,0,1,0,0,0,1,0,0,0,1]
      where 1 indicates rope is skipped on the layer.
      This allows defining arbitrary patterns of rope skipping.
      The pattern length must match the total number of transformer layers.
      Examples:
          "([1]+[0]*23)": Only first layer has rope skipped for a 24-layer network.
          "([0]*3+[1]*1)*2": Every 4 layers the rope is skipped on the last layer. Repeat twice.
    """
    if x is None or isinstance(x, int):
        return x
    assert isinstance(x, str)
    if '[' in x:
        # it's a custom pattern
        return _eval_pattern(x)
    else:
        # it's a single int but in str
        return int(x)


# code borrowed from NVIDIA/Megatron-LM
def moe_freq_type(x):
    """Frequency between MoE layers and Dense layers.

    Accepts either:
    - An integer N: Represents a 1:N ratio, meaning one expert layer for every N-1 dense layers
    - A string "N": Same as above, but provided as a string
    - A string containing a Python list expression that defines a custom pattern, e.g.:
      "([1]*3+[0]*1)*3" evaluates to [1,1,1,0,1,1,1,0,1,1,1,0]
      where 1 indicates an expert layer and 0 indicates a dense layer.
      This allows defining arbitrary patterns of expert and dense layers.
      The pattern length must match the total number of transformer layers.
      Examples:
          "([0]+[1]*23)": 1 dense layer followed by 23 experts layers
          "([1]*3+[0]*2)*2": Three expert layers followed by two dense layers, repeated twice.
    """
    if isinstance(x, int):
        return x
    assert isinstance(x, str)
    if '[' in x:
        # it's a custom pattern
        return _eval_pattern(x)
    else:
        # it's a single int but in str
        return int(x)


# code borrowed from NVIDIA/Megatron-LM
def tuple_type(x):
    """
    Convert a string to a tuple of integers.
    Examples:
        "1,2,3" -> (1, 2, 3)
        "(1,2,3)" -> (1, 2, 3)
    """
    if x is None or isinstance(x, tuple):
        return x
    assert isinstance(x, str)
    return tuple(int(i) for i in x.strip('()').split(','))


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
    moe_router_score_function: Literal['sigmoid', 'softmax'] = 'softmax'
    moe_router_bias_update_rate: float = 1e-3
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
    cp_comm_type: str = 'p2p'

    def __post_init__(self):
        self._format_config()
        if self.moe_router_dtype.lower() == 'none':
            self.moe_router_dtype = None
        if self.moe_shared_expert_intermediate_size == 0:
            self.moe_shared_expert_intermediate_size = None
        if self.num_moe_experts is not None:
            if self.moe_ffn_hidden_size is None:
                self.moe_ffn_hidden_size = self.ffn_hidden_size
        if self.rope_scaling is not None:
            self.rope_scaling = json_parse_to_dict(self.rope_scaling)
            if 'type' in self.rope_scaling and 'rope_type' not in self.rope_scaling:
                self.rope_scaling['rope_type'] = self.rope_scaling['type']

        if self.add_bias_linear:
            self.add_qkv_bias = True
        if self.swiglu:
            self.activation_func = F.silu
            self.gated_linear_unit = True
        if self.quick_geglu:
            # megatron-core>=0.14.0
            try:
                from megatron.core.fusions.fused_bias_geglu import quick_gelu
            except ImportError:
                from megatron.core.activations import quick_gelu
            assert not self.swiglu
            self.gated_linear_unit = True
            self.activation_func = quick_gelu
        super().__post_init__()
        self._check_npu()
        self.variable_seq_lengths = True

    def _format_config(self):
        if self.window_size is not None:
            self.window_size = tuple_type(self.window_size)
        if self.window_attn_skip_freq is not None:
            self.window_attn_skip_freq = moe_freq_type(self.window_attn_skip_freq)
        if self.no_rope_freq is not None:
            self.no_rope_freq = no_rope_freq_type(self.no_rope_freq)
        if self.moe_layer_freq is not None:
            self.moe_layer_freq = moe_freq_type(self.moe_layer_freq)

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


config_mapping = {
    'num_layers': ['num_hidden_layers'],
    'hidden_size': ['hidden_size'],
    'mlp_ffn_hidden_size': ['intermediate_size_mlp'],
    'ffn_hidden_size': ['intermediate_size'],
    'num_attention_heads': ['num_attention_heads'],
    'num_query_groups': ['num_key_value_heads'],
    'max_position_embeddings': ['max_position_embeddings'],
    'layernorm_epsilon': ['rms_norm_eps'],
    'rotary_base': ['rope_theta'],
    'padded_vocab_size': ['vocab_size'],
    'attention_dropout': ['attention_dropout'],
    'untie_embeddings_and_output_weights': ['tie_word_embeddings'],
    'swiglu': ['hidden_act'],
    'add_qkv_bias': ['attention_bias', 'qkv_bias', 'use_bias'],
    'add_bias_linear': ['mlp_bias'],
    'kv_channels': ['head_dim'],
    'hf_model_type': ['model_type'],
    # moe
    'moe_ffn_hidden_size': ['moe_intermediate_size'],
    'moe_shared_expert_intermediate_size': ['shared_expert_intermediate_size'],
    'moe_router_topk': ['num_experts_per_tok', 'moe_topk', 'moe_k'],
    'moe_router_num_groups': ['n_group'],
    'moe_router_group_topk': ['topk_group'],
    'num_moe_experts': ['num_experts', 'n_routed_experts', 'moe_num_experts', 'num_local_experts'],
    'moe_router_pre_softmax': ['norm_topk_prob'],
    # deepseek
    'q_lora_rank': ['q_lora_rank'],
    'kv_lora_rank': ['kv_lora_rank'],
    'moe_router_score_function': ['scoring_func'],
    'moe_router_bias_update_rate': ['aux_loss_alpha'],
    'qk_head_dim': ['qk_nope_head_dim'],
    'qk_pos_emb_head_dim': ['qk_rope_head_dim'],
    'v_head_dim': ['v_head_dim'],
    'moe_router_topk_scaling_factor': ['routed_scaling_factor'],
    'qk_layernorm': ['use_qk_norm'],
    # qwen3_next
    'linear_num_value_heads': ['linear_num_value_heads'],
    'linear_num_key_heads': ['linear_num_key_heads'],
    'linear_key_head_dim': ['linear_key_head_dim'],
    'linear_value_head_dim': ['linear_value_head_dim'],
    'linear_conv_kernel_dim': ['linear_conv_kernel_dim'],
    'full_attention_interval': ['full_attention_interval'],
    # other
    'original_max_position_embeddings': ['original_max_position_embeddings'],
    'partial_rotary_factor': ['partial_rotary_factor'],
    'first_k_dense_replace': ['first_k_dense_replace', 'moe_layer_start_index'],
    'n_shared_experts': ['n_shared_experts', 'num_shared_expert', 'moe_num_shared_experts'],
    'window_size': ['sliding_window'],
    'layer_types': ['layer_types'],
    'interleave_moe_layer_step': ['interleave_moe_layer_step'],
}


def _convert_config(config, _internal_call=False) -> Dict[str, Any]:
    megatron_config = {}
    for k, hf_keys in config_mapping.items():
        for hf_k in hf_keys:
            if hasattr(config, hf_k):
                hf_v = getattr(config, hf_k)
                if hf_v is None:
                    continue
                if k == 'rotary_base':
                    megatron_config[k] = int(hf_v)
                elif k in {'untie_embeddings_and_output_weights', 'moe_router_pre_softmax'}:
                    megatron_config[k] = not hf_v
                elif k == 'swiglu':
                    if hf_v == 'silu':
                        megatron_config[k] = True
                else:
                    if k == 'kv_lora_rank':
                        megatron_config['multi_latent_attention'] = True
                    elif k == 'hf_model_type':
                        if _internal_call:
                            k = 'llm_model_type'
                    megatron_config[k] = hf_v
                break
    for key in ['text_config', 'llm_config', 'thinker_config']:
        if hasattr(config, key):
            megatron_config.update(_convert_config(getattr(config, key), _internal_call=True))
    # compat llama3
    if getattr(config, 'rope_scaling', None) is not None:
        if isinstance(config.rope_scaling, int):
            megatron_config['rope_scaling'] = {'factor': config.rope_scaling, 'type': 'linear'},
        elif isinstance(config.rope_scaling, dict):
            megatron_config['rope_scaling'] = config.rope_scaling
    return megatron_config


def convert_hf_config(config) -> Dict[str, Any]:
    res = _convert_config(config)
    hf_model_type = res.get('hf_model_type')
    llm_model_type = res.get('llm_model_type') or hf_model_type
    res['llm_model_type'] = llm_model_type

    first_k_dense_replace = res.pop('first_k_dense_replace', None)
    n_shared_experts = res.pop('n_shared_experts', None)
    layer_types = res.pop('layer_types', None)
    mlp_ffn_hidden_size = res.pop('mlp_ffn_hidden_size', None)
    interleave_moe_layer_step = res.pop('interleave_moe_layer_step', None)
    window_size = res.pop('window_size', None)
    rope_scaling = res.get('rope_scaling') or {}
    if llm_model_type in {'qwen3', 'qwen3_moe', 'qwen3_next'} or hf_model_type in {
            'qwen3_omni_moe', 'qwen3_omni', 'qwen3_vl', 'qwen3_vl_moe', 'qwen3_5', 'qwen3_5_moe'
    }:
        res['qk_layernorm'] = True
    if llm_model_type in {'qwen2_moe', 'qwen3_moe', 'qwen3_next'
                          } or hf_model_type in {'qwen3_omni_moe', 'qwen3_vl_moe', 'qwen3_5_moe'}:
        res.pop('ffn_hidden_size', None)
        if llm_model_type in {'qwen2_moe', 'qwen3_next'} or hf_model_type == 'qwen3_5_moe':
            res['use_shared_expert_gate'] = True
    if llm_model_type in {
            'deepseek',
            'deepseek_v2',
            'deepseek_v3',
            'dots1',
    } or hf_model_type == 'kimi_vl':
        if llm_model_type != 'deepseek':
            res['qk_layernorm'] = True
        res['moe_router_load_balancing_type'] = 'seq_aux_loss'
        res.pop('num_query_groups', None)  # https://github.com/NVIDIA/Megatron-LM/issues/1475
        if llm_model_type == 'dots1':
            res['moe_router_score_function'] = 'sigmoid'
    elif llm_model_type == 'hunyuan':
        # Since HunYuan’s attention applies RoPE before using q/k_layernorm,
        # which is incompatible with megatron-core, support is not provided here.
        res['n_shared_experts'] = n_shared_experts
        for key in ['moe_ffn_hidden_size', 'n_shared_experts', 'moe_router_topk']:
            val = res.get(key)
            if isinstance(val, list) and val and min(val) == max(val):
                res[key] = val[0]
        n_shared_experts = res.pop('n_shared_experts')
    elif llm_model_type in {'ernie4_5', 'ernie4_5_moe', 'glm4'}:
        res['rotary_interleaved'] = True
    elif llm_model_type == 'gpt_oss':
        res['add_bias_linear'] = True
        res['bias_dropout_fusion'] = False
        res['softmax_type'] = 'learnable'
        res['swiglu'] = False
        res['quick_geglu'] = True
        res['activation_func_clamp_value'] = 7
        res['glu_linear_offset'] = 1
        res['window_size'] = f'{window_size},0'
        if layer_types is None:
            res['window_attn_skip_freq'] = '2'
        else:
            window_attn_skip_freq = ','.join(['1' if lt == 'sliding_attention' else '0' for lt in layer_types])
            res['window_attn_skip_freq'] = f'[{window_attn_skip_freq}]'
    elif llm_model_type in {'glm4_moe', 'glm4_moe_lite'} or hf_model_type == 'glm4v_moe':
        res['moe_router_score_function'] = 'sigmoid'
        if llm_model_type == 'glm4_moe_lite':
            res['qk_layernorm'] = True
            res.pop('num_query_groups', None)
    elif llm_model_type == 'qwen3_next' or hf_model_type in {'qwen3_5', 'qwen3_5_moe'}:
        full_attention_interval = res.pop('full_attention_interval', 4)
        num_layers = res['num_layers']
        res['layer_types'] = [
            'full_attention' if (i + 1) % full_attention_interval == 0 else 'linear_attention'
            for i in range(num_layers)
        ]
    elif llm_model_type == 'minimax_m2':
        res['add_qkv_bias'] = False
    elif hf_model_type == 'llama4':
        qk_layernorm = res.pop('qk_layernorm', False)
        if qk_layernorm:
            res['qk_l2_norm'] = True
        res['no_rope_freq'] = 4
        res['moe_apply_probs_on_input'] = True
        res['rotary_interleaved'] = True
        res['moe_router_score_function'] = 'sigmoid'
        res['moe_ffn_hidden_size'] = res['ffn_hidden_size']
        res['ffn_hidden_size'] = mlp_ffn_hidden_size
        res['moe_router_enable_expert_bias'] = False
        res['moe_shared_expert_intermediate_size'] = res['moe_ffn_hidden_size']
        if interleave_moe_layer_step > 1:
            moe_layer_freq = [
                '1' if i % interleave_moe_layer_step == (interleave_moe_layer_step - 1) else '0'
                for i in range(res['num_layers'])
            ]
            res['moe_layer_freq'] = f"[{','.join(moe_layer_freq)}]"
    elif hf_model_type == 'glm4v':
        res['rotary_interleaved'] = True
    if 'partial_rotary_factor' not in res and 'partial_rotary_factor' in rope_scaling:
        res['partial_rotary_factor'] = rope_scaling['partial_rotary_factor']
    if 'rotary_base' not in res and 'rope_theta' in rope_scaling:
        res['rotary_base'] = rope_scaling['rope_theta']
    if rope_scaling.get('mrope_section') is not None:
        res['position_embedding_type'] = 'mrope'
        res['mrope_section'] = rope_scaling['mrope_section']
        mrope_interleaved = rope_scaling.get('mrope_interleaved', False) or rope_scaling.get('interleaved', False)
        res['mrope_interleaved'] = mrope_interleaved

    if first_k_dense_replace is not None:
        res['moe_layer_freq'] = f'[0]*{first_k_dense_replace}+[1]*{res["num_layers"] - first_k_dense_replace}'
    if res.get('moe_router_score_function', 'softmax') == 'sigmoid' and 'moe_router_enable_expert_bias' not in res:
        res['moe_router_enable_expert_bias'] = True
    if n_shared_experts is not None and 'moe_shared_expert_intermediate_size' not in res:
        res['moe_shared_expert_intermediate_size'] = n_shared_experts * res['moe_ffn_hidden_size']
    return res


def get_mcore_model_config(args, hf_config):
    kwargs = convert_hf_config(hf_config)
    for f in fields(MegatronModelConfig):
        key, value = f.name, getattr(args, f.name, None)
        if value is None or isinstance(value, (list, tuple)) and len(value) == 0:
            continue
        kwargs[key] = value

    if args.task_type == 'seq_cls':
        args.problem_type = args.problem_type or getattr(hf_config, 'problem_type', None)
        logger.info(f'args.problem_type: {args.problem_type}')

    kwargs['pipeline_dtype'] = args.torch_dtype
    kwargs['num_layers_in_first_pipeline_stage'] = args.decoder_first_pipeline_num_layers
    kwargs['num_layers_in_last_pipeline_stage'] = args.decoder_last_pipeline_num_layers
    kwargs['fp8_param'] = args.fp8_param_gather
    kwargs['batch_p2p_comm'] = not args.overlap_p2p_comm
    swiglu = kwargs.get('swiglu', True)
    add_bias_linear = kwargs.get('add_bias_linear', False)
    position_embedding_type = kwargs.get('position_embedding_type', 'rope')
    num_moe_experts = kwargs.get('num_moe_experts', None)
    if position_embedding_type != 'rope':
        kwargs['apply_rope_fusion'] = False
    if not swiglu and not add_bias_linear:
        kwargs['bias_activation_fusion'] = False
    if add_bias_linear and num_moe_experts and args.moe_grouped_gemm:
        kwargs['bias_dropout_fusion'] = False
    if num_moe_experts is None:
        kwargs['expert_model_parallel_size'] = 1
        kwargs['expert_tensor_parallel_size'] = 1
    config = MegatronModelConfig(**kwargs)
    config.hf_config = hf_config
    config.args = args
    return config
