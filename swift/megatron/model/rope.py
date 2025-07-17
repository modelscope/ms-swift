from typing import Any, Dict, Optional

import torch
from megatron.training import get_args
from transformers import PretrainedConfig

from swift.utils import get_logger

logger = get_logger()


class DummyConfig:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _get_dummy_config(args):
    dummy_config = DummyConfig(
        rope_scaling=args.rope_scaling,
        rope_theta=args.rotary_base,
        max_position_embeddings=args.max_position_embeddings,
        head_dim=args.qk_pos_emb_head_dim if args.multi_latent_attention else args.kv_channels,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
    )
    original_max_position_embeddings = args.original_max_position_embeddings or args.rope_scaling.get(
        'original_max_position_embeddings')
    if original_max_position_embeddings is not None:
        dummy_config.original_max_position_embeddings = original_max_position_embeddings
    if args.partial_rotary_factor is not None:
        dummy_config.partial_rotary_factor = args.partial_rotary_factor
    return dummy_config


EXTENDED_ROPE_INIT_FUNCTIONS = {}


def _get_rope_type(rope_scaling: Dict[str, Any]):
    rope_type = rope_scaling['rope_type']
    if rope_type == 'dynamic' and rope_scaling.get('alpha') is not None:
        rope_type = 'dynamic_alpha'
    return rope_type


def get_rope_inv_freq(seq_len=None):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    args = get_args()
    ROPE_INIT_FUNCTIONS.update(EXTENDED_ROPE_INIT_FUNCTIONS)
    dummy_config = _get_dummy_config(args)
    rope_init_fn = ROPE_INIT_FUNCTIONS[_get_rope_type(args.rope_scaling)]
    inv_freq, attention_scaling = rope_init_fn(dummy_config, 'cpu', seq_len=seq_len)
    if attention_scaling is None:
        attention_scaling = 1.
    return inv_freq, attention_scaling


# borrowed from huggingface/transformers
def longrope_frequency_update(args, model, inv_freq, seq_len: int):
    if args.original_max_position_embeddings is not None:
        original_max_position_embeddings = args.original_max_position_embeddings
    else:
        original_max_position_embeddings = args.max_position_embeddings

    if not hasattr(model, 'long_inv_freq'):
        model.long_inv_freq, _ = get_rope_inv_freq(seq_len=original_max_position_embeddings + 1)
        model.original_inv_freq = inv_freq.clone()

    if seq_len > original_max_position_embeddings:
        inv_freq.data.copy_(model.long_inv_freq)
    else:
        inv_freq.data.copy_(model.original_inv_freq)


# borrowed from huggingface/transformers
def dynamic_frequency_update(args, model, inv_freq, seq_len: int):
    if not hasattr(model, 'max_seq_len_cached'):
        model.max_seq_len_cached = args.max_position_embeddings
        model.original_max_seq_len = args.max_position_embeddings
        model.original_inv_freq = inv_freq.clone()
    attention_scaling = None
    if seq_len > model.max_seq_len_cached:  # growth
        new_inv_freq, attention_scaling = get_rope_inv_freq(seq_len=seq_len)
        inv_freq.data.copy_(new_inv_freq)
        model.max_seq_len_cached = seq_len

    if seq_len < model.original_max_seq_len and model.max_seq_len_cached > model.original_max_seq_len:  # reset
        inv_freq.data.copy_(model.original_inv_freq)
        model.max_seq_len_cached = model.original_max_seq_len
    return attention_scaling


def dynamic_rope_update(model, inv_freq, seq_len: int):
    args = get_args()
    rope_type = _get_rope_type(args.rope_scaling)
    attention_scaling = None
    if rope_type == 'dynamic':
        attention_scaling = dynamic_frequency_update(args, model, inv_freq, seq_len)
    elif rope_type == 'longrope':
        attention_scaling = longrope_frequency_update(args, model, inv_freq, seq_len)
    return attention_scaling


def _compute_dynamic_alpha_ntk_parameters(
    config: Optional[PretrainedConfig] = None,
    device: Optional['torch.device'] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> tuple['torch.Tensor', float]:
    # Code borrowed from Tencent-Hunyuan/Hunyuan-A13B-Instruct
    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, 'partial_rotary_factor') else 1.0
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    alpha = config.rope_scaling['alpha']

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    base = base * alpha**(dim / (dim - 2))
    inv_freq = 1.0 / (base**(torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, attention_factor


EXTENDED_ROPE_INIT_FUNCTIONS['dynamic_alpha'] = _compute_dynamic_alpha_ntk_parameters
