import math
from collections import namedtuple
from typing import Any, Dict

import torch
from megatron.training import get_args

from swift.utils import get_logger

logger = get_logger()


def get_rope_inv_freq(device):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    args = get_args()
    DummyConfig = namedtuple(
        'DummyConfig',
        ['rope_scaling', 'rope_theta', 'max_position_embeddings', 'head_dim', 'hidden_size', 'num_attention_heads'])
    dummy_config = DummyConfig(
        rope_scaling=args.rope_scaling,
        rope_theta=args.rotary_base,
        max_position_embeddings=args.max_position_embeddings,
        head_dim=args.kv_channels,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
    )
    rope_init_fn = ROPE_INIT_FUNCTIONS[args.rope_scaling['rope_type']]
    inv_freq, attention_scaling = rope_init_fn(dummy_config, device)
    if attention_scaling is None:
        attention_scaling = 1.
    if attention_scaling != 1 and args.apply_rope_fusion:
        args.apply_rope_fusion = False
        logger.warning('`apply_rope_fusion` does not support `attention_scaling`. '
                       f'Setting `args.apply_rope_fusion`: {args.apply_rope_fusion}')
    return inv_freq, attention_scaling
