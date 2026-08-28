# Copyright (c) ModelScope Contributors. All rights reserved.
"""Distributed-setting and hub-endpoint helpers, copied from ``swift.utils.env``."""
import os
from typing import Tuple

from transformers.utils import strtobool


def use_hf_hub():
    return strtobool(os.environ.get('USE_HF', '0'))


def get_hf_endpoint():
    hf_endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co/')
    if hf_endpoint.endswith('/'):
        hf_endpoint = hf_endpoint[:-1]
    return hf_endpoint


def get_dist_setting() -> Tuple[int, int, int, int]:
    """return rank, local_rank, world_size, local_world_size"""
    rank = int(os.getenv('RANK', -1))
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE') or os.getenv('_PATCH_WORLD_SIZE') or 1)
    # compat deepspeed launch
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', None) or os.getenv('LOCAL_SIZE', 1))
    return rank, local_rank, world_size, local_world_size


def is_master():
    rank = get_dist_setting()[0]
    return rank in {-1, 0}


def is_dist():
    """Determine if the training is distributed"""
    rank, local_rank, _, _ = get_dist_setting()
    return rank >= 0 and local_rank >= 0
