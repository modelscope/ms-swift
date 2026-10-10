# Copyright (c) ModelScope Contributors. All rights reserved.
"""Central place to design and read swift/dev's environment variables.

Every env var dev reads is exposed here as a named accessor, so the full set is visible in one file.
This module only *reads* flags; the business logic that acts on a flag stays with its caller (e.g. the
multi-answer picking that consumes ``random_dataset_response()`` lives in the response converter).
"""
import os
from typing import Tuple


def _strtobool(val: str) -> int:
    """Parse a boolean env value: ``y/yes/t/true/on/1`` -> 1, ``n/no/f/false/off/0`` -> 0, else ValueError.

    Inlined rather than imported from ``transformers.utils`` so reading an env flag never pulls in
    transformers. Kept private: it is the parsing primitive for the readers below, not a general util.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    if val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    raise ValueError(f'invalid truth value {val!r}')


def use_hf_hub():
    """``USE_HF``: resolve models/datasets from HuggingFace Hub instead of ModelScope."""
    return _strtobool(os.environ.get('USE_HF', '0'))


def random_dataset_response() -> bool:
    """``RANDOM_DATASET_RESPONSE``: pick a random answer from a multi-answer response column."""
    return bool(_strtobool(os.environ.get('RANDOM_DATASET_RESPONSE', 'False')))


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
