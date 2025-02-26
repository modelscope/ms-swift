# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass

from swift.utils import get_logger

logger = get_logger()


@dataclass
class MergeArguments:
    """
    MergeArguments is a dataclass that holds configuration for merging models.

    Args:
        merge_lora (bool): Flag to indicate if LoRA merging is enabled. Default is False.
        safe_serialization(bool): Use safetensors or not, default `True`.
        max_shard_size(str): The max size of single shard file.
    """
    merge_lora: bool = False
    safe_serialization: bool = True
    max_shard_size: str = '5GB'
