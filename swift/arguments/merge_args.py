# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass

from swift.utils import get_logger

logger = get_logger()


@dataclass
class MergeArguments:
    """A dataclass that holds configuration for merging models.

    This dataclass stores all the arguments needed to configure the model merging process.

    Args:
        merge_lora (bool): Whether to merge LoRA adapters. This parameter supports `lora`, `llamapro`, and `longlora`.
            Defaults to False.
        safe_serialization (bool): Whether to use safetensors for serialization. Defaults to True.
        max_shard_size (str): The maximum size of a single saved shard file. Defaults to '5GB'.
    """
    merge_lora: bool = False
    safe_serialization: bool = True
    max_shard_size: str = '5GB'
