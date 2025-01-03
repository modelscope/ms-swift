# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Optional

from swift.utils import get_logger, is_merge_kit_available

logger = get_logger()


@dataclass
class MergeArguments:
    """
    MergeArguments is a dataclass that holds configuration for merging models.

    Args:
        merge_lora (bool): Flag to indicate if LoRA merging is enabled. Default is False.
        safe_serialization(bool): Use safetensors or not, default `True`.
        max_shard_size(str): The max size of single shard file.
        instruct_model (Optional[str]): Path or ID of the instruct model. Use when `use_merge_kit` is True.
        instruct_model_revision (Optional[str]): Revision of the instruct model. Use when `use_merge_kit` is True.
    """
    merge_lora: bool = False
    safe_serialization: bool = True
    max_shard_size: str = '5GB'

    instruct_model: Optional[str] = None
    instruct_model_revision: Optional[str] = None
