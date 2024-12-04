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
        use_merge_kit (bool): Flag to indicate merge with `mergekit`. Default is False.
        instruct_model (Optional[str]): Path or ID of the instruct model. Use when `use_merge_kit` is True.
        instruct_model_revision (Optional[str]): Revision of the instruct model. Use when `use_merge_kit` is True.
    """
    merge_lora: bool = False
    safe_serialization: bool = True
    max_shard_size: str = '5GB'

    use_merge_kit: bool = False
    instruct_model: Optional[str] = None
    instruct_model_revision: Optional[str] = None

    def __post_init__(self):
        if self.use_merge_kit:
            assert is_merge_kit_available(), ('please install mergekit by pip install '
                                              'git+https://github.com/arcee-ai/mergekit.git')
            logger.info('Important: You are using mergekit, please remember '
                        'the LoRA should be trained against the base model,'
                        'and pass its instruct model by --instruct_model xxx when merging')
            assert self.instruct_model, 'Please pass in the instruct model'

            self.merge_yaml = """
models:
  - model: {merged_model}
    parameters:
      weight: 1
      density: 1
  - model: {instruct_model}
    parameters:
      weight: 1
      density: 1
merge_method: ties
base_model: {base_model}
parameters:
  weight: 1
  density: 1
  normalize: true
  int8_mask: true
tokenizer_source: {merged_model}
dtype: bfloat16
"""
