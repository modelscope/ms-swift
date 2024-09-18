# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import field
from typing import List, Literal, Optional, Union

from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset

from swift.llm.dataset.loader import DATASET_MAPPING
from swift.utils import (get_logger)

logger = get_logger()
DATASET_TYPE = Union[HfDataset, HfIterableDataset]


class DataArguments:

    # dataset_id or dataset_name or dataset_path or ...
    dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    val_dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    dataset_seed: Optional[int] = None
    max_length: int = 2048  # -1: no limit

    dataset_test_ratio: float = 0.01

    truncation_strategy: Literal['delete', 'truncation_left'] = 'delete'
    check_dataset_strategy: Literal['none', 'discard', 'error', 'warning'] = 'none'

    custom_register_path: Optional[str] = None  # .py
    custom_dataset_info: Optional[str] = None  # .json

    # Chinese name and English name
    model_name: List[str] = field(default_factory=lambda: [None, None], metadata={'help': "e.g. ['小黄', 'Xiao Huang']"})
    model_author: List[str] = field(
        default_factory=lambda: [None, None], metadata={'help': "e.g. ['魔搭', 'ModelScope']"})


class TemplateArguments:
    template_type: str = field(
        default='AUTO', metadata={'help': f"template_type choices: {list(TEMPLATE_MAPPING.keys()) + ['AUTO']}"})
    system: Optional[str] = None
    tools_prompt: Literal['react_en', 'react_zh', 'toolbench'] = 'react_en'
    # multi-modal
    rescale_image: int = -1