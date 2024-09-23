# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from dataclasses import field, dataclass
from typing import List, Literal, Optional, Union

from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset

from swift.llm.dataset.loader import DATASET_MAPPING
from swift.llm.dataset.register import register_dataset_info_file
from swift.llm.model.model import get_default_template_type
from swift.llm.template import TEMPLATE_MAPPING
from swift.utils import get_logger

logger = get_logger()
DATASET_TYPE = Union[HfDataset, HfIterableDataset]


@dataclass
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

    def handle_custom_register(self) -> None:
        if self.custom_register_path is None:
            return
        folder, fname = os.path.split(self.custom_register_path)
        sys.path.append(folder)
        __import__(fname.rstrip('.py'))

    def handle_custom_dataset_info(self):
        if self.custom_dataset_info is None:
            return
        register_dataset_info_file(self.custom_dataset_info)

    def __post_init__(self: Union['SftArguments', 'InferArguments']):
        if self.max_length == -1:
            self.max_length = None
        if self.dataset_seed is None:
            self.dataset_seed = self.seed
        if len(self.val_dataset) > 0:
            self.dataset_test_ratio = 0.0
            logger.info('Using val_dataset, ignoring dataset_test_ratio')
        self.handle_custom_dataset_info()
        self.handle_custom_register()


@dataclass
class TemplateArguments:
    template_type: str = field(
        default='AUTO', metadata={'help': f"template_type choices: {list(TEMPLATE_MAPPING.keys()) + ['AUTO']}"})
    system: Optional[str] = None
    tools_prompt: str = 'react_en'
    # multi-modal
    rescale_image: int = -1

    def select_template(self):
        if self.template_type == 'AUTO':
            self.template_type = get_default_template_type(self.model_type)
            logger.info(f'Setting template_type: {self.template_type}')

    def __post_init__(self: Union['SftArguments', 'InferArguments']):
        self.select_template()
