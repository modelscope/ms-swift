# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from swift.llm import DATASET_MAPPING, TEMPLATE_MAPPING, register_dataset_info_file
from swift.utils import get_logger

logger = get_logger()


@dataclass
class DataArguments:

    # dataset_id or dataset_name or dataset_path or ...
    dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    val_dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    val_dataset_ratio: float = 0.01  # If val_dataset is empty, use a split from the dataset as the validation set.
    dataset_seed: Optional[int] = None
    max_length: Optional[int] = None

    truncation_strategy: Literal['delete', 'truncation_left'] = 'truncation_left'
    check_dataset_strategy: Literal['none', 'discard', 'error', 'warning'] = 'none'

    custom_register_path: Optional[str] = None  # .py
    custom_dataset_info: Optional[str] = None  # .json

    # Chinese name and English name
    model_name: List[str] = field(default_factory=lambda: [None, None], metadata={'help': "e.g. ['小黄', 'Xiao Huang']"})
    model_author: List[str] = field(
        default_factory=lambda: [None, None], metadata={'help': "e.g. ['魔搭', 'ModelScope']"})

    def handle_custom_register(self) -> None:
        """Register custom .py file to datasets"""
        if self.custom_register_path is None:
            return
        folder, fname = os.path.split(self.custom_register_path)
        sys.path.append(folder)
        __import__(fname.rstrip('.py'))

    def handle_custom_dataset_info(self):
        """register custom dataset_info.json to datasets"""
        if self.custom_dataset_info is None:
            return
        register_dataset_info_file(self.custom_dataset_info)

    def __post_init__(self: Union['SftArguments', 'InferArguments']):
        if self.dataset_seed is None:
            self.dataset_seed = self.seed
        if len(self.val_dataset) > 0:
            self.val_dataset_ratio = 0.0
            logger.info('Using val_dataset, ignoring val_dataset_ratio')
        self.handle_custom_dataset_info()
        self.handle_custom_register()


@dataclass
class TemplateArguments:
    template_type: str = field(
        default='auto', metadata={'help': f"template_type choices: {list(TEMPLATE_MAPPING.keys()) + ['AUTO']}"})
    system: Optional[str] = None
    tools_prompt: str = 'react_en'
    # multi-modal
    rescale_image: int = -1

    def select_template(self: Union['SftArguments', 'InferArguments']):
        # [TODO:]
        """If setting template to `auto`, find a proper one"""
        if self.template_type == 'auto':
            from swift.llm.model.register import ModelInfoReader
            self.template_type = ModelInfoReader.get_default_template_type(self.model_type)
            logger.info(f'Setting template_type: {self.template_type}')

    def __post_init__(self):
        self.select_template()
