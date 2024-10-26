# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from swift.llm import DATASET_MAPPING, register_dataset_info
from swift.utils import get_logger

logger = get_logger()


@dataclass
class DataArguments:
    # dataset_id or dataset_name or dataset_path or ...
    dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    val_dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    split_dataset_ratio: float = 0.01  # If val_dataset is empty, use a split from the dataset as the validation set.
    dataset_seed: Optional[int] = None

    num_proc: int = 1
    load_from_cache_file: bool = False
    download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists'
    # Chinese name and English name
    model_name: List[str] = field(default_factory=lambda: [None, None], metadata={'help': "e.g. ['小黄', 'Xiao Huang']"})
    model_author: List[str] = field(
        default_factory=lambda: [None, None], metadata={'help': "e.g. ['魔搭', 'ModelScope']"})
    streaming: bool = False
    streaming_val_size: int = 0
    streaming_buffer_size: int = 16384

    custom_register_path: Optional[str] = None  # .py
    custom_dataset_info: Optional[str] = None  # .json

    def _init_custom_register(self) -> None:
        """Register custom .py file to datasets"""
        if self.custom_register_path is None:
            return
        folder, fname = os.path.split(self.custom_register_path)
        sys.path.append(folder)
        __import__(fname.rstrip('.py'))

    def _init_custom_dataset_info(self):
        """register custom dataset_info.json to datasets"""
        if self.custom_dataset_info is None:
            return
        register_dataset_info(self.custom_dataset_info)

    def __post_init__(self):
        if self.dataset_seed is None:
            self.dataset_seed = self.seed
        if len(self.val_dataset) > 0:
            self.split_dataset_ratio = 0.
            logger.info(f'Using val_dataset, setting split_dataset_ratio: {self.split_dataset_ratio}')
        self._init_custom_register()
        self._init_custom_dataset_info()

