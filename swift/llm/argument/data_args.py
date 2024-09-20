# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from dataclasses import field, dataclass
from typing import List, Literal, Optional, Union

from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset

from swift.llm import TEMPLATE_MAPPING
from swift.llm.dataset.loader import DATASET_MAPPING, dataset_name_exists
from swift.llm.dataset.register import register_dataset_info_file
from swift.llm.model.model import get_default_template_type
from swift.utils import (get_logger)

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

    def register_self_cognition(self, sample_size) -> None:

        # compatibility. (Deprecated)
        idx_list = dataset_name_exists(self.dataset, 'self-cognition')
        assert len(idx_list) <= 1
        self.use_self_cognition = len(idx_list) == 1
        if sample_size > 0:
            d = f'self-cognition#{sample_size}'
            if len(idx_list) == 1:
                self.dataset[idx_list[0]] = d
            else:
                self.dataset.append(d)
            self.use_self_cognition = True
        # check
        if self.use_self_cognition:
            for k in ['model_name', 'model_author']:
                v = getattr(self, k)
                if isinstance(v, str):
                    v = [v]
                elif v is None:
                    v = [None, None]
                if len(v) == 1:
                    v = v * 2
                if v[0] is None and v[1] is None:
                    raise ValueError('Please set self.model_name self.model_author. '
                                     'For example: `--model_name 小黄 "Xiao Huang" --model_author 魔搭 ModelScope`. '
                                     'Representing the model name and model author in Chinese and English.')
                setattr(self, k, v)

    def __post_init__(self):
        if self.max_length == -1:
            self.max_length = None
        if len(self.val_dataset) > 0:
            self.dataset_test_ratio = 0.0
            logger.info('Using val_dataset, ignoring dataset_test_ratio')


class TemplateArguments:
    template_type: str = field(
        default='AUTO', metadata={'help': f"template_type choices: {list(TEMPLATE_MAPPING.keys()) + ['AUTO']}"})
    system: Optional[str] = None
    tools_prompt: Literal['react_en', 'react_zh', 'toolbench'] = 'react_en'
    # multi-modal
    rescale_image: int = -1

    def prepare_template(self, model_type):
        if self.template_type == 'AUTO':
            self.template_type = get_default_template_type(model_type)
            logger.info(f'Setting template_type: {self.template_type}')
