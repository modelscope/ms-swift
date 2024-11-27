# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from swift.llm import DATASET_MAPPING, register_dataset_info
from swift.utils import get_logger
from .utils import to_abspath

logger = get_logger()


@dataclass
class DataArguments:
    """
    DataArguments class is a dataclass that holds various arguments related to dataset handling and processing.

    Args:
        dataset (List[str]): List of dataset identifiers or paths.
        val_dataset (List[str]): List of validation dataset identifiers or paths.
        split_dataset_ratio (float): Ratio to split the dataset for validation if val_dataset is empty. Default is 0.01.
        data_seed (Optional[int]): Seed for dataset shuffling. Default is None.
        dataset_num_proc (int): Number of processes to use for data loading and preprocessing. Default is 1.
        load_from_cache_file (bool): Flag to load dataset from cache file. Default is False.
        download_mode (Literal): Mode for downloading datasets. Default is 'reuse_dataset_if_exists'.
        model_name (List[str]): List containing Chinese and English names of the model. Default is [None, None].
        model_author (List[str]): List containing Chinese and English names of the model author.
            Default is [None, None].
        streaming (bool): Flag to enable streaming of datasets. Default is False.
        custom_register_path (Optional[str]): Path to custom .py file for dataset registration. Default is None.
        custom_dataset_info (Optional[str]): Path to custom dataset_info.json file. Default is None.
    """
    # dataset_id or dataset_dir or dataset_path
    dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    val_dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    split_dataset_ratio: float = 0.01

    data_seed: Optional[int] = None
    dataset_num_proc: int = 1
    streaming: bool = False

    load_from_cache_file: bool = False
    download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists'
    strict: bool = False
    # Chinese name and English name
    model_name: List[str] = field(default_factory=lambda: [None, None], metadata={'help': "e.g. ['小黄', 'Xiao Huang']"})
    model_author: List[str] = field(
        default_factory=lambda: [None, None], metadata={'help': "e.g. ['魔搭', 'ModelScope']"})

    custom_register_path: Optional[str] = None  # .py
    custom_dataset_info: Optional[str] = None  # .json

    def _init_custom_register(self) -> None:
        """Register custom .py file to datasets"""
        if self.custom_register_path is None:
            return
        self.custom_register_path = to_abspath(self.custom_register_path, True)
        folder, fname = os.path.split(self.custom_register_path)
        sys.path.append(folder)
        __import__(fname.rstrip('.py'))

    def _init_custom_dataset_info(self):
        """register custom dataset_info.json to datasets"""
        if self.custom_dataset_info is None:
            return
        register_dataset_info(self.custom_dataset_info)

    def __post_init__(self):
        if self.data_seed is None:
            self.data_seed = self.seed
        if len(self.val_dataset) > 0:
            self.split_dataset_ratio = 0.
            logger.info(f'Using val_dataset, setting split_dataset_ratio: {self.split_dataset_ratio}')
        self._init_custom_register()
        self._init_custom_dataset_info()

    def get_dataset_kwargs(self):
        return {
            'seed': self.data_seed,
            'num_proc': self.dataset_num_proc,
            'streaming': self.streaming,
            'use_hf': self.use_hf,
            'hub_token': self.hub_token,
            'load_from_cache_file': self.load_from_cache_file,
            'download_mode': self.download_mode,
            'strict': self.strict,
            'model_name': self.model_name,
            'model_author': self.model_author,
        }
