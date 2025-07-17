# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from swift.llm import DATASET_MAPPING, register_dataset_info
from swift.utils import get_logger

logger = get_logger()


@dataclass
class DataArguments:
    """
    DataArguments class is a dataclass that holds various arguments related to dataset handling and processing.

    Args:
        dataset (List[str]): List of dataset_id, dataset_path or dataset_dir
        val_dataset (List[str]): List of validation dataset_id, dataset_path or dataset_dir
        split_dataset_ratio (float): Ratio to split the dataset for validation if val_dataset is empty. Default is 0..
        data_seed (Optional[int]): Seed for dataset shuffling. Default is None.
        dataset_num_proc (int): Number of processes to use for data loading and preprocessing. Default is 1.
        streaming (bool): Flag to enable streaming of datasets. Default is False.
        download_mode (Literal): Mode for downloading datasets. Default is 'reuse_dataset_if_exists'.
        columns: Used for manual column mapping of datasets.
        model_name (List[str]): List containing Chinese and English names of the model. Default is None.
        model_author (List[str]): List containing Chinese and English names of the model author. Default is None.
        custom_dataset_info (Optional[str]): Path to custom dataset_info.json file. Default is None.
    """
    # dataset_id or dataset_dir or dataset_path
    dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    val_dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    split_dataset_ratio: float = 0.

    data_seed: int = 42
    dataset_num_proc: int = 1
    load_from_cache_file: bool = True
    dataset_shuffle: bool = True
    val_dataset_shuffle: bool = False
    streaming: bool = False
    interleave_prob: Optional[List[float]] = None
    stopping_strategy: Literal['first_exhausted', 'all_exhausted'] = 'first_exhausted'
    shuffle_buffer_size: int = 1000

    download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists'
    columns: Optional[Union[dict, str]] = None
    strict: bool = False
    remove_unused_columns: bool = True
    # Chinese name and English name
    model_name: Optional[List[str]] = field(default=None, metadata={'help': "e.g. ['小黄', 'Xiao Huang']"})
    model_author: Optional[List[str]] = field(default=None, metadata={'help': "e.g. ['魔搭', 'ModelScope']"})

    custom_dataset_info: List[str] = field(default_factory=list)  # .json

    def _init_custom_dataset_info(self):
        """register custom dataset_info.json to datasets"""
        if isinstance(self.custom_dataset_info, str):
            self.custom_dataset_info = [self.custom_dataset_info]
        for path in self.custom_dataset_info:
            register_dataset_info(path)

    def __post_init__(self):
        self.columns = self.parse_to_dict(self.columns)
        if len(self.val_dataset) > 0 or self.streaming and self.split_dataset_ratio > 0:
            self.split_dataset_ratio = 0.
            if len(self.val_dataset) > 0:
                msg = 'len(args.val_dataset) > 0'
            else:
                msg = 'args.streaming is True'
            logger.info(f'Because {msg}, setting split_dataset_ratio: {self.split_dataset_ratio}')
        self._init_custom_dataset_info()

    def get_dataset_kwargs(self):
        return {
            'seed': self.data_seed,
            'num_proc': self.dataset_num_proc,
            'load_from_cache_file': self.load_from_cache_file,
            'streaming': self.streaming,
            'interleave_prob': self.interleave_prob,
            'stopping_strategy': self.stopping_strategy,
            'shuffle_buffer_size': self.shuffle_buffer_size,
            'use_hf': self.use_hf,
            'hub_token': self.hub_token,
            'download_mode': self.download_mode,
            'columns': self.columns,
            'strict': self.strict,
            'model_name': self.model_name,
            'model_author': self.model_author,
            'remove_unused_columns': self.remove_unused_columns,
        }
