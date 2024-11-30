# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch

from swift.utils import get_logger
from .base_args import to_abspath
from .deploy_args import DeployArguments

logger = get_logger()


@dataclass
class EvalArguments(DeployArguments):
    """
    EvalArguments is a dataclass that extends InferArguments and is used to define
    the arguments required for evaluating a model.

    Args:
        eval_dataset (List[str]): List of evaluation datasets. Default is an empty list.
        eval_limit (Optional[str]): Limit number of each evaluation dataset. Default is None.
    """
    eval_dataset: List[str] = field(default_factory=list)
    eval_limit: Optional[int] = None
    eval_output_dir: str = 'eval_output'

    temperature: Optional[float] = 0.
    verbose: bool = False
    max_batch_size: Optional[int] = None
    # If eval_url is set, ms-swift will not perform deployment operations and
    # will directly use the URL for evaluation.
    eval_url: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self._init_eval_dataset()
        self.eval_output_dir = to_abspath(self.eval_output_dir)
        logger.info(f'eval_output_dir: {self.eval_output_dir}')

    def _init_eval_dataset(self):
        if isinstance(self.eval_dataset, str):
            self.eval_dataset = [self.eval_dataset]

        from evalscope.backend.opencompass import OpenCompassBackendManager
        from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
        self.opencompass_dataset = set(OpenCompassBackendManager.list_datasets())
        self.vlmeval_dataset = set(VLMEvalKitBackendManager.list_supported_datasets())
        eval_dataset_mapping = {dataset.lower(): dataset for dataset in self.opencompass_dataset | self.vlmeval_dataset}
        self.eval_dataset_oc = []
        self.eval_dataset_vlm = []
        for dataset in self.eval_dataset:
            dataset = eval_dataset_mapping.get(dataset.lower(), dataset)
            if dataset in self.opencompass_dataset:
                self.eval_dataset_oc.append(dataset)
            elif dataset in self.vlmeval_dataset:
                self.eval_dataset_vlm.append(dataset)
            else:
                raise ValueError(f'eval_dataset: {dataset} is not supported.\n'
                                 f'opencompass_dataset: {OpenCompassBackendManager.list_datasets()}.\n\n'
                                 f'vlmeval_dataset: {VLMEvalKitBackendManager.list_supported_datasets()}.')

        logger.info(f'opencompass dataset: {self.eval_dataset_oc}')
        logger.info(f'vlmeval dataset: {self.eval_dataset_vlm}')

    def _init_result_path(self) -> None:
        if not self.model and not self.ckpt_dir:
            self.result_jsonl = to_abspath('./eval_result.jsonl')
            return
        self.time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        result_dir = self.ckpt_dir or self.model_dir
        self.result_jsonl = to_abspath(os.path.join(result_dir, 'eval_result.jsonl'))
        super()._init_result_path()

    def _init_torch_dtype(self) -> None:
        if self.eval_url:
            self.model_dir = self.eval_output_dir
            return
        super()._init_torch_dtype()
