# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional

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
        do_sample (bool): Flag to indicate if sampling should be done. Default is False.
    """
    eval_dataset: List[str] = field(default_factory=list)
    eval_limit: Optional[int] = None

    eval_result_dir: Optional[str] = None
    do_sample: bool = False
    verbose: bool = False
    max_batch_size: int = 16

    def __post_init__(self):
        super().__post_init__()
        self._init_eval_dataset()
        self._init_eval_result_dir()

    def _init_eval_dataset(self):
        if isinstance(self.eval_dataset, str):
            self.eval_dataset = [self.eval_dataset]
        self.url = f'http://127.0.0.1:{self.port}/v1/chat/completions'

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

    def _init_eval_result_dir(self) -> None:
        if self.eval_result_dir is not None:
            return
        self.eval_result_dir = self.get_result_path('eval_result', '')
        os.makedirs(self.eval_result_dir, exist_ok=True)
        logger.info(f'args.eval_result_dir: {self.eval_result_dir}')

    def _init_result_path(self) -> None:
        return
