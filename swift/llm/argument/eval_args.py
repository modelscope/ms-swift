# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from .infer_args import InferArguments

# [TODO]


@dataclass
class EvalArguments(InferArguments):
    """
    EvalArguments is a dataclass that extends InferArguments and is used to define
    the arguments required for evaluating a model.

    Args:
        eval_dataset (List[str]): List of evaluation datasets. Default is an empty list.
        eval_limit (Optional[str]): Limit number of each evaluation dataset. Default is None.
        do_sample (bool): Flag to indicate if sampling should be done. Default is False.
    """
    eval_dataset: List[str] = field(default_factory=list)
    eval_limit: Optional[str] = None

    eval_result_path: Optional[str] = None
    do_sample: bool = False

    def __post_init__(self):
        super().__post_init__()
        self._init_eval_dataset()
        self._init_eval_result_path()

    def _init_eval_dataset(self):
        from evalscope.backend.opencompass import OpenCompassBackendManager
        from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
        self.opencompass_dataset = set(OpenCompassBackendManager.list_datasets())
        self.vlmeval_dataset = set(VLMEvalKitBackendManager.list_supported_datasets())
        self.eval_dataset_mapping = {
            dataset.lower(): dataset
            for dataset in self.opencompass_dataset | self.vlmeval_dataset
        }
        self.eval_dataset_oc = []
        self.eval_dataset_vlm = []
        for dataset in self.eval_dataset:
            dataset = self.eval_dataset_mapping[dataset.lower()]
            if dataset in self.opencompass_dataset:
                self.opencompass_dataset.append(dataset)
            elif dataset in self.vlmeval_dataset:
                self.eval_dataset_vlm.append(dataset)
            else:
                raise ValueError(
                    f'eval_dataset: {dataset} is not supported.\n'
                    f'opencompass_dataset: {OpenCompassBackendManager.list_datasets()}.\n'
                    f'vlmeval_dataset: {OpenCompassBackendManager.VLMEvalKitBackendManager.list_supported_datasets()}.')

    def _init_eval_result_path(self) -> None:
        if self.eval_result_path is not None:
            return
        self.eval_result_path = self.get_result_path('eval_result')
        logger.info(f'args.eval_result_path: {self.eval_result_path}')
