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
    EvalArguments is a dataclass that extends DeployArguments and is used to define
    the arguments required for evaluating a model.

    Args:
        eval_dataset (List[str]): List of evaluation datasets. Default is an empty list.
        eval_limit (Optional[str]): Limit number of each evaluation dataset. Default is None.
        local_dataset(bool): Download extra dataset from opencompass, default False.
        eval_output_dir (str): The eval output dir.
        temperature (float): The temperature.
        verbose (bool): Output verbose information.
        eval_url(str): The extra eval url, use this as --model.
    """
    eval_dataset: List[str] = field(default_factory=list)
    eval_limit: Optional[int] = None
    eval_output_dir: str = 'eval_output'
    eval_backend: Literal['Native', 'OpenCompass', 'VLMEvalKit'] = 'Native'
    local_dataset: bool = False

    temperature: Optional[float] = 0.
    verbose: bool = False
    eval_num_proc: Optional[int] = None
    # If eval_url is set, ms-swift will not perform deployment operations and
    # will directly use the URL for evaluation.
    eval_url: Optional[str] = None

    def _init_eval_url(self):
        # [compat]
        if self.eval_url and 'chat/completions' in self.eval_url:
            self.eval_url = self.eval_url.split('/chat/completions', 1)[0]

    def __post_init__(self):
        super().__post_init__()
        self._init_eval_url()
        self._init_eval_dataset()
        self.eval_output_dir = to_abspath(self.eval_output_dir)
        logger.info(f'eval_output_dir: {self.eval_output_dir}')

    @staticmethod
    def list_eval_dataset():
        from evalscope.constants import EvalBackend
        from evalscope.benchmarks.benchmark import BENCHMARK_MAPPINGS
        from evalscope.backend.opencompass import OpenCompassBackendManager
        from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
        return {
            EvalBackend.NATIVE: list(BENCHMARK_MAPPINGS.keys()),
            EvalBackend.OPEN_COMPASS: OpenCompassBackendManager.list_datasets(),
            EvalBackend.VLM_EVAL_KIT: VLMEvalKitBackendManager.list_supported_datasets()
        }

    def _init_eval_dataset(self):
        if isinstance(self.eval_dataset, str):
            self.eval_dataset = [self.eval_dataset]

        all_eval_dataset = self.list_eval_dataset()
        dataset_mapping = {dataset.lower(): dataset for dataset in all_eval_dataset[self.eval_backend]}
        valid_dataset = []
        for dataset in self.eval_dataset:
            if dataset.lower() not in dataset_mapping:
                raise ValueError(
                    f'eval_dataset: {dataset} is not supported.\n'
                    f'eval_backend: {self.eval_backend} supported datasets: {all_eval_dataset[self.eval_backend]}')
            valid_dataset.append(dataset_mapping[dataset.lower()])
        self.eval_dataset = valid_dataset

        logger.info(f'eval_backend: {self.eval_backend}')
        logger.info(f'eval_dataset: {self.eval_dataset}')

    def _init_result_path(self, folder_name: str) -> None:
        self.time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        result_dir = self.ckpt_dir or f'result/{self.model_suffix}'
        os.makedirs(result_dir, exist_ok=True)
        self.result_jsonl = to_abspath(os.path.join(result_dir, 'eval_result.jsonl'))
        if not self.eval_url:
            super()._init_result_path('eval_result')

    def _init_torch_dtype(self) -> None:
        if self.eval_url:
            self.model_dir = self.eval_output_dir
            return
        super()._init_torch_dtype()
