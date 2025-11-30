# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

from swift.utils import get_logger, json_parse_to_dict
from .base_args import to_abspath
from .deploy_args import DeployArguments

logger = get_logger()


@dataclass
class EvalArguments(DeployArguments):
    """A dataclass that extends DeployArguments to define model evaluation arguments.

    These arguments control the evaluation process, including the choice of backend, datasets, generation parameters,
    and other configurations.

    Args:
        eval_dataset (List[str]): List of evaluation datasets. Please refer to the evaluation documentation for
            available options. Defaults to [].
        eval_limit (Optional[int]): The number of samples to take from each evaluation dataset. If None, all samples
            are used. Defaults to None.
        eval_dataset_args (Optional[Union[Dict, str]]): Evaluation dataset parameters, in JSON format, can be set for
            multiple datasets. Defaults to None.
        eval_generation_config (Optional[Union[Dict, str]]): The model's inference configuration for evaluation,
            provided as a JSON string (e.g., '{"max_new_tokens": 512}'). Defaults to None.
        eval_output_dir (str): The directory to store evaluation results. Defaults to 'eval_output'.
        eval_backend (str): The evaluation backend. Can be 'Native', 'OpenCompass', or 'VLMEvalKit'. Defaults to
            'Native'.
        local_dataset (bool): Whether to automatically download extra datasets required for certain evaluations
            (e.g., CMB). If True, a 'data' folder will be created in the current directory for the datasets. This
            download occurs only once, and subsequent runs will use the cache. Defaults to False.
            Note: By default, evaluation uses datasets from `~/.cache/opencompass`. When this is set to True, the
            `data` folder in the current directory is used instead.
        temperature (float): The temperature for sampling, which overrides the default generation config. Defaults
            to 0.0.
        verbose (bool): Whether to output verbose information during the evaluation process. Defaults to False.
        eval_num_proc (int): The maximum number of concurrent clients for evaluation. Defaults to 16.
        extra_eval_args (Optional[Union[Dict, str]]): Additional evaluation arguments, provided as a JSON string.
            These are only effective when using the 'Native' backend. Refer to the documentation for more details on
            available arguments. Defaults to {}.
        eval_url (Optional[str]): The URL for the evaluation service (e.g., 'http://localhost:8000/v1'). If not
            specified, evaluation runs on the locally deployed model. See documentation for more examples. Defaults
            to None.
    """
    eval_dataset: List[str] = field(default_factory=list)
    eval_limit: Optional[int] = None
    eval_dataset_args: Optional[Union[Dict, str]] = None
    eval_generation_config: Optional[Union[Dict, str]] = field(default_factory=dict)
    eval_output_dir: str = 'eval_output'
    eval_backend: Literal['Native', 'OpenCompass', 'VLMEvalKit'] = 'Native'
    local_dataset: bool = False

    temperature: Optional[float] = 0.
    verbose: bool = False
    eval_num_proc: int = 16
    extra_eval_args: Optional[Union[Dict, str]] = field(default_factory=dict)
    # If eval_url is set, ms-swift will not perform deployment operations and
    # will directly use the URL for evaluation.
    eval_url: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        self._init_eval_url()
        self._init_eval_dataset()
        self.eval_dataset_args = json_parse_to_dict(self.eval_dataset_args)
        self.eval_generation_config = json_parse_to_dict(self.eval_generation_config)
        self.extra_eval_args = json_parse_to_dict(self.extra_eval_args)
        self.eval_output_dir = to_abspath(self.eval_output_dir)
        logger.info(f'eval_output_dir: {self.eval_output_dir}')

    def _init_eval_url(self):
        # [compat]
        if self.eval_url and 'chat/completions' in self.eval_url:
            self.eval_url = self.eval_url.split('/chat/completions', 1)[0]

    @staticmethod
    def list_eval_dataset(eval_backend=None):
        from evalscope.constants import EvalBackend
        from evalscope.api.registry import BENCHMARK_REGISTRY
        from evalscope.backend.opencompass import OpenCompassBackendManager
        res = {
            EvalBackend.NATIVE: list(sorted(BENCHMARK_REGISTRY.keys())),
            EvalBackend.OPEN_COMPASS: sorted(OpenCompassBackendManager.list_datasets()),
        }
        try:
            from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager
            vlm_datasets = VLMEvalKitBackendManager.list_supported_datasets()
            res[EvalBackend.VLM_EVAL_KIT] = sorted(vlm_datasets)
        except ImportError:
            # fix cv2 import error
            if eval_backend == 'VLMEvalKit':
                raise
        return res

    def _init_eval_dataset(self):
        if isinstance(self.eval_dataset, str):
            self.eval_dataset = [self.eval_dataset]

        all_eval_dataset = self.list_eval_dataset(self.eval_backend)
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
