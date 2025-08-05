# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Literal, Optional

from swift.llm import safe_snapshot_download
from swift.utils import find_free_port, get_dist_setting, get_logger
from .base_args import BaseArguments
from .infer_args import InferArguments

logger = get_logger()


@dataclass
class DeployArguments(InferArguments):
    """
    DeployArguments is a dataclass that extends InferArguments and is used to define
    the arguments required for deploying a model.

    Args:
        host (str): The host address to bind the server to. Default is '0.0.0.0'.
        port (int): The port number to bind the server to. Default is 8000.
        api_key (Optional[str]): The API key for authentication. Default is None.
        ssl_keyfile (Optional[str]): The path to the SSL key file. Default is None.
        ssl_certfile (Optional[str]): The path to the SSL certificate file. Default is None.
        owned_by (str): The owner of the deployment. Default is 'swift'.
        served_model_name (Optional[str]): The name of the model being served. Default is None.
        verbose (bool): Whether to log request information. Default is True.
        log_interval (int): The interval for printing global statistics. Default is 20.
        max_logprobs(int): Max number of logprobs to return
    """
    host: str = '0.0.0.0'
    port: int = 8000
    api_key: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None

    owned_by: str = 'swift'
    served_model_name: Optional[str] = None
    verbose: bool = True  # Whether to log request_info
    log_interval: int = 20  # Interval for printing global statistics
    log_level: Literal['critical', 'error', 'warning', 'info', 'debug', 'trace'] = 'info'

    max_logprobs: int = 20
    vllm_use_async_engine: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.port = find_free_port(self.port)

    def _init_adapters(self):
        if isinstance(self.adapters, str):
            self.adapters = [self.adapters]
        self.adapter_mapping = {}
        adapters = []
        for i, adapter in enumerate(self.adapters):
            adapter_path = adapter.split('=')
            if len(adapter_path) == 1:
                adapter_path = (None, adapter_path[0])
            adapter_name, adapter_path = adapter_path
            adapter_path = safe_snapshot_download(adapter_path, use_hf=self.use_hf, hub_token=self.hub_token)
            if adapter_name is None:
                adapters.append(adapter_path)
            else:
                self.adapter_mapping[adapter_name] = adapter_path
        self.adapters = adapters

    def _init_ckpt_dir(self, adapters=None):
        return super()._init_ckpt_dir(self.adapters + list(self.adapter_mapping.values()))

    def _init_stream(self):
        return BaseArguments._init_stream(self)

    def _init_eval_human(self):
        pass

    def _init_result_path(self, folder_name: str) -> None:
        if folder_name == 'infer_result':
            folder_name = 'deploy_result'
        return super()._init_result_path(folder_name)


@dataclass
class RolloutArguments(DeployArguments):
    vllm_use_async_engine: Optional[bool] = None
    use_gym_env: Optional[bool] = None
    # only for GRPO rollout with AsyncEngine, see details in swift/plugin/multi_turn
    multi_turn_scheduler: Optional[str] = None
    max_turns: Optional[int] = None

    # GYM env
    gym_env: Optional[str] = None
    context_manager: Optional[str] = None

    def __post_init__(self):
        try:
            from trl.scripts.vllm_serve import WeightSyncWorkerExtension
        except ImportError as e:
            raise ImportError("Could not import 'WeightSyncWorkerExtension' from 'trl.scripts.vllm_serve'. "
                              "Please upgrade your 'trl' package by 'pip install -U trl'") from e
        super().__post_init__()

        if self.vllm_use_async_engine is None:
            if self.multi_turn_scheduler or self.use_gym_env:
                self.vllm_use_async_engine = True
            else:
                self.vllm_use_async_engine = False

        if self.vllm_pipeline_parallel_size > 1:
            raise ValueError('RolloutArguments does not support pipeline parallelism, '
                             'please set vllm_pipeline_parallel_size to 1.')

        local_world_size = get_dist_setting()[3]
        used_world_size = self.vllm_data_parallel_size * self.vllm_tensor_parallel_size
        assert local_world_size >= used_world_size, (
            f'Error: local_world_size ({local_world_size}) must be greater than or equal to '
            f'the product of vllm_data_parallel_size ({self.vllm_data_parallel_size}) and '
            f'vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}). '
            f'Current used_world_size = {used_world_size}.')

        if local_world_size > used_world_size:
            logger.warning_once(
                f'local_world_size ({local_world_size}) is greater than used_world_size ({used_world_size}). '
                'Only the first {used_world_size} ranks will be used for rollout. '
                'To fully utilize resources, set vllm_tensor_parallel_size * vllm_data_parallel_size = world_size. '
                f'world_size: {local_world_size}, '
                f'vllm_tensor_parallel_size: {self.vllm_tensor_parallel_size}, '
                f'vllm_data_parallel_size: {self.vllm_data_parallel_size}, '
                f'used_world_size: {used_world_size}.')
