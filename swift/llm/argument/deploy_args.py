# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Optional

from swift.llm import safe_snapshot_download
from swift.utils import find_free_port, get_logger
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

    max_logprobs: int = 20
    use_async_engine: bool = True

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
        pass

    def _init_eval_human(self):
        pass

    def _init_result_path(self, folder_name: str) -> None:
        if folder_name == 'infer_result':
            folder_name = 'deploy_result'
        return super()._init_result_path(folder_name)


@dataclass
class RolloutArguments(DeployArguments):
    use_async_engine: Optional[bool] = None
    # only for GRPO rollout with AsyncEngine, see details in swift/plugin/multi_turn
    multi_turn_scheduler: Optional[str] = None
    max_turns: Optional[int] = None

    def __post_init__(self):
        super().__post_init__()

        if self.use_async_engine is None:
            if self.multi_turn_scheduler:
                self.use_async_engine = True
            else:
                self.use_async_engine = False
