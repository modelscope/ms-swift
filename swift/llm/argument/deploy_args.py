from dataclasses import dataclass
from typing import Optional

from .infer_args import InferArguments


@dataclass
class DeployArguments(InferArguments):
    host: str = '0.0.0.0'
    port: int = 8000
    api_key: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None

    owned_by: str = 'swift'
    served_model_name: Optional[str] = None
    verbose: bool = True  # Whether to log request_info
    log_interval: int = 10  # Interval for printing global statistics

    def _init_stream(self):
        pass

    def _init_eval_human(self):
        pass

    def _init_result_dir(self, folder_name: str = 'deploy_result') -> None:
        super()._init_result_dir(folder_name=folder_name)
