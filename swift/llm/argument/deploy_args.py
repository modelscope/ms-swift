from dataclasses import dataclass
from typing import Optional

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
        log_interval (int): The interval for printing global statistics. Default is 10.
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

    def __post_init__(self):
        super().__post_init__()
        self.port = find_free_port(self.port)

    def _init_stream(self):
        pass

    def _init_eval_human(self):
        pass

    def _init_result_path(self) -> None:
        if self.result_path is not None:
            return
        self.result_path = self.get_result_path('deploy_result')
        logger.info(f'args.result_path: {self.result_path}')
