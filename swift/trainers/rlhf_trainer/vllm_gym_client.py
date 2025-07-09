# Copyright (c) Alibaba, Inc. and its affiliates.

# Code partially sourced from Hugging Face TRL

import atexit
import logging
import socket
import time
from typing import List, Optional
from urllib.parse import urlparse

import requests
import torch
from dacite import from_dict
from requests import ConnectionError
from torch import nn

from swift.llm import AdapterRequest, RolloutInferRequest, Template
from swift.llm.infer.protocol import ChatCompletionResponse, RequestConfig, GymRolloutResponseChoice
from swift.plugin import Metric
from swift.utils import is_vllm_ascend_available, is_vllm_available

if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator  # noqa

logger = logging.getLogger(__name__)


class VLLMGymClient:
    """
    A client class to interact with a gym-enabled vLLM server for GRPO training with environment interaction.

    This class provides methods to infer completions with gym environments, initialize and manage weight update groups, 
    and update model weights in a distributed setting. Before using it, start the gym vLLM server with 
    `swift rollout --use_gym_engine true`.

    Args:
        base_url (`str` or `None`, *optional*, defaults to `None`):
            Base URL for the vLLM server (e.g., `"http://localhost:8000"`). If provided, `host` and `server_port` are
            ignored.
        host (`str`, *optional*, defaults to `"0.0.0.0"`):
            IP address of the vLLM server. Ignored if `base_url` is provided.
        server_port (`int`, *optional*, defaults to `8000`):
            Port number of the vLLM server. Ignored if `base_url` is provided.
        group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group.
        connection_timeout (`float`, *optional*, defaults to `0.0`):
            Total timeout duration in seconds to wait for the server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.
    """

    def __init__(self,
                 base_url: Optional[str] = None,
                 host: str = '0.0.0.0',
                 server_port: int = 8000,
                 group_port: int = 51216,
                 connection_timeout: float = 0.0):
        if not is_vllm_available():
            raise ImportError('vLLM is not installed. Please install it with `pip install vllm`.')

        self.session = requests.Session()
        if base_url is not None:
            # Parse the base_url to extract host and port
            parsed_url = urlparse(base_url)
            self.host = socket.gethostbyname(parsed_url.hostname)
            scheme = parsed_url.scheme or 'http'
            self.base_url = f'{scheme}://{parsed_url.netloc}{parsed_url.path}'
        else:
            self.host = host
            self.server_port = server_port
            self.base_url = f'http://{self.host}:{self.server_port}'

        self.group_port = group_port
        self.check_server(connection_timeout)  # check server and fail after timeout

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        """
        Check gym server availability with retries on failure, within a total timeout duration. If the server is not up
        after the total timeout duration, raise a `ConnectionError`.

        Args:
            retry_interval (`float`, *optional*, defaults to `2.0`):
                Interval in seconds between retries.
            total_timeout (`float`, *optional*, defaults to `0.0`):
                Total timeout duration in seconds.
        """
        url = f'{self.base_url}/health/'
        start_time = time.time()  # Record the start time

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:
                # Check if the total timeout duration has passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The gym vLLM server can't be reached at {self.base_url} after {total_timeout} seconds. Make "
                        'sure the server is running by running `swift rollout --use_gym_engine true`.') from exc
            else:
                if response.status_code == 200:
                    # Verify this is a gym server
                    health_data = response.json()
                    if health_data.get('engine_type') != 'gym':
                        raise ConnectionError(
                            f"Server at {self.base_url} is not a gym server. "
                            "Please start the server with `swift rollout --use_gym_engine true`.")
                    
                    if 'X-Forwarded-For' in response.headers:
                        self.host = response.headers['X-Forwarded-For']
                    logger.info('Gym server is up!')
                    return None

            # Retry logic: wait before trying again
            logger.info(f'Gym server is not up yet. Retrying in {retry_interval} seconds...')
            time.sleep(retry_interval)

    def infer(
        self,
        infer_requests: List[RolloutInferRequest],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[AdapterRequest] = None,
    ) -> List[ChatCompletionResponse]:
        """
        Perform inference with gym environment interaction.
        
        This method sends rollout requests to the gym server, which will:
        1. Sample multiple trajectories per request using the specified environment
        2. Apply dynamic sampling if configured to ensure reward variance
        3. Return responses with accumulated rewards from environment interaction
        
        Args:
            infer_requests: List of rollout inference requests
            request_config: Configuration for the inference request
            metrics: Optional metrics to collect during inference
            template: Optional template for formatting
            use_tqdm: Whether to show progress bar
            adapter_request: Optional adapter configuration
            
        Returns:
            List of ChatCompletionResponse with gym trajectory results
        """
        url = f'{self.base_url}/infer/'
        response = self.session.post(
            url,
            json={
                'infer_requests': infer_requests,
                'request_config': request_config,
                'metrics': metrics,
                'template': template,
                'use_tqdm': use_tqdm,
                'adapter_request': adapter_request,
            },
        )
        if response.status_code == 200:
            # Parse gym-specific response format
            results = []
            for resp in response.json():
                # Gym responses always use RolloutResponseChoice
                choices = []
                for choice_data in resp['choices']:
                    choice = GymRolloutResponseChoice(
                        index=choice_data['index'],
                        message=choice_data['message'],
                        finish_reason=choice_data['finish_reason'],
                        logprobs=choice_data.get('logprobs'),
                        messages=choice_data.get('messages', []),
                        trajectory_id=choice_data.get('trajectory_id'),
                        total_reward=choice_data.get('total_reward', 0.0),
                        step_rewards=choice_data.get('step_rewards', [])
                    )
                    choices.append(choice)
                
                result = ChatCompletionResponse(
                    model=resp['model'],
                    choices=choices,
                    usage=resp.get('usage'),
                    id=resp.get('id'),
                    created=resp.get('created'),
                    object=resp.get('object', 'chat.completion')
                )
                results.append(result)
            
            return results
        else:
            raise Exception(f'Gym inference request failed: {response.status_code}, {response.text}')

    def init_communicator(self):
        """
        Initializes the weight update group in a distributed setup for model synchronization with gym server.
        """
        # Get the tensor parallel size from the gym server
        url = f'{self.base_url}/get_world_size/'
        response = requests.get(url)
        if response.status_code == 200:
            vllm_world_size = response.json()['world_size']
        else:
            raise Exception(f'Request failed: {response.status_code}, {response.text}')

        world_size = vllm_world_size + 1  # add the client to the world
        self.rank = vllm_world_size  # the client's rank is the last process

        # Initialize weight update group
        url = f'{self.base_url}/init_communicator/'
        # In the server side, the host is set to 0.0.0.0
        response = self.session.post(url, json={'host': '0.0.0.0', 'port': self.group_port, 'world_size': world_size})
        if response.status_code != 200:
            raise Exception(f'Request failed: {response.status_code}, {response.text}')

        # Brief delay to allow server initialization. While not strictly required (client socket will retry on
        # connection failure), this prevents log warnings like:
        # [W416 23:24:57.460001114 socket.cpp:204] [c10d] The hostname of the client socket cannot be retrieved. err=-3
        time.sleep(0.1)

        # Set up the communication group for weight broadcasting
        pg = StatelessProcessGroup.create(host=self.host, port=self.group_port, rank=self.rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(pg, device=0)

        # When the client object is deleted, close the weight update group
        atexit.register(self.close_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor):
        """
        Updates a specific named parameter in the gym model and broadcasts it to other processes.

        Args:
            name (`str`):
                Name of the layer whose weights are being updated.
            weights (`torch.Tensor`):
                Tensor containing the updated weights.
        """
        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f'{self.base_url}/update_named_param/'
        response = self.session.post(url, json={'name': name, 'dtype': dtype, 'shape': shape})
        if response.status_code != 200:
            raise Exception(f'Request failed: {response.status_code}, {response.text}')

        # Broadcast the weights to the other processes
        self.pynccl_comm.broadcast(weights, src=self.rank)
        self.pynccl_comm.group.barrier()

    def update_model_params(self, model: nn.Module):
        """
        Updates all parameters of the given gym model by calling `update_named_param` for each parameter in the model.

        Args:
            model (`nn.Module`):
                Model whose parameters (weights/biases) are to be updated.
        """
        for name, param in model.named_parameters():
            # Update each parameter individually
            self.update_named_param(name, param.data)

    def reset_prefix_cache(self):
        """
        Resets the prefix cache for the gym model.
        """
        url = f'{self.base_url}/reset_prefix_cache/'
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(f'Request failed: {response.status_code}, {response.text}')

    def get_engine_type(self):
        """
        Get the engine type from the gym server. Should return 'GymAsyncLLMEngine'.
        """
        url = f'{self.base_url}/get_engine_type/'
        response = self.session.post(url)
        if response.status_code == 200:
            result = response.json()['engine_type']
            # Gym engine is always async
            self.use_async_engine = result == 'GymAsyncLLMEngine'
            return result
        else:
            raise Exception(f'Request failed: {response.status_code}, {response.text}')

    def close_communicator(self):
        """
        Closes the weight update group and cleans up the communication group.
        """
        url = f'{self.base_url}/close_communicator/'

        try:
            response = self.session.post(url)
        except ConnectionError:
            # The server might be already down, so we don't need to close the communicator
            pass
        else:
            if response.status_code != 200:
                raise Exception(f'Request failed: {response.status_code}, {response.text}')