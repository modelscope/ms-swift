import atexit
import logging
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union
from urllib.parse import urlparse

import requests
import torch
from packaging import version
from pydantic import ValidationError
from requests import ConnectionError
from torch import nn
from transformers.utils import is_torch_cuda_available

from swift.llm import AdapterRequest, RolloutInferRequest, Template
from swift.llm.infer.protocol import ChatCompletionResponse, RequestConfig, RolloutOutput
from swift.plugin import Metric
from swift.utils import is_trl_available, is_vllm_ascend_available, is_vllm_available

if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator  # noqa

if is_trl_available():
    import trl
    trl_verison = version.parse(trl.__version__)

logger = logging.getLogger(__name__)


class VLLMClient:

    def __init__(self,
                 base_urls: Optional[List[str]] = None,
                 hosts: List[str] = ['0.0.0.0'],
                 server_ports: List[int] = [8000],
                 group_ports: Union[int, List[int]] = 51216,
                 connection_timeout: float = 240.0):
        if not is_vllm_available():
            raise ImportError('vLLM is not installed. Please install it with `pip install vllm`.')

        if base_urls is not None:
            self.base_urls = []
            self.hosts = []
            for url in base_urls:
                parsed_url = urlparse(url)
                host = socket.gethostbyname(parsed_url.hostname)
                scheme = parsed_url.scheme or 'http'
                base_url_i = f'{scheme}://{parsed_url.netloc}{parsed_url.path}'
                self.base_urls.append(base_url_i)
                self.hosts.append(host)
        else:
            if len(hosts) != len(server_ports):
                raise ValueError('host and server_port must have same length when lists are provided')
            self.base_urls = [f'http://{h}:{p}' for h, p in zip(hosts, server_ports)]
            self.hosts = hosts

        self.num_servers = len(self.base_urls)

        self.sessions = [requests.Session() for _ in range(self.num_servers)]

        if isinstance(group_ports, int):
            self.group_ports = [group_ports + i for i in range(self.num_servers)]
        elif isinstance(group_ports, list) and len(group_ports) == self.num_servers:
            self.group_ports = group_ports
        else:
            raise ValueError('group_port must be int or list of length num_servers')

        self.pynccl_comms = []
        self.check_server(connection_timeout)

    def check_server(self, total_timeout: float = 0.0, retry_interval: float = 2.0):
        server_status = [False] * self.num_servers

        def check_single_server(i):
            start_time = time.time()
            url = f'{self.base_urls[i]}/health/'
            while True:
                try:
                    response = requests.get(url, timeout=retry_interval)
                    if response.status_code == 200:
                        server_status[i] = True
                        return
                except Exception:
                    pass

                elapsed = time.time() - start_time
                if elapsed >= total_timeout:
                    return

                time.sleep(retry_interval)

        threads = []
        for i in range(self.num_servers):
            t = threading.Thread(target=check_single_server, args=(i, ))
            t.daemon = True
            t.start()
            threads.append(t)

        for t in threads:
            t.join(total_timeout)

        if not all(server_status):
            failed_servers = [self.base_urls[i] for i, status in enumerate(server_status) if not status]
            raise ConnectionError(f'Servers not reachable after {total_timeout}s: {failed_servers}')

    def infer(
        self,
        infer_requests: List[RolloutInferRequest],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[AdapterRequest] = None,
    ):
        if not hasattr(self, 'use_async_engine') or not hasattr(self, 'use_gym_env'):
            self.get_engine_type()

        n = len(infer_requests)
        chunk_size = (n + self.num_servers - 1) // self.num_servers
        chunks = [infer_requests[i:i + chunk_size] for i in range(0, n, chunk_size)]
        chunks += [[]] * (self.num_servers - len(chunks))

        results = [None] * self.num_servers
        errors = [None] * self.num_servers

        def process_chunk(i, chunk):
            try:
                response = self.sessions[i].post(
                    f'{self.base_urls[i]}/infer/',
                    json={
                        'infer_requests': chunk,
                        'request_config': request_config,
                        'metrics': metrics,
                        'template': template,
                        'use_tqdm': use_tqdm,
                        'adapter_request': adapter_request,
                    },
                )

                if response.status_code != 200:
                    errors[i] = Exception(f'Server {i} failed: {response.status_code}, {response.text}')
                    return

                resp_data = response.json()
                parsed: List[Union[RolloutOutput, ChatCompletionResponse]] = []

                for item in resp_data:
                    try:
                        parsed.append(RolloutOutput.model_validate(item))
                    except ValidationError:
                        parsed.append(ChatCompletionResponse(**item))
                results[i] = parsed
            except Exception as e:
                errors[i] = e

        with ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            futures = [executor.submit(process_chunk, i, chunk) for i, chunk in enumerate(chunks)]
            for future in futures:
                future.result()

        all_errors = [e for e in errors if e is not None]
        if all_errors:
            raise RuntimeError(f'Multiple errors: {all_errors}')

        return [res for server_results in results for res in server_results]

    def init_communicator(self, device: Union[int, str] = 0):
        self.pynccl_comms = []
        for i in range(self.num_servers):
            response = self.sessions[i].get(f'{self.base_urls[i]}/get_world_size/')
            if response.status_code != 200:
                raise Exception(f'Server {i} failed: {response.text}')
            vllm_world_size = response.json()['world_size']

            world_size = vllm_world_size + 1
            rank = vllm_world_size
            kwargs = {}
            if trl_verison >= version.parse('0.20.0'):
                if not is_torch_cuda_available():
                    raise NotImplementedError('trl >= 0.20.0 only support CUDA deivce. Please use trl < 0.20.0')
                client_device_uuid = str(torch.cuda.get_device_properties(device).uuid)
                kwargs['client_device_uuid'] = client_device_uuid

            response = self.sessions[i].post(
                f'{self.base_urls[i]}/init_communicator/',
                json={
                    'host': '0.0.0.0',
                    'port': self.group_ports[i],
                    'world_size': world_size,
                    **kwargs
                })
            if response.status_code != 200:
                raise Exception(f'Server {i} init failed: {response.text}')

            time.sleep(0.1)

            pg = StatelessProcessGroup.create(
                host=self.hosts[i], port=self.group_ports[i], rank=rank, world_size=world_size)
            comm = PyNcclCommunicator(pg, device=0)
            self.pynccl_comms.append(comm)

        atexit.register(self.close_communicator)

    def update_named_param(self, name: str, weights: torch.Tensor):
        dtype = str(weights.dtype)
        shape = tuple(weights.shape)

        errors = [None] * self.num_servers

        def _update_single_server(i):
            try:
                response = self.sessions[i].post(
                    f'{self.base_urls[i]}/update_named_param/',
                    json={
                        'name': name,
                        'dtype': dtype,
                        'shape': shape
                    },
                )
                if response.status_code != 200:
                    raise Exception(f'Server {i} update failed: {response.text}')

                self.pynccl_comms[i].broadcast(weights, src=self.pynccl_comms[i].rank)
                self.pynccl_comms[i].group.barrier()
            except Exception as e:
                errors[i] = e

        with ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            futures = [executor.submit(_update_single_server, i) for i in range(self.num_servers)]
            for future in futures:
                future.result()

        all_errors = [e for e in errors if e is not None]
        if all_errors:
            raise RuntimeError(f'Multiple errors: {all_errors}')

    def update_model_params(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.update_named_param(name, param.data)

    def reset_prefix_cache(self):
        errors = [None] * self.num_servers

        def _reset_single_server(i):
            try:
                response = self.sessions[i].post(f'{self.base_urls[i]}/reset_prefix_cache/')
                if response.status_code != 200:
                    raise Exception(f'Server {i} reset failed: {response.text}')
            except Exception as e:
                errors[i] = e

        with ThreadPoolExecutor(max_workers=self.num_servers) as executor:
            futures = [executor.submit(_reset_single_server, i) for i in range(self.num_servers)]
            for future in futures:
                future.result()
        all_errors = [e for e in errors if e is not None]
        if all_errors:
            raise RuntimeError(f'Multiple errors on reset_prefix_cache: {all_errors}')

    def get_engine_type(self):
        # assume that all server has same engine type
        response = self.sessions[0].post(f'{self.base_urls[0]}/get_engine_type/')
        if response.status_code != 200:
            raise Exception(f'Engine type request failed: {response.text}')

        result = response.json()
        self.use_async_engine = result['engine_type'] == 'AsyncLLMEngine'
        self.enable_multi_turn = result.get('enable_multi_turn', False)
        self.use_gym_env = result.get('gym_env', False)
        return result

    def close_communicator(self):
        for i in range(self.num_servers):
            try:
                response = self.sessions[i].post(f'{self.base_urls[i]}/close_communicator/')
                if response.status_code != 200:
                    logger.warning(f'Server {i} close failed: {response.text}')
            except Exception as e:
                logger.warning(f'Error closing server {i} communicator: {str(e)}')
