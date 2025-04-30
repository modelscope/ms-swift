# Copyright (c) Alibaba, Inc. and its affiliates.
# Code partially sourced from Hugging Face TRL

import inspect
import multiprocessing
import os
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import List, Optional, Union

import torch
import uvicorn
from aiohttp import ClientConnectorError
from fastapi import FastAPI

from swift.llm import DeployArguments, InferArguments, SwiftPipeline
from swift.llm.template.template_inputs import RolloutInferRequest
from swift.utils import get_logger
from .infer_engine import InferClient, VllmEngine
from .protocol import InitCommunicatorRequest, RequestConfig, UpdateWeightsRequest

try:
    from vllm.utils import get_open_port
    from trl.scripts.vllm_serve import chunk_list

except ImportError:
    pass

logger = get_logger()


def safe_set_start_method():
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')


def llm_worker(args: DeployArguments, data_parallel_rank: int, master_port: int, connection: Connection) -> None:
    # Set required environment variables for DP to work with vLLM
    os.environ['VLLM_DP_RANK'] = str(data_parallel_rank)
    os.environ['VLLM_DP_RANK_LOCAL'] = str(data_parallel_rank)
    os.environ['VLLM_DP_SIZE'] = str(args.data_parallel_size)
    os.environ['VLLM_DP_MASTER_PORT'] = str(master_port)
    engine = SwiftRolloutDeploy.get_infer_engine(args)

    # Send ready signal to parent process
    connection.send({'status': 'ready'})

    while True:
        # Wait for commands from the parent process
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            engine.inner_model_executor.collective_rpc(method='close_communicator')
            break

        # Handle commands
        if command['type'] in ['call', 'fire_and_forget']:
            method_name = command['method']
            args, kwargs = command.get('args', ()), command.get('kwargs', {})
            method = getattr(engine, method_name, None) or getattr(engine.engine, method_name, None)
            result = method(*args, **kwargs)
            if command['type'] == 'call':
                connection.send(result)
        elif command['type'] == 'shutdown':
            break


class SwiftRolloutDeploy(SwiftPipeline):
    args_class = DeployArguments
    args: args_class

    def _register_rl_rollout_app(self):
        self.app.get('/health/')(self.health)
        self.app.get('/get_world_size/')(self.get_world_size)
        self.app.post('/init_communicator/')(self.init_communicator)
        self.app.post('/update_named_param/')(self.update_named_param)
        self.app.post('/reset_prefix_cache/')(self.reset_prefix_cache)
        self.app.post('/close_communicator/')(self.close_communicator)
        self.app.post('/infer/', response_model=None)(self.infer)

    def __init__(self, args: Union[List[str], DeployArguments, None] = None):
        super().__init__(args)
        assert self.args.data_parallel_size == 1, 'currently, DP is not supported, please set --data_parallel_size 1'
        safe_set_start_method()
        self.app = FastAPI(lifespan=self.lifespan)
        self._register_rl_rollout_app()
        self.master_port = get_open_port()
        self.connections = []
        self.processes = []
        self._start_data_parallel_workers()

    def _start_data_parallel_workers(self):
        for data_parallel_rank in range(self.args.data_parallel_size):
            parent_conn, child_conn = Pipe()
            process = Process(target=llm_worker, args=(self.args, data_parallel_rank, self.master_port, child_conn))
            process.start()
            self.connections.append(parent_conn)
            self.processes.append(process)

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # Wait for all workers to send "ready"
        ready_connections = set()
        while len(ready_connections) < self.args.data_parallel_size:
            for connection in self.connections:
                msg = connection.recv()
                if isinstance(msg, dict) and msg.get('status') == 'ready':
                    ready_connections.add(connection)

        yield

        # Wait for processes to terminate
        for process in self.processes:
            process.join(timeout=10)  # Wait for 10 seconds for the process to terminate
            if process.is_alive():
                logger.warning(f'Process {process} is still alive after 10 seconds, attempting to terminate...')
                process.terminate()
                process.join()  # ensure process termination after calling terminate()

    @staticmethod
    def get_infer_engine(args: InferArguments, **kwargs):
        kwargs.update({
            'model_id_or_path': args.model,
            'model_type': args.model_type,
            'revision': args.model_revision,
            'torch_dtype': args.torch_dtype,
        })
        infer_backend = kwargs.pop('infer_backend', None) or args.infer_backend
        if infer_backend != 'vllm':
            infer_backend = 'vllm'
            logger.info('Currently, rollout only supports the vLLM backend. Set vLLM backend')
        use_async_engine = args.use_async_engine
        if use_async_engine:
            use_async_engine = False
            logger.info("currently, rollout don't support async engine, set use_async_engine False")
        kwargs.update(args.get_vllm_engine_kwargs())
        kwargs.update({'use_async_engine': use_async_engine})
        # used for RL external rollout backend
        engine_kwargs = kwargs.get('engine_kwargs', {})
        # for RL rollout model weight sync
        engine_kwargs.update({'worker_extension_cls': 'trl.scripts.vllm_serve.WeightSyncWorkerExtension'})
        engine_kwargs.update({'data_parallel_size': args.data_parallel_size})
        kwargs['engine_kwargs'] = engine_kwargs
        return VllmEngine(**kwargs)

    async def health(self):
        """
        Health check endpoint to verify that the server is running.
        """
        return {'status': 'ok'}

    async def get_world_size(self):
        """
        Retrieves the world size from the LLM engine.

        Returns:
            `dict`:
                A dictionary containing the world size.

        Example response:
        ```json
        {"world_size": 8}
        ```
        """
        return {'world_size': self.args.tensor_parallel_size * self.args.data_parallel_size}

    async def init_communicator(self, request: InitCommunicatorRequest):
        """
        Initializes the communicator for synchronizing model weights between a client and multiple server
        workers.

        Args:
            request (`InitCommunicatorRequest`):
                - `host` (`str`): Hostname or IP address of the master node.
                - `port` (`int`): Port number to be used for communication.
                - `world_size` (`int`): Total number of participating processes in the group.
        """
        world_size = self.args.tensor_parallel_size * self.args.data_parallel_size + 1

        # The function init_communicator is called this way: init_communicator(host, port, world_size)
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc(method="init_communicator", args=(host, port, world_size))
        kwargs = {'method': 'init_communicator', 'args': (request.host, request.port, world_size)}
        for connection in self.connections:
            connection.send({'type': 'fire_and_forget', 'method': 'collective_rpc', 'kwargs': kwargs})

        return {'message': 'Request received, initializing communicator'}

    async def update_named_param(self, request: UpdateWeightsRequest):
        """
        Updates the model weights with the provided tensor.

        Once this endpoint is called, the client process should broadcast the updated weights to all server workers.

        Args:
            request (`UpdateWeightsRequest`):
                - `name` (`str`): Name of the weight tensor being updated.
                - `dtype` (`str`): Data type of the weight tensor (e.g., `"torch.float32"`).
                - `shape` (list of `int`): Shape of the weight

        """
        # The function update_named_param is called this way: update_named_param("name", torch.float32, (10, 10))
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc("update_named_param", args=("name", torch.float32, (10, 10)))
        dtype = torch.__getattribute__(request.dtype.split('.')[-1])
        kwargs = {'method': 'update_named_param', 'args': (request.name, dtype, tuple(request.shape))}
        for connection in self.connections:
            connection.send({'type': 'fire_and_forget', 'method': 'collective_rpc', 'kwargs': kwargs})

        return {'message': 'Request received, updating named parameter'}

    async def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        for connection in self.connections:
            connection.send({'type': 'call', 'method': 'reset_prefix_cache'})
        # Wait for and collect all results
        all_outputs = [connection.recv() for connection in self.connections]
        success = all(output for output in all_outputs)
        return {'message': 'Request received, resetting prefix cache status: ' + str(success)}

    async def close_communicator(self):
        """
        Closes the weight update group and cleans up associated resources.
        """
        kwargs = {'method': 'close_communicator'}
        for connection in self.connections:
            connection.send({'type': 'fire_and_forget', 'method': 'collective_rpc', 'kwargs': kwargs})
        return {'message': 'Request received, closing communicator'}

    async def infer(
        self,
        infer_requests: List[RolloutInferRequest],
        request_config: Optional[RequestConfig] = None,
        *,
        use_tqdm: Optional[bool] = None,
    ):
        chunked_infer_requests = chunk_list(infer_requests, self.args.data_parallel_size)

        # Send the prompts to each worker
        for connection, infer_requests in zip(self.connections, chunked_infer_requests):
            # When the number of prompts is less than data_parallel_size, some workers will receive empty prompts.
            # However, vLLM requires that we always send at least one prompt. So we send a placeholder prompt to comply
            # with vLLM's requirement, and we later ignore the result.
            if not infer_requests:
                infer_requests = RolloutInferRequest(messages=None)
            kwargs = {'infer_requests': infer_requests, 'request_config': request_config, 'use_tqdm': use_tqdm}
            connection.send({'type': 'call', 'method': 'infer', 'kwargs': kwargs})

        all_outputs = [connection.recv() for connection in self.connections]
        # Handle empty prompts (see above)
        all_outputs = [output for output, prompts in zip(all_outputs, chunked_infer_requests) if infer_requests]
        all_outputs = list(chain.from_iterable(all_outputs))  # from list of list to single list

        return all_outputs

    def run(self):
        args = self.args
        uvicorn.run(self.app, host=args.host, port=args.port, log_level=args.log_level)


def rollout_main(args: Union[List[str], DeployArguments, None] = None) -> None:
    SwiftRolloutDeploy(args).main()


def is_accessible(port: int):
    infer_client = InferClient(port=port)
    try:
        infer_client.get_model_list()
    except ClientConnectorError:
        return False
    return True


@contextmanager
def run_rollout(args: DeployArguments, return_url: bool = False):
    if isinstance(args, DeployArguments) and args.__class__.__name__ == 'DeployArguments':
        deploy_args = args
    else:
        args_dict = asdict(args)
        parameters = inspect.signature(DeployArguments).parameters
        for k in list(args_dict.keys()):
            if k not in parameters or args_dict[k] is None:
                args_dict.pop(k)
        deploy_args = DeployArguments(**args_dict)

    mp = multiprocessing.get_context('spawn')
    process = mp.Process(target=rollout_main, args=(deploy_args, ))
    process.start()
    try:
        while not is_accessible(deploy_args.port):
            time.sleep(1)
        yield f'http://127.0.0.1:{deploy_args.port}/v1' if return_url else deploy_args.port
    finally:
        process.terminate()
        logger.info('The deployment process has been terminated.')
