# Copyright (c) Alibaba, Inc. and its affiliates.
# Code partially sourced from Hugging Face TRL

import asyncio
import inspect
import multiprocessing
import os
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Dict, List, Optional, Union

import torch
import uvicorn
from aiohttp import ClientConnectorError
from fastapi import FastAPI

from swift.llm import InferArguments, RolloutArguments, SwiftPipeline
from swift.llm.template.template_inputs import RolloutInferRequest
from swift.utils import get_device, get_logger
from .infer_engine import GRPOVllmEngine, InferClient
from .protocol import InitCommunicatorRequest, RequestConfig, UpdateWeightsRequest

try:
    from vllm.utils import get_open_port
    from trl.scripts.vllm_serve import chunk_list

except ImportError:
    pass
"""
This module defines the execution logic for `swift rollout`.
It adds weight synchronization logic based on `vLLMEngine`.

Usage:
    swift rollout \
        --model xxx \
        --tensor_parallel_size xxx \
        --data_parallel_size xxx \
        --use_async_engine true/false \
        --... \
        --other_vllm_arguments

Note:
- Rollout is intended solely for GRPO training sampling.
- For inference or deployment, please use the `swift infer` or `swift deploy` commands.
"""

logger = get_logger()


def safe_set_start_method():
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')


def llm_worker(args: RolloutArguments, data_parallel_rank: int, master_port: int, connection: Connection) -> None:
    # Set required environment variables for DP to work with vLLM
    os.environ['VLLM_DP_RANK'] = str(data_parallel_rank)
    os.environ['VLLM_DP_RANK_LOCAL'] = str(data_parallel_rank)
    os.environ['VLLM_DP_SIZE'] = str(args.data_parallel_size)
    os.environ['VLLM_DP_MASTER_PORT'] = str(master_port)
    kwargs = {}
    if args.tensor_parallel_size == 1 and args.data_parallel_size > 1:
        kwargs['device'] = get_device(str(data_parallel_rank))
    kwargs['template'] = args.get_template(None)
    engine = SwiftRolloutDeploy.get_infer_engine(args, **kwargs)

    # Send ready signal to parent process
    connection.send({'status': 'ready'})

    while True:
        # Wait for commands from the parent process
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            engine.engine.collective_rpc(method='close_communicator')
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


async def async_llm_worker(args: RolloutArguments, data_parallel_rank: int, master_port: int,
                           connection: Connection) -> None:
    engine = SwiftRolloutDeploy.get_infer_engine(args)
    # Send ready signal to parent process
    connection.send({'status': 'ready'})

    loop = asyncio.get_running_loop()
    while True:
        try:
            command = await loop.run_in_executor(None, connection.recv)
        except KeyboardInterrupt:
            await engine.engine.collective_rpc(method='close_communicator')
            break

        # Handle commands
        if command['type'] in ['call', 'fire_and_forget']:
            method_name = command['method']
            args, kwargs = command.get('args', ()), command.get('kwargs', {})
            method = getattr(engine, method_name, None) or getattr(engine.engine, method_name, None)
            try:
                result = await method(*args, **kwargs)
            except Exception as e:
                logger.error(f'Method execution failed: {e}')
                result = None

            if command['type'] == 'call':
                connection.send(result)
        elif command['type'] == 'shutdown':
            break


def llm_worker_entry(*args, **kwargs):
    asyncio.run(async_llm_worker(*args, **kwargs))


class SwiftRolloutDeploy(SwiftPipeline):
    args_class = RolloutArguments
    args: args_class

    def _register_rl_rollout_app(self):
        self.app.get('/health/')(self.health)
        self.app.get('/get_world_size/')(self.get_world_size)
        self.app.post('/init_communicator/')(self.init_communicator)
        self.app.post('/update_named_param/')(self.update_named_param)
        self.app.post('/reset_prefix_cache/')(self.reset_prefix_cache)
        self.app.post('/close_communicator/')(self.close_communicator)
        self.app.post('/infer/', response_model=None)(self.infer)
        self.app.post('/get_engine_type/')(self.get_engine_type)

    def __init__(self, args: Union[List[str], RolloutArguments, None] = None):
        super().__init__(args)
        self.use_async_engine = self.args.use_async_engine
        self.num_connections = 1 if self.use_async_engine else self.args.data_parallel_size
        safe_set_start_method()
        self.app = FastAPI(lifespan=self.lifespan)
        self._register_rl_rollout_app()
        self.master_port = get_open_port()
        self.connections = []
        self.processes = []
        self._start_data_parallel_workers()

    def _start_data_parallel_workers(self):
        for data_parallel_rank in range(self.num_connections):
            parent_conn, child_conn = Pipe()
            worker_func = llm_worker_entry if self.use_async_engine else llm_worker
            process = Process(target=worker_func, args=(self.args, data_parallel_rank, self.master_port, child_conn))
            process.start()
            self.connections.append(parent_conn)
            self.processes.append(process)

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        # Wait for all workers to send "ready"
        ready_connections = set()

        while len(ready_connections) < self.num_connections:
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
    def get_infer_engine(args: InferArguments, template=None, **kwargs):
        kwargs.update({
            'model_id_or_path': args.model,
            'model_type': args.model_type,
            'revision': args.model_revision,
            'torch_dtype': args.torch_dtype,
            'template': template,
            'use_async_engine': args.use_async_engine,
            'multi_turn_scheduler': args.multi_turn_scheduler,
            'max_turn': args.max_turns
        })
        infer_backend = kwargs.pop('infer_backend', None) or args.infer_backend
        if infer_backend != 'vllm':
            infer_backend = 'vllm'
            logger.info('Currently, rollout only supports the vLLM backend. Set vLLM backend')
        kwargs.update(args.get_vllm_engine_kwargs())
        # used for RL external rollout backend
        engine_kwargs = kwargs.get('engine_kwargs', {})
        # for RL rollout model weight sync
        engine_kwargs.update({'worker_extension_cls': 'trl.scripts.vllm_serve.WeightSyncWorkerExtension'})
        if args.use_async_engine and args.data_parallel_size > 1:
            engine_kwargs['data_parallel_size'] = args.data_parallel_size
        kwargs['engine_kwargs'] = engine_kwargs

        return GRPOVllmEngine(**kwargs)

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

    async def get_engine_type(self):
        """
        Health check endpoint to verify that the server is running.
        """
        if self.use_async_engine:
            return {'engine_type': 'AsyncLLMEngine'}
        else:
            return {'engine_type': 'LLMEngine'}

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
        infer_requests: List[Union[Dict, RolloutInferRequest]],
        request_config: Optional[RequestConfig] = None,
        *,
        use_tqdm: Optional[bool] = None,
    ):
        chunked_infer_requests = chunk_list(infer_requests, self.num_connections)

        # Send the prompts to each worker
        for i, (connection, requests) in enumerate(zip(self.connections, chunked_infer_requests)):
            # When the number of prompts is less than data_parallel_size, some workers will receive empty prompts.
            # However, vLLM requires that we always send at least one prompt. So we send a placeholder prompt to comply
            # with vLLM's requirement, and we later ignore the result.
            if not requests:
                requests = RolloutInferRequest(messages=[{'role': 'user', 'content': '<placeholder>'}])
            # different seed bewteen vLLM Engine
            if request_config.seed:
                request_config.seed += i * len(requests)
            kwargs = {'infer_requests': requests, 'request_config': request_config, 'use_tqdm': use_tqdm}
            method = 'async_infer' if self.use_async_engine else 'infer'
            connection.send({'type': 'call', 'method': method, 'kwargs': kwargs})

        all_outputs = [connection.recv() for connection in self.connections]
        # Handle empty prompts (see above)
        all_outputs = [output for output, requests in zip(all_outputs, chunked_infer_requests) if requests]
        all_outputs = list(chain.from_iterable(all_outputs))  # from list of list to single list

        return all_outputs

    def run(self):
        args = self.args
        uvicorn.run(self.app, host=args.host, port=args.port, log_level=args.log_level)


def rollout_main(args: Union[List[str], RolloutArguments, None] = None) -> None:
    SwiftRolloutDeploy(args).main()


def is_accessible(port: int):
    infer_client = InferClient(port=port)
    try:
        infer_client.get_model_list()
    except ClientConnectorError:
        return False
    return True


@contextmanager
def run_rollout(args: RolloutArguments, return_url: bool = False):
    if isinstance(args, RolloutArguments) and args.__class__.__name__ == 'RolloutArguments':
        deploy_args = args
    else:
        args_dict = asdict(args)
        parameters = inspect.signature(RolloutArguments).parameters
        for k in list(args_dict.keys()):
            if k not in parameters or args_dict[k] is None:
                args_dict.pop(k)
        deploy_args = RolloutArguments(**args_dict)

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
