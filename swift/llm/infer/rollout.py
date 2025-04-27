# Copyright (c) Alibaba, Inc. and its affiliates.
# Code partially sourced from Hugging Face TRL

import asyncio
import inspect
import multiprocessing
import time
from contextlib import contextmanager, asynccontextmanager
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection

from dataclasses import asdict
from http import HTTPStatus
from threading import Thread
from typing import List, Optional, Union

import json
import torch
import uvicorn
from aiohttp import ClientConnectorError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from swift.llm import AdapterRequest, DeployArguments, InferRequest
from swift.llm.infer.protocol import MultiModalRequestMixin
from swift.llm.template.template_inputs import RolloutInferRequest
from swift.plugin import InferStats
from swift.utils import JsonlWriter, get_logger
from .deploy import SwiftDeploy
from .infer_engine import InferClient
from .protocol import (ChatCompletionRequest, ChatCompletionResponse, CompletionRequest, InitCommunicatorRequest, Model,
                       ModelList, RequestConfig, UpdateWeightsRequest)

logger = get_logger()
import os


def chunk_list(lst: list, n: int) -> list[list]:
    """
    Split list `lst` into `n` evenly distributed sublists.
    Example:
        >>> chunk_list([1, 2, 3, 4, 5, 6], 2)
        [[1, 2, 3], [4, 5, 6]]
        >>> chunk_list([1, 2, 3, 4, 5, 6], 4)
        [[1, 2], [3, 4], [5], [6]]
        >>> chunk_list([1, 2, 3, 4, 5, 6], 8)
        [[1], [2], [3], [4], [5], [6], [], []]
    """
    k, r = divmod(len(lst), n)
    return [lst[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]


def llm_worker(
    args: DeployArguments, data_parallel_rank: int, master_port: int, connection: Connection
) -> None:
    # Set required environment variables for DP to work with vLLM
    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    llm = Engine(
        model=args.model,
        revision=args.revision,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        dtype=args.dtype,
        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
        # This is particularly useful here because we generate completions from the same prompts.
        enable_prefix_caching=args.enable_prefix_caching,
        max_model_len=args.max_model_len,
        worker_extension_cls="trl.scripts.vllm_serve.WeightSyncWorkerExtension",
    )

    # Send ready signal to parent process
    connection.send({"status": "ready"})

    while True:
        # Wait for commands from the parent process
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            llm.collective_rpc(method="close_communicator")
            break

        # Handle commands
        if command["type"] in ["call", "fire_and_forget"]:
            method_name = command["method"]
            args, kwargs = command.get("args", ()), command.get("kwargs", {})
            method = getattr(llm, method_name)
            result = method(*args, **kwargs)
            if command["type"] == "call":
                connection.send(result)
        elif command["type"] == "shutdown":
            break

class SwiftRolloutDeploy(SwiftDeploy):
    args_class = DeployArguments
    args: args_class

    def _register_rl_rollout_app(self):
        self.app.get('/health/')(self.health_check)
        self.app.get('/get_world_size/')(self.get_world_size)
        self.app.post('/init_communicator/')(self.init_communicator)
        self.app.post('/update_named_param/')(self.update_named_param)
        self.app.post('/reset_prefix_cache/')(self.reset_prefix_cache)
        self.app.post('/close_communicator/')(self.close_communicator)
        self.app.post('/infer/', response_model=None)(self.infer)

    def __init__(self, args: Union[List[str], DeployArguments, None] = None) -> None:
        # TODO: rewrite super init, load more engine
        super().__init__(args)
        self._register_rl_rollout_app()

    def rollout_main(self):
        from vllm.utils import get_open_port
        master_port = get_open_port()
        connections = []
        processes = []

        for data_parallel_rank in range(self.args.data_parallel_size):
            parent_connection, child_connection = Pipe()
            process = Process(target=llm_worker, args=(self.args, data_parallel_rank, master_port, child_connection))
            process.start()
            connections.append(parent_connection)
            processes.append(process)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Wait for all workers to send "ready"
            ready_connections = set()
            while len(ready_connections) < self.args.data_parallel_size:
                for connection in connections:
                    msg = connection.recv()
                    if isinstance(msg, dict) and msg.get("status") == "ready":
                        ready_connections.add(connection)

        yield

        # Wait for processes to terminate
        for process in processes:
            process.join(timeout=10)  # Wait for 10 seconds for the process to terminate
            if process.is_alive():
                logger.warning(f"Process {process} is still alive after 10 seconds, attempting to terminate...")
                process.terminate()
                process.join()  # ensure process termination after calling terminate()

        self.app = FastAPI(lifespan=lifespan)

    def _get_model_list(self):
        args = self.args
        model_list = [args.served_model_name or args.model_suffix]
        if args.adapter_mapping:
            model_list += [name for name in args.adapter_mapping.keys()]
        return model_list

    async def get_available_models(self):
        model_list = self._get_model_list()
        data = [Model(id=model_id, owned_by=self.args.owned_by) for model_id in model_list]
        return ModelList(data=data)

    async def _check_model(self, request: ChatCompletionRequest) -> Optional[str]:
        available_models = await self.get_available_models()
        model_list = [model.id for model in available_models.data]
        if request.model not in model_list:
            return f'`{request.model}` is not in the model_list: `{model_list}`.'

    def _check_api_key(self, raw_request: Request) -> Optional[str]:
        api_key = self.args.api_key
        if api_key is None:
            return
        authorization = dict(raw_request.headers).get('authorization')
        error_msg = 'API key error'
        if authorization is None or not authorization.startswith('Bearer '):
            return error_msg
        request_api_key = authorization[7:]
        if request_api_key != api_key:
            return error_msg

    def _check_max_logprobs(self, request):
        args = self.args
        if isinstance(request.top_logprobs, int) and request.top_logprobs > args.max_logprobs:
            return (f'The value of top_logprobs({request.top_logprobs}) is greater than '
                    f'the server\'s max_logprobs({args.max_logprobs}).')

    @staticmethod
    def create_error_response(status_code: Union[int, str, HTTPStatus], message: str) -> JSONResponse:
        status_code = int(status_code)
        return JSONResponse({'message': message, 'object': 'error'}, status_code)

    def _post_process(self, request_info, response, return_cmpl_response: bool = False):
        args = self.args

        for i in range(len(response.choices)):
            if not hasattr(response.choices[i], 'message') or not isinstance(response.choices[i].message.content,
                                                                             (tuple, list)):
                continue
            for j, content in enumerate(response.choices[i].message.content):
                if content['type'] == 'image':
                    b64_image = MultiModalRequestMixin.to_base64(content['image'])
                    response.choices[i].message.content[j]['image'] = f'data:image/jpg;base64,{b64_image}'

        is_finished = all(response.choices[i].finish_reason for i in range(len(response.choices)))
        if 'stream' in response.__class__.__name__.lower():
            request_info['response'] += response.choices[0].delta.content
        else:
            request_info['response'] = response.choices[0].message.content
        if return_cmpl_response:
            response = response.to_cmpl_response()
        if is_finished:
            if args.log_interval > 0:
                self.infer_stats.update(response)
            if self.jsonl_writer:
                self.jsonl_writer.append(request_info)
            if self.args.verbose:
                logger.info(request_info)
        return response

    def _set_request_config(self, request_config) -> None:
        default_request_config = self.args.get_request_config()
        if default_request_config is None:
            return
        for key, val in asdict(request_config).items():
            default_val = getattr(default_request_config, key)
            if default_val is not None and (val is None or isinstance(val, (list, tuple)) and len(val) == 0):
                setattr(request_config, key, default_val)

    async def create_chat_completion(self,
                                     request: ChatCompletionRequest,
                                     raw_request: Request,
                                     *,
                                     return_cmpl_response: bool = False):
        args = self.args
        error_msg = (await self._check_model(request) or self._check_api_key(raw_request)
                     or self._check_max_logprobs(request))
        if error_msg:
            return self.create_error_response(HTTPStatus.BAD_REQUEST, error_msg)
        infer_kwargs = self.infer_kwargs.copy()
        adapter_path = args.adapter_mapping.get(request.model)
        if adapter_path:
            infer_kwargs['adapter_request'] = AdapterRequest(request.model, adapter_path)

        infer_request, request_config = request.parse()
        self._set_request_config(request_config)
        request_info = {'response': '', 'infer_request': infer_request.to_printable()}

        def pre_infer_hook(kwargs):
            request_info['generation_config'] = kwargs['generation_config']
            return kwargs

        infer_kwargs['pre_infer_hook'] = pre_infer_hook
        try:
            res_or_gen = await self.infer_async(infer_request, request_config, template=self.template, **infer_kwargs)
        except Exception as e:
            import traceback
            logger.info(traceback.format_exc())
            return self.create_error_response(HTTPStatus.BAD_REQUEST, str(e))
        if request_config.stream:

            async def _gen_wrapper():
                async for res in res_or_gen:
                    res = self._post_process(request_info, res, return_cmpl_response)
                    yield f'data: {json.dumps(asdict(res), ensure_ascii=False)}\n\n'
                yield 'data: [DONE]\n\n'

            return StreamingResponse(_gen_wrapper(), media_type='text/event-stream')
        else:
            return self._post_process(request_info, res_or_gen, return_cmpl_response)

    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        chat_request = ChatCompletionRequest.from_cmpl_request(request)
        return await self.create_chat_completion(chat_request, raw_request, return_cmpl_response=True)

    async def health_check(self):
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
        kwargs = {"method": "init_communicator", "args": (request.host, request.port, world_size)}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})


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
        # The function is called this way: update_named_param(name="name", dtype=torch.float32, shape=(10, 10))
        # So with collect_rpc we need to call it this way:
        # self.infer_engine.engine.model_executor.collective_rpc("update_named_param", \
        # args=("name", torch.float32, (10, 10)))
        # And with background_tasks.add_task we need to call it this way:
        # background_tasks.add_task(self.infer_engine.engine.model_executor.collective_rpc, \
        # "update_named_param", args=("name", torch.float32, (10, 10)))
        dtype = torch.__getattribute__(request.dtype.split('.')[-1])
        background_tasks.add_task(
            self.infer_engine.engine.model_executor.collective_rpc,
            'update_named_param',
            args=(request.name, dtype, request.shape))

        return {'message': 'Request received, updating named parameter'}

    async def reset_prefix_cache(self):
        """
        Resets the prefix cache for the model.
        """
        success = self.infer_engine.engine.reset_prefix_cache()
        return {'message': 'Request received, resetting prefix cache status: ' + str(success)}

    async def close_communicator(self):
        """
        Closes the weight update group and cleans up associated resources.
        """
        self.infer_engine.engine.model_executor.collective_rpc('close_communicator')
        return {'message': 'Request received, closing communicator'}

    async def infer(
        self,
        infer_requests: List[RolloutInferRequest],
        request_config: Optional[RequestConfig] = None,
        *,
        use_tqdm: Optional[bool] = None,
    ):
        # TODO: split infer_requests into DP size chunks
        chunked_infer_requests = chunk_list(infer_requests, self.args.data_parallel_size)

        # Send the prompts to each worker
        for connection, prompts in zip(connections, chunked_prompts):
            # When the number of prompts is less than data_parallel_size, some workers will receive empty prompts.
            # However, vLLM requires that we always send at least one prompt. So we send a placeholder prompt to comply
            # with vLLM's requirement, and we later ignore the result.
            if not prompts:
                prompts = ["<placeholder>"]
            kwargs = {"prompts": prompts, "sampling_params": sampling_params}
            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})
            
        res = self.infer_engine.infer(infer_requests, request_config, use_tqdm=use_tqdm)
        return res

    def run(self):
        args = self.args
        self.jsonl_writer = JsonlWriter(args.result_path) if args.result_path else None
        logger.info(f'model_list: {self._get_model_list()}')
        uvicorn.run(
            self.app, host=args.host, port=args.port, ssl_keyfile=args.ssl_keyfile, ssl_certfile=args.ssl_certfile)


def deploy_main(args: Union[List[str], DeployArguments, None] = None) -> None:
    SwiftDeploy(args).main()


def is_accessible(port: int):
    infer_client = InferClient(port=port)
    try:
        infer_client.get_model_list()
    except ClientConnectorError:
        return False
    return True


@contextmanager
def run_deploy(args: DeployArguments, return_url: bool = False):
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
    process = mp.Process(target=deploy_main, args=(deploy_args, ))
    process.start()
    try:
        while not is_accessible(deploy_args.port):
            time.sleep(1)
        yield f'http://127.0.0.1:{deploy_args.port}/v1' if return_url else deploy_args.port
    finally:
        process.terminate()
        logger.info('The deployment process has been terminated.')
