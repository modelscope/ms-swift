# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import inspect
import multiprocessing
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from http import HTTPStatus
from threading import Thread
from typing import List, Optional, Union

import json
import uvicorn
from aiohttp import ClientConnectorError
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from swift.llm import AdapterRequest, DeployArguments
from swift.llm.infer.protocol import MultiModalRequestMixin
from swift.plugin import InferStats
from swift.utils import JsonlWriter, get_logger
from .infer import SwiftInfer
from .infer_engine import InferClient
from .protocol import ChatCompletionRequest, CompletionRequest, Model, ModelList

logger = get_logger()


class SwiftDeploy(SwiftInfer):
    args_class = DeployArguments
    args: args_class

    def _register_app(self):
        self.app.get('/v1/models')(self.get_available_models)
        self.app.post('/v1/chat/completions')(self.create_chat_completion)
        self.app.post('/v1/completions')(self.create_completion)

    def __init__(self, args: Union[List[str], DeployArguments, None] = None) -> None:
        super().__init__(args)
        self.infer_engine.strict = True
        self.infer_stats = InferStats()
        self.app = FastAPI(lifespan=self.lifespan)
        self._register_app()

    async def _log_stats_hook(self):
        while True:
            await asyncio.sleep(self.args.log_interval)
            self._compute_infer_stats()
            self.infer_stats.reset()

    def _compute_infer_stats(self):
        global_stats = self.infer_stats.compute()
        for k, v in global_stats.items():
            global_stats[k] = round(v, 8)
        logger.info(global_stats)

    def lifespan(self, app: FastAPI):
        args = self.args
        if args.log_interval > 0:
            thread = Thread(target=lambda: asyncio.run(self._log_stats_hook()), daemon=True)
            thread.start()
        try:
            yield
        finally:
            if args.log_interval > 0:
                self._compute_infer_stats()

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
