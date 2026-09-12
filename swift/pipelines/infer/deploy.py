# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import base64
import binascii
import inspect
import json
import multiprocessing
import os
import re
import tempfile
import time
import uvicorn
from aiohttp import ClientConnectorError
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from http import HTTPStatus
from threading import Thread
from typing import Any, List, Optional, Union
from urllib.parse import urlsplit

from swift.arguments import DeployArguments, InferArguments
from swift.infer_engine import AdapterRequest, InferClient, RequestConfig
from swift.infer_engine.protocol import (ChatCompletionRequest, CompletionRequest, EmbeddingRequest, Model, ModelList,
                                         MultiModalRequestMixin, RolloutInferRequest)
from swift.metrics import InferStats
from swift.utils import JsonlWriter, SafeMediaPath, SafeUrlFetcher, get_logger
from .infer import SwiftInfer

logger = get_logger()


class SwiftDeploy(SwiftInfer):
    args_class = DeployArguments
    args: args_class

    @staticmethod
    def get_infer_engine(args: InferArguments, template=None, **kwargs):
        if isinstance(args, DeployArguments) and args.infer_backend == 'vllm':
            engine_kwargs = (kwargs.get('engine_kwargs') or {}).copy()
            if args.vllm_data_parallel_size > 1:
                if not args.vllm_use_async_engine:
                    raise ValueError('vLLM data parallel requires `vllm_use_async_engine=True` in deploy mode.')
                engine_kwargs.setdefault('data_parallel_size', args.vllm_data_parallel_size)
                logger.info(f'Enable vLLM data parallel with size {args.vllm_data_parallel_size}.')
            if args.max_logprobs is not None:
                engine_kwargs['max_logprobs'] = args.max_logprobs
            kwargs['engine_kwargs'] = engine_kwargs
        return SwiftInfer.get_infer_engine(args, template, **kwargs)

    def _register_app(self):
        self.app.get('/health')(self.health)
        self.app.get('/health/')(self.health)
        self.app.get('/ping')(self.ping)
        self.app.post('/ping')(self.ping)
        self.app.get('/v1/models')(self.get_available_models)
        self.app.post('/v1/chat/completions')(self.create_chat_completion)
        self.app.post('/v1/completions')(self.create_completion)
        self.app.post('/v1/embeddings')(self.create_embedding)
        self.app.post('/infer/')(self.infer_handler)

    def __init__(self, args: Optional[Union[List[str], DeployArguments]] = None) -> None:
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

    async def health(self) -> Response:
        """Health check endpoint."""
        if self.infer_engine is not None:
            return Response(status_code=200)
        else:
            return Response(status_code=503)

    async def ping(self) -> Response:
        """Ping check endpoint. Required for SageMaker compatibility."""
        return await self.health()

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

    @staticmethod
    def _materialize_media(value: Any, media_type: str, temp_dir: str) -> Any:
        if isinstance(value, (list, tuple)):
            item_type = 'image' if media_type == 'video' else media_type
            return [SwiftDeploy._materialize_media(item, item_type, temp_dir) for item in value]
        if isinstance(value, dict):
            if 'url' in value:
                source_key = 'url'
            elif value.get('bytes'):
                source_key = 'bytes'
            elif 'path' in value:
                source_key = 'path'
            elif 'bytes' in value:
                source_key = 'bytes'
            else:
                raise ValueError(f'Refusing media object {value!r}: expected a url, bytes, or path field.')
            value[source_key] = SwiftDeploy._materialize_media(value[source_key], media_type, temp_dir)
            return value
        if not isinstance(value, str):
            return value

        stripped = value.strip()
        match = re.match(r'(https?)://', stripped, re.IGNORECASE)
        if match:
            url = match.group(1).lower() + stripped[len(match.group(1)):]
            timeout = float(os.getenv('SWIFT_TIMEOUT', '20'))
            request_kwargs = {'timeout': timeout} if timeout > 0 else {}
            media_bytes = SafeUrlFetcher.read(url, **request_kwargs)
            suffix = os.path.splitext(urlsplit(url).path)[1]
            if not re.fullmatch(r'\.[A-Za-z0-9]{1,10}', suffix):
                suffix = {'image': '.jpg', 'audio': '.wav', 'video': '.mp4'}[media_type]
            with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=suffix, delete=False) as f:
                f.write(media_bytes)
                return f.name

        scheme = urlsplit(stripped).scheme.lower()
        if scheme == 'data':
            return value
        if scheme:
            raise ValueError(f'Refusing media URI {value!r}: only http://, https://, and data: are supported.')
        if os.path.isfile(value):
            return SafeMediaPath.check(value)
        try:
            base64.b64decode(stripped, validate=True)
            return value
        except (ValueError, binascii.Error):
            return SafeMediaPath.check(value)

    @staticmethod
    @contextmanager
    def _prepare_request_media(infer_request):
        with tempfile.TemporaryDirectory(prefix='swift-media-') as temp_dir:
            for media_type in ('image', 'audio', 'video'):
                key = f'{media_type}s'
                values = getattr(infer_request, key, None)
                if values:
                    setattr(infer_request, key,
                            [SwiftDeploy._materialize_media(value, media_type, temp_dir) for value in values])

            for message in infer_request.messages:
                content = message.get('content')
                if not isinstance(content, list):
                    continue
                for item in content:
                    key = item.get('type', '')
                    media_type = key[:-len('_url')] if key.endswith('_url') else key
                    if media_type not in {'image', 'audio', 'video'} or key not in item:
                        continue
                    item[key] = SwiftDeploy._materialize_media(item[key], media_type, temp_dir)
            yield infer_request

    def _post_process(self, request_info, response, return_cmpl_response: bool = False):
        args = self.args

        for i in range(len(response.choices)):
            if not hasattr(response.choices[i], 'message') or not isinstance(response.choices[i].message.content,
                                                                             (tuple, list)):
                continue
            for j, content in enumerate(response.choices[i].message.content):
                if isinstance(content, dict) and content['type'] == 'image':
                    b64_image = MultiModalRequestMixin.to_base64(content['image'])
                    response.choices[i].message.content[j]['image'] = f'data:image/jpg;base64,{b64_image}'

        is_finished = all(response.choices[i].finish_reason for i in range(len(response.choices)))
        if 'stream' in response.__class__.__name__.lower():
            request_info['response'] += response.choices[0].delta.content or ''
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
        media_stack = ExitStack()
        try:
            media_stack.enter_context(self._prepare_request_media(infer_request))
            res_or_gen = await self.infer_async(infer_request, request_config, **infer_kwargs)
        except asyncio.CancelledError:
            media_stack.close()
            raise
        except Exception as e:
            media_stack.close()
            import traceback
            logger.info(traceback.format_exc())
            return self.create_error_response(HTTPStatus.BAD_REQUEST, str(e))
        if request_config.stream:

            async def _gen_wrapper():
                try:
                    async for res in res_or_gen:
                        res = self._post_process(request_info, res, return_cmpl_response)
                        yield f'data: {json.dumps(asdict(res), ensure_ascii=False)}\n\n'
                    yield 'data: [DONE]\n\n'
                finally:
                    media_stack.close()

            return StreamingResponse(_gen_wrapper(), media_type='text/event-stream')
        try:
            if hasattr(res_or_gen, 'choices'):
                # instance of ChatCompletionResponse
                return self._post_process(request_info, res_or_gen, return_cmpl_response)
            return res_or_gen
        finally:
            media_stack.close()

    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        chat_request = ChatCompletionRequest.from_cmpl_request(request)
        return await self.create_chat_completion(chat_request, raw_request, return_cmpl_response=True)

    async def create_embedding(self, request: EmbeddingRequest, raw_request: Request):
        chat_request = ChatCompletionRequest.from_cmpl_request(request)
        return await self.create_chat_completion(chat_request, raw_request, return_cmpl_response=True)

    async def infer_handler(self, raw_request: Request):
        error_msg = self._check_api_key(raw_request)
        if error_msg:
            return self.create_error_response(HTTPStatus.BAD_REQUEST, error_msg)
        body = await raw_request.json()
        infer_requests = [RolloutInferRequest(**r) for r in body.get('infer_requests', [])]
        rc_data = body.get('request_config')
        request_config = RequestConfig(**rc_data) if rc_data else RequestConfig()
        if request_config.stream:
            return self.create_error_response(HTTPStatus.BAD_REQUEST, '`/infer/` does not support streaming requests.')
        with ExitStack() as media_stack:
            for infer_request in infer_requests:
                media_stack.enter_context(self._prepare_request_media(infer_request))
            return await asyncio.gather(*[self.infer_async(req, request_config) for req in infer_requests])

    def _warn_if_unauthenticated(self):
        """Warn when the service is reachable from other hosts with authentication disabled.

        Such a service lets anyone who can route to the port run inference, and a chat request can ask the
        server to fetch a media URL on the caller's behalf, so it should not be exposed as-is.
        """
        args = self.args
        if args.api_key is not None or args.host in {'127.0.0.1', 'localhost', '::1'}:
            return
        logger.warning(f'The server is listening on {args.host}:{args.port} without an API key, so anyone able '
                       'to reach this port can use it. Pass `--api_key` to require one, and `--host 127.0.0.1` '
                       'to accept local connections only (put a gateway in front for remote access).')

    def run(self):
        args = self.args
        self.jsonl_writer = JsonlWriter(args.result_path) if args.result_path else None
        logger.info(f'model_list: {self._get_model_list()}')
        self._warn_if_unauthenticated()
        uvicorn.run(
            self.app,
            host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            log_level=args.log_level)


def deploy_main(args: Optional[Union[List[str], DeployArguments]] = None) -> None:
    SwiftDeploy(args).main()


def is_accessible(port: int):
    infer_client = InferClient(port=port)
    try:
        infer_client.get_model_list()
    except ClientConnectorError:
        return False
    return True


def _deploy_main(args):
    args._import_external_plugins()
    return deploy_main(args)


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
    process = mp.Process(target=_deploy_main, args=(deploy_args, ))
    process.start()
    try:
        while not is_accessible(deploy_args.port):
            time.sleep(1)
        yield f'http://127.0.0.1:{deploy_args.port}/v1' if return_url else deploy_args.port
    finally:
        process.terminate()
        logger.info('The deployment process has been terminated.')
