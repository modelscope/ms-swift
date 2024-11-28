# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from threading import Thread
from typing import Any, Dict, List, Optional, Union

import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from swift.llm import TEMPLATE_MAPPING, DeployArguments, Template, merge_lora
from swift.plugin import InferStats
from swift.utils import get_logger
from .infer import SwiftInfer
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
        self.infer_states = InferStats()
        self.app = FastAPI(lifespan=self.lifespan)
        self._register_app()

    async def _log_stats_hook(self, log_interval: int):
        while True:
            await asyncio.sleep(log_interval)
            global_stats = self.infer_states.compute()
            self.infer_states.reset()
            for k, v in global_stats.items():
                global_stats[k] = round(v, 8)
            logger.info(global_stats)

    def lifespan(self, app: FastAPI):
        args = self.args
        if args.log_interval > 0:
            thread = Thread(target=lambda: asyncio.run(self._log_stats_hook(args.log_interval)))
            thread.start()
        yield

    async def get_available_models(self):
        args = self.args
        model_list = [args.served_model_name or args.model_suffix]
        if args.lora_request_list is not None:
            model_list += [lora_request.lora_name for lora_request in args.lora_request_list]
        data = [Model(id=model_id, owned_by=args.owned_by) for model_id in model_list]
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
        if isinstance(request.top_logprobs, int) and request.top_logprobs > self.args.max_logprobs:
            return (f'The value of top_logprobs({request.top_logprobs}) is greater than '
                    f'the server\'s max_logprobs({args.max_logprobs}).')

    @staticmethod
    def create_error_response(status_code: Union[int, str, HTTPStatus], message: str) -> JSONResponse:
        status_code = int(status_code)
        return JSONResponse({'message': message, 'object': 'error'}, status_code)

    def _post_process(self, request_info, response, return_cmpl_response: bool = False):
        args = self.args
        if args.log_interval > 0:
            self.infer_states.update(response)
        if return_cmpl_response:
            response = response.to_cmpl_response()

        if self.jsonl_writer:
            data = {'response': asdict(response), **request_info}
            self.jsonl_writer.append(data)
        return response

    @contextmanager
    def patch_infer_engine(self, infer_request, request_info):
        # log generation_config
        verbose = self.args.verbose
        _origin_add_stop_words = self.infer_engine.__class__._add_stop_words

        def _add_stop_words(self, generation_config, *args, **kwargs):
            res = _origin_add_stop_words(self, generation_config, *args, **kwargs)
            printable_infer_request = infer_request.to_printable()
            request_info.update({'infer_request': printable_infer_request, 'generation_config': generation_config})
            if verbose:
                logger.info(request_info)
            return res

        self.infer_engine.__class__._add_stop_words = _add_stop_words
        try:
            yield
        finally:
            self.infer_engine.__class__._add_stop_words = _origin_add_stop_words

    async def create_chat_completion(self,
                                     request: ChatCompletionRequest,
                                     raw_request: Request,
                                     *,
                                     return_cmpl_response: bool = False):
        error_msg = (await self._check_model(request) or self._check_api_key(raw_request)
                     or self._check_max_logprobs(request))
        if error_msg:
            return self.create_error_response(HTTPStatus.BAD_REQUEST, error_msg)

        infer_request, request_config = request.parse()
        request_info = {}
        try:
            with self.patch_infer_engine(infer_request, request_info):
                res_or_gen = await self.infer_async(infer_request, request_config, template=self.template)
        except ValueError as e:
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
        uvicorn.run(
            self.app, host=args.host, port=args.port, ssl_keyfile=args.ssl_keyfile, ssl_certfile=args.ssl_certfile)


def deploy_main(args: Union[List[str], DeployArguments, None] = None) -> None:
    SwiftDeploy(args).main()
