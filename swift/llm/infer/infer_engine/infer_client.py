# Copyright (c) Alibaba, Inc. and its affiliates.

import asyncio
from copy import deepcopy
from dataclasses import asdict
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import aiohttp
import json
from dacite import from_dict
from requests.exceptions import HTTPError

from swift.plugin import Metric
from ..protocol import (ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse, InferRequest,
                        ModelList, RequestConfig)
from .infer_engine import InferEngine


class InferClient(InferEngine):

    def __init__(self,
                 host: str = '127.0.0.1',
                 port: str = '8000',
                 api_key: str = 'EMPTY',
                 *,
                 timeout: Optional[int] = None) -> None:
        self.api_key = api_key
        self.host = host
        self.port = port
        self.timeout = timeout
        self.models = []
        for model in self.get_model_list().data:
            self.models.append(model.id)
        assert len(self.models) > 0, f'self.models: {self.models}'

    def get_model_list(self, *, url: Optional[str] = None) -> ModelList:
        return asyncio.run(self.get_model_list_async(url=url))

    def _get_request_kwargs(self) -> Dict[str, Any]:
        request_kwargs = {}
        if isinstance(self.timeout, int) and self.timeout >= 0:
            request_kwargs['timeout'] = self.timeout
        if self.api_key is not None:
            request_kwargs['headers'] = {'Authorization': f'Bearer {self.api_key}'}
        return request_kwargs

    async def get_model_list_async(self, *, url: Optional[str] = None) -> ModelList:
        if url is None:
            url = f'http://{self.host}:{self.port}/v1/models'
        async with aiohttp.ClientSession() as session:
            async with session.get(url, **self._get_request_kwargs()) as resp:
                resp_obj = await resp.json()
        return from_dict(ModelList, resp_obj)

    def infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        model: Optional[str] = None,
        url: Optional[str] = None,
        use_tqdm: Optional[bool] = None
    ) -> Union[List[ChatCompletionResponse], Iterator[List[Optional[ChatCompletionStreamResponse]]]]:
        return super().infer(infer_requests, request_config, metrics, model=model, url=url, use_tqdm=use_tqdm)

    @staticmethod
    def _prepare_request_data(model: str, infer_request: InferRequest, request_config: RequestConfig) -> Dict[str, Any]:
        return asdict(ChatCompletionRequest(model, **asdict(infer_request), **asdict(request_config)))

    @staticmethod
    def _parse_stream_data(data: bytes) -> Optional[str]:
        data = data.decode(encoding='utf-8')
        data = data.strip()
        if len(data) == 0:
            return
        assert data.startswith('data:'), f'data: {data}'
        return data[5:].strip()

    async def infer_async(
            self,
            infer_request: InferRequest,
            request_config: Optional[RequestConfig] = None,
            *,
            model: Optional[str] = None,
            url: Optional[str] = None) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        request_config = deepcopy(request_config or RequestConfig())
        if model is None:
            if len(self.models) == 1:
                model = self.models[0]
            else:
                raise ValueError(f'Please explicitly specify the model. Available models: {self.models}.')
        if url is None:
            url = f'http://{self.host}:{self.port}/v1/chat/completions'

        request_data = self._prepare_request_data(model, infer_request, request_config)
        if request_config.stream:

            async def _gen_stream() -> AsyncIterator[ChatCompletionStreamResponse]:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=request_data, **self._get_request_kwargs()) as resp:
                        async for data in resp.content:
                            data = self._parse_stream_data(data)
                            if data == '[DONE]':
                                break
                            if data is not None:
                                resp_obj = json.loads(data)
                                if resp_obj['object'] == 'error':
                                    raise HTTPError(resp_obj['message'])
                                yield from_dict(ChatCompletionStreamResponse, resp_obj)

            return _gen_stream()
        else:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=request_data, **self._get_request_kwargs()) as resp:
                    resp_obj = await resp.json()
                    if resp_obj['object'] == 'error':
                        raise HTTPError(resp_obj['message'])
                    return from_dict(ChatCompletionResponse, resp_obj)
