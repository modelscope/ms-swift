# Copyright (c) Alibaba, Inc. and its affiliates.
import os
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
                 port: int = 8000,
                 api_key: str = 'EMPTY',
                 *,
                 base_url: Optional[str] = None,
                 timeout: Optional[int] = 86400) -> None:
        """
        Initialize the InferClient.

        Args:
            host (str): The hostname of the inference server. Defaults to '127.0.0.1'.
            port (str): The port of the inference server. Defaults to '8000'.
            api_key (str): The API key for authentication. Defaults to 'EMPTY'.
            timeout (Optional[int]): The timeout for requests in seconds. Defaults to None.
        """
        self.api_key = api_key
        self.host = host
        self.port = port
        self.timeout = timeout
        if base_url is None:
            base_url = f'http://{self.host}:{self.port}/v1'
        self.base_url = base_url
        self._models = None

    @property
    def models(self):
        if self._models is None:
            models = []
            for model in self.get_model_list().data:
                models.append(model.id)
            assert len(models) > 0, f'models: {models}'
            self._models = models
        return self._models

    def get_model_list(self) -> ModelList:
        """Get model list from the inference server.
        """
        coro = self.get_model_list_async()
        return self.safe_asyncio_run(coro)

    def _get_request_kwargs(self) -> Dict[str, Any]:
        request_kwargs = {}
        if isinstance(self.timeout, (int, float)) and self.timeout > 0:
            request_kwargs['timeout'] = self.timeout
        if self.api_key is not None:
            request_kwargs['headers'] = {'Authorization': f'Bearer {self.api_key}'}
        return request_kwargs

    async def get_model_list_async(self) -> ModelList:
        url = f"{self.base_url.rstrip('/')}/models"
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
            use_tqdm: Optional[bool] = None
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
        """
        Perform inference using the specified model.

        Args:
            infer_requests (List[InferRequest]): A list of inference requests.
            request_config (Optional[RequestConfig]): Configuration for the request. Defaults to None.
            metrics (Optional[List[Metric]]): The usage information to return. Defaults to None.
            model (Optional[str]): The model name to be used for inference. Defaults to None.
            use_tqdm (Optional[bool]): Whether to use tqdm for progress tracking. Defaults to None.

        Returns:
            List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
            The inference responses or an iterator of streaming responses.
        """
        return super().infer(infer_requests, request_config, metrics, model=model, use_tqdm=use_tqdm)

    @staticmethod
    def _prepare_request_data(model: str, infer_request: InferRequest, request_config: RequestConfig) -> Dict[str, Any]:
        if not isinstance(infer_request, dict):
            infer_request = asdict(infer_request)
        res = asdict(ChatCompletionRequest(model, **infer_request, **asdict(request_config)))
        # ignore empty
        empty_request = ChatCompletionRequest('', [])
        for k in list(res.keys()):
            if res[k] == getattr(empty_request, k):
                res.pop(k)
        return res

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
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        request_config = deepcopy(request_config or RequestConfig())
        if model is None:
            if len(self.models) == 1:
                model = self.models[0]
            else:
                raise ValueError(f'Please explicitly specify the model. Available models: {self.models}.')
        url = f"{self.base_url.rstrip('/')}/chat/completions"

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
