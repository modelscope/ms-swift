# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import AsyncIterator, Iterator, List, Optional, Union

from swift.plugin import Metric
from ..protocol import ChatCompletionResponse, ChatCompletionStreamResponse, InferRequest, RequestConfig
from .base import BaseInferEngine


class InferClient(BaseInferEngine):

    def __init__(self,
                 model: str,
                 host: str = '127.0.0.1',
                 port: str = '8000',
                 api_key: str = 'EMPTY',
                 *,
                 url: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key

        if url is not None:
            url = f'http://{host}:{port}/v1/chat/completions'
        self.url = url

    @staticmethod
    def get_model_list():
        pass

    @staticmethod
    async def get_model_list_async():
        pass

    def infer(self,
              infer_requests: List[InferRequest],
              request_config: Optional[RequestConfig] = None,
              metrics: Optional[List[Metric]] = None,
              *,
              use_tqdm: Optional[bool] = None,
              **kwargs) -> Union[List[ChatCompletionResponse], Iterator[List[Optional[ChatCompletionStreamResponse]]]]:
        pass

    async def infer_async(self, *args,
                          **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        pass
