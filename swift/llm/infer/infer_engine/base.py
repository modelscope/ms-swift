# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, List, Optional, Union

from swift.plugin import Metric
from ..protocol import ChatCompletionResponse, ChatCompletionStreamResponse, InferRequest, RequestConfig


class BaseInferEngine(ABC):

    @abstractmethod
    def infer(self,
              infer_requests: List[InferRequest],
              request_config: Optional[RequestConfig] = None,
              metrics: Optional[List[Metric]] = None,
              *,
              use_tqdm: Optional[bool] = None,
              **kwargs) -> Union[List[ChatCompletionResponse], Iterator[List[Optional[ChatCompletionStreamResponse]]]]:
        pass

    @abstractmethod
    async def infer_async(self,
                          infer_request: InferRequest,
                          request_config: Optional[RequestConfig] = None,
                          **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        pass
