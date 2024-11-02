# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, List, Optional, Union

from swift.llm import InferRequest
from swift.plugin import Metric
from ..protocol import ChatCompletionResponse, ChatCompletionStreamResponse, RequestConfig


class BaseInferEngine(ABC):

    @abstractmethod
    def infer(self,
              infer_requests: List[InferRequest],
              request_config: Optional[RequestConfig] = None,
              metrics: Optional[List[Metric]] = None,
              *,
              use_tqdm: Optional[bool] = None,
              **kwargs) -> Union[List[ChatCompletionResponse], Iterator[List[Optional[ChatCompletionStreamResponse]]]]:
        """
        This method performs inference on a list of inference requests.

        The method takes a list of inference requests and processes them according to the provided configuration.
        It can optionally use tqdm for progress visualization and accept additional keyword arguments.

        Args:
            infer_requests (List[InferRequest]): A list of inference requests to be processed.
            request_config (Optional[RequestConfig]): Configuration for the request, if any.
            metrics (Optional[List[Metric]]): A list of usage information to return.
            use_tqdm (Optional[bool]): Whether to use tqdm for progress visualization.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[List[ChatCompletionResponse], Iterator[List[Optional[ChatCompletionStreamResponse]]]]:
                The result of the inference.
        """
        pass

    @abstractmethod
    async def infer_async(self,
                          infer_request: InferRequest,
                          request_config: Optional[RequestConfig] = None,
                          **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        """
        This method performs asynchronous inference on a single inference request.

        The method takes an inference request and processes it according to the provided configuration.
        It can accept additional keyword arguments.

        Args:
            infer_request (InferRequest): An inference request to be processed.
            request_config (Optional[RequestConfig]): Configuration for the request, if any.
            **kwargs: Additional keyword arguments.

        Returns:
            Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]: The result of
                the asynchronous inference.
        """
        pass
