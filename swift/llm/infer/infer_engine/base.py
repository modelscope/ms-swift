# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, List, Union

from ..protocol import ChatCompletionResponse, ChatCompletionStreamResponse


class BaseInferEngine(ABC):

    @abstractmethod
    def infer(self, *args,
              **kwargs) -> Union[List[ChatCompletionResponse], Iterator[List[ChatCompletionStreamResponse]]]:
        pass

    @abstractmethod
    async def infer_async(self, *args,
                          **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        pass
