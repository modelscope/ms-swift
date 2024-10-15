# Copyright (c) Alibaba, Inc. and its affiliates.

import asyncio
import time
from abc import ABC, abstractmethod
from queue import Empty, Queue
from threading import Thread
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Tuple, Union

import torch
from tqdm import tqdm

from swift.plugin import Metric
from ..model import get_default_torch_dtype, get_model_tokenizer
from ..template import Template
from .protocol import ChatCompletionResponse, ChatCompletionStreamResponse, InferRequest, RequestConfig


class InferEngine(ABC):

    def _prepare_model_tokenizer(self,
                                 model_id_or_path: str,
                                 torch_dtype: Optional[torch.dtype],
                                 load_model: bool,
                                 *,
                                 model_type: Optional[str] = None,
                                 **kwargs) -> None:
        use_hf = kwargs.pop('use_hf', None)
        revision = kwargs.pop('revision', None)
        model, tokenizer = get_model_tokenizer(
            model_id_or_path,
            torch_dtype,
            load_model=load_model,
            model_type=model_type,
            download_model=True,
            use_hf=use_hf,
            revision=revision)
        config = tokenizer.config
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.torch_dtype = config.torch_dtype

        self.model_type = tokenizer.model_type
        self.model_dir = tokenizer.model_dir
        self.is_multimodal = tokenizer.is_multimodal
        self.is_moe = tokenizer.is_moe
        self.chat_template = tokenizer.chat_template
        self.generation_template = tokenizer.generation_template

    @staticmethod
    async def _run_infer(i, task, queue, stream: bool = False):
        # task with queue
        if stream:
            async for stream_response in await task:
                queue.put((i, stream_response))
        else:
            queue.put((i, await task))
        queue.put((i, None))

    @staticmethod
    async def _batch_run(tasks):
        return await asyncio.gather(*tasks)

    @staticmethod
    def _infer_stream(tasks,
                      stream: bool = True,
                      use_tqdm: bool = True,
                      metrics: Optional[List[Metric]] = None) -> Iterator[List[ChatCompletionStreamResponse]]:
        if metrics is None:
            metrics = []
        for metric in metrics:
            metric.reset()

        queue = Queue()
        new_tasks = [InferEngine._run_infer(i, task, queue, stream) for i, task in enumerate(tasks)]
        thread = Thread(target=lambda: asyncio.run(InferEngine._batch_run(new_tasks)))
        thread.start()

        prog_bar = tqdm(total=len(new_tasks), dynamic_ncols=True, disable=not use_tqdm)
        n_finished = 0
        outputs = [None] * len(new_tasks)

        while n_finished < len(new_tasks):
            i, output = queue.get()
            if output is None:  # is_finished
                n_finished += 1
                prog_bar.update()
            else:
                if outputs[i] is not None:  # The logic will only apply to the stream.
                    yield outputs
                    outputs = [None] * len(new_tasks)
                outputs[i] = output
            for metric in metrics:
                metric.update(output)
        yield outputs

    @staticmethod
    def _infer(tasks, use_tqdm: bool = True, metrics: Optional[List[Metric]] = None) -> List[ChatCompletionResponse]:
        for outputs in InferEngine._infer_stream(tasks, False, use_tqdm, metrics):
            pass
        return outputs

    @abstractmethod
    async def infer_async(self,
                          template: Template,
                          infer_request: InferRequest,
                          request_config: Optional[RequestConfig] = None,
                          *,
                          request_id: Optional[str] = None,
                          **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        pass

    @torch.inference_mode()
    def infer(self,
              template: Template,
              infer_requests: List[InferRequest],
              request_config: Optional[RequestConfig] = None,
              *,
              use_tqdm: Optional[bool] = None,
              metrics: Optional[List[Metric]] = None,
              **kwargs) -> Union[List[ChatCompletionResponse], Iterator[List[ChatCompletionStreamResponse]]]:
        tasks = [
            self.infer_async(template, infer_request, request_config, **kwargs) for infer_request in infer_requests
        ]
        if use_tqdm is None:
            use_tqdm = not request_config.stream
        if request_config.stream:
            return self._infer_stream(tasks, True, use_tqdm, metrics)
        else:
            return self._infer(tasks, use_tqdm, metrics)
