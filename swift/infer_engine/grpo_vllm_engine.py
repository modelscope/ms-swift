# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

from swift.metrics import Metric
from swift.template import Template
from .protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage, InferRequest, RequestConfig,
                       RolloutOutput)
from .utils import AdapterRequest
from .vllm_engine import VllmEngine

try:
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '86400'
    from vllm.lora.request import LoRARequest
except Exception:
    raise


class GRPOVllmEngine(VllmEngine):

    def infer(
        self,
        infer_requests: List[Union[InferRequest, Dict[str, Any]]],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[AdapterRequest] = None,
    ) -> List[RolloutOutput]:
        if not adapter_request and self.enable_lora:
            lora_int_ids = list(self.engine.list_loras())
            if lora_int_ids:
                # since max_lora = 1, pick the first lora
                adapter_request = LoRARequest(
                    lora_name=f'{lora_int_ids[0]}',
                    lora_int_id=lora_int_ids[0],
                    lora_path='dummy_lora_path',
                )

        res = super().infer(
            infer_requests,
            request_config,
            metrics,
            use_tqdm=use_tqdm,
            adapter_request=adapter_request,
        )
        if not isinstance(res, list):
            res = [res]
        for i, result in enumerate(res):
            if not isinstance(result, RolloutOutput):
                if not isinstance(result, ChatCompletionResponse):
                    raise TypeError('Result must be a ChatCompletionResponse or RolloutOutput instance.')
                res[i] = RolloutOutput(response=result)

        return res

    async def async_infer(self,
                          infer_requests: List[InferRequest],
                          request_config: Optional[RequestConfig] = None,
                          metrics: Optional[List[Metric]] = None,
                          *,
                          use_tqdm: Optional[bool] = None,
                          **kwargs) -> List[RolloutOutput]:
        if request_config is None:
            request_config = RequestConfig()
        assert request_config.n == 1

        tasks = [self.infer_async(infer_request, request_config, **kwargs) for infer_request in infer_requests]
        if use_tqdm is None:
            use_tqdm = len(infer_requests) > 1
        res = await self._batch_infer_stream(tasks, request_config.stream, use_tqdm, metrics)

        for i, result in enumerate(res):
            if not isinstance(result, RolloutOutput):
                if not isinstance(result, ChatCompletionResponse):
                    raise TypeError('Result must be a ChatCompletionResponse or RolloutOutput instance.')
                res[i] = RolloutOutput(response=result)

        return res

    async def _batch_infer_stream(self,
                                  tasks,
                                  stream: bool = True,
                                  use_tqdm: bool = True,
                                  metrics: Optional[List[Metric]] = None):
        assert not stream
        prog_bar = tqdm_asyncio(total=len(tasks), dynamic_ncols=True, disable=not use_tqdm)

        async def _new_run(task):
            try:
                res = await task
            except Exception as e:
                if getattr(self, 'strict', True):
                    raise
                res = e
            prog_bar.update()
            self._update_metrics(res, metrics)
            return res

        new_tasks = [_new_run(task) for task in tasks]
        return await self.batch_run(new_tasks)

    def _create_chat_completion_response(self, result, inputs, request_config, request_id) -> ChatCompletionResponse:
        assert result is not None
        num_generated_tokens = sum(len(output.token_ids) for output in result.outputs)
        usage_info = self._get_usage_info(len(result.prompt_token_ids), num_generated_tokens)
        choices = []
        for output in result.outputs:
            output.token_ids = list(output.token_ids)
            response = self.template.decode(output.token_ids)
            logprobs = self._get_logprobs(output.logprobs, output.token_ids, request_config.top_logprobs)
            toolcall = self._get_toolcall(response)

            token_ids = output.token_ids if request_config.return_details else None
            choice = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
                finish_reason=output.finish_reason,
                logprobs=logprobs,
                token_ids=token_ids,
            )
            choices.append(choice)
        prompt_token_ids = None
        images_size = None
        if request_config.return_details:
            prompt_token_ids = result.prompt_token_ids
            images = inputs['template_inputs'].images
            if all(isinstance(image, Image.Image) for image in images):
                images_size = [image.size for image in images]
        return ChatCompletionResponse(
            model=self.model_name,
            choices=choices,
            usage=usage_info,
            id=request_id,
            prompt_token_ids=prompt_token_ids,
            images_size=images_size)

    def _add_adapter(self, adapter_request: Optional[Union[AdapterRequest, LoRARequest]] = None):
        assert self.enable_lora, f'adapter_request: {adapter_request}, self.enable_lora: {self.enable_lora}'
        from vllm.lora.request import LoRARequest
        if isinstance(adapter_request, AdapterRequest):
            return super()._add_adapter(adapter_request)
        elif isinstance(adapter_request, LoRARequest):
            return adapter_request
        else:
            raise ValueError(f'Invalid adapter request: {adapter_request}')
