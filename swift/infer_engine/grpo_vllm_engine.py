# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import inspect
import os
import pickle
import torch
from collections import Counter
from copy import deepcopy
from PIL import Image
from tqdm.asyncio import tqdm_asyncio
from typing import Any, Dict, List, Optional, Union

from swift.loss_scale import LossScale
from swift.metrics import Metric
from swift.rlhf_trainers.utils import VLLM_LORA_INT_ID, VLLM_LORA_NAME, VLLM_LORA_PATH
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

    def _get_encode_indices(self, infer_requests):
        indices = list(range(len(infer_requests)))
        template = self.template
        # Reuse only deterministic text encoding with the standard Swift template.
        if (type(template) is not Template or template.template_backend != 'swift'
                or template.template_meta.template_type == 'dummy' or type(template.loss_scale) is not LossScale):
            return indices
        tokenizer = template.tokenizer
        if not tokenizer.is_fast or getattr(tokenizer.backend_tokenizer.model, 'dropout', None):
            return indices

        seen = {}
        for i, request in enumerate(infer_requests):
            if request.images or request.audios or request.videos or request.objects or request.tools:
                continue
            if any(message['role'] not in ('system', 'user', 'assistant')
                   or not isinstance(message['content'], (str, type(None))) or message.get('tool_calls')
                   for message in request.messages):
                continue
            # Preserve nested values, their types and field order; UUID is restored per request.
            key = pickle.dumps({k: v for k, v in vars(request).items() if k != 'uuid'})
            indices[i] = seen.setdefault(key, i)
        return indices

    def _get_grpo_lora_request(self, adapter_request: Optional[AdapterRequest] = None):
        if not adapter_request and self.enable_lora:
            lora_loaded = VLLM_LORA_INT_ID in self.engine.list_loras()
            if lora_loaded:
                adapter_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH)
        return adapter_request

    async def _get_grpo_lora_request_async(self, adapter_request: Optional[AdapterRequest] = None):
        if not adapter_request and self.enable_lora:
            loaded_loras = self.engine.list_loras()
            if inspect.isawaitable(loaded_loras):
                loaded_loras = await loaded_loras
            lora_loaded = VLLM_LORA_INT_ID in loaded_loras
            if lora_loaded:
                adapter_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH)
        return adapter_request

    def infer(
        self,
        infer_requests: List[Union[InferRequest, Dict[str, Any]]],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[AdapterRequest] = None,
    ) -> List[RolloutOutput]:
        adapter_request = self._get_grpo_lora_request(adapter_request)

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

        if not infer_requests:
            return []
        # Resolve once per batch, including when no adapter is loaded.
        kwargs['adapter_request'] = await self._get_grpo_lora_request_async(kwargs.get('adapter_request'))
        kwargs['_lora_request_resolved'] = True
        indices = self._get_encode_indices(infer_requests)
        counts = Counter(indices)
        encodings = {}

        async def infer_shared(infer_request, index):
            if index not in encodings:
                self.template.set_mode('vllm')
                loop = asyncio.get_running_loop()
                encodings[index] = loop.run_in_executor(None, self.template.encode, infer_requests[index], True)
            # Cancellation of one generation must not cancel its siblings' encoding.
            inputs = deepcopy(await asyncio.shield(encodings[index]))
            if 'uuid' in vars(infer_request):
                inputs['template_inputs'].extra_kwargs['uuid'] = infer_request.uuid
            return await self.infer_async(infer_request, request_config, _inputs=inputs, **kwargs)

        tasks = [
            infer_shared(request, index) if counts[index] > 1 else self.infer_async(request, request_config, **kwargs)
            for request, index in zip(infer_requests, indices)
        ]
        if use_tqdm is None:
            use_tqdm = len(infer_requests) > 1
        res = await self._batch_infer_stream(tasks, request_config.stream, use_tqdm, metrics)

        for i, result in enumerate(res):
            if not isinstance(result, RolloutOutput):
                if not isinstance(result, ChatCompletionResponse):
                    raise TypeError('Result must be a ChatCompletionResponse or RolloutOutput instance.')
                res[i] = RolloutOutput(response=result)

        return res

    async def infer_async(
        self,
        infer_request: InferRequest,
        request_config: Optional[RequestConfig] = None,
        *,
        adapter_request: Optional[AdapterRequest] = None,
        pre_infer_hook=None,
        _lora_request_resolved: bool = False,
        _inputs=None,
    ):
        if not _lora_request_resolved:
            adapter_request = await self._get_grpo_lora_request_async(adapter_request)
        return await super().infer_async(
            infer_request,
            request_config,
            adapter_request=adapter_request,
            pre_infer_hook=pre_infer_hook,
            _inputs=_inputs,
        )

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
            response = self.template.decode_generate_ids(output.token_ids, template_inputs=inputs['template_inputs'])
            logprobs = self._get_logprobs(output.logprobs, output.token_ids, request_config.top_logprobs)
            toolcall = self._get_toolcall(response, template_inputs=inputs['template_inputs'])

            token_ids = output.token_ids if request_config.return_details else None
            choice = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
                finish_reason=output.finish_reason,
                logprobs=logprobs,
                token_ids=token_ids,
                routed_experts=getattr(output, 'routed_experts', None))
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
