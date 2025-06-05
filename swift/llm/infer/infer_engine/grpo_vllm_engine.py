# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import os
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch
import tqdm

from swift.llm import InferRequest, Template, VllmEngine
from swift.plugin import Metric, multi_turns
from ..protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
                        ChatCompletionStreamResponse, ChatMessage, DeltaMessage, RequestConfig, random_uuid)
# from ..protocol import ChatCompletionResponse, random_uuid
from .utils import AdapterRequest

try:
    # After setting the environment variables, import vllm. This way of writing allows lint to pass.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '3600'
    from vllm import SamplingParams

except Exception:
    raise


class GRPOVllmEngine(VllmEngine):

    def __init__(
        self,
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        *,
        use_async_engine: bool = False,
        model_type: Optional[str] = None,
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        revision: Optional[str] = None,
        # engine_kwargs
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 256,
        disable_custom_all_reduce: bool = True,
        enforce_eager: bool = False,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        device: str = 'auto',
        seed: Optional[int] = None,
        # lora
        enable_lora: bool = False,
        max_loras: int = 1,
        max_lora_rank: int = 16,
        enable_prefix_caching: bool = False,
        enable_sleep_mode: bool = False,
        distributed_executor_backend: Optional[str] = None,
        quantization: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        template: Optional[Template] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_id_or_path=model_id_or_path,
            torch_dtype=torch_dtype,
            use_async_engine=use_async_engine,
            model_type=model_type,
            use_hf=use_hf,
            hub_token=hub_token,
            revision=revision,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            disable_custom_all_reduce=disable_custom_all_reduce,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
            device=device,
            seed=seed,
            enable_lora=enable_lora,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
            enable_prefix_caching=enable_prefix_caching,
            enable_sleep_mode=enable_sleep_mode,
            distributed_executor_backend=distributed_executor_backend,
            quantization=quantization,
            engine_kwargs=engine_kwargs,
            template=template,
        )

        multi_turn: Union[Callable, str] = kwargs.get('multi_turn_func', None)
        if isinstance(multi_turn, str):
            assert multi_turn in multi_turns
            self.multi_turn_scheduler = multi_turn
        else:
            from swift.plugin.multi_turn import FunctionScheduler
            assert callable(multi_turn)
            self.multi_turn_scheduler = FunctionScheduler(multi_turn)

        self.is_multi_turn = True  # TODO

    async def _infer_full_async(
        self,
        template: Template,
        inputs: Dict[str, Any],
        generation_config: SamplingParams,
        adapter_request: Optional[AdapterRequest] = None,
    ) -> ChatCompletionResponse:
        request_id = random_uuid()
        result_generator = self._add_request(inputs, generation_config, request_id, adapter_request=adapter_request)
        result = None
        async for result in result_generator:
            pass
        return self._create_chat_completion_response(result, template, generation_config, request_id)

    async def infer_async(
        self,
        infer_request: InferRequest,
        request_config: Optional[RequestConfig] = None,
        *,
        template: Optional[Template] = None,
        adapter_request: Optional[AdapterRequest] = None,
        pre_infer_hook=None,
    ) -> Union[ChatCompletionResponse]:
        if not self.is_multi_turn:
            return await self.infer_async(
                infer_request,
                request_config,
                template=template,
                adapter_request=adapter_request,
                pre_infer_hook=pre_infer_hook)
        if not self.use_async_engine:
            raise ValueError('If you want to use `infer_async`, you need to pass `use_async_engine` as True.')
        template.set_mode('vllm')
        async def _single_round_infer(current_request: InferRequest) -> ChatCompletionResponse:
            nonlocal template, request_config, adapter_request, pre_infer_hook
            current_config = deepcopy(request_config or RequestConfig())

            if template is None:
                template = self.default_template

            loop = asyncio.get_running_loop()
            with torch.inference_mode():
                inputs = await loop.run_in_executor(None, template.encode, current_request)

            self.set_default_max_tokens(current_config, inputs)
            generation_config = self._prepare_generation_config(current_config)
            self._add_stop_words(generation_config, current_config, template.template_meta)

            kwargs = {
                'template': template,
                'inputs': inputs,
                'generation_config': generation_config,
                'adapter_request': adapter_request,
            }
            if pre_infer_hook:
                kwargs = pre_infer_hook(kwargs)

            return await self._infer_full_async(**kwargs)

        if not self.is_multi_turn:
            return await _single_round_infer(infer_request)

        history = []
        current_request = infer_request

        for turn in range(self.multi_turn_scheduler.max_turns):
            response = await _single_round_infer(current_request)
            history.append(response)

            should_stop = self.multi_turn_scheduler.check_finished(response)

            if should_stop:
                break

            current_request = self._prepare_next_turn_request(
                original_request=infer_request, current_request=current_request, response=response, history=history)

    def _batch_infer_stream(
        self,
        tasks,
        stream: bool = True,
        use_tqdm: bool = True,
        metrics: Optional[List[Metric]] = None
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
        """
        Code from swift.llm.infer.InferEngine.
        Modify the _new_run method to enable multi-round rollout,
        which is used only when use_async_engine is set to true.

        Call stack:
        self.infer -> InferEngine.infer -> vLLMEngine.infer_async (template encode here)
        -> vLLMEngine._infer_full_async -> vLLMEngine._add_request -> self.engine.generate
        -> self._batch_infer_stream (this func)

        TODO: pass the whole inputs to judge
        """
        assert self.use_async_engine
        prog_bar = tqdm(total=len(tasks), dynamic_ncols=True, disable=not use_tqdm)

        async def _new_run(task):
            try:
                res = await task
            except Exception as e:
                if getattr(self, 'strict', True):
                    raise
                res = e

            if isinstance(res, Exception) or self.multi_turn_scheduler.check_finished(res):
                prog_bar.update()
                self._update_metrics(res, metrics)
            else:
                result = self.multi_turn_scheduler.step(res)
                new_inputs = None  # TODO
                self._add_request
            return res

        new_tasks = [_new_run(task) for task in tasks]
        return self.safe_asyncio_run(self.batch_run(new_tasks))

    def _add_request(self,
                     inputs: Dict[str, Any],
                     generation_config: SamplingParams,
                     request_id: str,
                     adapter_request: Optional[AdapterRequest] = None):
        if not self.use_async_engine:
            return super()._add_request(inputs, generation_config, request_id, adapter_request)
        kwargs = {}
        if self.enable_lora and adapter_request:
            from vllm.lora.request import LoRARequest
            adapter_name = adapter_request.name
            adapter_path = adapter_request.path
            if adapter_name in self._adapters_pool:
                kwargs['lora_request'] = self._adapters_pool[adapter_name]
            else:
                kwargs['lora_request'] = LoRARequest(
                    lora_name=adapter_name, lora_path=adapter_path, lora_int_id=len(self._adapters_pool) + 1)
                self._adapters_pool[adapter_name] = kwargs['lora_request']
        input_ids = inputs['input_ids']
        llm_inputs = {'prompt_token_ids': input_ids}
        mm_data = {}
        for key in ['images', 'audios', 'videos']:
            media_data = inputs.get(key) or []
            if media_data:
                if self._version_ge('0.6'):
                    mm_data = {key.rstrip('s'): media_data[0] if len(media_data) == 1 else media_data}
                else:
                    assert len(media_data) == 1, (
                        f'The current version of vllm only supports single {key}. Please upgrade to vllm >= 0.6.0')
                    mm_data = {key.rstrip('s'): media_data[0]}
        if mm_data:
            llm_inputs['multi_modal_data'] = mm_data

        return self.engine.generate(llm_inputs, generation_config, request_id, **kwargs)

    # def async_infer(self,
    #           infer_requests: List[InferRequest],
    #           request_config: Optional[RequestConfig] = None,
    #           metrics: Optional[List[Metric]] = None,
    #           *,
    #           use_tqdm: Optional[bool] = None,
    #           **kwargs) -> List[ChatCompletionResponse]:
    #     if request_config is None:
    #         request_config = RequestConfig()
    #     tasks = [self.infer_async(infer_request, request_config, **kwargs) for infer_request in infer_requests]
    #     if use_tqdm is None:
    #         use_tqdm = not request_config.stream and len(infer_requests) > 1
    #     return self._batch_infer_stream(tasks, request_config.stream, use_tqdm, metrics)

    # def _batch_infer_stream(
    #     self,
    #     tasks,
    #     stream: bool = True,
    #     use_tqdm: bool = True,
    #     metrics: Optional[List[Metric]] = None
    # ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:

    #     prog_bar = tqdm(total=len(tasks), dynamic_ncols=True, disable=not use_tqdm)
    #     if stream:
    #         return [self.async_iter_to_iter(task, prog_bar, metrics) for task in tasks]
    #     else:

    #         async def _new_run(task):
    #             try:
    #                 res = await task
    #             except Exception as e:
    #                 if getattr(self, 'strict', True):
    #                     raise
    #                 res = e
    #             prog_bar.update()
    #             self._update_metrics(res, metrics)
    #             return res

    #         new_tasks = [_new_run(task) for task in tasks]
    #         return self.safe_asyncio_run(self.batch_run(new_tasks))

    # async def generate_async(self, prompts, sampling_params):
    #     from vllm.utils import random_uuid

    #     request_id = random_uuid()
    #     results_generator = self.engine.generate(prompts, sampling_params, request_id)
    #     final_output = None
    #     async for request_output in results_generator:
    #         final_output = request_output
    #     return final_output

    # async def get_responses(self):
    #     """
    #     Synchronously get all completed agent results from the queue.
    #     Waits for all tasks to complete before returning results.
    #     Returns: List of all completed agent results.
    #     """
    #     # Get all results from the queue
    #     results = []
    #     while not self.result_queue.empty():
    #         try:
    #             results.append(await self.result_queue.get())
    #         except asyncio.QueueEmpty:
    #             break
    #     return results
