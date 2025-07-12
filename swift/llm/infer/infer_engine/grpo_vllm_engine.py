# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, List, Optional, Union

import torch
from tqdm.asyncio import tqdm_asyncio

from swift.llm import InferRequest, RolloutInferRequest, Template, VllmEngine
from swift.plugin import Metric, multi_turns
from swift.plugin.multi_turn import MultiTurnScheduler
from ..protocol import ChatCompletionResponse, ChatMessage, RequestConfig, RolloutResponseChoice
from .utils import AdapterRequest

try:
    # After setting the environment variables, import vllm. This way of writing allows lint to pass.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '86400'

except Exception:
    raise

# TODO: check the sleep issue with AsyncEngine: https://github.com/vllm-project/vllm/issues/17103


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
        self.max_turns = kwargs.get('max_turns')

        multi_turn_scheduler: Union[MultiTurnScheduler, str] = kwargs.get('multi_turn_scheduler', None)
        if multi_turn_scheduler:
            if isinstance(multi_turn_scheduler, str):
                assert multi_turn_scheduler in multi_turns
                self.multi_turn_scheduler: MultiTurnScheduler = multi_turns[multi_turn_scheduler](
                    template=template, max_turns=self.max_turns)
            else:
                assert isinstance(multi_turn_scheduler, MultiTurnScheduler)
                self.multi_turn_scheduler: MultiTurnScheduler = multi_turn_scheduler
        else:
            self.multi_turn_scheduler = None

    def infer(
        self,
        infer_requests: List[Union[InferRequest, Dict[str, Any]]],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[AdapterRequest] = None,
    ) -> List[ChatCompletionResponse]:
        assert not self.use_async_engine, 'for Async Engine, use infer_async instead'
        return super().infer(
            infer_requests,
            request_config,
            metrics,
            template=template,
            use_tqdm=use_tqdm,
            adapter_request=adapter_request,
        )

    async def async_infer(self,
                          infer_requests: List[Union[RolloutInferRequest, Dict[str, Any]]],
                          request_config: Optional[RequestConfig] = None,
                          metrics: Optional[List[Metric]] = None,
                          *,
                          use_tqdm: Optional[bool] = None,
                          **kwargs) -> List[ChatCompletionResponse]:
        if request_config is None:
            request_config = RequestConfig()
        # in GRPO n always equals 1
        assert request_config.n == 1

        # change here, multi turn loop
        async def _infer_async_single(infer_request: Union[RolloutInferRequest, Dict[str, Any]],
                                      request_config: Optional[RequestConfig] = None,
                                      **kwargs):
            if isinstance(infer_request, Dict):
                infer_request = RolloutInferRequest(**infer_request)
            current_request = infer_request
            current_turn = 1
            while True:
                messages = current_request.messages
                if current_turn == 1 or not messages[-1]['content']:
                    # If it's the first turn or the last message content is empty(dummy), remove the response
                    InferRequest.remove_response(messages)

                result: ChatCompletionResponse = await self.infer_async(current_request, request_config, **kwargs)
                result_choice: RolloutResponseChoice = result.choices[0]

                completion = result_choice.message.content
                if messages[-1]['role'] == 'assistant':
                    messages[-1]['content'] += completion
                else:
                    messages.append({'role': 'assistant', 'content': completion})

                if self.multi_turn_scheduler:
                    should_stop = self.multi_turn_scheduler.check_finished(current_request, result_choice, current_turn)
                else:
                    should_stop = True

                if self.max_turns:
                    should_stop = should_stop or (current_turn >= self.max_turns)

                if should_stop:
                    result_choice.messages = messages
                    return result

                current_request = self.multi_turn_scheduler.step(current_request, result_choice, current_turn)
                assert isinstance(current_request, RolloutInferRequest)
                if current_request.messages[-1]['role'] == 'assistant':
                    # NOTE: engine will discard last response during inference
                    # https://github.com/modelscope/ms-swift/blob/v3.5.1/swift/llm/template/base.py#L416-L419
                    # To allow the engine to continue generating content, add a dummy response here.
                    current_request.messages.append({'role': 'assistant', 'content': None})

                current_turn += 1

        tasks = [_infer_async_single(infer_request, request_config, **kwargs) for infer_request in infer_requests]
        if use_tqdm is None:
            use_tqdm = len(infer_requests) > 1
        return await self._batch_infer_stream(tasks, request_config.stream, use_tqdm, metrics)

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

    def _create_chat_completion_response(self, result, template: Template, generation_config,
                                         request_id) -> ChatCompletionResponse:
        assert result is not None
        num_generated_tokens = sum(len(output.token_ids) for output in result.outputs)
        usage_info = self._get_usage_info(len(result.prompt_token_ids), num_generated_tokens)
        choices = []
        for output in result.outputs:
            output.token_ids = list(output.token_ids)
            response = template.decode(output.token_ids)
            logprobs = self._get_logprobs(output.logprobs, output.token_ids, generation_config.top_logprobs)
            toolcall = self._get_toolcall(response, template)
            choice = RolloutResponseChoice(
                index=output.index,
                message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
                finish_reason=output.finish_reason,
                logprobs=logprobs)
            choices.append(choice)
        return ChatCompletionResponse(model=self.model_name, choices=choices, usage=usage_info, id=request_id)
