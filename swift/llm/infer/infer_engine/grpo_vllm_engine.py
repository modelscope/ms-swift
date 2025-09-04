# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

from swift.llm import InferRequest, Template, VllmEngine
from swift.plugin import Metric
from ..protocol import ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage, RequestConfig, RolloutOutput
from .utils import AdapterRequest

try:
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '86400'
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
        enable_expert_parallel: bool = False,
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 256,
        disable_custom_all_reduce: bool = True,
        enforce_eager: bool = False,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        task_type: Optional[str] = None,
        disable_cascade_attn: bool = False,
        load_format: str = 'auto',
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
            enable_expert_parallel=enable_expert_parallel,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            disable_custom_all_reduce=disable_custom_all_reduce,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
            seed=seed,
            task_type=task_type,
            disable_cascade_attn=disable_cascade_attn,
            load_format=load_format,
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

    def infer(
        self,
        infer_requests: List[Union[InferRequest, Dict[str, Any]]],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[AdapterRequest] = None,
    ) -> List[RolloutOutput]:
        res = super().infer(
            infer_requests,
            request_config,
            metrics,
            template=template,
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
                          **kwargs) -> List[ChatCompletionResponse]:
        if request_config is None:
            request_config = RequestConfig()
        assert request_config.n == 1

        tasks = [self.infer_async(infer_request, request_config, **kwargs) for infer_request in infer_requests]
        if use_tqdm is None:
            use_tqdm = len(infer_requests) > 1
        return self._batch_infer_stream(tasks, request_config.stream, use_tqdm, metrics)

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

    def _create_chat_completion_response(self, result, inputs, template: Template, request_config,
                                         request_id) -> ChatCompletionResponse:
        assert result is not None
        num_generated_tokens = sum(len(output.token_ids) for output in result.outputs)
        usage_info = self._get_usage_info(len(result.prompt_token_ids), num_generated_tokens)
        choices = []
        for output in result.outputs:
            output.token_ids = list(output.token_ids)
            response = template.decode(output.token_ids)
            logprobs = self._get_logprobs(output.logprobs, output.token_ids, request_config.top_logprobs)
            toolcall = self._get_toolcall(response, template)

            token_ids = template.skip_stop_tokens(output.token_ids) if request_config.return_details else None
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
