# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, Optional, Callable

import torch
import asyncio
from swift.llm import Template, VllmEngine
from .utils import AdapterRequest
from ..protocol import (ChatCompletionResponse, random_uuid)
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
        callback: Callable = None,
    ) -> None:
        assert not use_async_engine  # TODO
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
        self.callback = callback

    async def _infer_full_async(
        self,
        template: Template,
        inputs: Dict[str, Any],
        generation_config: SamplingParams,
        adapter_request: Optional[AdapterRequest] = None,
        **kwargs
    ) -> ChatCompletionResponse:
        request_id = random_uuid()
        max_turns = 3
        current_turn = 1
        infer_request = kwargs.pop('infer_request')
        request_config = kwargs.pop('request_config', None)
        while current_turn <= max_turns:
            result_generator = self._add_request(inputs, generation_config, request_id, adapter_request=adapter_request)

            result = None
            async for result in result_generator:
                pass
            if True:
                loop = asyncio.get_running_loop()
                infer_request.messages.append({'role': 'assistant', 'content': self.tokenizer.decode(tuple(list(result.outputs[0].token_ids)[:-1] + self.tokenizer.encode(result.request_id)))})
                with torch.inference_mode():
                    inputs = await loop.run_in_executor(None, template.encode, infer_request)
                self.set_default_max_tokens(request_config, inputs)
                generation_config = self._prepare_generation_config(request_config)
                self._add_stop_words(generation_config, request_config, template.template_meta)
            current_turn += 1
        return self._create_chat_completion_response(result, template, generation_config, request_id)
