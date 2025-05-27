# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from copy import copy, deepcopy
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
import tqdm
from packaging import version

from swift.llm import InferRequest, Template, VllmEngine
from swift.plugin import Metric
from ..protocol import ChatCompletionResponse, ChatCompletionStreamResponse, RequestConfig
from .utils import AdapterRequest

try:
    # After setting the environment variables, import vllm. This way of writing allows lint to pass.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '3600'
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

    def infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[AdapterRequest] = None,
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
        request_config = deepcopy(request_config or RequestConfig())
        if template is None:
            template = self.default_template
        template.set_mode('vllm')
        batched_inputs, error_list = self._batch_encode(
            infer_requests, template=template, strict=getattr(self, 'strict', True))
        self.set_default_max_tokens(request_config, batched_inputs)
        request_id_list = []
        for inputs in batched_inputs:
            request_id = str(self._request_count)
            request_id_list.append(request_id)
            self._request_count += 1
            generation_config = self._prepare_generation_config(request_config)
            self._add_stop_words(generation_config, request_config, template.template_meta)
            self._add_request(inputs, generation_config, request_id, adapter_request=adapter_request)
        prog_bar = tqdm(total=len(batched_inputs), dynamic_ncols=True, disable=not use_tqdm)
        outputs = {}
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs[output.request_id] = output
                    prog_bar.update()
        prog_bar.close()
        outputs = [outputs[request_id] for request_id in request_id_list]
        res = [
            self._create_chat_completion_response(result, template, generation_config, request_id)
            for request_id, result in zip(request_id_list, outputs)
        ]
        self._update_metrics(res, metrics)
        return res
