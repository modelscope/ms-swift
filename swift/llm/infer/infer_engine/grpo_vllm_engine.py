# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from copy import copy, deepcopy
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from packaging import version

from swift.llm import InferRequest, Template, VllmEngine, get_model_tokenizer
from swift.plugin import Metric
from ..protocol import ChatCompletionResponse, ChatCompletionStreamResponse, RequestConfig
from .patch import patch_auto_config, patch_auto_tokenizer
from .utils import AdapterRequest, patch_vllm_memory_leak

try:
    # After setting the environment variables, import vllm. This way of writing allows lint to pass.
    os.environ['VLLM_USE_V1'] = os.environ.get('VLLM_USE_V1', '0')
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '3600'
    import vllm
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, EngineArgs, LLM
except Exception:
    raise


class GRPOVllmEngine(VllmEngine):

    def __init__(
        self,
        model_id_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        *,
        use_async_engine: bool = True,
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
        disable_custom_all_reduce: bool = False,
        enforce_eager: bool = False,
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,
        device: str = 'auto',
        # lora
        enable_lora: bool = False,
        max_loras: int = 1,
        max_lora_rank: int = 16,
        enable_prefix_caching: bool = False,
        num_infer_workers: int = 1,
        enable_sleep_mode: bool = False,
        distributed_executor_backend: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        patch_vllm_memory_leak()
        self.use_async_engine = use_async_engine
        self.processor = get_model_tokenizer(
            model_id_or_path,
            torch_dtype,
            load_model=False,
            download_model=True,
            model_type=model_type,
            use_hf=use_hf,
            hub_token=hub_token,
            revision=revision)[1]
        self._post_init()

        self._prepare_engine_kwargs(
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            disable_custom_all_reduce=disable_custom_all_reduce,
            enforce_eager=enforce_eager,
            limit_mm_per_prompt=limit_mm_per_prompt,
            enable_lora=enable_lora,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
            enable_prefix_caching=enable_prefix_caching,
            device=device,
            distributed_executor_backend=distributed_executor_backend,
            enable_sleep_mode=enable_sleep_mode,
            engine_kwargs=engine_kwargs,
        )
        self._prepare_engine()
        self._load_generation_config()

    def _prepare_engine(self) -> None:
        with patch_auto_tokenizer(self.tokenizer), patch_auto_config(self.config):
            engine = LLM(**self.engine_args.__dict__)
        self.engine = engine

    @property
    def inner_model(self):
        return self.engine.llm_engine.model_executor.driver_worker.model_runner.model

    @property
    def inner_model_executor(self):
        return self.engine.llm_engine.model_executor

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

        prompts = []
        for inputs in batched_inputs:
            llm_inputs = {'prompt_token_ids': inputs['input_ids']}
            mm_data = {}
            for key in ['images', 'audios', 'videos']:
                media_data = inputs.get(key) or []
                if media_data:
                    if version.parse(vllm.__version__) < version.parse('0.6'):
                        assert len(media_data) == 1, (
                            f'The current version of vllm only supports single {key}. Please upgrade to vllm >= 0.6.0')
                        mm_data = {key.rstrip('s'): media_data[0]}
                    else:
                        mm_data = {key.rstrip('s'): media_data[0] if len(media_data) == 1 else media_data}
            if mm_data:
                llm_inputs['multi_modal_data'] = mm_data
            prompts.append(llm_inputs)

        generation_configs = []
        seed = request_config.seed
        assert seed >= 0, 'Seed is needed for GRPOVllmEngine.'
        for i, _ in enumerate(prompts):
            request_config = copy(request_config)
            request_config.seed = seed + i
            generation_config = self._prepare_generation_config(request_config)
            self._add_stop_words(generation_config, request_config, template.template_meta)
            generation_configs.append(generation_config)
        outputs = self.engine.generate(prompts, generation_configs, use_tqdm=False)
        return [
            self._create_chat_completion_response(result, template, generation_configs[0], '') for result in outputs
        ]
