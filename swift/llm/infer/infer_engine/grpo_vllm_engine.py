# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import os
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch
import tqdm

from swift.llm import InferRequest, Template, VllmEngine
from swift.plugin import Metric, multi_turns
from swift.plugin.multi_turn import MultiTurnScheduler
from ..protocol import ChatCompletionResponse, RequestConfig, random_uuid
from .utils import AdapterRequest

try:
    # After setting the environment variables, import vllm. This way of writing allows lint to pass.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '3600'
    from vllm import SamplingParams

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
        if use_async_engine:
            self._patch_vllm_init_worker_distributed_environment()

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

        multi_turn_func: Union[MultiTurnScheduler, str] = kwargs.get('multi_turn_func', None)
        if multi_turn_func:
            if isinstance(multi_turn_func, str):
                assert multi_turn_func in multi_turns
                self.multi_turn_scheduler: MultiTurnScheduler = multi_turn_func
            else:
                assert isinstance(multi_turn_func, MultiTurnScheduler)
                self.multi_turn_scheduler: MultiTurnScheduler = multi_turn_func
        else:
            self.multi_turn_scheduler = None

        self.max_turns = kwargs.get('max_turns')  # TODO: check in argument initilization

    def infer(
        self,
        infer_requests: List[InferRequest],
        request_config: Optional[RequestConfig] = None,
        metrics: Optional[List[Metric]] = None,
        *,
        template: Optional[Template] = None,
        use_tqdm: Optional[bool] = None,
        adapter_request: Optional[AdapterRequest] = None,
    ) -> List[ChatCompletionResponse]:
        if self.use_async_engine:
            return self._infer_async(
                infer_requests,
                request_config,
                metrics,
                template=template,
                use_tqdm=use_tqdm,
                adapter_request=adapter_request,
            )
        else:
            return super().infer(
                infer_requests,
                request_config,
                metrics,
                template=template,
                use_tqdm=use_tqdm,
                adapter_request=adapter_request,
            )

    def _infer_async(self,
                     infer_requests: List[InferRequest],
                     request_config: Optional[RequestConfig] = None,
                     metrics: Optional[List[Metric]] = None,
                     *,
                     use_tqdm: Optional[bool] = None,
                     **kwargs) -> List[ChatCompletionResponse]:
        if request_config is None:
            request_config = RequestConfig()
        # in GRPO n always equals 1
        assert request_config.n == 1

        async def _infer_async_single(infer_request: InferRequest,
                                      request_config: Optional[RequestConfig] = None,
                                      current_turn: int = 1,
                                      **kwargs):
            current_request = infer_request

            while True:
                result = await self.infer_async(current_request, request_config, **kwargs)
                should_stop = self.multi_turn_scheduler.check_finished(current_request, result, current_turn)

                if should_stop:
                    return result

                current_request = self.multi_turn_scheduler.step(current_request, result, current_turn)

        tasks = [_infer_async_single(infer_request, request_config, **kwargs) for infer_request in infer_requests]
        if use_tqdm is None:
            use_tqdm = not request_config.stream and len(infer_requests) > 1
        return self._batch_infer_stream(tasks, request_config.stream, use_tqdm, metrics)

    def _patch_vllm_init_worker_distributed_environment(self):
        """Patched version of init_worker_distributed_environment.
        Behavior:
            1. Reads WORLD_SIZE from environment variables
            2. Temporarily overrides parallel_config.world_size
            3. Calls original initialization
            4. Restores original world_size
        """

        import vllm.envs as envs
        from functools import wraps

        vllm_use_v1 = envs.VLLM_USE_V1
        if vllm_use_v1:
            from vllm.v1.worker.gpu_worker import init_worker_distributed_environment as original_init
        else:
            from vllm.worker.worker import init_worker_distributed_environment as original_init

        @wraps(original_init)
        def patched_init(
            vllm_config,
            rank: int,
            distributed_init_method: Optional[str] = None,
            local_rank: int = -1,
        ) -> None:
            world_size = os.environ.get('WORLD_SIZE')
            if world_size is not None and world_size.isdigit():
                world_size = int(world_size)
                original_world_size = vllm_config.parallel_config.world_size

                try:
                    vllm_config.parallel_config.world_size = world_size
                    return original_init(vllm_config, rank, distributed_init_method, local_rank)
                finally:
                    vllm_config.parallel_config.world_size = original_world_size
            else:
                return original_init(vllm_config, rank, distributed_init_method, local_rank)

        if vllm_use_v1:
            from vllm.v1.worker import gpu_worker
            gpu_worker.init_worker_distributed_environment = patched_init
        else:
            from vllm.worker import worker
            worker.init_worker_distributed_environment = patched_init
