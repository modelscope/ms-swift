# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch
from tqdm.asyncio import tqdm_asyncio

from swift.llm import InferRequest, RolloutInferRequest, Template, VllmEngine
from swift.plugin import Metric, multi_turns
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler
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

        # Get sampling controller configurations from kwargs
        multi_turn_scheduler = kwargs.get('multi_turn_scheduler', None)
        use_gym_env = kwargs.get('use_gym_env', False)
        if use_gym_env:
            self.gym_env = kwargs.get('gym_env', None)
            self.context_manager = kwargs.get('context_manager', None)
        # Ensure mutual exclusivity of sampling controllers
        if use_gym_env and multi_turn_scheduler is not None:
            raise ValueError('gym_env and multi_turn_scheduler are mutually exclusive sampling controllers')

        self.use_gym_env = use_gym_env

        # Initialize sampling controller
        if use_gym_env:
            self.multi_turn_scheduler = None
        elif multi_turn_scheduler is not None:
            if isinstance(multi_turn_scheduler, str):
                assert multi_turn_scheduler in multi_turns
                self.multi_turn_scheduler: MultiTurnScheduler = multi_turns[multi_turn_scheduler](
                    template=template, max_turns=self.max_turns)
            else:
                assert isinstance(multi_turn_scheduler, MultiTurnScheduler)
                self.multi_turn_scheduler: MultiTurnScheduler = multi_turn_scheduler
        else:
            self.multi_turn_scheduler = None

    def _create_env(self, env_config: Dict) -> Env:
        """Create environment instance for gym sampling."""
        env_name = env_config.get('name', None)
        if not env_name:
            env_name = self.gym_env
        if env_name not in envs:
            raise ValueError((f"Environment '{env_name}' not found in envs registry. "
                              f'Available: {list(envs.keys())}'))
        return envs[env_name](env_config)

    def _create_context_manager(self, ctx_config: Dict) -> ContextManager:
        """Create context manager for gym sampling."""
        ctx_name = ctx_config.get('name', None)
        if not ctx_name:
            ctx_name = self.context_manager

        if not ctx_name:
            ctx_name = 'dummyContextManager'

        if ctx_name not in context_managers:
            raise ValueError((f"Context manager '{ctx_name}' not found in registry. "
                              f'Available: {list(context_managers.keys())}'))
        return context_managers[ctx_name](ctx_config)

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
        assert not self.use_async_engine, 'for Async Engine, use infer_async instead'
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
                    raise TypeError("Result must be a ChatCompletionResponse or RolloutOutput instance.")                
                res[i] = RolloutOutput(results=result)

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

    def _create_chat_completion_response(self, result: 'ChatCompletionResponse', template: Template, request_config,
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
            choice_cls = ChatCompletionResponseChoice

            token_ids = template.skip_stop_tokens(output.token_ids) if request_config.return_details else None
            choice = choice_cls(
                index=output.index,
                message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
                finish_reason=output.finish_reason,
                logprobs=logprobs,
                token_ids=token_ids,
            )
            choices.append(choice)
        prompt_token_ids = result.prompt_token_ids if request_config.return_details else None
        return ChatCompletionResponse(
            model=self.model_name, choices=choices, usage=usage_info, id=request_id, prompt_token_ids=prompt_token_ids)
