# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, List, Optional, Union
import asyncio
from copy import deepcopy

import torch
from tqdm.asyncio import tqdm_asyncio

from swift.llm import InferRequest, RolloutInferRequest, Template, VllmEngine
from swift.plugin import Metric
from swift.plugin.env import Env, envs
from swift.plugin.context_manager import context_managers
from ..protocol import ChatCompletionResponse, ChatMessage, RequestConfig, RolloutResponseChoice, GymRolloutResponseChoice
from .utils import AdapterRequest

try:
    # After setting the environment variables, import vllm. This way of writing allows lint to pass.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '86400'

except Exception:
    raise


class GymVllmEngine(VllmEngine):

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

        # Context manager setup
        context_manager_name = kwargs.get('context_manager', 'dummyContextManager')
        if context_manager_name not in context_managers:
            raise ValueError(f"Context manager '{context_manager_name}' not found in context_managers registry. "
                             f"Available: {list(context_managers.keys())}")

        self.context_manager = context_managers[context_manager_name]()
        self.max_turns = kwargs.get('max_turns', 3)

    def _create_env(self, env_config) -> Env:
        """Create environment instance."""
        if env_config.get('name', 'math_env') not in envs:
            raise ValueError(
                f"Environment '{env_config.get('name', None)}' not found in envs registry. Available: {list(envs.keys())}"
            )
        return envs[env_config.get('name', 'math_env')]()

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
        assert request_config.n == 1

        # change here, gym environment loop
        async def _infer_async_single(infer_request: Union[RolloutInferRequest, Dict[str, Any]],
                                      request_config: Optional[RequestConfig] = None,
                                      **kwargs):
            if isinstance(infer_request, Dict):
                infer_request = RolloutInferRequest(**infer_request)
            
            # Create environment
            env_config = infer_request.data_dict.get('env_config', {})
            env = self._create_env(env_config)
            
            try:
                # Environment reset
                observation, info, system_message = await env.reset(infer_request)
                
                # Initialize conversation
                messages = []
                if system_message:
                    messages.append({'role': 'system', 'content': system_message})
                messages.append({'role': 'user', 'content': observation})
                
                current_request = deepcopy(infer_request)
                current_request.messages = messages
                current_turn = 1
                done = False
                total_reward = 0.0
                step_rewards = []
                trajectory_id = f"{id(infer_request)}_{hash(str(infer_request))}"
                
                while True:
                    # Apply context management
                    current_request = self.context_manager.manage_context(current_request, trajectory_id)
                    
                    # Remove any previous assistant response for generation
                    InferRequest.remove_response(current_request.messages)
                    
                    # Generate LLM response using parent's infer_async
                    result: ChatCompletionResponse = await self.infer_async(current_request, request_config, **kwargs)
                    result_choice: RolloutResponseChoice = result.choices[0]
                    
                    completion = result_choice.message.content
                    messages.append({'role': 'assistant', 'content': completion})
                    
                    # Environment step
                    next_observation, reward, done, step_info = await env.step(result_choice)
                    
                    # Accumulate rewards
                    total_reward += reward
                    step_rewards.append(reward)
                    
                    if done or current_turn>self.max_turns:
                        break
                    messages.append({'role': 'user', 'content': next_observation})
                    current_request.messages = messages
                    
                    current_turn += 1
                
                # Create final result with gym-specific information
                final_choice = GymRolloutResponseChoice(
                    index=result_choice.index,
                    message=result_choice.message,
                    finish_reason=result_choice.finish_reason,
                    logprobs=result_choice.logprobs,
                    messages=messages,
                    trajectory_id=trajectory_id,
                    total_reward=total_reward,
                    step_rewards=step_rewards
                )
                
                return ChatCompletionResponse(
                    model=self.model_name,
                    choices=[final_choice],
                    usage=result.usage,
                    id=f"gym_{trajectory_id}"
                )
                
            finally:
                # Clean up environment
                await self._close_env_async(env)

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

    async def _close_env_async(self, env: Env):
        """Asynchronously close environment."""
        try:
            if hasattr(env, 'close') and asyncio.iscoroutinefunction(env.close):
                await env.close()
            elif hasattr(env, 'close'):
                env.close()
        except Exception:
            pass

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
            choice = GymRolloutResponseChoice(
                index=output.index,
                message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),
                finish_reason=output.finish_reason,
                logprobs=logprobs,
            )
            choices.append(choice)
        return ChatCompletionResponse(model=self.model_name, choices=choices, usage=usage_info, id=request_id)