# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, List, Optional, Union
import asyncio
from copy import deepcopy
import math

import torch
from tqdm.asyncio import tqdm_asyncio

from swift.llm import InferRequest, RolloutInferRequest, Template, VllmEngine
from swift.plugin import Metric
from swift.plugin.env import Env, envs
from swift.plugin.context_manager import ContextManager, context_managers
from ..protocol import ChatCompletionResponse, ChatMessage, RequestConfig, RolloutResponseChoice,GymRolloutResponseChoice
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
        use_gym_engine: bool = False,
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
            use_gym_engine=use_gym_engine,
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

        # Environment and context manager configuration from kwargs
        
        # Context manager setup - use dummy if not specified
        context_manager_name = kwargs.get('context_manager', 'dummyContextManager')
        if context_manager_name not in context_managers:
            raise ValueError(f"Context manager '{context_manager_name}' not found in context_managers registry. "
                           f"Available: {list(context_managers.keys())}")
        
        self.context_manager = context_managers[context_manager_name]()
        self.max_turns = kwargs.get('max_turns', 10)
        self.num_generations = kwargs.get('num_generations', 1)
        
        # Dynamic sampling parameters
        self.dynamic_sample = kwargs.get('dynamic_sample', False)
        self.max_resample_times = kwargs.get('max_resample_times', 5)
        self.variance_threshold = 1e-4  # If variance < this, resample

    def _create_env(self, env_config) -> Env:
        """Create environment instance."""
        if env_config.get("name",None) not in envs:
            raise ValueError(f"Environment '{env_config.get("name",None)}' not found in envs registry. Available: {list(envs.keys())}")
        return envs[env_config.get("name")](env_config)
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
        assert not self.use_gym_engine, 'for Async Engine, use infer_async instead'
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

        # Gym environment interaction for multiple trajectories per query with dynamic sampling
        async def _infer_async_single_query(infer_request: Union[RolloutInferRequest, Dict[str, Any]],
                                           request_config: Optional[RequestConfig] = None,
                                           **kwargs):
            if isinstance(infer_request, Dict):
                infer_request = RolloutInferRequest(**infer_request)
            
            # Dynamic sampling loop
            resample_count = 0
            while resample_count <= self.max_resample_times:
                # Sample multiple trajectories for this query
                results = []
                
                # Create multiple environments for parallel trajectory sampling
                env_config = infer_request.data_dict.get('env_config', {})

                envs = []
                try:
                    # Create independent environment instances for each trajectory
                    for i in range(self.num_generations):
                        env = self._create_env(env_config)
                        envs.append(env)
                    
                    # Sample multiple trajectories in parallel
                    trajectory_tasks = []
                    for i, env in enumerate(envs):
                        trajectory_id = f"{id(infer_request)}_{i}_{resample_count}"
                        request_copy = deepcopy(infer_request)
                        task = self._sample_single_trajectory(env, request_copy, trajectory_id, request_config, **kwargs)
                        trajectory_tasks.append(task)
                    
                    # Wait for all trajectories to complete
                    results = await asyncio.gather(*trajectory_tasks)
                    
                finally:
                    # Clean up all environments asynchronously
                    close_tasks = []
                    for env in envs:
                        close_tasks.append(self._close_env_async(env))
                    if close_tasks:
                        await asyncio.gather(*close_tasks, return_exceptions=True)
                
                # Check if dynamic sampling is needed
                if not self.dynamic_sample or resample_count >= self.max_resample_times:
                    return results
                
                # Extract rewards for variance check
                rewards = []
                for result in results:
                    if hasattr(result.choices[0], 'total_reward'):
                        rewards.append(result.choices[0].total_reward)
                    else:
                        # Fallback: assume reward is 0 if not available
                        rewards.append(0.0)
                
                # Calculate variance
                if len(rewards) > 1:
                    mean_reward = sum(rewards) / len(rewards)
                    variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
                    
                    if variance > self.variance_threshold:
                        # Good variance, return results
                        return results
                
                # Variance too low, resample
                resample_count += 1
                if resample_count <= self.max_resample_times:
                    print(f"Resampling trajectory for query (attempt {resample_count}/{self.max_resample_times}), "
                          f"variance: {variance if len(rewards) > 1 else 'N/A'}")
            
            # Max retries reached, return last results
            return results

        # Create tasks for all queries
        tasks = [_infer_async_single_query(infer_request, request_config, **kwargs) for infer_request in infer_requests]
        
        # Process with progress bar using the same pattern as GRPOVllmEngine
        if use_tqdm is None:
            use_tqdm = len(infer_requests) > 1
        
        results = await self._batch_infer_stream(tasks, use_tqdm, metrics)
        
        # Flatten results since each query returns multiple trajectories
        all_results = []
        for query_results in results:
            if isinstance(query_results, Exception):
                raise query_results
            all_results.extend(query_results)
        
        return all_results

    async def _close_env_async(self, env: Env):
        """Asynchronously close environment."""
        try:
            if hasattr(env, 'close') and asyncio.iscoroutinefunction(env.close):
                await env.close()
            elif hasattr(env, 'close'):
                # Fallback for sync close
                env.close()
        except Exception:
            pass  # Ignore cleanup errors

    async def _batch_infer_stream(self,
                                  tasks,
                                  use_tqdm: bool = True,
                                  metrics: Optional[List[Metric]] = None):
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

    async def _sample_single_trajectory(self, 
                                      env: Env, 
                                      infer_request: RolloutInferRequest,
                                      trajectory_id: str,
                                      request_config: RequestConfig,
                                      **kwargs) -> ChatCompletionResponse:
        """Sample a single trajectory using the given environment."""
        
        # Environment reset (async)
        observation, info, system_message = await env.reset(infer_request)
        
        # Initialize conversation with system message and initial observation
        messages = []
        if system_message:
            messages.append({'role': 'system', 'content': system_message})
        messages.append({'role': 'user', 'content': observation})
        
        current_turn = 1
        done = False
        total_reward = 0.0  # Track cumulative reward
        step_rewards = []   # Track individual step rewards
        
        while not done and current_turn <= self.max_turns:
            # Apply context management
            current_request = RolloutInferRequest(
                messages=messages.copy(),
                **infer_request.data_dict
            )
            
            current_request = self.context_manager.manage_context(current_request,trajectory_id)
            
            # Remove any previous assistant response for generation
            InferRequest.remove_response(current_request.messages)
            
            # Generate LLM response using parent's async_infer
            result: List[ChatCompletionResponse] = await super().async_infer([current_request], request_config, **kwargs)
            result_choice: RolloutResponseChoice = result[0].choices[0]  # async_infer returns list
            
            # Environment step
            next_observation, reward, done, step_info = await env.step(result_choice) # TODO 在step_info中记录每个reward func的信息作为日志
            
            # Accumulate rewards
            total_reward += reward
            step_rewards.append(reward)
            
            # Update conversation history
            messages.append({'role': 'assistant', 'content': result_choice.message.content})
            
            if not done:
                messages.append({'role': 'user', 'content': next_observation})
            
            current_turn += 1
        
        # Set final messages and trajectory info in result choice
        result_choice = GymRolloutResponseChoice(
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
            choices=[result_choice],
            usage=result[0].usage,
            id=f"gym_{trajectory_id}"
        )

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