"""模块功能概述：
该模块实现基于 vLLM 引擎的 GRPO（Group Relative Policy Optimization）推理引擎 `GRPOVllmEngine`，
用于强化学习场景下的多轮对话推理。提供：
- 继承自 VllmEngine 的高性能推理能力；
- 支持多轮对话调度（Multi-turn Scheduler）和 Gym 环境采样控制器；
- 支持轨迹生成、奖励累积和上下文管理；
- 异步推理接口，支持批量并发处理；
- 与 Swift 框架和强化学习训练流程的无缝集成。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，标注代码版权所有者
import asyncio  # 引入 asyncio 模块，用于异步编程和协程调度
import os  # 引入 os 模块，用于环境变量设置和文件路径操作
from copy import deepcopy  # 引入 deepcopy，用于对对象进行深拷贝，避免副作用
from typing import Any, Dict, List, Optional, Union  # 引入类型注解，明确参数与返回值类型

import torch  # 引入 PyTorch 库，用于张量操作和数据类型定义
from tqdm.asyncio import tqdm_asyncio  # 引入异步进度条，用于显示批量异步任务的进度

from swift.llm import InferRequest, RolloutInferRequest, Template, VllmEngine  # 引入 Swift 框架的推理请求、轨迹推理请求、模板和 vLLM 引擎
from swift.llm.infer.protocol import MultiModalRequestMixin  # 引入多模态请求混合类，用于处理图像、音频等多模态数据
from swift.plugin import Metric, multi_turns  # 引入指标采集器和多轮对话调度器注册表
from swift.plugin.context_manager import ContextManager, context_managers  # 引入上下文管理器和上下文管理器注册表
from swift.plugin.env import Env, envs  # 引入 Gym 环境接口和环境注册表
from swift.plugin.multi_turn import MultiTurnScheduler  # 引入多轮对话调度器基类
from ..protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatMessage, GymRolloutResponseChoice,  # 引入协议层数据类：响应、选项、消息
                        RequestConfig, RolloutResponseChoice)  # 引入请求配置、轨迹响应选项
from .utils import AdapterRequest  # 引入适配器请求工具类

try:  # 尝试设置 vLLM 环境变量（必须在导入 vLLM 之前设置）
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'  # 设置 vLLM 工作进程的多进程方法为 'spawn'（更安全，避免 fork 问题）
    os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '86400'  # 设置 vLLM 引擎迭代超时时间为 86400 秒（1 天，用于长时间推理任务）
except Exception:  # 若设置环境变量失败（通常不会发生）
    raise  # 抛出异常，阻止程序继续运行


class GRPOVllmEngine(VllmEngine):

    """类功能：
    定义基于 vLLM 的 GRPO 推理引擎类，继承 VllmEngine
    GRPOVllmEngine 负责基于 vLLM 引擎进行 GRPO 强化学习场景下的多轮对话推理。
    
    - 角色：推理引擎封装，继承 VllmEngine 的高性能推理能力，扩展支持多轮对话和强化学习采样；
    - 能力：支持异步推理、Gym 环境采样、多轮对话调度、轨迹生成、奖励累积；
    - 适用：强化学习训练流程中的策略评估和轨迹采样，如 GRPO、PPO 等算法；
    - 线程/协程安全：基于异步接口，支持并发请求。
    
    特点：
    - 高性能：继承 vLLM 的高吞吐量和低延迟特性
    - 灵活采样：支持两种采样控制器（Gym 环境和多轮调度器）
    - 轨迹管理：自动记录对话轨迹、奖励和环境信息
    - 上下文管理：支持动态上下文裁剪和管理，适应长对话
    """

    def __init__(  # 初始化 GRPOVllmEngine 实例，配置模型加载、引擎参数和采样控制器
        self,  # 实例自身引用
        model_id_or_path: str,  # 模型的 HuggingFace ID 或本地路径，用于加载模型和分词器
        torch_dtype: Optional[torch.dtype] = None,  # 模型推理使用的数据类型（如 float16、bfloat16），None 则自动推断
        *,  # 仅限关键字参数分隔符，后续参数必须以关键字形式传入
        use_async_engine: bool = False,  # 是否使用异步引擎（True 推荐用于 GRPO，支持并发请求）
        model_type: Optional[str] = None,  # 模型类型标识（如 'qwen', 'llama'），用于选择特定的模板和配置
        use_hf: Optional[bool] = None,  # 是否强制从 HuggingFace Hub 下载模型，None 则自动判断
        hub_token: Optional[str] = None,  # HuggingFace Hub 访问令牌，用于下载私有模型
        revision: Optional[str] = None,  # 模型的版本/分支名称，用于下载特定版本
        # engine_kwargs  # 分组注释：以下为 vLLM 引擎的核心配置参数
        gpu_memory_utilization: float = 0.9,  # GPU 显存利用率（0.9 表示使用 90% 显存用于 KV 缓存）
        tensor_parallel_size: int = 1,  # Tensor Parallelism 张量并行度，多 GPU 时用于分割模型
        pipeline_parallel_size: int = 1,  # Pipeline Parallelism 流水线并行度，多 GPU 时用于分层部署
        enable_expert_parallel: bool = False,  # 是否启用专家并行（用于 MoE 模型）
        max_model_len: Optional[int] = None,  # 最大模型长度（context window），None 则使用模型配置的默认值
        max_num_seqs: int = 256,  # 最大批处理序列数（并发请求数），影响吞吐量
        disable_custom_all_reduce: bool = True,  # 是否禁用自定义的 All-Reduce 算子（自定义算子可能更快但稳定性较差）
        enforce_eager: bool = False,  # 是否强制使用 Eager 模式（禁用 CUDA Graph，便于调试）
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,  # 每个 prompt 的多模态数据限制（如 {'image': 5} 表示最多 5 张图片）
        seed: Optional[int] = None,  # 随机种子，用于确保采样的可复现性
        task_type: Optional[str] = None,  # 任务类型（如 'embedding'），None 表示普通生成任务
        # lora  # 分组注释：以下为 LoRA 相关配置参数
        enable_lora: bool = False,  # 是否启用 LoRA 适配器支持
        max_loras: int = 1,  # 最大 LoRA 适配器数量（同时加载的 LoRA 数量）
        max_lora_rank: int = 16,  # 最大 LoRA 秩（影响显存占用和性能）
        enable_prefix_caching: bool = False,  # 是否启用前缀缓存（共享 prompt 前缀的 KV 缓存，提高吞吐量）
        enable_sleep_mode: bool = False,  # 是否启用睡眠模式（空闲时释放 GPU 显存）
        distributed_executor_backend: Optional[str] = None,  # 分布式执行后端（如 'ray', 'mp'），None 则自动选择
        quantization: Optional[str] = None,  # 量化方案（如 'awq', 'gptq', 'fp8'），None 表示不量化
        engine_kwargs: Optional[Dict[str, Any]] = None,  # 其他传递给 vLLM 引擎的额外参数字典
        template: Optional[Template] = None,  # 预设的对话模板，None 则根据 model_type 自动选择
        **kwargs,  # 额外的关键字参数，用于 GRPO 特定配置（max_turns、multi_turn_scheduler、gym_env 等）
    ) -> None:  # 无返回值
        """函数功能：
        初始化 `GRPOVllmEngine` 实例，完成以下步骤：
        1. 调用父类 VllmEngine 的初始化方法，设置 vLLM 引擎；
        2. 配置多轮对话相关参数（max_turns）；
        3. 根据配置选择并初始化采样控制器（Gym 环境或多轮调度器）；
        4. 确保采样控制器的互斥性（不能同时使用两种控制器）。
        
        参数：见上方签名注释。
        
        返回值：
        - None（初始化实例属性）
        
        示例：
        >>> # 使用多轮调度器
        >>> engine = GRPOVllmEngine('Qwen/Qwen-7B-Chat', use_async_engine=True,
        ...                          multi_turn_scheduler='basic', max_turns=5)
        >>> 
        >>> # 使用 Gym 环境
        >>> engine = GRPOVllmEngine('Qwen/Qwen-7B-Chat', use_async_engine=True,
        ...                          use_gym_env=True, gym_env='math_env', max_turns=10)
        """
        super().__init__(  # 调用父类 VllmEngine 的初始化方法
            model_id_or_path=model_id_or_path,  # 传递模型路径
            torch_dtype=torch_dtype,  # 传递数据类型
            use_async_engine=use_async_engine,  # 传递是否使用异步引擎
            model_type=model_type,  # 传递模型类型
            use_hf=use_hf,  # 传递是否使用 HuggingFace Hub
            hub_token=hub_token,  # 传递 HuggingFace 令牌
            revision=revision,  # 传递模型版本
            gpu_memory_utilization=gpu_memory_utilization,  # 传递 GPU 显存利用率
            tensor_parallel_size=tensor_parallel_size,  # 传递张量并行度
            pipeline_parallel_size=pipeline_parallel_size,  # 传递流水线并行度
            enable_expert_parallel=enable_expert_parallel,  # 传递是否启用专家并行
            max_model_len=max_model_len,  # 传递最大模型长度
            max_num_seqs=max_num_seqs,  # 传递最大批处理序列数
            disable_custom_all_reduce=disable_custom_all_reduce,  # 传递是否禁用自定义 All-Reduce
            enforce_eager=enforce_eager,  # 传递是否强制 Eager 模式
            limit_mm_per_prompt=limit_mm_per_prompt,  # 传递多模态数据限制
            seed=seed,  # 传递随机种子
            task_type=task_type,  # 传递任务类型
            enable_lora=enable_lora,  # 传递是否启用 LoRA
            max_loras=max_loras,  # 传递最大 LoRA 数量
            max_lora_rank=max_lora_rank,  # 传递最大 LoRA 秩
            enable_prefix_caching=enable_prefix_caching,  # 传递是否启用前缀缓存
            enable_sleep_mode=enable_sleep_mode,  # 传递是否启用睡眠模式
            distributed_executor_backend=distributed_executor_backend,  # 传递分布式执行后端
            quantization=quantization,  # 传递量化方案
            engine_kwargs=engine_kwargs,  # 传递额外引擎参数
            template=template,  # 传递对话模板
        )

        self.max_turns = kwargs.get('max_turns')  # 从额外参数中获取最大对话轮数（用于限制多轮对话长度）

        # Get sampling controller configurations from kwargs  # 获取采样控制器配置（从额外参数中）
        multi_turn_scheduler = kwargs.get('multi_turn_scheduler', None)  # 获取多轮调度器配置（字符串名称或实例，None 表示不使用）
        use_gym_env = kwargs.get('use_gym_env', False)  # 获取是否使用 Gym 环境（默认 False）
        
        if use_gym_env:  # 若启用 Gym 环境采样
            self.gym_env = kwargs.get('gym_env', None)  # 获取 Gym 环境名称（从环境注册表中查找）
            self.context_manager = kwargs.get('context_manager', None)  # 获取上下文管理器名称（用于管理对话上下文）
        
        # Ensure mutual exclusivity of sampling controllers  # 确保采样控制器的互斥性（不能同时使用两种）
        if use_gym_env and multi_turn_scheduler is not None:  # 若同时配置了 Gym 环境和多轮调度器
            raise ValueError('gym_env and multi_turn_scheduler are mutually exclusive sampling controllers')  # 抛出错误

        self.use_gym_env = use_gym_env  # 保存是否使用 Gym 环境的标志

        # Initialize sampling controller  # 初始化采样控制器（根据配置选择 Gym 环境或多轮调度器）
        if use_gym_env:  # 若使用 Gym 环境采样
            self.multi_turn_scheduler = None  # 多轮调度器设为 None（互斥）
        elif multi_turn_scheduler is not None:  # 若配置了多轮调度器
            if isinstance(multi_turn_scheduler, str):  # 若 multi_turn_scheduler 是字符串（调度器名称）
                assert multi_turn_scheduler in multi_turns  # 断言调度器名称在注册表中
                self.multi_turn_scheduler: MultiTurnScheduler = multi_turns[multi_turn_scheduler](  # 从注册表创建调度器实例
                    template=template, max_turns=self.max_turns)  # 传递模板和最大轮数
            else:  # 若 multi_turn_scheduler 是调度器实例
                assert isinstance(multi_turn_scheduler, MultiTurnScheduler)  # 断言是 MultiTurnScheduler 类型
                self.multi_turn_scheduler: MultiTurnScheduler = multi_turn_scheduler  # 直接使用传入的实例
        else:  # 若未配置任何采样控制器
            self.multi_turn_scheduler = None  # 多轮调度器设为 None（默认单轮推理）

    def _create_env(self, env_config: Dict) -> Env:
        """函数功能：
        私有方法，根据环境配置创建 Gym 环境实例，用于强化学习采样。
        
        参数：
        - env_config (Dict): 环境配置字典（包含 name 等参数）
        
        返回值：
        - Env: Gym 环境实例
        
        示例：
        >>> env = self._create_env({'name': 'math_env', 'difficulty': 'hard'})
        """
        env_name = env_config.get('name', None)  # 从配置中获取环境名称
        if not env_name:  # 若配置中未指定环境名称
            env_name = self.gym_env  # 使用初始化时配置的环境名称
        if env_name not in envs:  # 若环境名称不在环境注册表中
            raise ValueError((f"Environment '{env_name}' not found in envs registry. "  # 抛出错误
                              f'Available: {list(envs.keys())}'))  # 列出可用的环境名称
        return envs[env_name](env_config)  # 从注册表创建环境实例并返回

    def _create_context_manager(self, ctx_config: Dict) -> ContextManager:
        """函数功能：
        私有方法，根据上下文配置创建上下文管理器实例，用于管理多轮对话的上下文（如裁剪、摘要等）。
        
        参数：
        - ctx_config (Dict): 上下文管理器配置字典（包含 name 等参数）
        
        返回值：
        - ContextManager: 上下文管理器实例
        
        示例：
        >>> ctx_manager = self._create_context_manager({'name': 'sliding_window', 'window_size': 10})
        """
        ctx_name = ctx_config.get('name', None)  # 从配置中获取上下文管理器名称
        if not ctx_name:  # 若配置中未指定名称
            ctx_name = self.context_manager  # 使用初始化时配置的上下文管理器名称

        if not ctx_name:  # 若仍未指定名称
            ctx_name = 'dummyContextManager'  # 使用默认的虚拟上下文管理器（不做任何处理）

        if ctx_name not in context_managers:  # 若上下文管理器名称不在注册表中
            raise ValueError((f"Context manager '{ctx_name}' not found in registry. "  # 抛出错误
                              f'Available: {list(context_managers.keys())}'))  # 列出可用的上下文管理器名称
        return context_managers[ctx_name](ctx_config)  # 从注册表创建上下文管理器实例并返回

    def infer(
        self,  # 实例自身引用
        infer_requests: List[Union[InferRequest, Dict[str, Any]]],  # 推理请求列表（InferRequest 对象或字典）
        request_config: Optional[RequestConfig] = None,  # 请求配置（可选）
        metrics: Optional[List[Metric]] = None,  # 指标采集器列表（可选）
        *,  # 仅限关键字参数分隔符
        template: Optional[Template] = None,  # 对话模板（可选）
        use_tqdm: Optional[bool] = None,  # 是否使用进度条（可选）
        adapter_request: Optional[AdapterRequest] = None,  # LoRA 适配器请求（可选）
    ) -> List[ChatCompletionResponse]:  # 返回响应列表
        """函数功能：
        同步批量推理的公开接口，仅支持同步引擎。
        
        参数：见上方签名注释。
        
        返回值：
        - List[ChatCompletionResponse]: 响应列表
        
        示例：
        >>> requests = [InferRequest(messages=[{'role': 'user', 'content': '你好'}])]
        >>> responses = engine.infer(requests)
        """
        assert not self.use_async_engine, 'for Async Engine, use infer_async instead'  # 断言未使用异步引擎（GRPO 推荐使用异步）
        # 若使用了异步引擎，应该调用 async_infer 方法
        
        return super().infer(  # 调用父类 VllmEngine 的 infer 方法
            infer_requests,  # 传递请求列表
            request_config,  # 传递请求配置
            metrics,  # 传递指标采集器
            template=template,  # 传递模板
            use_tqdm=use_tqdm,  # 传递是否使用进度条
            adapter_request=adapter_request,  # 传递适配器请求
        )

    async def async_infer(self,
                          infer_requests: List[Union[RolloutInferRequest, Dict[str, Any]]],  # 轨迹推理请求列表
                          request_config: Optional[RequestConfig] = None,  # 请求配置（可选）
                          metrics: Optional[List[Metric]] = None,  # 指标采集器列表（可选）
                          *,  # 仅限关键字参数分隔符
                          use_tqdm: Optional[bool] = None,  # 是否使用进度条（可选）
                          **kwargs) -> List[ChatCompletionResponse]:  # 返回响应列表
        """函数功能：
        异步批量推理的公开接口，支持多轮对话和 Gym 环境采样。
        
        参数：见上方签名注释。
        
        返回值：
        - List[ChatCompletionResponse]: 响应列表（包含轨迹信息和奖励）
        
        示例：
        >>> requests = [RolloutInferRequest(messages=[{'role': 'user', 'content': '解决这个数学问题'}])]
        >>> responses = await engine.async_infer(requests)
        """
        if request_config is None:  # 若未提供请求配置
            request_config = RequestConfig()  # 使用默认配置
        assert request_config.n == 1  # 断言每个请求只生成 1 个候选（GRPO 不支持多候选）
        
        async def _infer_async_single(infer_request: Union[RolloutInferRequest, Dict[str, Any]],  # 单个异步推理（内部函数）
                                      request_config: Optional[RequestConfig] = None,  # 请求配置
                                      **kwargs):  # 额外参数
            """内部函数：处理单个轨迹推理请求。"""
            if isinstance(infer_request, Dict):  # 若请求是字典
                infer_request = RolloutInferRequest(**infer_request)  # 转换为 RolloutInferRequest 对象

            # Route to appropriate sampling controller  # 路由（选择分支）到适当的采样控制器
            if self.use_gym_env:  # 若使用 Gym 环境采样
                return await self._gym_sampling_controller(infer_request, request_config, **kwargs)  # 调用 Gym 采样控制器
            else:  # 若使用多轮调度器
                return await self._multi_turn_sampling_controller(infer_request, request_config, **kwargs)  # 调用多轮调度器

        tasks = [_infer_async_single(infer_request, request_config, **kwargs) for infer_request in infer_requests]  # 为每个请求创建异步任务
        if use_tqdm is None:  # 若未指定是否使用进度条
            use_tqdm = len(infer_requests) > 1  # 多个请求时默认显示进度条
        return await self._batch_infer_stream(tasks, request_config.stream, use_tqdm, metrics)  # 批量执行异步任务

    async def _gym_sampling_controller(self, infer_request: RolloutInferRequest, request_config: RequestConfig,
                                       **kwargs) -> ChatCompletionResponse:
        """函数功能：
        私有方法，基于 Gym 环境的采样控制器，用于强化学习场景下的多轮对话生成和轨迹采样。
        
        工作流程：
        1. 创建环境和上下文管理器；
        2. 重置环境，获取初始观察；
        3. 循环执行：应用上下文管理 -> LLM 生成响应 -> 环境执行动作 -> 累积奖励；
        4. 返回包含完整轨迹、奖励和环境信息的响应。
        
        参数：见上方签名注释。
        
        返回值：
        - ChatCompletionResponse: 包含轨迹信息、总奖励、步骤奖励的响应
        
        示例：
        >>> request = RolloutInferRequest(messages=[...], data_dict={'env_config': {'name': 'math_env'}})
        >>> response = await self._gym_sampling_controller(request, RequestConfig())
        """
        # Create environment and context manager  # 创建环境和上下文管理器
        env_config = infer_request.data_dict.get('env_config', {})  # 从请求的数据字典中获取环境配置
        env = self._create_env(env_config)  # 创建 Gym 环境实例
        ctx_config = infer_request.data_dict.get('ctx_config', {})  # 从请求的数据字典中获取上下文管理器配置
        context_manager = self._create_context_manager(ctx_config)  # 创建上下文管理器实例

        try:  # 尝试执行采样流程（使用 try-finally 确保环境被正确关闭）
            # Environment reset  # 环境重置
            observation, info, system_message = await env.reset(infer_request)  # 异步重置环境，获取初始观察、信息和系统消息
            # observation: 环境的初始观察（如问题描述）
            # info: 环境的初始信息（如难度等级）
            # system_message: 可选的系统提示消息

            # Initialize conversation  # 初始化对话
            messages = []  # 创建空的消息列表
            if system_message:  # 若环境返回了系统消息
                messages.append({'role': 'system', 'content': system_message})  # 添加系统消息到对话历史
            messages.append({'role': 'user', 'content': observation})  # 添加用户消息（初始观察）到对话历史

            current_request = deepcopy(infer_request)  # 深拷贝原始请求（避免修改原始对象）
            current_turn = 1  # 初始化当前轮数为 1
            done = False  # 初始化完成标志为 False
            total_reward = 0.0  # 初始化总奖励为 0
            step_rewards = []  # 初始化步骤奖励列表为空
            trajectory_id = f'{id(infer_request)}_{hash(str(infer_request))}'  # 生成唯一的轨迹 ID（基于请求对象的内存地址和哈希值）
            trajectory_info = [info]  # 初始化轨迹信息列表（包含初始环境信息）

            while True:  # 无限循环（直到环境完成或达到最大轮数）
                # Apply context management  # 应用上下文管理
                messages = context_manager.manage_context(messages, trajectory_id)  # 使用上下文管理器处理消息（如裁剪过长的对话）
                current_request.messages = messages  # 更新请求的消息列表
                # Remove any previous assistant response for generation  # 移除之前的 assistant 响应（为新一轮生成做准备）
                InferRequest.remove_response(current_request.messages)  # 调用静态方法移除最后的 assistant 消息（如果存在）

                # Generate LLM response  # 生成 LLM 响应
                result: ChatCompletionResponse = await self.infer_async(current_request, request_config, **kwargs)  # 异步调用 LLM 生成响应
                result_choice: RolloutResponseChoice = result.choices[0]  # 提取第一个（也是唯一的）响应选项

                completion = result_choice.message.content  # 提取生成的文本内容
                messages.append({'role': 'assistant', 'content': completion})  # 将 LLM 的响应添加到对话历史

                # Environment step  # 环境执行步骤
                next_observation, reward, done, step_info = await env.step(deepcopy(messages))  # 异步执行环境步骤，传入当前对话历史
                # next_observation: 下一个观察（如环境反馈）
                # reward: 本步骤获得的奖励
                # done: 是否完成（True 表示任务结束）
                # step_info: 本步骤的额外信息

                # Accumulate rewards  # 累积奖励
                total_reward += reward  # 将本步骤奖励加到总奖励
                step_rewards.append(reward)  # 将本步骤奖励添加到奖励列表
                trajectory_info.append(step_info)  # 将本步骤信息添加到轨迹信息列表

                if done or current_turn > self.max_turns:  # 若环境完成或超过最大轮数
                    break  # 退出循环

                messages.append({'role': 'user', 'content': next_observation})  # 将环境的下一个观察添加到对话历史（作为用户消息）
                current_request.messages = messages  # 更新请求的消息列表
                current_turn += 1  # 轮数加 1

            # Create final result with gym-specific information  # 创建包含 Gym 特定信息的最终结果
            final_choice = GymRolloutResponseChoice(  # 创建 Gym 轨迹响应选项对象
                index=result_choice.index,  # 选项索引
                message=result_choice.message,  # 最后一次生成的消息
                finish_reason=result_choice.finish_reason,  # 完成原因
                logprobs=result_choice.logprobs,  # 对数概率
                messages=messages,  # 完整的对话历史
                trajectory_id=trajectory_id,  # 轨迹 ID
                total_reward=total_reward,  # 总奖励
                step_rewards=step_rewards,  # 每步奖励列表
                trajectory_info=trajectory_info)  # 轨迹信息列表

            return ChatCompletionResponse(  # 返回聊天补全响应对象
                model=self.model_name, choices=[final_choice], usage=result.usage, id=f'gym_{trajectory_id}')  # 包含模型名、选项、用量和带前缀的轨迹 ID

        finally:  # 无论是否出现异常，都执行清理
            await self._close_env_async(env)  # 异步关闭环境（释放资源）

    async def _multi_turn_sampling_controller(self, infer_request: RolloutInferRequest, request_config: RequestConfig,
                                              **kwargs) -> ChatCompletionResponse:
        """函数功能：
        私有方法，基于多轮调度器的采样控制器，用于多轮对话生成（不依赖外部环境）。
        
        工作流程：
        1. 循环执行：LLM 生成响应 -> 调度器检查是否完成 -> 调度器生成下一轮输入；
        2. 支持调度器控制多轮对话的终止条件和下一轮输入生成；
        3. 返回包含完整对话历史和多轮信息的响应。
        
        参数：见上方签名注释。
        
        返回值：
        - ChatCompletionResponse: 包含完整对话历史和多轮信息的响应
        
        示例：
        >>> request = RolloutInferRequest(messages=[{'role': 'user', 'content': '开始对话'}])
        >>> response = await self._multi_turn_sampling_controller(request, RequestConfig())
        """
        current_request = infer_request  # 初始化当前请求为原始请求
        current_turn = 1  # 初始化当前轮数为 1
        info_dict = {}  # 初始化信息字典为空（用于存储多轮对话的额外信息）
        
        while True:  # 无限循环（直到调度器决定停止或达到最大轮数）
            messages = current_request.messages  # 获取当前请求的消息列表
            if current_turn == 1 or not messages[-1]['content']:  # 若是第一轮 或 最后一条消息内容为空（虚拟消息）
                # If it's the first turn or the last message content is empty(dummy), remove the response  # 移除响应（为生成做准备）
                InferRequest.remove_response(messages)  # 调用静态方法移除最后的 assistant 消息

            result: ChatCompletionResponse = await self.infer_async(current_request, request_config, **kwargs)  # 异步调用 LLM 生成响应
            result_choice: RolloutResponseChoice = result.choices[0]  # 提取第一个（也是唯一的）响应选项

            completion = result_choice.message.content  # 提取生成的文本内容
            if messages[-1]['role'] == 'assistant':  # 若最后一条消息是 assistant（虚拟消息）
                messages[-1]['content'] += completion  # 将新生成的内容追加到虚拟消息（支持续写）
            else:  # 若最后一条消息不是 assistant
                messages.append({'role': 'assistant', 'content': completion})  # 添加新的 assistant 消息

            if self.multi_turn_scheduler:  # 若配置了多轮调度器
                should_stop = self.multi_turn_scheduler.check_finished(current_request, result_choice, current_turn)  # 调用调度器检查是否应该停止
                # 调度器根据请求、响应和轮数判断是否完成
            else:  # 若未配置多轮调度器
                should_stop = True  # 默认停止（单轮推理）

            if self.max_turns:  # 若设置了最大轮数
                should_stop = should_stop or (current_turn >= self.max_turns)  # 达到最大轮数时停止

            if should_stop:  # 若应该停止
                result_choice.messages = messages  # 将完整的对话历史设置到响应选项
                info_dict['num_turns'] = current_turn  # 记录实际执行的轮数
                
                for key, values in info_dict.items():  # 遍历信息字典（处理多模态数据和其他信息）
                    if key in ['images', 'audios', 'videos']:  # 若是多模态数据
                        if not isinstance(values, list):  # 若不是列表
                            values = [values]  # 转换为列表
                        for i, value in enumerate(values):  # 遍历多模态数据
                            values[i] = MultiModalRequestMixin.to_base64(value)  # 将多模态数据转换为 base64 编码
                    
                    if hasattr(result_choice, key):  # 若响应选项有该属性
                        setattr(result_choice, key, values)  # 直接设置属性值
                    else:  # 若响应选项没有该属性
                        result_choice.multi_turn_infos[key] = values  # 添加到 multi_turn_infos 字典
                
                return result  # 返回最终结果

            ret = self.multi_turn_scheduler.step(current_request, result_choice, current_turn)  # 调用调度器生成下一轮请求
            # 调度器根据当前请求、响应和轮数生成下一轮的请求（可能修改消息、添加新输入等）
            
            if isinstance(ret, tuple):  # 若调度器返回元组（请求 + 信息字典）
                current_request, info_dict = ret  # 解包元组
            else:  # 若调度器只返回请求
                current_request = ret  # 使用新请求
                info_dict = {}  # 清空信息字典

            assert isinstance(current_request, RolloutInferRequest)  # 断言返回的是 RolloutInferRequest 对象
            
            if current_request.messages[-1]['role'] == 'assistant':  # 若最后一条消息是 assistant
                # Add a dummy response to allow engine to continue generating  # 添加虚拟响应以允许引擎继续生成
                current_request.messages.append({'role': 'assistant', 'content': None})  # 添加内容为 None 的 assistant 消息（作为占位符）

            current_turn += 1  # 轮数加 1

    async def _batch_infer_stream(self,
                                  tasks,  # 异步任务列表
                                  stream: bool = True,  # 是否流式（当前不支持，必须为 False）
                                  use_tqdm: bool = True,  # 是否使用进度条
                                  metrics: Optional[List[Metric]] = None):  # 指标采集器列表（可选）
        """函数功能：
        私有方法，批量异步推理，支持进度条显示和指标采集。
        
        参数：见上方签名注释。
        
        返回值：
        - List[ChatCompletionResponse]: 响应列表
        
        示例：
        >>> tasks = [self._gym_sampling_controller(...) for _ in range(10)]
        >>> responses = await self._batch_infer_stream(tasks, stream=False, use_tqdm=True)
        """
        assert not stream  # 断言非流式模式（GRPO 当前不支持批量流式推理）
        prog_bar = tqdm_asyncio(total=len(tasks), dynamic_ncols=True, disable=not use_tqdm)  # 创建异步进度条

        async def _new_run(task):
            """内部函数：包装执行单个异步任务以支持异常处理和进度更新。"""
            try:  # 尝试执行任务
                res = await task  # 等待任务完成
            except Exception as e:  # 若任务抛出异常
                if getattr(self, 'strict', True):  # 若启用严格模式（默认）
                    raise  # 重新抛出异常
                res = e  # 否则将异常作为结果（容错模式）
            prog_bar.update()  # 更新进度条（任务完成）
            self._update_metrics(res, metrics)  # 更新指标（统计用量、耗时等）
            return res  # 返回结果

        new_tasks = [_new_run(task) for task in tasks]  # 为每个任务创建包装任务
        return await self.batch_run(new_tasks)  # 批量运行任务（使用父类的 batch_run 方法）

    async def _close_env_async(self, env: Env):
        """函数功能：
        私有方法，异步关闭 Gym 环境，释放资源。支持同步和异步的 close 方法。
        
        参数：
        - env (Env): Gym 环境实例
        
        返回值：
        - None
        
        示例：
        >>> await self._close_env_async(env)
        """
        try:  # 尝试关闭环境（忽略所有异常，确保清理流程继续）
            if hasattr(env, 'close') and asyncio.iscoroutinefunction(env.close):  # 若环境有 close 方法且是异步函数
                await env.close()  # 异步关闭环境
            elif hasattr(env, 'close'):  # 若环境有 close 方法但是同步函数
                env.close()  # 同步关闭环境
        except Exception:  # 若关闭过程中出现任何异常
            pass  # 忽略异常（避免影响主流程）

    def _create_chat_completion_response(self, result, template: Template, request_config,
                                         request_id) -> ChatCompletionResponse:  # 请求 ID
        """函数功能：
        私有方法，覆盖父类。
        从 vLLM 的输出中创建聊天补全响应，根据引擎配置选择合适的响应类型。
        
        参数：
        - result: vLLM 的输出结果对象
        - template (Template): 对话模板
        - request_config: 请求配置
        - request_id: 请求 ID
        
        返回值：
        - ChatCompletionResponse: 聊天补全响应对象
        
        示例：
        >>> response = self._create_chat_completion_response(result, template, request_config, 'req_123')
        """
        assert result is not None  # 断言结果不为空
        num_generated_tokens = sum(len(output.token_ids) for output in result.outputs)  # 计算生成的总 token 数（所有输出的 token 数之和）
        usage_info = self._get_usage_info(len(result.prompt_token_ids), num_generated_tokens)  # 构造用量信息（prompt token 数 + 生成 token 数）
        
        choices = []  # 初始化选项列表为空
        for output in result.outputs:  # 遍历所有输出（通常只有一个）
            output.token_ids = list(output.token_ids)  # 将 token_ids 转换为列表（vLLM 返回的可能是其他类型）
            response = template.decode(output.token_ids)  # 使用模板解码 token IDs 为文本
            logprobs = self._get_logprobs(output.logprobs, output.token_ids, request_config.top_logprobs)  # 获取对数概率信息
            toolcall = self._get_toolcall(response, template)  # 从响应文本中提取工具调用信息

            if self.use_gym_env:  # 若使用 Gym 环境采样
                choice_cls = GymRolloutResponseChoice  # 使用 Gym 轨迹响应选项类
            elif self.use_async_engine:  # 若使用异步引擎（但不是 Gym 环境）
                choice_cls = RolloutResponseChoice  # 使用轨迹响应选项类
            else:  # 若使用同步引擎
                choice_cls = ChatCompletionResponseChoice  # 使用标准响应选项类

            token_ids = template.skip_stop_tokens(output.token_ids) if request_config.return_details else None  # 若需要返回详细信息，则跳过停止 token
            choice = choice_cls(  # 创建响应选项对象（根据引擎类型选择合适的类）
                index=output.index,  # 选项索引
                message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),  # 消息对象
                finish_reason=output.finish_reason,  # 完成原因
                logprobs=logprobs,  # 对数概率
                token_ids=token_ids,  # token IDs（若 return_details=True）
            )
            choices.append(choice)  # 将选项添加到列表
        
        prompt_token_ids = result.prompt_token_ids if request_config.return_details else None  # 若需要返回详细信息，则提取 prompt token_ids
        return ChatCompletionResponse(  # 返回聊天补全响应对象
            model=self.model_name, choices=choices, usage=usage_info, id=request_id, prompt_token_ids=prompt_token_ids)  # 包含模型名、选项、用量、请求 ID 和 prompt token_ids
