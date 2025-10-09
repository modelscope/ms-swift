"""模块功能概述：
该模块实现基于 vLLM 引擎的高性能推理引擎 `VllmEngine`，用于在生产环境中进行大语言模型的推理。提供：
- 同步和异步推理接口，支持批量处理和单个请求；
- 流式和非流式响应模式，适配不同的应用场景；
- LoRA 动态适配器支持，实现多租户场景下的模型定制；
- 嵌入向量生成能力，支持文本嵌入任务；
- 多模态输入处理（图像、音频、视频），支持视觉语言模型。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，标注代码版权所有者
import asyncio  # 引入 asyncio 模块，用于异步编程和协程调度
import inspect  # 引入 inspect 模块，用于运行时检查函数签名和参数
import os  # 引入 os 模块，用于环境变量设置和文件路径操作
from contextlib import nullcontext  # 引入 nullcontext，作为空上下文管理器（无操作）
from copy import deepcopy  # 引入 deepcopy，用于对配置进行深拷贝，避免副作用
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union  # 引入类型注解，明确参数与返回值类型

import torch  # 引入 PyTorch 库，用于张量操作和数据类型判断
from packaging import version  # 引入 version 模块，用于版本号比较
from tqdm import tqdm  # 引入 tqdm，用于显示进度条
from transformers import GenerationConfig  # 引入 GenerationConfig，用于加载和管理生成配置
from transformers.utils import is_torch_npu_available  # 引入 NPU 可用性检查函数

from swift.llm import InferRequest, Template, TemplateMeta, get_model_tokenizer  # 引入 Swift 框架的推理请求、模板和模型加载工具
from swift.plugin import Metric  # 引入 Metric 类型，用于传递统计/用量指标采集器
from swift.utils import get_device, get_dist_setting, get_logger, is_dist  # 引入设备获取、分布式设置和日志工具
from ..protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,  # 引入协议层数据类：响应、选项、流式响应
                        ChatCompletionStreamResponse, ChatMessage, DeltaMessage, EmbeddingResponse,  # 引入聊天消息、增量消息、嵌入响应
                        EmbeddingResponseData, RequestConfig, random_uuid)  # 引入嵌入数据、请求配置和 UUID 生成器
from .infer_engine import InferEngine  # 引入父类 InferEngine，提供通用推理接口与工具方法
from .patch import patch_auto_config, patch_auto_tokenizer  # 引入配置和分词器补丁，用于修复兼容性问题
from .utils import AdapterRequest, InferStreamer, patch_npu_vllm, patch_vllm_memory_leak  # 引入适配器请求、流式输出器和补丁工具

logger = get_logger()  # 获取全局日志记录器实例，用于记录运行时信息
try:  # 尝试导入 vLLM 库，并在导入前设置必要的环境变量
    # After setting the environment variables, import vllm. This way of writing allows lint to pass.  # 注释：先设置环境变量再导入，以兼容 lint 检查
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'  # 设置 vLLM 工作进程的启动方法为 'spawn'，确保多进程兼容性（特别是 Windows 和某些 CUDA 场景）
    os.environ['VLLM_ENGINE_ITERATION_TIMEOUT_S'] = '86400'  # 设置 vLLM 引擎迭代超时为 86400 秒（1 天），避免长请求被过早终止
    import vllm  # 引入 vllm 库，提供高性能的 LLM 推理引擎
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, EngineArgs, LLMEngine  # 引入异步/同步引擎参数、引擎实例和采样参数类
except Exception:  # 若导入失败，捕获异常
    raise  # 直接抛出异常，不进行额外处理（确保调用方能够感知到依赖缺失）

dtype_mapping = {torch.float16: 'float16', torch.bfloat16: 'bfloat16', torch.float32: 'float32'}  # 定义 PyTorch 数据类型到 vLLM 字符串表示的映射字典，用于引擎初始化时的类型转换


class VllmEngine(InferEngine):  # 定义基于 vLLM 的推理引擎类，继承通用推理引擎基类

    """类功能：
    `VllmEngine` 负责基于 vLLM 引擎进行高性能的大语言模型推理。
    
    - 角色：推理引擎封装，连接 vLLM 底层实现与上层应用接口；
    - 能力：支持同步/异步推理、流式/非流式输出、批量处理、LoRA 适配器、多模态输入、嵌入生成；
    - 适用：生产环境中的高吞吐、低延迟 LLM 服务，支持多租户场景；
    - 线程/协程安全：异步引擎模式下支持并发请求，同步模式需外部协调。
    """

    def __init__(  # 初始化 VllmEngine 实例，配置模型加载、引擎参数、LoRA 和任务类型
        self,  # 实例自身引用
        model_id_or_path: str,  # 模型的 HuggingFace ID 或本地路径，用于加载模型和分词器
        torch_dtype: Optional[torch.dtype] = None,  # 模型推理使用的数据类型（如 float16、bfloat16），None 则自动推断
        *,  # 仅限关键字参数分隔符，后续参数必须以关键字形式传入
        use_async_engine: bool = False,  # 是否使用异步引擎，True 则创建 AsyncLLMEngine，False 则创建同步 LLMEngine
        model_type: Optional[str] = None,  # 模型类型标识（如 'qwen', 'llama'），用于选择特定的模板和配置
        use_hf: Optional[bool] = None,  # 是否强制从 HuggingFace Hub 下载模型，None 则自动判断
        hub_token: Optional[str] = None,  # HuggingFace Hub 访问令牌，用于下载私有模型
        revision: Optional[str] = None,  # 模型的版本/分支名称，用于下载特定版本
        # engine_kwargs  # 分组注释：以下为 vLLM 引擎的核心配置参数
        gpu_memory_utilization: float = 0.9,  # GPU 显存利用率（0~1），0.9 表示使用 90% 的可用显存用于 KV 缓存
        tensor_parallel_size: int = 1,  # 张量并行规模，将模型权重在多个 GPU 上分片（需 GPU 数量为其倍数）
        pipeline_parallel_size: int = 1,  # 流水线并行规模，将模型层在多个 GPU 上分段（需 GPU 数量为其倍数）
        enable_expert_parallel: bool = False,  # 是否启用专家并行（用于 MoE 模型，如 Mixtral），需 vLLM 版本支持
        max_model_len: Optional[int] = None,  # 模型支持的最大序列长度，None 则从模型配置自动读取
        max_num_seqs: int = 256,  # 最大并发序列数，控制批处理大小和显存占用
        disable_custom_all_reduce: bool = True,  # 是否禁用自定义的 all-reduce 通信算子（True 使用 NCCL 标准实现，更稳定）
        enforce_eager: bool = False,  # 是否强制使用 eager 模式（不使用 CUDA Graph），调试时有用但性能较低
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,  # 多模态输入的数量限制（如 {'image': 5}），None 则不限制
        seed: Optional[int] = None,  # 随机种子，用于可复现的生成（影响采样过程）
        task_type: Optional[str] = None,  # embedding  # 任务类型，'embedding' 表示嵌入生成任务，None 表示文本生成
        # lora  # 分组注释：以下为 LoRA 适配器相关配置
        enable_lora: bool = False,  # 是否启用 LoRA 支持，True 则可在推理时动态加载不同的 LoRA 适配器
        max_loras: int = 1,  # 最大同时支持的 LoRA 适配器数量（影响显存预留）
        max_lora_rank: int = 16,  # LoRA 的最大秩（rank），用于预分配计算资源
        enable_prefix_caching: bool = False,  # 是否启用前缀缓存（自动缓存相同前缀的 KV 值，节省计算）
        enable_sleep_mode: bool = False,  # 是否启用睡眠模式（空闲时释放资源），需 vLLM 版本支持
        distributed_executor_backend: Optional[str] = None,  # 分布式执行后端（'ray' 或 'mp'），None 则自动选择
        quantization: Optional[str] = None,  # 量化方法（如 'awq', 'gptq', 'fp8'），None 则不使用量化
        engine_kwargs: Optional[Dict[str, Any]] = None,  # 其他传递给 vLLM 引擎的额外参数字典
        template: Optional[Template] = None,  # 预设的对话模板，None 则根据 model_type 自动选择
    ) -> None:  # 无返回值
        """函数功能：
        初始化 `VllmEngine` 实例，完成以下步骤：
        1. 加载分词器和配置（不加载模型权重，由 vLLM 接管）；
        2. 准备 vLLM 引擎参数（显存、并行、LoRA、量化等）；
        3. 创建 vLLM 引擎实例（同步或异步）；
        4. 加载生成配置（从 generation_config.json）；
        5. 应用补丁修复已知问题。
        
        参数：见上方签名注释。
        
        返回值：
        - None
        
        示例：
        >>> engine = VllmEngine(
        ...     'Qwen/Qwen-7B-Chat',
        ...     torch_dtype=torch.float16,
        ...     use_async_engine=True,
        ...     tensor_parallel_size=2
        ... )
        >>> # 引擎已就绪，可调用 infer_async 或 infer 方法
        """
        if engine_kwargs is None:  # 若未提供额外引擎参数
            engine_kwargs = {}  # 初始化为空字典，避免后续操作报错
        patch_vllm_memory_leak()  # 应用内存泄漏补丁，修复 vLLM 中已知的内存泄漏问题（如循环引用）
        self.use_async_engine = use_async_engine  # 保存异步引擎标志，供后续方法判断使用
        self.processor = get_model_tokenizer(  # 加载分词器（processor）和模型配置，返回值为 (model, tokenizer, config) 元组
            model_id_or_path,  # 模型路径或 HuggingFace ID
            torch_dtype,  # 数据类型（如 float16）
            load_model=False,  # 不加载模型权重（vLLM 会自行加载），仅获取分词器和配置
            download_model=True,  # 允许从 HuggingFace Hub 下载模型文件（若本地不存在）
            model_type=model_type,  # 模型类型（如 'qwen'），用于匹配特定逻辑
            use_hf=use_hf,  # 是否强制使用 HuggingFace Hub
            hub_token=hub_token,  # HuggingFace 访问令牌
            revision=revision,  # 模型版本/分支
            task_type=task_type)[1]  # 取元组的第二个元素（tokenizer），索引 [1] 提取分词器
        self._post_init(template)  # 调用父类的后初始化方法，设置默认模板、模型信息等（继承自 InferEngine）

        self._prepare_engine_kwargs(  # 准备 vLLM 引擎的初始化参数（封装为 EngineArgs 或 AsyncEngineArgs）
            gpu_memory_utilization=gpu_memory_utilization,  # 显存利用率
            tensor_parallel_size=tensor_parallel_size,  # 张量并行规模
            pipeline_parallel_size=pipeline_parallel_size,  # 流水线并行规模
            enable_expert_parallel=enable_expert_parallel,  # 专家并行（MoE 模型）
            max_model_len=max_model_len,  # 最大序列长度
            max_num_seqs=max_num_seqs,  # 最大并发序列数
            disable_custom_all_reduce=disable_custom_all_reduce,  # 禁用自定义 all-reduce
            enforce_eager=enforce_eager,  # 强制 eager 模式
            limit_mm_per_prompt=limit_mm_per_prompt,  # 多模态输入限制
            enable_lora=enable_lora,  # 启用 LoRA
            max_loras=max_loras,  # 最大 LoRA 数量
            max_lora_rank=max_lora_rank,  # 最大 LoRA 秩
            enable_prefix_caching=enable_prefix_caching,  # 启用前缀缓存
            seed=seed,  # 随机种子
            distributed_executor_backend=distributed_executor_backend,  # 分布式后端
            enable_sleep_mode=enable_sleep_mode,  # 睡眠模式
            quantization=quantization,  # 量化方法
            task=task_type,  # 任务类型（传递给引擎）
            **engine_kwargs,  # 展开额外参数传递给引擎
        )
        context = nullcontext()  # 默认使用空上下文管理器（无操作）
        if is_torch_npu_available() and (tensor_parallel_size == 1 or pipeline_parallel_size == 1):  # 若检测到华为 NPU 且未使用多卡并行
            context = patch_npu_vllm(get_device())  # 应用 NPU 专用补丁，修复 vLLM 在 NPU 上的兼容性问题
        with context:  # 进入补丁上下文（或空上下文）
            self._prepare_engine()  # 创建 vLLM 引擎实例（LLMEngine 或 AsyncLLMEngine）
        self._load_generation_config()  # 从模型目录加载 generation_config.json，设置默认生成参数
        self._fix_vllm_bug()  # 应用针对特定 vLLM 版本的 bug 修复（如 tokenizer.__len__ 慢的问题）
        self.patch_remove_log()  # 移除 vLLM 的某些日志输出，减少干扰
        self._request_count = 0  # 初始化请求计数器，用于生成唯一的 request_id

    def _prepare_engine(self) -> None:  # 创建 vLLM 引擎实例（内部方法）
        """函数功能：
        创建 vLLM 引擎实例（同步或异步），并应用必要的补丁。
        使用补丁上下文确保 vLLM 正确加载自定义的分词器和配置。
        
        参数：
        - 无（使用实例属性 self.tokenizer, self.config, self.engine_args）
        
        返回值：
        - None（设置 self.engine 属性）
        
        示例：
        >>> # 内部调用，通常在 __init__ 中自动执行
        >>> self._prepare_engine()
        >>> assert self.engine is not None
        """
        with patch_auto_tokenizer(self.tokenizer), patch_auto_config(self.config):  # 临时补丁：让 vLLM 的 AutoTokenizer/AutoConfig 使用我们已加载的实例，避免重复加载
            llm_engine_cls = AsyncLLMEngine if self.use_async_engine else LLMEngine  # 根据标志选择引擎类（异步或同步）
            engine = llm_engine_cls.from_engine_args(self.engine_args)  # 从 EngineArgs 创建引擎实例（加载模型权重、初始化 KV 缓存等）
        self.engine = engine  # 保存引擎实例到实例属性，供后续推理使用

    def _prepare_engine_kwargs(  # 准备 vLLM 引擎的初始化参数（内部方法）
        self,  # 实例自身引用
        gpu_memory_utilization: float = 0.9,  # GPU 显存利用率，0.9 表示使用 90% 的显存
        tensor_parallel_size: int = 1,  # 张量并行规模（模型权重分片数）
        pipeline_parallel_size: int = 1,  # 流水线并行规模（模型层分段数）
        enable_expert_parallel: bool = False,  # 是否启用专家并行（MoE 模型）
        max_model_len: Optional[int] = None,  # 最大序列长度，None 则自动推断
        max_num_seqs: int = 256,  # 最大并发序列数
        disable_custom_all_reduce: bool = True,  # 禁用自定义 all-reduce 算子
        enforce_eager: bool = False,  # 强制 eager 模式（不使用 CUDA Graph）
        limit_mm_per_prompt: Optional[Dict[str, Any]] = None,  # 多模态输入限制（如每个 prompt 最多几张图片）
        seed: Optional[int] = None,  # 随机种子
        enable_lora: bool = False,  # 是否启用 LoRA 支持
        max_loras: int = 1,  # 最大同时支持的 LoRA 数量
        max_lora_rank: int = 16,  # LoRA 的最大秩
        enable_prefix_caching: bool = False,  # 是否启用前缀缓存
        distributed_executor_backend: Optional[str] = None,  # 分布式执行后端（'ray' 或 'mp'）
        enable_sleep_mode: bool = False,  # 是否启用睡眠模式
        task: Optional[str] = None,  # 任务类型（'embedding' 或 None）
        **engine_kwargs,  # 其他传递给 vLLM 的额外参数
    ) -> None:  # 无返回值
        """函数功能：
        准备 vLLM 引擎的初始化参数，根据 vLLM 版本动态检测支持的参数，
        并构造 EngineArgs 或 AsyncEngineArgs 对象。
        
        参数：见上方签名注释。
        
        返回值：
        - None（设置 self.engine_args 和 self.enable_lora 属性）
        
        示例：
        >>> # 内部调用，通常在 __init__ 中自动执行
        >>> self._prepare_engine_kwargs(tensor_parallel_size=2, enable_lora=True)
        >>> assert self.engine_args.tensor_parallel_size == 2
        """
        if task == 'embedding':  # 若任务类型为 'embedding'
            task = 'embed'  # 标准化为 vLLM 接受的 'embed' 字符串
        disable_log_stats = engine_kwargs.pop('disable_log_stats', True)  # 从额外参数中提取 disable_log_stats（默认 True，禁用统计日志）
        if self.use_async_engine:  # 若使用异步引擎
            engine_cls = AsyncEngineArgs  # 选择异步引擎参数类
            engine_kwargs['disable_log_requests'] = True  # 禁用请求日志（异步引擎专有参数）
        else:  # 若使用同步引擎
            engine_cls = EngineArgs  # 选择同步引擎参数类
        parameters = inspect.signature(engine_cls).parameters  # 获取引擎参数类的构造函数签名，用于动态检测支持的参数（兼容不同 vLLM 版本）
        if 'enable_lora' in parameters and enable_lora:  # 若引擎支持 enable_lora 参数且用户启用了 LoRA
            engine_kwargs['enable_lora'] = enable_lora  # 添加 enable_lora 参数
            engine_kwargs['max_loras'] = max_loras  # 添加 max_loras 参数
            engine_kwargs['max_lora_rank'] = max_lora_rank  # 添加 max_lora_rank 参数
        else:  # 若引擎不支持 enable_lora 或用户未启用
            assert not enable_lora, 'The current version of vLLM does not support `enable_lora`. Please upgrade vLLM.'  # 断言用户未启用 LoRA，否则提示升级 vLLM

        if 'limit_mm_per_prompt' in parameters and limit_mm_per_prompt:  # 若引擎支持 limit_mm_per_prompt 参数且用户设置了限制
            engine_kwargs['limit_mm_per_prompt'] = limit_mm_per_prompt  # 添加多模态输入限制参数
        else:  # 若引擎不支持或用户未设置
            assert not limit_mm_per_prompt, (  # 断言用户未设置，否则提示升级 vLLM
                'The current version of VLLM does not support `limit_mm_per_prompt`. Please upgrade VLLM.')
        for key in ['enable_expert_parallel', 'enable_sleep_mode']:  # 遍历可选参数列表（专家并行和睡眠模式）
            if key in parameters:  # 若引擎支持该参数
                engine_kwargs[key] = locals()[key]  # 从局部变量中获取对应值并添加到 engine_kwargs
        for key in ['task', 'seed']:  # 遍历另一组可选参数（任务类型和随机种子）
            val = locals()[key]  # 从局部变量中获取对应值
            if val is not None:  # 若值不为 None
                engine_kwargs[key] = val  # 添加到 engine_kwargs

        model_info = self.model_info  # 获取模型信息对象（包含数据类型等元数据）
        arch_mapping = {'deepseek_vl2': ['DeepseekVLV2ForCausalLM'], 'glm4v': ['GLM4VForCausalLM']}  # 定义需要特殊架构名称覆盖的模型类型映射表
        if self.model_meta.model_type in arch_mapping:  # 若当前模型类型需要覆盖架构名称
            architectures = arch_mapping[self.model_meta.model_type]  # 获取对应的架构名称列表
            engine_kwargs['hf_overrides'] = {'architectures': architectures}  # 添加 HuggingFace 配置覆盖，确保 vLLM 使用正确的模型架构类
        engine_args = engine_cls(  # 创建引擎参数对象（EngineArgs 或 AsyncEngineArgs）
            model=self.model_dir,  # 模型目录路径
            dtype=dtype_mapping[model_info.torch_dtype],  # 数据类型（转换为 vLLM 接受的字符串格式）
            gpu_memory_utilization=gpu_memory_utilization,  # GPU 显存利用率
            tensor_parallel_size=tensor_parallel_size,  # 张量并行规模
            pipeline_parallel_size=pipeline_parallel_size,  # 流水线并行规模
            max_model_len=max_model_len,  # 最大序列长度
            max_num_seqs=max_num_seqs,  # 最大并发序列数
            disable_log_stats=disable_log_stats,  # 禁用统计日志
            disable_custom_all_reduce=disable_custom_all_reduce,  # 禁用自定义 all-reduce
            enforce_eager=enforce_eager,  # 强制 eager 模式
            trust_remote_code=True,  # 信任远程代码（允许加载自定义模型代码）
            enable_prefix_caching=enable_prefix_caching,  # 启用前缀缓存
            distributed_executor_backend=distributed_executor_backend,  # 分布式执行后端
            **engine_kwargs,  # 展开额外参数
        )
        self.engine_args = engine_args  # 保存引擎参数对象到实例属性
        self.enable_lora = enable_lora  # 保存 LoRA 启用标志
        if max_model_len is not None:  # 若用户显式设置了最大序列长度
            self.max_model_len = max_model_len  # 保存到实例属性
            logger.info(f'Setting max_model_len: {max_model_len}')  # 记录日志

    def _fix_vllm_bug(self) -> None:  # 修复 vLLM 已知 bug（内部方法）
        """函数功能：
        修复 vLLM 0.4~0.6 版本中 tokenizer.__len__ 调用过慢的 bug。
        通过缓存 tokenizer 长度并替换 __len__ 方法来加速。
        
        参数：
        - 无（使用 self.tokenizer）
        
        返回值：
        - None
        
        示例：
        >>> # 内部调用，通常在 __init__ 中自动执行
        >>> self._fix_vllm_bug()
        """
        # fix vllm==0.4 bug (very slow)  # 注释：修复 vLLM 0.4 版本的性能问题
        tokenizer = self.tokenizer  # 获取分词器实例
        if self._version_ge(  # 若当前 vLLM 版本 >= 0.4 且 < 0.6，且分词器未被缓存包装
                '0.4') and not self._version_ge('0.6') and not tokenizer.__class__.__name__.startswith('Cached'):  # 检查版本范围和分词器类型
            _tokenizer_len = len(tokenizer)  # 提前计算并缓存分词器长度（原始调用可能很慢）
            __old_len__ = tokenizer.__class__.__len__  # 保存原始 __len__ 方法的引用

            def __len__(self) -> int:  # 定义新的 __len__ 方法
                if self is tokenizer:  # 若调用者是我们的分词器实例
                    return _tokenizer_len  # 直接返回缓存的长度，避免重复计算
                else:  # 若调用者是其他分词器实例
                    return __old_len__(self)  # 调用原始 __len__ 方法

            tokenizer.__class__.__len__ = __len__  # 替换分词器类的 __len__ 方法（猴子补丁）
    
    def _load_generation_config(self) -> None:  # 加载模型的生成配置（内部方法）
        """函数功能：
        从模型目录加载 generation_config.json 文件，并转换为 vLLM 的 SamplingParams 对象。
        若文件不存在，则使用默认的 SamplingParams。
        
        参数：
        - 无（使用 self.model_dir）
        
        返回值：
        - None（设置 self.generation_config 属性）
        
        示例：
        >>> # 内部调用，通常在 __init__ 中自动执行
        >>> self._load_generation_config()
        >>> assert isinstance(self.generation_config, SamplingParams)
        """
        generation_config_path = os.path.join(self.model_dir, 'generation_config.json')  # 拼接生成配置文件的完整路径
        if os.path.isfile(generation_config_path):  # 若配置文件存在
            generation_config = GenerationConfig.from_pretrained(self.model_dir)  # 从模型目录加载 HuggingFace 的 GenerationConfig 对象
            kwargs = generation_config.to_dict()  # 将配置对象转换为字典
            max_new_tokens = kwargs.get('max_new_tokens')  # 提取 max_new_tokens 参数
            if max_new_tokens is not None:  # 若存在该参数
                kwargs['max_tokens'] = max_new_tokens  # 转换为 vLLM 的 max_tokens 参数（命名差异）
            top_k = kwargs.get('top_k')  # 提取 top_k 参数
            if top_k == 0:  # 若 top_k 为 0（HuggingFace 中表示不使用 top-k）
                kwargs['top_k'] = -1  # 转换为 vLLM 的 -1（vLLM 使用 -1 表示不限制）
            parameters = inspect.signature(SamplingParams).parameters  # 获取 SamplingParams 的构造函数签名，用于过滤无效参数
            for k, v in kwargs.copy().items():  # 遍历配置字典的拷贝（避免在迭代中修改字典）
                if k not in parameters or v is None:  # 若参数不被 SamplingParams 支持，或值为 None
                    kwargs.pop(k)  # 移除该参数，避免传递时报错
            self.generation_config = SamplingParams(**kwargs)  # 创建 SamplingParams 对象并保存到实例属性
        else:  # 若配置文件不存在
            self.generation_config = SamplingParams()  # 使用默认的 SamplingParams

    def _add_stop_words(self, generation_config: SamplingParams, request_config: RequestConfig,  # 添加停止词到生成配置（内部方法）
                        template_meta: TemplateMeta) -> None:  # 模板元数据，包含模板的停止词列表
        """函数功能：
        将请求配置、默认配置和模板配置中的停止词合并，并设置到生成配置中。
        同时设置停止词的 token ID 列表（兼容 vLLM v1 引擎）。
        
        参数：
        - generation_config (SamplingParams): 待修改的生成配置对象
        - request_config (RequestConfig): 当前请求的配置
        - template_meta (TemplateMeta): 模板元数据
        
        返回值：
        - None（直接修改 generation_config）
        
        示例：
        >>> # 内部调用，通常在推理前自动执行
        >>> self._add_stop_words(gen_config, req_config, template_meta)
        """
        stop_words = (request_config.stop or []) + (self.generation_config.stop or []) + template_meta.stop_words  # 合并三个来源的停止词列表（请求配置 + 默认配置 + 模板配置）
        generation_config.stop = self._get_stop_words(stop_words)  # 调用父类方法处理停止词（去重、规范化），并设置到生成配置的 stop 属性
        # stop parameter is not effective in v1 engine (test version: vllm 0.8.5.post)  # 注释：vLLM v1 引擎中 stop 参数不生效，需额外设置 stop_token_ids
        generation_config.stop_token_ids = self._get_stop_token_ids(stop_words)  # 将停止词转换为 token ID 列表，并设置到 stop_token_ids 属性（兼容 v1 引擎）

    @staticmethod  # 声明为静态方法，不依赖实例状态
    def _version_ge(base_version: str):  # 比较 vLLM 版本是否大于等于指定版本
        """函数功能：
        比较当前 vLLM 版本是否大于等于指定的基准版本。
        
        参数：
        - base_version (str): 基准版本号（如 '0.4'）
        
        返回值：
        - bool: True 表示当前版本 >= 基准版本，False 表示当前版本 < 基准版本
        
        示例：
        >>> VllmEngine._version_ge('0.4')  # 若当前 vLLM 版本为 0.5.0
        True
        """
        vllm_version = vllm.__version__  # 获取当前 vLLM 的版本号字符串
        if vllm_version is None or 'dev' in vllm_version:  # 若版本号为 None 或包含 'dev'（开发版）
            return True  # 保守处理，假定开发版支持所有功能
        return version.parse(vllm_version) >= version.parse(base_version)  # 使用 packaging 库解析版本号并比较

    def _add_request(self,  # 向 vLLM 引擎添加推理请求（内部方法）
                     inputs: Dict[str, Any],  # 输入字典，包含 input_ids、images、audios、videos 等
                     generation_config: SamplingParams,  # 生成配置（温度、top_p、停止词等）
                     request_id: str,  # 请求的唯一标识符
                     adapter_request: Optional[AdapterRequest] = None):  # 可选的 LoRA 适配器请求（包含适配器名称和路径）
        """函数功能：
        向 vLLM 引擎添加推理请求，处理 LoRA 适配器、多模态输入和不同任务类型。
        根据 vLLM 版本选择不同的 API 调用方式。
        
        参数：见上方签名注释。
        
        返回值：
        - 同步引擎：返回 None（请求添加到队列）
        - 异步引擎：返回 AsyncIterator（异步生成器）
        - 嵌入任务：返回 AsyncIterator（异步生成器）
        
        示例：
        >>> # 内部调用，通常在推理过程中自动执行
        >>> result = self._add_request(inputs, gen_config, req_id)
        """
        kwargs = {}  # 初始化额外参数字典（用于传递 LoRA 请求等）
        if self.enable_lora and adapter_request:  # 若启用了 LoRA 且提供了适配器请求
            from vllm.lora.request import LoRARequest  # 动态导入 LoRARequest 类（避免未启用时的依赖问题）
            adapter_name = adapter_request.name  # 获取适配器名称
            adapter_path = adapter_request.path  # 获取适配器路径
            if adapter_name in self._adapters_pool:  # 若适配器已在池中缓存
                kwargs['lora_request'] = self._adapters_pool[adapter_name]  # 复用已有的 LoRARequest 对象
            else:  # 若适配器未缓存
                kwargs['lora_request'] = LoRARequest(  # 创建新的 LoRARequest 对象
                    lora_name=adapter_name, lora_path=adapter_path, lora_int_id=len(self._adapters_pool) + 1)  # lora_int_id 为唯一整数 ID（从 1 开始递增）
                self._adapters_pool[adapter_name] = kwargs['lora_request']  # 缓存到适配器池中，供后续请求复用
        input_ids = inputs['input_ids']  # 提取输入的 token ID 列表
        if self._version_ge('0.4.3'):  # 若 vLLM 版本 >= 0.4.3（使用新的 API）
            llm_inputs = {'prompt_token_ids': input_ids}  # 构造新版 API 的输入字典
            mm_data = {}  # 初始化多模态数据字典
            for key in ['images', 'audios', 'videos']:  # 遍历三种多模态输入类型
                media_data = inputs.get(key) or []  # 获取对应的媒体数据列表（若不存在则为空列表）
                if media_data:  # 若存在媒体数据
                    if self._version_ge('0.6'):  # 若 vLLM 版本 >= 0.6（支持多个媒体文件）
                        mm_data = {key.rstrip('s'): media_data[0] if len(media_data) == 1 else media_data}  # 单个文件时取第一个，多个文件时传整个列表，key 去掉末尾的 's'（如 'images' -> 'image'）
                    else:  # 若 vLLM 版本 < 0.6（仅支持单个媒体文件）
                        assert len(media_data) == 1, (  # 断言仅有一个媒体文件
                            f'The current version of vllm only supports single {key}. Please upgrade to vllm >= 0.6.0')  # 提示用户升级 vLLM
                        mm_data = {key.rstrip('s'): media_data[0]}  # 取第一个媒体文件
            if mm_data:  # 若存在多模态数据
                llm_inputs['multi_modal_data'] = mm_data  # 添加到输入字典
            if self.task_type == 'embedding':  # 若任务类型为嵌入生成
                from vllm.pooling_params import PoolingParams  # 动态导入 PoolingParams 类
                if 'task' in inspect.signature(PoolingParams).parameters:  # 若 PoolingParams 支持 task 参数（新版 vLLM）
                    pooling_params = PoolingParams(task='embed')  # 创建带 task='embed' 的 PoolingParams
                else:  # 若不支持 task 参数（旧版 vLLM）
                    pooling_params = PoolingParams()  # 使用默认的 PoolingParams
                return self.engine.encode(llm_inputs, pooling_params, request_id)  # 调用引擎的 encode 方法生成嵌入向量
            elif self.use_async_engine:  # 若使用异步引擎且任务类型为文本生成
                return self.engine.generate(llm_inputs, generation_config, request_id, **kwargs)  # 调用异步引擎的 generate 方法，返回异步生成器
            else:  # 若使用同步引擎且任务类型为文本生成
                return self.engine.add_request(request_id, llm_inputs, generation_config, **kwargs)  # 调用同步引擎的 add_request 方法，将请求添加到队列
        else:  # 若 vLLM 版本 < 0.4.3（使用旧的 API）
            if self.use_async_engine:  # 若使用异步引擎
                return self.engine.generate(None, generation_config, request_id, input_ids, **kwargs)  # 调用旧版异步 API（prompt 参数为 None，直接传 input_ids）
            else:  # 若使用同步引擎
                return self.engine.add_request(request_id, None, generation_config, input_ids, **kwargs)  # 调用旧版同步 API

    def _get_logprobs(self,  # 获取并格式化对数概率信息（内部方法，覆盖父类）
                      logprobs_list: Optional[List[Dict[int, float]]],  # vLLM 返回的原始对数概率列表（每个 token 对应一个字典）
                      token_ids: List[int],  # 生成的 token ID 列表
                      top_logprobs: Optional[int] = None) -> Optional[Dict[str, Any]]:  # 需要返回的 top-k 对数概率数量
        """函数功能：
        从 vLLM 的原始对数概率数据中提取和格式化信息，转换为标准格式。
        vLLM 返回的 logprob 是对象，需要提取其 .logprob 属性。
        
        参数：
        - logprobs_list: vLLM 返回的对数概率列表
        - token_ids: 生成的 token ID 列表
        - top_logprobs: 需要返回的 top-k 数量
        
        返回值：
        - Optional[Dict[str, Any]]: 格式化后的对数概率字典，或 None
        
        示例：
        >>> # 内部调用，通常在构造响应时自动执行
        >>> logprobs = self._get_logprobs(raw_logprobs, tokens, top_k=5)
        """
        if logprobs_list is None or len(token_ids) == 0:  # 若对数概率列表为空或没有生成 token
            return None  # 返回 None
        if len(token_ids) > 0:  # 若有生成的 token
            logprobs_list = logprobs_list[-len(token_ids):]  # 只保留对应生成 token 的对数概率（截取列表尾部）
        for logprobs in logprobs_list:  # 遍历每个 token 的对数概率字典
            for token_id, logprob in logprobs.items():  # 遍历字典中的每个 (token_id, logprob_obj) 对
                logprobs[token_id] = logprob.logprob  # 提取 logprob 对象的 .logprob 属性（浮点数），覆盖原对象
        return super()._get_logprobs(logprobs_list, token_ids, top_logprobs)  # 调用父类方法进行进一步格式化

    def _prepare_generation_config(self, request_config: RequestConfig) -> SamplingParams:  # 准备生成配置参数（内部方法）
        """函数功能：
        根据请求配置和默认配置，构造 vLLM 的 SamplingParams 对象。
        优先使用请求配置中的参数，若未指定则使用默认配置。
        
        参数：
        - request_config (RequestConfig): 当前请求的配置
        
        返回值：
        - SamplingParams: vLLM 的采样参数对象
        
        示例：
        >>> # 内部调用，通常在推理前自动执行
        >>> gen_config = self._prepare_generation_config(req_config)
        """
        kwargs = {'max_tokens': request_config.max_tokens}  # 初始化参数字典，设置最大生成 token 数
        for key in ['temperature', 'top_k', 'top_p', 'repetition_penalty']:  # 遍历采样相关参数
            new_value = getattr(request_config, key)  # 从请求配置中获取参数值
            if new_value is None:  # 若请求配置中未指定该参数
                kwargs[key] = getattr(self.generation_config, key)  # 使用默认配置的值
            else:  # 若请求配置中指定了该参数
                kwargs[key] = new_value  # 使用请求配置的值

        if request_config.logprobs:  # 若请求需要返回对数概率
            kwargs['logprobs'] = 1  # 至少返回 top-1 对数概率
            if request_config.top_logprobs is not None:  # 若指定了 top-k 对数概率数量
                kwargs['logprobs'] = max(1, request_config.top_logprobs)  # 使用指定值（至少为 1）

        # TODO: beam search  # 注释：beam search 支持待实现
        for key in ['n', 'best_of', 'frequency_penalty', 'presence_penalty', 'seed']:  # 遍历其他采样参数
            kwargs[key] = getattr(request_config, key)  # 从请求配置中获取并添加到参数字典

        res = SamplingParams(**kwargs)  # 创建 SamplingParams 对象

        if hasattr(res, 'output_kind') and res.n > 1:  # 若 SamplingParams 有 output_kind 属性（vLLM v1 引擎）且 n > 1（需要生成多个候选）
            # fix n > 1 in V1 Engine  # 注释：修复 v1 引擎中 n > 1 时的问题
            from vllm.sampling_params import RequestOutputKind  # 导入输出类型枚举
            res.output_kind = RequestOutputKind.FINAL_ONLY  # 设置为仅返回最终结果（不返回中间步骤）
        return res  # 返回采样参数对象

    @property  # 声明为只读属性
    def inner_model(self):  # 获取 vLLM 引擎内部的模型实例
        """函数功能：
        返回 vLLM 引擎内部的原始 PyTorch 模型实例。
        用于高级调试和模型权重访问。
        
        参数：
        - 无
        
        返回值：
        - torch.nn.Module: 底层 PyTorch 模型
        
        示例：
        >>> model = engine.inner_model
        >>> # 可访问模型权重、层等
        """
        return self.engine.model_executor.driver_worker.worker.model_runner.model  # 通过引擎的执行器层层访问到底层模型实例

    @property  # 声明为只读属性
    def inner_model_executor(self):  # 获取 vLLM 引擎的模型执行器
        """函数功能：
        返回 vLLM 引擎的模型执行器对象。
        用于访问分布式执行、worker 管理等高级功能。
        
        参数：
        - 无
        
        返回值：
        - ModelExecutor: vLLM 的模型执行器对象
        
        示例：
        >>> executor = engine.inner_model_executor
        >>> # 可访问分布式执行相关功能
        """
        return self.engine.model_executor  # 返回引擎的模型执行器对象

    async def _infer_stream_async(  # 异步流式推理（内部方法）
        self,  # 实例自身引用
        template: Template,  # 对话模板对象
        inputs: Dict[str, Any],  # 输入字典（包含 input_ids、多模态数据等）
        generation_config: SamplingParams,  # 生成配置参数
        adapter_request: Optional[AdapterRequest],  # 可选的 LoRA 适配器请求
        request_config: RequestConfig,  # 请求配置（包含流式、对数概率等选项）
    ) -> AsyncIterator[ChatCompletionStreamResponse]:  # 返回异步迭代器，产出流式响应
        """函数功能：
        以异步流式方式执行推理，逐步产出增量响应。
        适用于需要实时显示生成结果的场景（如聊天界面）。
        
        参数：见上方签名注释。
        
        返回值：
        - AsyncIterator[ChatCompletionStreamResponse]: 异步迭代器，逐步产出流式响应对象
        
        示例：
        >>> # 内部调用，通常在 infer_async 中自动执行
        >>> async for chunk in self._infer_stream_async(...):
        ...     print(chunk.choices[0].delta.content)
        """
        request_id = random_uuid()  # 生成唯一的请求 ID
        result_generator = self._add_request(inputs, generation_config, request_id, adapter_request=adapter_request)  # 向 vLLM 引擎添加请求，获取结果生成器
        infer_streamers = [InferStreamer(template) for _ in range(generation_config.n)]  # 为每个候选创建流式输出器（处理增量 token 的打印）
        token_idxs = [0 for _ in range(generation_config.n)]  # 初始化每个候选的 token 索引（用于跟踪已处理的 token 数量）
        async for result in result_generator:  # 异步迭代 vLLM 引擎的输出结果
            res = self._create_chat_completion_stream_response(result, template, request_config, request_id,  # 构造流式响应对象
                                                               infer_streamers, token_idxs)  # 传入流式输出器和 token 索引
            if res is None:  # 若响应为 None（无新增内容）
                continue  # 跳过本次迭代
            yield res  # 产出流式响应对象给调用方

    def _create_chat_completion_stream_response(self, result, template, request_config, request_id, infer_streamers,  # 构造流式聊天补全响应（内部方法）
                                                token_idxs) -> Optional[ChatCompletionStreamResponse]:  # token 索引列表（跟踪每个候选已处理的 token 数）
        """函数功能：
        从 vLLM 的原始输出构造流式聊天补全响应对象。
        处理增量文本、对数概率、工具调用等信息。
        
        参数：
        - result: vLLM 引擎的原始输出结果
        - template: 对话模板对象
        - request_config: 请求配置
        - request_id: 请求 ID
        - infer_streamers: 流式输出器列表
        - token_idxs: token 索引列表（可变，会被修改）
        
        返回值：
        - Optional[ChatCompletionStreamResponse]: 流式响应对象，或 None（若无新增内容）
        
        示例：
        >>> # 内部调用，通常在流式推理中自动执行
        >>> response = self._create_chat_completion_stream_response(...)
        """
        is_diff = False  # 标记是否有新增文本内容
        is_finished = False  # 标记是否有候选完成生成
        for output in result.outputs:  # 遍历 vLLM 结果中的每个输出（多个候选时会有多个 output）
            output.token_ids = list(output.token_ids)  # 将 token_ids 转换为列表（可能原本是 tuple）
            output.delta_text = infer_streamers[output.index].get_printable_text(output.token_ids, output.finished())  # 使用流式输出器提取增量可打印文本（处理特殊 token、部分 UTF-8 字符等）
            output.is_finished = output.finish_reason is not None  # 判断该输出是否完成（有 finish_reason 表示完成）
            is_diff |= bool(output.delta_text)  # 更新 is_diff 标记（任一输出有新增文本则为 True）
            is_finished |= output.is_finished  # 更新 is_finished 标记（任一输出完成则为 True）
        if not is_diff and not is_finished:  # 若既无新增文本也无完成标记
            return  # 返回 None，跳过本次响应（避免发送空响应）

        num_generated_tokens = sum(len(output.token_ids) for output in result.outputs)  # 计算所有候选的总生成 token 数
        usage_info = self._get_usage_info(len(result.prompt_token_ids), num_generated_tokens)  # 构造用量信息（prompt token 数、生成 token 数）
        choices = []  # 初始化选项列表
        for output in result.outputs:  # 遍历每个输出
            logprobs = self._get_logprobs(output.logprobs, output.token_ids[token_idxs[output.index]:],  # 获取本次新增 token 的对数概率（从上次索引到当前索引的 token）
                                          request_config.top_logprobs)  # 传入 top_logprobs 参数
            token_idxs[output.index] = len(output.token_ids)  # 更新该候选的 token 索引为当前总长度（下次从这里继续）
            toolcall = None  # 初始化工具调用为 None
            if output.is_finished:  # 若该输出已完成
                toolcall = self._get_toolcall(template.decode(output.token_ids), template)  # 解码完整输出并提取工具调用信息（仅在完成时提取）
            choice = ChatCompletionResponseStreamChoice(  # 构造流式选项对象
                index=output.index,  # 候选索引（从 0 开始）
                delta=DeltaMessage(role='assistant', content=output.delta_text, tool_calls=toolcall),  # 增量消息（角色、增量文本、工具调用）
                finish_reason=output.finish_reason,  # 完成原因（'stop', 'length' 等，未完成时为 None）
                logprobs=logprobs)  # 对数概率信息
            choices.append(choice)  # 添加到选项列表
        return ChatCompletionStreamResponse(model=self.model_name, choices=choices, usage=usage_info, id=request_id)  # 返回完整的流式响应对象

    def _create_embedding_response(self, result, template, generation_config, request_id) -> EmbeddingResponse:  # 构造嵌入响应（内部方法）
        """函数功能：
        从 vLLM 的嵌入任务输出构造嵌入响应对象。
        提取嵌入向量并转换为标准格式。
        
        参数：
        - result: vLLM 引擎的嵌入任务输出结果
        - template: 对话模板对象（嵌入任务中未使用，保持签名一致）
        - generation_config: 生成配置（嵌入任务中未使用，保持签名一致）
        - request_id: 请求 ID
        
        返回值：
        - EmbeddingResponse: 嵌入响应对象
        
        示例：
        >>> # 内部调用，通常在嵌入任务推理后自动执行
        >>> response = self._create_embedding_response(result, template, gen_config, req_id)
        """
        assert result is not None  # 断言结果不为 None（嵌入任务必须有输出）
        embedding = result.outputs.data.cpu().numpy().tolist()  # 将嵌入张量从 GPU 移到 CPU，转为 NumPy 数组，再转为 Python 列表
        usage_info = self._get_usage_info(len(result.prompt_token_ids), 0)  # 构造用量信息（嵌入任务不生成 token，生成数为 0）
        return EmbeddingResponse(  # 返回嵌入响应对象
            model=self.model_name, data=[EmbeddingResponseData(embedding=embedding)], usage=usage_info, id=request_id)  # 包含模型名、嵌入数据、用量、请求 ID

    def _create_chat_completion_response(  # 构造非流式聊天补全响应（内部方法）
        self,  # 实例自身引用
        result,  # vLLM 引擎的原始输出结果（完整生成结果）
        template,  # 对话模板对象
        request_config,  # 请求配置
        request_id,  # 请求 ID
    ) -> ChatCompletionResponse:  # 返回完整的聊天补全响应对象
        """函数功能：
        从 vLLM 的完整输出构造非流式聊天补全响应对象。
        处理多个候选、解码文本、对数概率、工具调用等信息。
        
        参数：见上方签名注释。
        
        返回值：
        - ChatCompletionResponse: 完整的聊天补全响应对象
        
        示例：
        >>> # 内部调用，通常在非流式推理后自动执行
        >>> response = self._create_chat_completion_response(...)
        """
        assert result is not None  # 断言结果不为 None（必须有输出）
        num_generated_tokens = sum(len(output.token_ids) for output in result.outputs)  # 计算所有候选的总生成 token 数
        usage_info = self._get_usage_info(len(result.prompt_token_ids), num_generated_tokens)  # 构造用量信息
        choices = []  # 初始化选项列表
        for output in result.outputs:  # 遍历每个输出候选
            output.token_ids = list(output.token_ids)  # 将 token_ids 转换为列表
            response = template.decode(output.token_ids)  # 使用模板解码 token_ids 为文本（处理特殊 token、格式化等）
            logprobs = self._get_logprobs(output.logprobs, output.token_ids, request_config.top_logprobs)  # 获取对数概率信息
            toolcall = self._get_toolcall(response, template)  # 从解码文本中提取工具调用信息（若存在）
            token_ids = template.skip_stop_tokens(output.token_ids) if request_config.return_details else None  # 若需要返回详细信息，则跳过停止 token 并返回 token_ids，否则为 None
            choice = ChatCompletionResponseChoice(  # 构造响应选项对象
                index=output.index,  # 候选索引
                message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),  # 完整消息（角色、内容、工具调用）
                finish_reason=output.finish_reason,  # 完成原因
                logprobs=logprobs,  # 对数概率信息
                token_ids=token_ids)  # token_ids（若 return_details=True）
            choices.append(choice)  # 添加到选项列表
        prompt_token_ids = result.prompt_token_ids if request_config.return_details else None  # 若需要返回详细信息，则返回 prompt 的 token_ids，否则为 None
        return ChatCompletionResponse(  # 返回完整的聊天补全响应对象
            model=self.model_name, choices=choices, usage=usage_info, id=request_id, prompt_token_ids=prompt_token_ids)  # 包含模型名、选项列表、用量、请求 ID、prompt token_ids

    async def _infer_full_async(  # 异步非流式推理（内部方法）
        self,  # 实例自身引用
        template: Template,  # 对话模板对象
        inputs: Dict[str, Any],  # 输入字典
        generation_config: SamplingParams,  # 生成配置参数
        adapter_request: Optional[AdapterRequest],  # 可选的 LoRA 适配器请求
        request_config: RequestConfig,  # 请求配置
    ) -> Union[ChatCompletionResponse, EmbeddingResponse]:  # 返回聊天补全响应或嵌入响应
        """函数功能：
        以异步非流式方式执行推理，等待完整结果后一次性返回。
        适用于不需要实时显示的场景（如批处理、API 调用等）。
        
        参数：见上方签名注释。
        
        返回值：
        - Union[ChatCompletionResponse, EmbeddingResponse]: 完整响应对象（根据任务类型）
        
        示例：
        >>> # 内部调用，通常在 infer_async 中自动执行
        >>> response = await self._infer_full_async(...)
        """
        request_id = random_uuid()  # 生成唯一的请求 ID
        result_generator = self._add_request(inputs, generation_config, request_id, adapter_request=adapter_request)  # 向 vLLM 引擎添加请求，获取结果生成器
        result = None  # 初始化结果为 None
        async for result in result_generator:  # 异步迭代结果生成器，直到完成
            pass  # 不处理中间结果，仅等待最终结果（最后一次迭代的 result 即为完整结果）
        if self.task_type == 'embedding':  # 若任务类型为嵌入生成
            return self._create_embedding_response(result, template, generation_config, request_id)  # 构造并返回嵌入响应
        else:  # 若任务类型为文本生成
            return self._create_chat_completion_response(result, template, request_config, request_id)  # 构造并返回聊天补全响应

    def _batch_infer_stream(self, *args, **kwargs):  # 批量流式推理（内部方法，覆盖父类）
        """函数功能：
        执行批量流式推理，覆盖父类方法以修复 vLLM 的一个已知问题。
        在调用父类方法前，重置并行 worker 任务状态。
        
        参数：
        - *args: 位置参数，透传给父类方法
        - **kwargs: 关键字参数，透传给父类方法
        
        返回值：
        - 与父类方法相同（批量流式响应）
        
        示例：
        >>> # 内部调用，通常在批量流式推理时自动执行
        >>> results = self._batch_infer_stream(...)
        """
        if hasattr(self.engine, 'engine'):  # 若引擎有嵌套的 engine 属性（某些 vLLM 版本的结构）
            self.engine.engine.model_executor.parallel_worker_tasks = None  # 重置并行 worker 任务为 None（修复已知的并发问题）
        return super()._batch_infer_stream(*args, **kwargs)  # 调用父类的批量流式推理方法

    def infer(  # 同步推理入口（公开方法）
        self,  # 实例自身引用
        infer_requests: List[InferRequest],  # 批量推理请求列表
        request_config: Optional[RequestConfig] = None,  # 可选的统一推理配置
        metrics: Optional[List[Metric]] = None,  # 可选的指标采集器列表
        *,  # 仅限关键字参数分隔符
        template: Optional[Template] = None,  # 可选的对话模板，None 则使用默认模板
        use_tqdm: Optional[bool] = None,  # 是否显示进度条，None 则自动判断（批量时显示）
        adapter_request: Optional[AdapterRequest] = None,  # 可选的 LoRA 适配器请求
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:  # 返回响应列表（非流式或流式迭代器）
        """函数功能：
        同步推理入口，支持批量处理和流式/非流式输出。
        异步引擎模式下，调用父类方法；同步引擎模式下，使用 vLLM 的 step() API。
        
        参数：见上方签名注释。
        
        返回值：
        - List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]: 
          与输入等长的响应列表，元素为非流式响应或流式响应迭代器。
        
        示例：
        >>> engine = VllmEngine('Qwen/Qwen-7B-Chat', use_async_engine=False)
        >>> reqs = [InferRequest(messages=[{"role": "user", "content": "hi"}])]
        >>> responses = engine.infer(reqs)
        >>> print(responses[0].choices[0].message.content)
        """
        if self.use_async_engine:  # 若使用异步引擎
            return super().infer(  # 调用父类方法（父类会自动转为异步并阻塞等待）
                infer_requests,  # 推理请求列表
                request_config,  # 请求配置
                metrics,  # 指标采集器
                template=template,  # 模板
                use_tqdm=use_tqdm,  # 进度条
                adapter_request=adapter_request,  # LoRA 适配器
            )
        else:  # 若使用同步引擎（需手动调用 step() API）
            request_config = deepcopy(request_config or RequestConfig())  # 拷贝请求配置，避免副作用
            if request_config.stream and len(infer_requests) > 1:  # 若需要流式输出且有多个请求
                raise ValueError('If you want to use stream batch inference, you need to set use_async_engine to True.')  # 抛出异常，同步引擎不支持批量流式（需异步引擎）
            if use_tqdm is None:  # 若未指定是否显示进度条
                use_tqdm = len(infer_requests) > 1  # 批量请求时默认显示
            rank = get_dist_setting()[0]  # 获取当前进程的分布式排名
            if is_dist() and rank % self.engine_args.tensor_parallel_size != 0:  # 若使用分布式且当前进程不是张量并行组的主进程
                use_tqdm = False  # 禁用进度条（避免多进程重复显示）
            if template is None:  # 若未指定模板
                template = self.default_template  # 使用默认模板
            template.set_mode('vllm')  # 设置模板为 vLLM 模式（影响编码行为）
            batched_inputs, error_list = self._batch_encode(  # 批量编码请求，返回输入列表和错误列表
                infer_requests, template=template, strict=getattr(self, 'strict', True))  # 严格模式下编码错误会抛出异常
            request_id_list = []  # 初始化请求 ID 列表
            for i, inputs in enumerate(batched_inputs):  # 遍历每个编码后的输入
                request_id = str(self._request_count)  # 生成请求 ID（使用计数器）
                request_id_list.append(request_id)  # 添加到列表
                self._request_count += 1  # 递增计数器
                _request_config = deepcopy(request_config)  # 为每个请求拷贝配置（避免互相影响）
                self.set_default_max_tokens(_request_config, inputs)  # 根据输入长度设置默认的 max_tokens
                generation_config = self._prepare_generation_config(_request_config)  # 准备生成配置
                if generation_config.seed is not None:  # 若设置了随机种子
                    generation_config.seed += i  # 为每个请求使用不同的种子（确保多样性）
                self._add_stop_words(generation_config, _request_config, template.template_meta)  # 添加停止词
                self._add_request(inputs, generation_config, request_id, adapter_request=adapter_request)  # 向引擎添加请求
            prog_bar = tqdm(total=len(batched_inputs), dynamic_ncols=True, disable=not use_tqdm)  # 创建进度条
            outputs = {}  # 初始化输出字典（request_id -> output）
            if request_config.stream:  # 若需要流式输出

                def _gen_wrapper():  # 定义流式生成器包装函数
                    """内部生成器：逐步调用引擎的 step() 方法，产出流式响应。"""
                    infer_streamers = [InferStreamer(template) for _ in range(generation_config.n)]  # 为每个候选创建流式输出器
                    token_idxs = [0 for _ in range(generation_config.n)]  # 初始化每个候选的 token 索引
                    while self.engine.has_unfinished_requests():  # 循环直到所有请求完成
                        result = self.engine.step()  # 执行一步推理（同步 API），返回当前步骤的结果列表
                        if not result:  # 若结果为空（无输出）
                            continue  # 跳过本次循环
                        result = result[0]  # 取第一个结果（同步引擎单请求流式时仅有一个结果）
                        res = self._create_chat_completion_stream_response(result, template, request_config, request_id,  # 构造流式响应
                                                                           infer_streamers, token_idxs)  # 传入流式输出器和索引
                        if res is None:  # 若响应为 None（无新增内容）
                            continue  # 跳过本次循环
                        yield res  # 产出流式响应给调用方
                        if result.finished:  # 若该结果已完成
                            break  # 终止生成器

                    self._update_metrics(res, metrics)  # 更新指标（使用最后一个响应）

                return [_gen_wrapper()]  # 返回包含生成器的列表（长度为 1，对应单个请求）
            else:  # 若需要非流式输出（批量处理）
                while self.engine.has_unfinished_requests():  # 循环直到所有请求完成
                    step_outputs = self.engine.step()  # 执行一步推理，返回当前步骤的所有输出
                    for output in step_outputs:  # 遍历每个输出
                        if output.finished:  # 若该输出已完成
                            outputs[output.request_id] = output  # 保存到输出字典
                            prog_bar.update()  # 更新进度条（完成一个请求）
                prog_bar.close()  # 关闭进度条
                outputs = [outputs[request_id] for request_id in request_id_list]  # 按请求顺序重新排列输出（保持与输入顺序一致）
                res = [  # 为每个输出构造聊天补全响应
                    self._create_chat_completion_response(result, template, request_config, request_id)  # 构造响应对象
                    for request_id, result in zip(request_id_list, outputs)  # 遍历请求 ID 和对应输出
                ]
                self._update_metrics(res, metrics)  # 更新指标（批量）
                return self._add_error_list(res, error_list)  # 将编码错误添加回结果列表（对应编码失败的请求）

    async def infer_async(  # 异步推理入口（公开方法）
        self,  # 实例自身引用
        infer_request: InferRequest,  # 单个推理请求
        request_config: Optional[RequestConfig] = None,  # 可选的推理配置
        *,  # 仅限关键字参数分隔符
        template: Optional[Template] = None,  # 可选的对话模板
        adapter_request: Optional[AdapterRequest] = None,  # 可选的 LoRA 适配器请求
        pre_infer_hook=None,  # 可选的推理前钩子函数（用于修改推理参数）
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:  # 返回非流式响应或异步流式迭代器
        """函数功能：
        异步推理入口，支持单个请求的流式/非流式输出。
        需在初始化时设置 use_async_engine=True。
        
        参数：见上方签名注释。
        
        返回值：
        - Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
          非流式返回 ChatCompletionResponse；流式返回 AsyncIterator。
        
        示例：
        >>> engine = VllmEngine('Qwen/Qwen-7B-Chat', use_async_engine=True)
        >>> req = InferRequest(messages=[{"role": "user", "content": "hi"}])
        >>> # 非流式
        >>> response = await engine.infer_async(req)
        >>> print(response.choices[0].message.content)
        >>> # 流式
        >>> async for chunk in await engine.infer_async(req, RequestConfig(stream=True)):
        ...     print(chunk.choices[0].delta.content, end='')
        """
        if not self.use_async_engine:  # 若未启用异步引擎
            raise ValueError('If you want to use `infer_async`, you need to pass `use_async_engine` as True.')  # 抛出异常
        request_config = deepcopy(request_config or RequestConfig())  # 拷贝请求配置
        if template is None:  # 若未指定模板
            template = self.default_template  # 使用默认模板

        template.set_mode('vllm')  # 设置模板为 vLLM 模式
        loop = asyncio.get_running_loop()  # 获取当前事件循环
        with torch.inference_mode():  # 启用推理模式（禁用梯度计算）
            inputs = await loop.run_in_executor(None, template.encode, infer_request)  # 在线程池中异步执行编码（避免阻塞事件循环）
        self.set_default_max_tokens(request_config, inputs)  # 设置默认的 max_tokens
        generation_config = self._prepare_generation_config(request_config)  # 准备生成配置
        self._add_stop_words(generation_config, request_config, template.template_meta)  # 添加停止词
        kwargs = {  # 构造推理参数字典
            'template': template,  # 模板
            'inputs': inputs,  # 输入
            'generation_config': generation_config,  # 生成配置
            'adapter_request': adapter_request,  # LoRA 适配器
            'request_config': request_config,  # 请求配置
        }
        if pre_infer_hook:  # 若提供了推理前钩子
            kwargs = pre_infer_hook(kwargs)  # 调用钩子修改参数（可用于自定义逻辑）
        if request_config.stream:  # 若需要流式输出
            return self._infer_stream_async(**kwargs)  # 返回异步流式迭代器
        else:  # 若需要非流式输出
            return await self._infer_full_async(**kwargs)  # 等待并返回完整响应

    @staticmethod  # 声明为静态方法（不依赖实例状态）
    def patch_remove_log():  # 补丁方法：移除 vLLM 的某些日志输出
        """函数功能：
        应用补丁以移除 vLLM 异步引擎的某些日志输出。
        替换 `_log_task_completion` 函数，静默处理 CancelledError，避免干扰日志。
        
        参数：
        - 无
        
        返回值：
        - None
        
        示例：
        >>> # 内部调用，通常在 __init__ 中自动执行
        >>> VllmEngine.patch_remove_log()
        """
        from vllm.engine import async_llm_engine  # 动态导入 vLLM 的异步引擎模块

        async_llm_engine._origin_log_task_completion = async_llm_engine._log_task_completion  # 保存原始的任务完成日志函数

        def new_log_task_completion(task, error_callback) -> None:  # 定义新的任务完成日志函数
            """新的任务完成日志函数：静默处理 CancelledError，避免不必要的日志输出。"""
            try:  # 尝试获取任务结果
                return_value = task.result()  # 获取任务返回值（若任务正常完成）
                raise AssertionError(f'The engine background task should never finish without an '  # 抛出断言错误（后台任务不应正常完成）
                                     f'exception. {return_value}')  # 包含返回值信息
            except asyncio.exceptions.CancelledError:  # 若任务被取消
                pass  # 静默处理，不输出日志（避免干扰）

        async_llm_engine._log_task_completion = new_log_task_completion  # 替换为新的日志函数（猴子补丁）
