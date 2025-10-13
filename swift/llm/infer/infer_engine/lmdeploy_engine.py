"""模块功能概述：
该模块实现基于 LMDeploy 引擎的高性能推理引擎 `LmdeployEngine`，用于在生产环境中进行大语言模型的推理。提供：
- 异步推理接口，支持流式和非流式响应模式；
- 基于 LMDeploy 的 TurbomindEngine 或 PytorchEngine 后端；
- 多模态输入处理（图像等），支持视觉语言模型；
- 会话管理和上下文安全执行机制；
- 与 Swift 框架的无缝集成。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，标注代码版权所有者
import asyncio  # 引入 asyncio 模块，用于异步编程和协程调度
import inspect  # 引入 inspect 模块，用于运行时检查函数签名和参数RTTI
import os  # 引入 os 模块，用于文件路径操作和环境变量访问
import time  # 引入 time 模块，用于生成时间戳（会话 ID）
from contextlib import contextmanager  # 引入 contextmanager 装饰器，用于创建上下文管理器
from copy import deepcopy  # 引入 deepcopy，用于对配置进行深拷贝，避免副作用
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union  # 引入类型注解，明确参数与返回值类型

import lmdeploy  # 引入 lmdeploy 库，高性能 LLM 推理框架
import torch  # 引入 PyTorch 库，用于张量操作和推理模式
from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig, VisionConfig, pipeline  # 引入 LMDeploy 的引擎配置、视觉配置和 pipeline 工厂函数
from lmdeploy.api import autoget_backend_config  # 引入自动获取后端配置的函数
from lmdeploy.serve import async_engine  # 引入 LMDeploy 的异步引擎模块
from packaging import version  # 引入 version 模块，用于版本比较
from transformers import GenerationConfig  # 引入 HuggingFace 的生成配置类
from transformers.utils.versions import require_version  # 引入版本检查函数

from swift.llm import InferRequest, Template, TemplateMeta, get_model_tokenizer  # 引入 Swift 框架的推理请求、模板、模板元数据和模型加载工具
from swift.plugin import Metric  # 引入 Metric 类型，用于传递统计/用量指标采集器
from swift.utils import get_logger, get_seed  # 引入日志工具和随机种子生成工具
from ..protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,  # 引入协议层数据类：响应、选项、流式响应
                        ChatCompletionStreamResponse, ChatMessage, DeltaMessage, RequestConfig)  # 引入聊天消息、增量消息、请求配置
from .infer_engine import InferEngine  # 引入父类 InferEngine，提供通用推理接口与工具方法
from .patch import patch_auto_config, patch_auto_tokenizer  # 引入补丁函数，用于在 pipeline 初始化时注入自定义配置和分词器
from .utils import InferStreamer  # 引入流式输出器，用于处理增量 token 的解码

try:  # 尝试导入 EngineGenerationConfig（LMDeploy < 0.6）
    from lmdeploy import EngineGenerationConfig as LmdeployGenerationConfig  # 引入 LMDeploy 的生成配置类（旧版本命名）
except ImportError:  # 若导入失败（LMDeploy >= 0.6）
    # compat lmdeploy >= 0.6.*  # 兼容注释：LMDeploy 0.6 及以上版本
    from lmdeploy import GenerationConfig as LmdeployGenerationConfig  # 引入 LMDeploy 的生成配置类（新版本命名）

logger = get_logger()  # 获取日志记录器实例，用于输出日志信息


class LmdeployEngine(InferEngine):

    """类功能：
    定义基于 LMDeploy 的推理引擎类，继承通用推理引擎基类
    LmdeployEngine 负责基于 LMDeploy 引擎进行高性能的大语言模型推理。
    
    - 角色：推理引擎封装，使用 LMDeploy（TurbomindEngine 或 PytorchEngine）进行推理；
    - 能力：支持异步推理、流式/非流式输出、多模态输入、会话管理；
    - 适用：生产环境中的高性能推理服务，支持 Turbomind（推荐）和 PyTorch 两种后端；
    - 线程/协程安全：基于异步接口，支持并发请求。
    
    特点：
    - 高性能：LMDeploy 针对推理场景优化，吞吐量高、延迟低
    - 多模态：支持视觉语言模型（如 GLM4V、DeepseekVL 等）
    - 会话管理：每个请求使用唯一会话 ID，确保上下文隔离
    """

    def __init__(  # 初始化 LmdeployEngine 实例，配置模型加载、引擎参数和后端选择
        self,  # 实例自身引用
        model_id_or_path: str,  # 模型的 HuggingFace ID 或本地路径，用于加载模型和分词器
        torch_dtype: Optional[torch.dtype] = None,  # 模型推理使用的数据类型（如 float16、bfloat16），None 则自动推断
        *,  # 仅限关键字参数分隔符，后续参数必须以关键字形式传入
        model_type: Optional[str] = None,  # 模型类型标识（如 'qwen', 'llama'），用于选择特定的模板和配置
        use_hf: Optional[bool] = None,  # 是否强制从 HuggingFace Hub 下载模型，None 则自动判断
        hub_token: Optional[str] = None,  # HuggingFace Hub 访问令牌，用于下载私有模型
        revision: Optional[str] = None,  # 模型的版本/分支名称，用于下载特定版本
        # engine_kwargs  # 分组注释：以下为 LMDeploy 引擎的核心配置参数
        tp: int = 1,  # Tensor Parallelism 张量并行度，多 GPU 时用于分割模型（例如 tp=4 表示 4 卡并行）
        session_len: Optional[int] = None,  # 会话最大长度（context window），None 则使用模型配置的默认值
        cache_max_entry_count: float = 0.8,  # KV 缓存的最大条目数比例（0.8 表示使用 80% 显存作为缓存）
        quant_policy: int = 0,  # 量化策略（0=不量化，4=4bit 量化，8=8bit 量化）
        vision_batch_size: int = 1,  # 视觉编码器的最大批大小（多模态模型使用）
        engine_kwargs: Optional[Dict[str, Any]] = None,  # 其他传递给 LMDeploy 引擎的额外参数字典
        template: Optional[Template] = None,  # 预设的对话模板，None 则根据 model_type 自动选择
        devices: Optional[List[int]] = None,  # 指定使用的 GPU 设备列表（例如 [0, 1] 表示使用 GPU 0 和 1）
    ) -> None:  # 无返回值
        """函数功能：
        初始化 `LmdeployEngine` 实例，完成以下步骤：
        1. 加载分词器（不加载模型权重，由 LMDeploy 接管）；
        2. 准备 LMDeploy 引擎参数（Tensor Parallelism、KV 缓存、量化等）；
        3. 创建 LMDeploy pipeline 实例（TurbomindEngine 或 PytorchEngine）；
        4. 加载生成配置（从 generation_config.json）。
        
        参数：见上方签名注释。
        
        返回值：
        - None（初始化实例属性）
        
        示例：
        >>> # 单卡推理
        >>> engine = LmdeployEngine('Qwen/Qwen-7B-Chat', tp=1, session_len=4096)
        >>> 
        >>> # 多卡推理（4 卡 Tensor Parallelism）
        >>> engine = LmdeployEngine('meta-llama/Llama-2-70b-chat-hf', tp=4, devices=[0,1,2,3])
        >>> 
        >>> # 多模态模型
        >>> engine = LmdeployEngine('OpenGVLab/InternVL-Chat-V1-5', vision_batch_size=8)
        """
        if engine_kwargs is None:  # 若未提供额外引擎参数
            engine_kwargs = {}  # 初始化为空字典，避免后续操作报错
        
        self.processor = get_model_tokenizer(  # 加载分词器（processor）和模型配置，返回值为 (model, tokenizer, config) 元组
            model_id_or_path,  # 模型路径或 HuggingFace ID
            torch_dtype,  # 数据类型（如 float16）
            load_model=False,  # 不加载模型权重（LMDeploy 会自行加载），仅获取分词器和配置
            download_model=True,  # 允许从 HuggingFace Hub 下载模型文件（若本地不存在）
            model_type=model_type,  # 模型类型（如 'qwen'），用于匹配特定逻辑
            use_hf=use_hf,  # 是否强制使用 HuggingFace Hub
            hub_token=hub_token,  # HuggingFace 访问令牌
            revision=revision)[1]  # 取元组的第二个元素（tokenizer），索引 [1] 提取分词器
        
        self._post_init(template)  # 调用父类的后初始化方法，设置默认模板、模型信息等（继承自 InferEngine）

        if self.max_model_len is not None:  # 若设置了最大模型长度（从配置中读取）
            self.max_model_len -= 1  # 减 1（为 LMDeploy 的特殊 token 预留空间，如 EOS）
        
        self._prepare_engine_kwargs(  # 调用内部方法准备 LMDeploy 引擎参数
            tp=tp,  # 张量并行度
            session_len=session_len,  # 会话最大长度
            cache_max_entry_count=cache_max_entry_count,  # KV 缓存比例
            quant_policy=quant_policy,  # 量化策略
            vision_batch_size=vision_batch_size,  # 视觉编码器批大小
            devices=devices,  # GPU 设备列表
            **engine_kwargs)  # 展开额外参数

        self.config.torch_dtype = torch_dtype or self.model_info.torch_dtype  # 设置配置的数据类型
        # 优先使用用户指定的 torch_dtype，否则使用模型默认的数据类型

        self._prepare_engine()  # 调用内部方法创建 LMDeploy pipeline 实例
        self._load_generation_config()  # 调用内部方法加载生成配置

    def _prepare_engine_kwargs(self,  # 准备 LMDeploy 引擎参数（内部方法）
                               tp: int = 1,  # 张量并行度
                               session_len: Optional[int] = None,  # 会话最大长度
                               cache_max_entry_count: float = 0.8,  # KV 缓存比例
                               quant_policy: int = 0,  # 量化策略
                               vision_batch_size: int = 1,  # 视觉编码器批大小
                               devices: Optional[List[int]] = None,  # GPU 设备列表
                               **engine_kwargs):  # 其他额外参数
        """函数功能：
        准备 LMDeploy 引擎的配置参数，包括后端配置和 pipeline 配置。
        
        参数：见上方签名注释。
        
        返回值：
        - None（设置实例属性 backend_config 和 pipeline_kwargs）
        
        示例：
        >>> # 内部调用
        >>> self._prepare_engine_kwargs(tp=2, session_len=8192)
        """
        engine_kwargs['tp'] = tp  # 设置张量并行度（用于多 GPU 推理）
        engine_kwargs['session_len'] = session_len  # 设置会话最大长度（context window）
        engine_kwargs['cache_max_entry_count'] = cache_max_entry_count  # 设置 KV 缓存的最大条目数比例
        engine_kwargs['quant_policy'] = quant_policy  # 设置量化策略（0/4/8）
        
        if 'devices' in inspect.signature(TurbomindEngineConfig).parameters:  # 若 TurbomindEngineConfig 支持 devices 参数（版本检查）
            engine_kwargs['devices'] = devices  # 设置 GPU 设备列表
        
        backend_config = TurbomindEngineConfig(**engine_kwargs)  # 创建 TurbomindEngineConfig 配置对象
        # TurbomindEngineConfig 是 LMDeploy 的 Turbomind 后端配置（推荐后端，性能最优）
        
        backend_config = autoget_backend_config(self.model_dir, backend_config)  # 自动获取最优后端配置
        # autoget_backend_config 会根据模型类型和硬件环境自动选择最优的后端（Turbomind 或 PyTorch）
        
        self.backend_config = backend_config  # 保存后端配置
        logger.info(f'backend_config: {backend_config}')  # 记录后端配置信息到日志

        pipeline_kwargs = {}  # 初始化 pipeline 参数字典
        is_multimodal = self.model_meta.is_multimodal  # 检查模型是否为多模态模型
        
        if is_multimodal:  # 若模型支持多模态（如视觉语言模型）
            require_version(  # 检查 LMDeploy 版本要求
                'lmdeploy<0.9', 'LmdeployEngine will no longer maintain inference for '  # 要求 LMDeploy < 0.9
                'multimodal models in lmdeploy>=0.9.')  # 提示信息：LMDeploy 0.9 及以上不再维护多模态推理
            
            vision_config = VisionConfig(max_batch_size=vision_batch_size)  # 创建视觉配置对象
            # VisionConfig 用于配置视觉编码器（如图像预处理、批大小等）
            
            pipeline_kwargs['vision_config'] = vision_config  # 添加视觉配置到 pipeline 参数
            logger.info(f'vision_config: {vision_config}')  # 记录视觉配置信息到日志
        
        self.pipeline_kwargs = pipeline_kwargs  # 保存 pipeline 参数

    @contextmanager  # 声明为上下文管理器装饰器
    def _patch_pipeline(self):
        """函数功能：
        上下文管理器，临时修补 pipeline 的模型匹配逻辑，用于临时替换 LMDeploy 的 best_match_model 函数，
        确保 pipeline 使用我们指定的 model_type 而非自动检测。
        
        参数：
        - 无
        
        返回值：
        - Generator（上下文管理器）
        
        示例：
        >>> # 内部调用
        >>> with self._patch_pipeline():
        ...     engine = pipeline(...)
        """
        _old_best_match_model = async_engine.best_match_model  # 保存原始的 best_match_model 函数引用

        def _best_match_model(*args, **kwargs) -> Optional[str]:  # 定义替换函数：返回指定的 model_type
            """替换函数：直接返回我们配置的 model_type，跳过自动检测。"""
            return self.model_info.model_type  # 返回我们已知的模型类型（例如 'qwen'）

        async_engine.best_match_model = _best_match_model  # 临时替换为我们的函数
        try:  # 尝试执行上下文内的代码
            yield  # 让上下文管理器的调用方执行代码（pipeline 初始化）
        finally:  # 无论是否出现异常，都恢复原始函数
            async_engine.best_match_model = _old_best_match_model  # 恢复原始函数引用

    def _prepare_engine(self):
        """函数功能：
        创建 LMDeploy pipeline 实例，应用必要的补丁以确保正确加载。
        设置实例属性 engine。
        
        示例：
        >>> # 内部调用
        >>> self._prepare_engine()
        """
        with patch_auto_tokenizer(self.tokenizer), patch_auto_config(self.config), self._patch_pipeline():  # 应用三个补丁
            # patch_auto_tokenizer: 注入已加载的 tokenizer（避免 LMDeploy 重新加载）
            # patch_auto_config: 注入已加载的 config（避免 LMDeploy 重新加载）
            # _patch_pipeline: 注入 model_type（避免 LMDeploy 自动检测错误）
            engine = pipeline(self.model_dir, backend_config=self.backend_config, **self.pipeline_kwargs)  # 创建 pipeline 实例
            # pipeline 是 LMDeploy 的工厂函数，根据 backend_config 自动选择 TurbomindEngine 或 PytorchEngine
        
        self.engine = engine  # 保存 pipeline 实例

    def _load_generation_config(self):
        """函数功能：
        从模型目录加载 generation_config.json，转换为 LMDeploy 的生成配置对象。
        设置实例属性 generation_config。
        
        示例：
        >>> # 内部调用
        >>> self._load_generation_config()
        """
        generation_config_path = os.path.join(self.model_dir, 'generation_config.json')  # 构造生成配置文件路径
        
        if os.path.isfile(generation_config_path):  # 若配置文件存在
            generation_config = GenerationConfig.from_pretrained(self.model_dir)  # 加载 HuggingFace 的生成配置
            kwargs = generation_config.to_dict()  # 转换为字典
            
            max_new_tokens = kwargs.get('max_new_tokens')  # 获取 max_new_tokens 参数
            if max_new_tokens is None:  # 若 max_new_tokens 为 None
                kwargs.pop('max_new_tokens', None)  # 从字典中移除（LMDeploy 不接受 None 值）
            
            parameters = inspect.signature(LmdeployGenerationConfig).parameters  # 获取 LmdeployGenerationConfig 的参数签名
            for k, v in kwargs.copy().items():  # 遍历配置字典的副本（避免迭代时修改字典）
                if k not in parameters or v is None:  # 若参数不在 LmdeployGenerationConfig 的签名中 或 值为 None
                    kwargs.pop(k)  # 从字典中移除（过滤不支持的参数）
            
            self.generation_config = LmdeployGenerationConfig(**kwargs)  # 创建 LMDeploy 的生成配置对象
        else:  # 若配置文件不存在
            self.generation_config = LmdeployGenerationConfig()  # 使用默认配置

    def _add_stop_words(self, generation_config: LmdeployGenerationConfig, request_config: RequestConfig,
                        template_meta: TemplateMeta) -> None:  # 模板元数据
        """函数功能：
        将请求配置、默认配置和模板配置中的停止词合并，并设置到生成配置中。
        
        参数：见上方签名注释。
        
        返回值：
        - None（直接修改 generation_config）
        """
        stop_words = (request_config.stop or []) + (self.generation_config.stop_words or []) + template_meta.stop_words  # 合并停止词列表
        # 三个来源：请求配置 + 默认配置 + 模板配置
        
        generation_config.stop_words = self._get_stop_token_ids(stop_words)  # 将停止词（字符串）转换为 token IDs，并设置到生成配置
        # _get_stop_token_ids 是父类方法，将字符串停止词编码为 token ID 列表
        
        # compat lmdeploy >= 0.6.*  # 兼容注释：LMDeploy 0.6 及以上版本
        generation_config.stop_token_ids = generation_config.stop_words  # 同时设置 stop_token_ids 字段（兼容新版本 API）

    def _prepare_generation_config(self, request_config: RequestConfig) -> LmdeployGenerationConfig:
        """函数功能：
        根据请求配置和默认配置，构造 LMDeploy 的生成配置对象。
        
        参数：
        - request_config (RequestConfig): 请求配置
        
        返回值：
        - LmdeployGenerationConfig: LMDeploy 的生成配置对象
        """
        kwargs = {'max_new_tokens': request_config.max_tokens}  # 初始化参数字典，设置最大生成 token 数
        
        for key in ['temperature', 'top_k', 'top_p', 'repetition_penalty']:  # 遍历采样参数
            new_value = getattr(request_config, key)  # 从请求配置中获取参数值
            if new_value is None:  # 若请求配置中未指定
                kwargs[key] = getattr(self.generation_config, key)  # 使用默认配置的值
            else:  # 若请求配置中指定了
                kwargs[key] = new_value  # 使用请求配置的值
        
        if request_config.seed is None:  # 若请求配置中未指定随机种子
            request_config.seed = get_seed()  # 生成随机种子
        kwargs['random_seed'] = request_config.seed  # 设置随机种子
        
        if request_config.temperature == 0:  # 若 temperature 为 0（表示贪心解码）
            kwargs['temperature'] = 1  # avoid unnecessary process  # 设置为 1（避免不必要的计算）
            kwargs['top_k'] = 1  # 设置 top_k 为 1（只选择概率最高的 token，等效于贪心）

        if request_config.logprobs:  # 若需要返回对数概率
            kwargs['logprobs'] = 1  # 默认返回 1 个候选的对数概率
            if request_config.top_logprobs is not None:  # 若指定了 top_logprobs
                kwargs['logprobs'] = max(1, request_config.top_logprobs)  # 使用指定值（至少为 1）

        res = LmdeployGenerationConfig(**kwargs)  # 创建 LMDeploy 的生成配置对象
        return res  # 返回生成配置

    async def _infer_stream_async(
        self,  # 实例自身引用
        template: Template,  # 对话模板
        inputs: Dict[str, Any],  # 模型输入字典（包含 input_ids, attention_mask, etc.）
        generation_config: LmdeployGenerationConfig,  # LMDeploy 的生成配置
        request_config: RequestConfig,  # 请求配置
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        """函数功能：
        执行异步流式推理，逐步产出增量响应（类似 ChatGPT 的打字机效果）。
        使用 LMDeploy 的异步 API，支持会话管理和上下文隔离。
        
        参数：见上方签名注释。
        
        返回值：
        - AsyncIterator[ChatCompletionStreamResponse]: 异步流式响应迭代器
        """
        session_id = time.time_ns()  # 生成唯一的会话 ID（使用纳秒时间戳，确保唯一性）
        kwargs = {'stream_output': True, 'gen_config': generation_config, 'sequence_start': True, 'sequence_end': True}  # 初始化推理参数字典
        # stream_output=True: 启用流式输出
        # gen_config: 生成配置
        # sequence_start=True: 标记会话开始（LMDeploy 需要）
        # sequence_end=True: 标记会话结束（LMDeploy 需要）
        
        if version.parse(lmdeploy.__version__) >= version.parse('0.6.5'):  # 若 LMDeploy 版本 >= 0.6.5（新版本 API）
            async with self.engine.model_inst(session_id) as inst:  # 创建模型实例上下文（异步上下文管理器）
                context = self.engine.safe_run(inst, session_id, **inputs, **kwargs)  # 创建安全运行上下文
        else:  # 若 LMDeploy 版本 < 0.6.5（旧版本 API）
            context = self.engine.safe_run(session_id)  # 创建安全运行上下文（旧版本 API）

        infer_streamer = InferStreamer(template)  # 创建流式输出器（用于处理增量 token 的解码）
        token_idx = 0  # 初始化 token 索引（跟踪已处理的 token 数量）
        
        async with context as gen:  # 进入上下文管理器（异步生成器）
            if version.parse(lmdeploy.__version__) < version.parse('0.6.5'):  # 若使用旧版本 API
                generator = await self.engine.get_generator(False, session_id)  # 获取生成器实例
                gen = generator.async_stream_infer(session_id=session_id, **inputs, **kwargs)  # 启动异步流式推理
            
            is_finished = False  # 初始化完成标志为 False
            while not is_finished:  # 循环直到生成完成
                try:  # 尝试获取下一个输出
                    output = await gen.__anext__()  # 从异步生成器获取下一个输出
                    # output 包含：token_ids（生成的 token 列表）、logprobs（对数概率）、num_token（生成的 token 数）、status（状态）
                except StopAsyncIteration:  # 若异步生成器耗尽（生成完成）
                    is_finished = True  # 标记为完成
                
                delta_text = infer_streamer.get_printable_text(output.token_ids, is_finished)  # 使用流式输出器提取增量可打印文本
                # infer_streamer 跟踪已输出的 token，只返回新增的可打印部分
                
                if not delta_text and not is_finished:  # 若无新增文本且未完成
                    continue  # 跳过本次迭代（不产出响应）

                logprobs = self._get_logprobs(output.logprobs, output.token_ids[token_idx:],  # 获取新增 token 的对数概率
                                              request_config.top_logprobs)  # 传入 top_logprobs 参数
                # output.token_ids[token_idx:]: 提取新增的 token IDs（从上次处理的位置到当前）
                
                token_idx = len(output.token_ids)  # 更新 token 索引为当前总长度

                usage_info = self._get_usage_info(len(inputs['input_ids']), output.num_token)  # 构造用量信息
                # len(inputs['input_ids']): prompt 的 token 数
                # output.num_token: 已生成的 token 数
                
                toolcall = None  # 初始化工具调用为 None
                if is_finished:  # 若已完成生成
                    toolcall = self._get_toolcall(template.decode(output.token_ids), template)  # 解码完整输出并提取工具调用信息
                
                finish_reason = self._get_finish_reason(generation_config.max_new_tokens, output.num_token,  # 获取完成原因
                                                        output.status.name == 'FINISH')  # 检查状态是否为 FINISH
                
                choices = [  # 构造选项列表
                    ChatCompletionResponseStreamChoice(  # 创建流式响应选项对象
                        index=0,  # 选项索引（单个候选时为 0）
                        delta=DeltaMessage(role='assistant', content=delta_text, tool_calls=toolcall),  # 增量消息
                        finish_reason=finish_reason,  # 完成原因
                        logprobs=logprobs)  # 对数概率信息
                ]
                yield ChatCompletionStreamResponse(model=self.model_name, choices=choices, usage=usage_info)  # 产出流式响应

    async def _infer_full_async(
        self,  # 实例自身引用
        template: Template,  # 对话模板
        inputs: Dict[str, Any],  # 模型输入字典（包含 input_ids 等）
        generation_config: LmdeployGenerationConfig,  # LMDeploy 的生成配置
        request_config: RequestConfig,  # 请求配置
    ) -> ChatCompletionResponse:  # 返回聊天补全响应
        """函数功能：
        执行异步非流式推理，等待完整生成后一次性返回响应。
        使用 LMDeploy 的异步 API，支持会话管理和上下文隔离。
        
        参数：见上方签名注释。
        
        返回值：
        - ChatCompletionResponse: 聊天补全响应
        """
        session_id = time.time_ns()  # 生成唯一的会话 ID（使用纳秒时间戳，确保唯一性）
        kwargs = {'stream_output': False, 'gen_config': generation_config, 'sequence_start': True, 'sequence_end': True}  # 初始化推理参数字典
        # stream_output=False: 禁用流式输出（等待完整生成）
        # gen_config: 生成配置
        # sequence_start=True: 标记会话开始
        # sequence_end=True: 标记会话结束
        
        if version.parse(lmdeploy.__version__) >= version.parse('0.6.5'):  # 若 LMDeploy 版本 >= 0.6.5（新版本 API）
            async with self.engine.model_inst(session_id) as inst:  # 创建模型实例上下文（异步上下文管理器）
                async with self.engine.safe_run(inst, session_id, **inputs, **kwargs) as gen:  # 创建安全运行上下文
                    async for output in gen:  # 遍历所有输出（虽然非流式，但 LMDeploy 仍以流式方式迭代）
                        pass  # 跳过中间输出，只保留最后一个（output 变量会保留最后一次迭代的值）
                
                if self.engine.backend == 'pytorch':  # 若后端是 PyTorch（而非 Turbomind）
                    # manually end pytorch session  # 手动结束 PyTorch 会话
                    await inst.async_end(session_id)  # 调用异步方法结束会话（PyTorch 后端需要手动清理）

        else:  # 若 LMDeploy 版本 < 0.6.5（旧版本 API）
            async with self.engine.safe_run(session_id):  # 创建安全运行上下文（旧版本 API）
                generator = await self.engine.get_generator(False, session_id)  # 获取生成器实例
                async for output in generator.async_stream_infer(session_id=session_id, **inputs, **kwargs):  # 启动异步流式推理
                    pass  # 跳过中间输出，只保留最后一个

        response = template.decode(output.token_ids)  # 使用模板解码生成的 token IDs 为文本
        logprobs = self._get_logprobs(output.logprobs, output.token_ids, request_config.top_logprobs)  # 获取对数概率信息

        usage_info = self._get_usage_info(len(inputs['input_ids']), output.num_token)  # 构造用量信息
        toolcall = self._get_toolcall(response, template)  # 从解码文本中提取工具调用信息
        finish_reason = self._get_finish_reason(generation_config.max_new_tokens, output.num_token,  # 获取完成原因
                                                output.status.name == 'FINISH')  # 检查状态是否为 FINISH
        token_ids = template.skip_stop_tokens(output.token_ids) if request_config.return_details else None  # 若需要返回详细信息，则跳过停止 token 并返回 token_ids
        
        choices = [  # 构造选项列表
            ChatCompletionResponseChoice(  # 创建响应选项对象（非流式，包含完整消息）
                index=0,  # 选项索引（单个候选时为 0）
                message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),  # 完整消息（而非 delta）
                finish_reason=finish_reason,  # 完成原因
                logprobs=logprobs,  # 对数概率信息
                token_ids=token_ids)  # token_ids（若 return_details=True）
        ]
        prompt_token_ids = inputs['input_ids'] if request_config.return_details else None  # 若需要返回详细信息，则提取 prompt token_ids
        return ChatCompletionResponse(  # 返回聊天补全响应对象
            model=self.model_name, choices=choices, usage=usage_info, prompt_token_ids=prompt_token_ids)  # 包含模型名、选项、用量、prompt token_ids

    async def infer_async(self,
                          infer_request: InferRequest,  # 推理请求对象
                          request_config: Optional[RequestConfig] = None,  # 请求配置（可选）
                          *,  # 仅限关键字参数分隔符
                          template: Optional[Template] = None,  # 对话模板（可选）
                          pre_infer_hook=None,  # 推理前钩子函数（可选）
                          **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
        """函数功能：
        异步推理的公开接口，支持流式和非流式模式，支持多模态输入。
        
        工作流程：
        1. 准备请求配置和模板；
        2. 使用模板编码输入（在线程池中执行，避免阻塞事件循环）；
        3. 处理多模态输入（如图像）：
           - 将图像转换为 PIL 格式；
           - 使用视觉编码器预处理和编码；
           - 根据后端（Turbomind/PyTorch）准备不同格式的输入；
        4. 准备生成配置和停止词；
        5. 调用流式或非流式推理方法。
        
        参数：见上方签名注释。
        
        返回值：
        - Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
          若 stream=False，返回完整响应；若 stream=True，返回异步流式响应迭代器
        
        示例：
        >>> # 非流式推理
        >>> request = InferRequest(messages=[{'role': 'user', 'content': '你好'}])
        >>> response = await engine.infer_async(request)
        >>> 
        >>> # 流式推理
        >>> async for chunk in await engine.infer_async(request, RequestConfig(stream=True)):
        ...     print(chunk.choices[0].delta.content)
        """
        request_config = deepcopy(request_config or RequestConfig())  # 深拷贝请求配置（若未提供则使用默认配置）
        # 深拷贝避免修改原始配置对象，确保多次调用间的独立性
        
        if template is None:  # 若未提供模板
            template = self.default_template  # 使用默认模板

        template.set_mode('lmdeploy')  # 设置模板模式为 'lmdeploy'（LMDeploy 需要特定的输入格式）

        loop = asyncio.get_running_loop()  # 获取当前运行的事件循环
        with torch.inference_mode():  # 启用 PyTorch 推理模式（禁用梯度计算，节省显存）
            inputs = await loop.run_in_executor(None, template.encode, infer_request)  # 在线程池中执行编码（避免阻塞事件循环）
            # template.encode 是 CPU 密集型操作，应在线程池中执行
        
        images = inputs.pop('images', None)  # 从输入中提取图像（若存在），并从输入字典中移除

        if images:  # 若存在图像（多模态输入）
            if version.parse(lmdeploy.__version__) >= version.parse('0.6.5'):  # 若 LMDeploy 版本 >= 0.6.5（新版本 API）
                messages = self.engine._convert_prompts(('', images))  # 将图像转换为 LMDeploy 内部格式
                messages = await self.engine.async_convert_to_pil_images(messages)  # 异步转换为 PIL 图像
                results = await self.engine.vl_encoder.preprocess(messages)  # 使用视觉编码器预处理图像
                
                if self.engine.backend == 'turbomind':  # 若后端是 Turbomind
                    results = await self.engine.vl_encoder.async_infer(results)  # 异步编码图像（Turbomind 需要提前编码）
                    inputs['images'] = [result['content'] for result in results if result['role'] == 'forward'][0]  # 提取编码后的图像特征
                    # 过滤 role='forward' 的结果并取第一个（图像特征）
                    await template.prepare_lmdeploy_turbomind_inputs(inputs)  # 准备 Turbomind 格式的输入
                else:  # 若后端是 PyTorch
                    inputs['images'] = results[1]['content']  # 提取预处理后的图像（PyTorch 后端不需要提前编码）
                    await template.prepare_lmdeploy_pytorch_inputs(inputs)  # 准备 PyTorch 格式的输入
            else:  # 若 LMDeploy 版本 < 0.6.5（旧版本 API）
                inputs['images'] = await self.engine.vl_encoder.async_infer(images)  # 异步编码图像
                await template.prepare_lmdeploy_turbomind_inputs(inputs)  # 准备 Turbomind 格式的输入

        self.set_default_max_tokens(request_config, inputs)  # 设置默认的 max_tokens（若未指定，根据输入长度推断）
        generation_config = self._prepare_generation_config(request_config)  # 准备生成配置
        self._add_stop_words(generation_config, request_config, template.template_meta)  # 添加停止词
        
        kwargs.update({  # 更新参数字典，准备传递给内部推理方法
            'template': template,  # 模板
            'inputs': inputs,  # 模型输入
            'generation_config': generation_config,  # 生成配置
            'request_config': request_config  # 请求配置
        })
        
        if pre_infer_hook:  # 若提供了推理前钩子函数
            kwargs = pre_infer_hook(kwargs)  # 调用钩子函数（允许外部修改参数）

        if request_config.stream:  # 若启用流式模式
            return self._infer_stream_async(**kwargs)  # 返回流式推理的异步迭代器
        else:  # 若禁用流式模式（非流式）
            return await self._infer_full_async(**kwargs)  # 等待并返回完整响应

    def _batch_infer_stream(self, *args, **kwargs):
        """函数功能：
        批量流式推理的内部方法，清理 LMDeploy 的内部状态以确保批量推理正确执行。
        
        参数：
        - *args, **kwargs: 传递给父类方法的参数
        
        返回值：
        - 父类方法的返回值
        """
        if hasattr(self.engine, 'vl_encoder'):  # 若引擎有视觉编码器（多模态模型）
            self.engine.vl_encoder._loop_task = None  # 清理视觉编码器的事件循环任务（避免批量推理时冲突）
        
        if hasattr(self.engine, 'free_insts'):  # 若引擎有 free_insts 属性
            self.engine.free_insts = None  # 清理空闲实例列表（避免批量推理时冲突）
        
        return super()._batch_infer_stream(*args, **kwargs)  # 调用父类的批量流式推理方法

    def infer(
        self,  # 实例自身引用
        infer_requests: List[InferRequest],  # 推理请求列表
        request_config: Optional[RequestConfig] = None,  # 请求配置（可选）
        metrics: Optional[List[Metric]] = None,  # 指标采集器列表（可选）
        *,  # 仅限关键字参数分隔符
        template: Optional[Template] = None,  # 对话模板（可选）
        use_tqdm: Optional[bool] = None,  # 是否使用进度条（可选）
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:  # 返回响应列表
        """函数功能：
        批量推理的公开接口，支持同步批量处理多个请求。
        
        参数：见上方签名注释。
        
        返回值：
        - List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
          响应列表，每个元素对应一个请求的响应（非流式）或流式响应迭代器（流式）
        
        示例：
        >>> requests = [
        ...     InferRequest(messages=[{'role': 'user', 'content': '你好'}]),
        ...     InferRequest(messages=[{'role': 'user', 'content': '世界'}])
        ... ]
        >>> responses = engine.infer(requests)
        """
        return super().infer(infer_requests, request_config, metrics, template=template, use_tqdm=use_tqdm)  # 调用父类的批量推理方法
