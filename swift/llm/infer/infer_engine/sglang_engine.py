"""模块功能概述：
该模块实现基于 SGLang 引擎的高性能推理引擎 `SglangEngine`，用于在生产环境中进行大语言模型的推理。提供：
- 异步推理接口，支持流式和非流式响应模式；
- 基于 SGLang Runtime (SRT) 的高性能推理后端；
- 支持多种并行策略（TP/PP/DP/EP）和优化技术；
- 支持嵌入模型的向量生成任务；
- 与 Swift 框架的无缝集成。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，标注代码版权所有者
import asyncio  # 引入 asyncio 模块，用于异步编程和协程调度
import inspect  # 引入 inspect 模块，用于运行时检查函数签名和参数
import os  # 引入 os 模块，用于文件路径操作和环境变量访问
from copy import deepcopy  # 引入 deepcopy，用于对配置进行深拷贝，避免副作用
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union  # 引入类型注解，明确参数与返回值类型

import sglang as sgl  # 引入 SGLang 库，高性能 LLM 推理框架
import torch  # 引入 PyTorch 库，用于张量操作和推理模式
from sglang.srt.sampling.sampling_params import SamplingParams  # 引入 SGLang 的采样参数类
from sglang.srt.server_args import ServerArgs  # 引入 SGLang 的服务器参数类
from transformers import GenerationConfig  # 引入 HuggingFace 的生成配置类

from swift.llm import InferRequest, Template, TemplateMeta, get_model_tokenizer  # 引入 Swift 框架的推理请求、模板、模板元数据和模型加载工具
from swift.plugin import Metric  # 引入 Metric 类型，用于传递统计/用量指标采集器
from swift.utils import get_logger  # 引入日志工具
from ..protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,  # 引入协议层数据类：响应、选项、流式响应
                        ChatCompletionStreamResponse, ChatMessage, DeltaMessage, EmbeddingResponse,  # 引入聊天消息、增量消息、嵌入响应
                        EmbeddingResponseData, RequestConfig, random_uuid)  # 引入嵌入响应数据、请求配置、随机 UUID 生成器
from .infer_engine import InferEngine  # 引入父类 InferEngine，提供通用推理接口与工具方法
from .utils import InferStreamer  # 引入流式输出器，用于处理增量 token 的解码

logger = get_logger()  # 获取日志记录器实例，用于输出日志信息


class SglangEngine(InferEngine):

    """类功能：
    定义基于 SGLang 的推理引擎类，继承通用推理引擎基类，负责基于 SGLang 引擎进行高性能的大语言模型推理。
    
    - 角色：推理引擎封装，使用 SGLang Runtime (SRT) 进行推理；
    - 能力：支持异步推理、流式/非流式输出、嵌入生成、多种并行策略；
    - 适用：生产环境中的高性能推理服务，支持大规模模型部署；
    - 线程/协程安全：基于异步接口，支持并发请求。
    
    特点：
    - 高性能：SGLang 采用 RadixAttention 和高效的调度算法，吞吐量高、延迟低
    - 并行策略：支持 [Tensor Parallelism (TP)、Pipeline Parallelism (PP)]、Data Parallelism (DP)、Expert Parallelism (EP)
    - 灵活配置：支持 CUDA Graph、量化、KV Cache 优化等多种优化技术
    - 嵌入支持：支持生成文本嵌入向量（用于 RAG、相似度计算等）
    """

    def __init__(  # 初始化 SglangEngine 实例，配置模型加载、引擎参数和并行策略
        self,  # 实例自身引用
        model_id_or_path: str,  # 模型的 HuggingFace ID 或本地路径，用于加载模型和分词器
        torch_dtype: Optional[torch.dtype] = None,  # 模型推理使用的数据类型（如 float16、bfloat16），None 则自动推断
        *,  # 仅限关键字参数分隔符，后续参数必须以关键字形式传入
        model_type: Optional[str] = None,  # 模型类型标识（如 'qwen', 'llama'），用于选择特定的模板和配置
        use_hf: Optional[bool] = None,  # 是否强制从 HuggingFace Hub 下载模型，None 则自动判断
        hub_token: Optional[str] = None,  # HuggingFace Hub 访问令牌，用于下载私有模型
        revision: Optional[str] = None,  # 模型的版本/分支名称，用于下载特定版本
        # engine kwargs  # 分组注释：以下为 SGLang 引擎的核心配置参数
        tp_size: int = 1,  # Tensor Parallelism 张量并行度，多 GPU 时用于分割模型（例如 tp_size=4 表示 4 卡并行）
        pp_size: int = 1,  # Pipeline Parallelism 流水线并行度，多 GPU 时用于分层部署（例如 pp_size=2 表示 2 层流水线）
        dp_size: int = 1,  # Data Parallelism 数据并行度，用于并发处理多个请求（例如 dp_size=2 表示 2 个副本）
        ep_size: int = 1,  # Expert Parallelism 专家并行度，用于 MoE 模型的专家分割（例如 ep_size=4 表示 4 卡专家并行）
        enable_ep_moe: bool = False,  # 是否启用 MoE（Mixture of Experts）的专家并行
        mem_fraction_static: Optional[float] = None,  # 静态显存分配比例（如 0.9 表示使用 90% 显存），None 则自动分配
        context_length: Optional[int] = None,  # 上下文窗口长度（context window），None 则使用模型配置的默认值
        disable_cuda_graph: bool = False,  # 是否禁用 CUDA Graph 优化（CUDA Graph 可显著降低小批量延迟）
        quantization: Optional[str] = None,  # 量化方案（如 'int8', 'int4', 'fp8'），None 表示不量化
        task_type: Optional[str] = None,  # 任务类型（如 'embedding' 表示嵌入生成），None 表示普通生成任务
        kv_cache_dtype: str = 'auto',  # KV Cache 的数据类型（'auto', 'fp16', 'int8' 等），影响显存占用和精度
        enable_dp_attention: bool = False,  # 是否启用数据并行的注意力计算（用于跨设备的注意力共享）
        disable_custom_all_reduce: bool = True,  # 是否禁用自定义的 All-Reduce 算子（自定义算子可能更快但稳定性较差）
        log_level='error',  # 日志级别（'debug', 'info', 'warning', 'error'），控制 SGLang 的日志输出
        engine_kwargs: Optional[Dict[str, Any]] = None,  # 其他传递给 SGLang 引擎的额外参数字典
        template: Optional[Template] = None,  # 预设的对话模板，None 则根据 model_type 自动选择
    ):  # 无返回值
        """函数功能：
        初始化 `SglangEngine` 实例，完成以下步骤：
        1. 加载分词器（不加载模型权重，由 SGLang 接管）；
        2. 准备 SGLang 服务器参数（并行策略、显存管理、优化选项等）；
        3. 创建 SGLang Engine 实例；
        4. 加载生成配置（从 generation_config.json）。
        
        参数：见上方签名注释。
        
        返回值：
        - None（初始化实例属性）
        
        示例：
        >>> # 单卡推理
        >>> engine = SglangEngine('Qwen/Qwen-7B-Chat', tp_size=1, context_length=4096)
        >>> 
        >>> # 多卡张量并行（4 卡 TP）
        >>> engine = SglangEngine('meta-llama/Llama-2-70b-chat-hf', tp_size=4)
        >>> 
        >>> # 嵌入模型
        >>> engine = SglangEngine('BAAI/bge-large-zh-v1.5', task_type='embedding')
        """
        if engine_kwargs is None:  # 若未提供额外引擎参数
            engine_kwargs = {}  # 初始化为空字典，避免后续操作报错
        
        self.processor = get_model_tokenizer(  # 加载分词器（processor）和模型配置，返回值为 (model, tokenizer, config) 元组
            model_id_or_path,  # 模型路径或 HuggingFace ID
            torch_dtype,  # 数据类型（如 float16）
            load_model=False,  # 不加载模型权重（SGLang 会自行加载），仅获取分词器和配置
            download_model=True,  # 允许从 HuggingFace Hub 下载模型文件（若本地不存在）
            model_type=model_type,  # 模型类型（如 'qwen'），用于匹配特定逻辑
            use_hf=use_hf,  # 是否强制使用 HuggingFace Hub
            hub_token=hub_token,  # HuggingFace 访问令牌
            revision=revision,  # 模型版本/分支
            task_type=task_type)[1]  # 取元组的第二个元素（tokenizer），索引 [1] 提取分词器
        
        self._post_init(template)  # 调用父类的后初始化方法，设置默认模板、模型信息等（继承自 InferEngine）
        
        if context_length is not None:  # 若显式指定了上下文长度
            self.max_model_len = context_length  # 覆盖模型配置的默认值
            logger.info(f'Setting max_model_len: {context_length}')  # 记录日志
        
        if self.max_model_len is not None:  # 若设置了最大模型长度（从配置或参数）
            self.max_model_len -= 1  # 减 1（为特殊 token 预留空间，如 EOS）
        
        parameters = inspect.signature(ServerArgs).parameters  # 获取 ServerArgs 的参数签名（用于版本兼容性检查）
        if 'pp_size' in parameters:  # 若 ServerArgs 支持 pp_size 参数（版本检查）
            engine_kwargs['pp_size'] = pp_size  # 添加流水线并行度参数
        
        self.server_args = ServerArgs(  # 创建 SGLang 服务器参数对象
            model_path=self.model_dir,  # 模型目录路径
            dtype=self.model_info.torch_dtype,  # 数据类型（从模型配置中获取）
            tp_size=tp_size,  # 张量并行度
            dp_size=dp_size,  # 数据并行度
            ep_size=ep_size,  # 专家并行度
            enable_ep_moe=enable_ep_moe,  # 是否启用 MoE 专家并行
            mem_fraction_static=mem_fraction_static,  # 静态显存分配比例
            context_length=context_length,  # 上下文窗口长度
            disable_cuda_graph=disable_cuda_graph,  # 是否禁用 CUDA Graph
            quantization=quantization,  # 量化方案
            kv_cache_dtype=kv_cache_dtype,  # KV Cache 数据类型
            enable_dp_attention=enable_dp_attention,  # 是否启用数据并行注意力
            disable_custom_all_reduce=disable_custom_all_reduce,  # 是否禁用自定义 All-Reduce
            log_level=log_level,  # 日志级别
            skip_tokenizer_init=True,  # 跳过分词器初始化（我们已经加载了分词器）
            **engine_kwargs,  # 展开额外参数
        )
        
        self.task_type = task_type  # 保存任务类型
        if task_type == 'embedding':  # 若任务类型为嵌入生成
            self.server_args.is_embedding = True  # 设置服务器参数的嵌入标志为 True
        
        self.engine = sgl.Engine(server_args=self.server_args)  # 创建 SGLang Engine 实例
        # Engine 是 SGLang 的核心类，负责模型加载、调度、推理等

        self._load_generation_config()  # 调用内部方法加载生成配置

    def _load_generation_config(self) -> None:
        """函数功能：
        从模型目录加载 generation_config.json，转换为 SGLang 的采样参数配置字典。
        设置实例属性 generation_config。
        
        示例：
        >>> # 内部调用
        >>> self._load_generation_config()
        """
        generation_config_path = os.path.join(self.model_dir, 'generation_config.json')  # 构造生成配置文件路径
        
        if os.path.isfile(generation_config_path):  # 若配置文件存在
            generation_config = GenerationConfig.from_pretrained(self.model_dir)  # 加载 HuggingFace 的生成配置
        else:  # 若配置文件不存在
            generation_config = GenerationConfig()  # 使用默认配置
        
        kwargs = generation_config.to_dict()  # 转换为字典
        top_k = kwargs.get('top_k')  # 获取 top_k 参数
        if top_k == 0:  # 若 top_k 为 0（HuggingFace 约定 0 表示禁用）
            kwargs['top_k'] = -1  # 转换为 SGLang 的约定（-1 表示禁用）

        parameters = inspect.signature(SamplingParams).parameters  # 获取 SamplingParams 的参数签名
        self.generation_config = {k: v for k, v in kwargs.items() if k in parameters and v is not None}  # 过滤参数
        # 只保留 SamplingParams 支持的参数，且值不为 None 的参数

    def _prepare_generation_config(self, request_config: RequestConfig) -> Dict[str, Any]:
        """函数功能：
        根据请求配置和默认配置，构造 SGLang 的采样参数字典。
        
        参数：
        - request_config (RequestConfig): 请求配置
        
        返回值：
        - Dict[str, Any]: 采样参数字典
        
        示例：
        >>> config = self._prepare_generation_config(RequestConfig(max_tokens=100, temperature=0.7))
        """
        kwargs = {'max_new_tokens': request_config.max_tokens}  # 初始化参数字典，设置最大生成 token 数
        
        for key in ['temperature', 'top_k', 'top_p', 'repetition_penalty']:  # 遍历采样参数
            new_value = getattr(request_config, key)  # 从请求配置中获取参数值
            if new_value is None:  # 若请求配置中未指定
                kwargs[key] = self.generation_config.get(key)  # 使用默认配置的值
            else:  # 若请求配置中指定了
                kwargs[key] = new_value  # 使用请求配置的值
        
        for key in ['n', 'frequency_penalty', 'presence_penalty']:  # 遍历其他采样参数
            kwargs[key] = getattr(request_config, key)  # 直接从请求配置中获取（这些参数没有默认值）

        return kwargs  # 返回采样参数字典

    def _add_stop_words(self, generation_config: Dict[str, Any], request_config: RequestConfig,
                        template_meta: TemplateMeta) -> None:  # 模板元数据
        """函数功能：
        将请求配置、默认配置和模板配置中的停止词合并，并设置到生成配置中。
        
        参数：见上方签名注释。
        
        返回值：
        - None（直接修改 generation_config）
        
        示例：
        >>> self._add_stop_words(generation_config, request_config, template.template_meta)
        """
        stop_words = (request_config.stop or []) + (self.generation_config.get('stop') or []) + template_meta.stop_words  # 合并停止词列表
        # 三个来源：请求配置 + 默认配置 + 模板配置
        
        generation_config['stop_token_ids'] = self._get_stop_token_ids(stop_words)  # 将停止词（字符串）转换为 token IDs，并设置到生成配置
        # _get_stop_token_ids 是父类方法，将字符串停止词编码为 token ID 列表

    def _create_chat_completion_response(self, output, template, return_details: bool = False):
        """函数功能：
        从 SGLang 的输出中提取信息，构造聊天补全响应对象。
        
        参数：
        - output: SGLang 的输出字典（包含 output_ids、meta_info 等）
        - template (Template): 对话模板
        - return_details (bool): 是否返回详细信息（token_ids 等）
        
        返回值：
        - ChatCompletionResponse: 聊天补全响应对象
        
        示例：
        >>> response = self._create_chat_completion_response(output, template, return_details=True)
        """
        assert output is not None  # 断言输出不为空（防御性编程）
        meta_info = output['meta_info']  # 提取元信息（包含用量、完成原因等）
        usage_info = self._get_usage_info(meta_info['prompt_tokens'], meta_info['completion_tokens'])  # 构造用量信息
        # meta_info['prompt_tokens']: prompt 的 token 数
        # meta_info['completion_tokens']: 生成的 token 数
        
        response = template.decode(output['output_ids'])  # 使用模板解码生成的 token IDs 为文本
        if template.template_meta.response_prefix:  # 若模板配置了响应前缀
            response = template.template_meta.response_prefix + response  # 添加前缀到响应文本
        
        toolcall = self._get_toolcall(response, template)  # 从响应文本中提取工具调用信息
        token_ids = template.skip_stop_tokens(output['output_ids']) if return_details else None  # 若需要返回详细信息，则跳过停止 token 并返回 token_ids
        
        choice = ChatCompletionResponseChoice(  # 创建响应选项对象
            index=0,  # 选项索引（单个候选时为 0）
            message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),  # 完整消息
            finish_reason=meta_info['finish_reason']['type'],  # 完成原因（从元信息中提取）
            logprobs=None,  # 对数概率（SGLang 当前不支持，设为 None）
            token_ids=token_ids)  # token_ids（若 return_details=True）
        
        prompt_token_ids = output.get('prompt_token_ids') if return_details else None  # 若需要返回详细信息，则提取 prompt token_ids
        return ChatCompletionResponse(  # 返回聊天补全响应对象
            model=self.model_name,  # 模型名称
            choices=[choice],  # 选项列表
            usage=usage_info,  # 用量信息
            id=random_uuid(),  # 随机生成的请求 ID
            prompt_token_ids=prompt_token_ids)  # prompt token_ids（若 return_details=True）

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

    async def infer_async(self,
                          infer_request: InferRequest,  # 推理请求对象
                          request_config: Optional[RequestConfig] = None,  # 请求配置（可选）
                          *,  # 仅限关键字参数分隔符
                          template: Optional[Template] = None,  # 对话模板（可选）
                          pre_infer_hook=None,  # 推理前钩子函数（可选）
                          **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:  # 返回响应或流式响应迭代器
        """函数功能：
        异步推理的公开接口，支持流式和非流式模式，支持嵌入生成。
        
        工作流程：
        1. 准备请求配置和模板；
        2. 使用模板编码输入（在线程池中执行，避免阻塞事件循环）；
        3. 准备生成配置和停止词；
        4. 根据配置调用流式、嵌入或非流式推理方法。
        
        参数：见上方签名注释。
        
        返回值：
        - Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
          若 stream=False 且非嵌入任务，返回完整响应；若 stream=True，返回异步流式响应迭代器；若为嵌入任务，返回 EmbeddingResponse
        
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

        template.set_mode('sglang')  # 设置模板模式为 'sglang'（SGLang 需要特定的输入格式）
        
        loop = asyncio.get_running_loop()  # 获取当前运行的事件循环
        with torch.inference_mode():  # 启用 PyTorch 推理模式（禁用梯度计算，节省显存）
            inputs = await loop.run_in_executor(None, template.encode, infer_request)  # 在线程池中执行编码（避免阻塞事件循环）
            # template.encode 是 CPU 密集型操作，应在线程池中执行
        
        if self.task_type == 'embedding':  # 若任务类型为嵌入生成
            inputs.pop('length', None)  # 移除 length 字段（嵌入任务不需要）

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
        elif self.task_type == 'embedding':  # 若任务类型为嵌入生成
            kwargs.pop('generation_config', None)  # 移除生成配置（嵌入任务不需要）
            return await self._infer_embedding_async(**kwargs)  # 等待并返回嵌入响应
        else:  # 若禁用流式模式（非流式）
            return await self._infer_full_async(**kwargs)  # 等待并返回完整响应

    async def _infer_embedding_async(self, template: Template, inputs: Dict[str, Any], **kwargs) -> EmbeddingResponse:
        """函数功能：
        执行异步嵌入生成推理，将输入文本转换为向量表示。
        
        参数：
        - template (Template): 对话模板
        - inputs (Dict[str, Any]): 模型输入字典（包含 input_ids、images、audios 等）
        - **kwargs: 其他参数（未使用）
        
        返回值：
        - EmbeddingResponse: 嵌入响应对象（包含向量数据）
        
        示例：
        >>> response = await self._infer_embedding_async(template, inputs)
        >>> embedding = response.data[0].embedding  # 获取向量
        """
        from sglang.srt.managers.io_struct import EmbeddingReqInput  # 引入 SGLang 的嵌入请求输入类
        obj = EmbeddingReqInput(  # 创建嵌入请求输入对象
            input_ids=inputs['input_ids'], image_data=inputs.get('images'), audio_data=inputs.get('audios'))  # 设置输入数据
        # input_ids: 文本的 token IDs
        # image_data: 图像数据（若为多模态嵌入模型）
        # audio_data: 音频数据（若为多模态嵌入模型）
        
        generator = self.engine.tokenizer_manager.generate_request(obj, None)  # 生成请求并获取生成器
        # tokenizer_manager 负责管理分词和请求调度
        
        output = await generator.__anext__()  # 从异步生成器获取输出（嵌入向量）
        usage_info = self._get_usage_info(output['meta_info']['prompt_tokens'], 0)  # 构造用量信息
        # 嵌入任务不生成新 token，所以 completion_tokens 为 0
        
        return EmbeddingResponse(  # 返回嵌入响应对象
            model=self.model_name,  # 模型名称
            data=[EmbeddingResponseData(embedding=output['embedding'])],  # 嵌入数据列表（包含向量）
            usage=usage_info,  # 用量信息
            id=random_uuid())  # 随机生成的请求 ID

    async def _infer_full_async(self, template: Template, inputs: Dict[str, Any], generation_config: Dict[str, Any],
                                request_config: RequestConfig) -> ChatCompletionResponse:  # 请求配置
        """函数功能：
        执行异步非流式推理，等待完整生成后一次性返回响应。
        
        参数：见上方签名注释。
        
        返回值：
        - ChatCompletionResponse: 聊天补全响应
        
        示例：
        >>> response = await self._infer_full_async(template, inputs, generation_config, request_config)
        """
        output = await self.engine.async_generate(**inputs, sampling_params=generation_config)  # 调用 SGLang 引擎的异步生成方法
        # **inputs: 展开输入字典（input_ids、images 等）
        # sampling_params: 采样参数配置
        
        output['prompt_token_ids'] = inputs['input_ids']  # 将 prompt token_ids 添加到输出（用于返回详细信息）
        return self._create_chat_completion_response(output, template, request_config.return_details)  # 创建并返回聊天补全响应

    async def _infer_stream_async(self, template: Template, inputs: Dict[str, Any], generation_config: Dict[str, Any],
                                  **kwargs) -> AsyncIterator[ChatCompletionStreamResponse]:  # 返回异步流式响应迭代器
        """函数功能：
        执行异步流式推理，逐步产出增量响应（类似 ChatGPT 的打字机效果）。
        
        参数：见上方签名注释。
        
        返回值：
        - AsyncIterator[ChatCompletionStreamResponse]: 异步流式响应迭代器
        
        示例：
        >>> async for chunk in self._infer_stream_async(template, inputs, generation_config):
        ...     print(chunk.choices[0].delta.content)
        """
        result_generator = await self.engine.async_generate(**inputs, sampling_params=generation_config, stream=True)  # 调用 SGLang 引擎的异步生成方法（流式模式）
        # stream=True: 启用流式输出
        
        infer_streamer = InferStreamer(template)  # 创建流式输出器（用于处理增量 token 的解码）
        
        async for output in result_generator:  # 遍历异步生成器的输出
            res = self._create_chat_completion_stream_response(output, template, infer_streamer)  # 创建流式响应
            if res is None:  # 若响应为 None（无新增文本且未完成）
                continue  # 跳过本次迭代
            yield res  # 产出流式响应

    def _create_chat_completion_stream_response(self, output, template,
                                                infer_streamer) -> Optional[ChatCompletionStreamResponse]:  # 流式输出器
        """函数功能：
        从 SGLang 的流式输出中提取信息，构造聊天补全流式响应对象。
        
        参数：
        - output: SGLang 的输出字典（包含 output_ids、meta_info 等）
        - template (Template): 对话模板
        - infer_streamer (InferStreamer): 流式输出器
        
        返回值：
        - Optional[ChatCompletionStreamResponse]: 流式响应对象，若无新增文本则返回 None
        
        示例：
        >>> response = self._create_chat_completion_stream_response(output, template, infer_streamer)
        """
        assert output is not None  # 断言输出不为空（防御性编程）
        meta_info = output['meta_info']  # 提取元信息（包含用量、完成原因等）
        finish_reason = meta_info['finish_reason']  # 获取完成原因
        is_finished = finish_reason is not None  # 判断是否已完成（finish_reason 不为 None 表示完成）
        
        delta_text = infer_streamer.get_printable_text(output['output_ids'], is_finished)  # 使用流式输出器提取增量可打印文本
        # infer_streamer 跟踪已输出的 token，只返回新增的可打印部分
        
        if not delta_text and not is_finished:  # 若无新增文本且未完成
            return  # 返回 None（不产出响应）
        
        toolcall = None  # 初始化工具调用为 None
        if is_finished:  # 若已完成生成
            finish_reason = finish_reason['type']  # 提取完成原因的类型（如 'stop', 'length'）
            toolcall = self._get_toolcall(template.decode(output['output_ids']), template)  # 解码完整输出并提取工具调用信息
        
        meta_info = output['meta_info']  # 再次提取元信息（确保使用最新的）
        usage_info = self._get_usage_info(meta_info['prompt_tokens'], meta_info['completion_tokens'])  # 构造用量信息
        
        # TODO: logprobs  # 待办：SGLang 未来版本可能支持对数概率，当前设为 None
        choice = ChatCompletionResponseStreamChoice(  # 创建流式响应选项对象
            index=0,  # 选项索引（单个候选时为 0）
            delta=DeltaMessage(role='assistant', content=delta_text, tool_calls=toolcall),  # 增量消息
            finish_reason=finish_reason,  # 完成原因
            logprobs=None)  # 对数概率（当前不支持，设为 None）

        return ChatCompletionStreamResponse(model=self.model_name, choices=[choice], usage=usage_info)  # 返回流式响应对象
