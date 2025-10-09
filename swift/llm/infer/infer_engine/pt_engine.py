"""模块功能概述：
该模块实现基于 PyTorch 的原生推理引擎 `PtEngine`，用于在生产环境中进行大语言模型的推理。提供：
- 同步和异步推理接口，支持批量处理和单个请求；
- 流式和非流式响应模式，适配不同的应用场景；
- 动态批处理（Dynamic Batching）能力，自动合并多个请求以提高吞吐量；
- LoRA/Swift 适配器支持，实现多租户场景下的模型定制；
- 多种任务类型支持（文本生成、序列分类、PRM、嵌入向量等）；
- 多模态输入处理（图像、音频、视频等），支持视觉语言模型。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，标注代码版权所有者
import asyncio  # 引入 asyncio 模块，用于异步编程和协程调度
import hashlib  # 引入 hashlib 模块，用于计算哈希值（用于任务池分组）
import inspect  # 引入 inspect 模块，用于运行时检查函数签名和参数
import pickle  # 引入 pickle 模块，用于对象序列化（用于任务池键计算）
import time  # 引入 time 模块，用于线程睡眠和时间控制
from copy import deepcopy  # 引入 deepcopy，用于对配置进行深拷贝，避免副作用
from queue import Queue  # 引入 Queue，用于线程间通信（任务队列）
from threading import Thread  # 引入 Thread，用于创建后台推理工作线程
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Union  # 引入类型注解，明确参数与返回值类型

import json  # 引入 json 模块，用于 JSON 序列化和反序列化
import torch  # 引入 PyTorch 库，用于张量操作和模型推理
from tqdm import tqdm  # 引入 tqdm，用于显示进度条
from transformers import GenerationConfig, LogitsProcessorList  # 引入 HuggingFace 的生成配置和 logits 处理器列表
from transformers.utils import is_torch_npu_available  # 引入 NPU 可用性检查函数

from swift.llm import InferRequest, Template, TemplateMeta, get_model_tokenizer, safe_snapshot_download, to_device  # 引入 Swift 框架的推理请求、模板、模型加载和设备转换工具
from swift.plugin import Metric  # 引入 Metric 类型，用于传递统计/用量指标采集器
from swift.tuners import Swift  # 引入 Swift 微调库，用于加载 LoRA 等适配器
from ..protocol import (ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,  # 引入协议层数据类：响应、选项、流式响应
                        ChatCompletionStreamResponse, ChatMessage, DeltaMessage, EmbeddingResponse,  # 引入聊天消息、增量消息、嵌入响应
                        EmbeddingResponseData, RequestConfig, random_uuid)  # 引入嵌入数据、请求配置和 UUID 生成器
from .infer_engine import InferEngine  # 引入父类 InferEngine，提供通用推理接口与工具方法
from .utils import AdapterRequest, InferStreamer, LogitsStreamer, TokensIteratorStreamer, prepare_generation_config  # 引入适配器请求、流式输出器和配置准备工具


class _GenerationConfig(GenerationConfig):  # 定义内部生成配置类，继承自 HuggingFace 的 GenerationConfig

    """类功能：
    `_GenerationConfig` 是 GenerationConfig 的增强版本，提供更好的字符串表示。
    
    - 角色：内部辅助类，用于美化生成配置的打印输出；
    - 能力：过滤元数据、版本信息，提供简洁的配置展示；
    - 适用：调试和日志记录场景。
    """

    def __repr__(self) -> str:  # 覆盖 __repr__ 方法，提供自定义的字符串表示
        """函数功能：
        返回生成配置的简洁字符串表示，去除不必要的元数据和版本信息。
        
        参数：
        - 无
        
        返回值：
        - str: 格式化的配置字符串
        
        示例：
        >>> config = _GenerationConfig(max_new_tokens=100, temperature=0.7)
        >>> print(config)
        GenerationConfig({'max_new_tokens': 100, 'temperature': 0.7})
        """
        parameters = inspect.signature(self.to_json_string).parameters  # 获取 to_json_string 方法的签名，检查支持的参数
        kwargs = {}  # 初始化参数字典
        if 'ignore_metadata' in parameters:  # 若 to_json_string 支持 ignore_metadata 参数（新版 transformers）
            kwargs['ignore_metadata'] = True  # 设置为 True，忽略元数据信息
        gen_kwargs = json.loads(self.to_json_string(**kwargs))  # 将配置转为 JSON 字符串，再解析为字典
        gen_kwargs.pop('transformers_version', None)  # 移除 transformers_version 字段（版本信息不重要）
        return f'GenerationConfig({gen_kwargs})'  # 返回格式化的字符串表示


class PtEngine(InferEngine):  # 定义基于 PyTorch 的推理引擎类，继承通用推理引擎基类

    """类功能：
    `PtEngine` 负责基于 PyTorch 进行大语言模型的原生推理。
    
    - 角色：推理引擎封装，使用 PyTorch 原生 API 进行推理；
    - 能力：支持同步/异步推理、流式/非流式输出、动态批处理、LoRA 适配器、多模态输入、多种任务类型；
    - 适用：生产环境中的灵活推理服务，支持多租户和多种任务场景；
    - 线程/协程安全：内部使用后台线程和任务队列，支持并发请求。
    """

    def __init__(  # 初始化 PtEngine 实例，配置模型加载、批处理、适配器和任务类型
            self,  # 实例自身引用
            model_id_or_path: str,  # 模型的 HuggingFace ID 或本地路径，用于加载模型和分词器
            torch_dtype: Optional[torch.dtype] = None,  # 模型推理使用的数据类型（如 float16、bfloat16），None 则自动推断
            *,  # 仅限关键字参数分隔符，后续参数必须以关键字形式传入
            adapters: List[str] = None,  # LoRA/Swift 适配器列表（模型路径或 HuggingFace ID），None 表示不使用适配器
            max_batch_size: int = 1,  # 最大批处理大小，0 或 1 表示无限制（处理所有请求）
            model_type: Optional[str] = None,  # 模型类型标识（如 'qwen', 'llama'），用于选择特定的模板和配置
            use_hf: Optional[bool] = None,  # 是否强制从 HuggingFace Hub 下载模型，None 则自动判断
            revision: Optional[str] = None,  # 模型的版本/分支名称，用于下载特定版本
            hub_token: Optional[str] = None,  # HuggingFace Hub 访问令牌，用于下载私有模型
            load_model: bool = True,  # 是否加载模型权重，False 则仅加载分词器和配置
            # model kwargs  # 分组注释：以下为模型加载的核心配置参数
            attn_impl: Optional[str] = None,  # 注意力机制实现方式（如 'flash_attn', 'sdpa'），None 则使用默认
            device_map: Optional[Union[str, Dict[str, Any]]] = None,  # 设备映射（'auto', 'cuda:0' 或自定义字典），用于多 GPU 分配
            task_type: Optional[str] = None,  # 任务类型（'causal_lm', 'seq_cls', 'prm', 'embedding'），None 则自动推断
            quantization_config=None,  # 量化配置（如 BitsAndBytesConfig），用于 INT8/INT4 量化
            model_kwargs: Optional[Dict[str, Any]] = None,  # 其他传递给模型加载的额外参数字典
            template: Optional[Template] = None,  # 预设的对话模板，None 则根据 model_type 自动选择
            **kwargs):  # 其他额外参数，透传给 get_model_tokenizer
        """函数功能：
        初始化 `PtEngine` 实例，完成以下步骤：
        1. 加载模型和分词器（使用 PyTorch 原生加载）；
        2. 配置批处理大小；
        3. 加载 LoRA/Swift 适配器（若提供）；
        4. 初始化任务队列和后台线程。
        
        参数：见上方签名注释。
        
        返回值：
        - None
        
        示例：
        >>> engine = PtEngine(
        ...     'Qwen/Qwen-7B-Chat',
        ...     torch_dtype=torch.float16,
        ...     max_batch_size=8,
        ...     adapters=['path/to/lora']
        ... )
        >>> # 引擎已就绪，可调用 infer_async 或 infer 方法
        """
        self.model, self.processor = get_model_tokenizer(  # 加载模型和分词器，返回 (model, processor) 元组
            model_id_or_path,  # 模型路径或 HuggingFace ID
            torch_dtype,  # 数据类型
            load_model=load_model,  # 是否加载模型权重
            model_type=model_type,  # 模型类型
            download_model=True,  # 允许从 HuggingFace Hub 下载模型文件
            use_hf=use_hf,  # 是否强制使用 HuggingFace Hub
            hub_token=hub_token,  # HuggingFace 访问令牌
            revision=revision,  # 模型版本/分支
            device_map=device_map,  # 设备映射
            quantization_config=quantization_config,  # 量化配置
            attn_impl=attn_impl,  # 注意力实现方式
            task_type=task_type,  # 任务类型
            model_kwargs=model_kwargs,  # 其他模型参数
            **kwargs)  # 展开额外参数
        self.max_batch_size = max_batch_size  # 保存最大批处理大小
        if isinstance(adapters, str):  # 若 adapters 是字符串（单个适配器）
            adapters = [adapters]  # 转换为列表
        self.adapters = adapters or []  # 保存适配器列表（空则为 []）
        for adapter in self.adapters:  # 遍历每个适配器路径
            self._add_adapter(safe_snapshot_download(adapter, use_hf=use_hf, hub_token=hub_token))  # 下载适配器并加载到模型
        self._post_init(template)  # 调用后初始化方法，设置任务队列、线程和配置

    def _post_init(self, template=None):  # 后初始化方法（内部方法）
        """函数功能：
        完成初始化的后续步骤，设置任务队列、线程池和生成配置。
        
        参数：
        - template (Optional[Template]): 对话模板，None 则自动选择
        
        返回值：
        - None
        
        示例：
        >>> # 内部调用，通常在 __init__ 中自动执行
        >>> self._post_init(template)
        """
        super()._post_init(template)  # 调用父类的后初始化方法，设置默认模板、模型信息等
        self.engine = self.model  # dummy  # 设置 engine 属性为 model（保持接口一致性，虽然 PtEngine 不需要额外引擎）
        self.generation_config = self.model.generation_config  # 从模型获取默认的生成配置
        self._queue = Queue()  # 初始化任务队列（线程安全的 FIFO 队列，用于接收异步推理请求）
        self._task_pool = {}  # 初始化任务池（字典，键为配置哈希，值为批处理任务列表）
        self._task_thread = None  # 初始化后台推理线程为 None（延迟启动）

    def _start_infer_worker(self):  # 启动后台推理工作线程（内部方法）
        """函数功能：
        创建并启动后台推理工作线程，用于处理异步推理请求。
        线程为守护线程，主线程退出时自动终止。
        
        参数：
        - 无
        
        返回值：
        - None
        
        示例：
        >>> # 内部调用，通常在首次异步请求时自动执行
        >>> self._start_infer_worker()
        """
        self._task_thread = Thread(target=self._infer_worker, daemon=True)  # 创建后台线程，目标函数为 _infer_worker，设置为守护线程
        self._task_thread.start()  # 启动线程

    def _fetch_infer_requests(self):  # 获取并组织推理请求（内部方法，动态批处理核心）
        """函数功能：
        从任务队列中获取请求，按配置哈希分组到任务池，然后提取一批请求进行处理。
        实现动态批处理（Dynamic Batching）：相同配置的请求会被合并处理。
        
        参数：
        - 无（使用实例属性 self._queue 和 self._task_pool）
        
        返回值：
        - None: 若任务池为空，无请求可处理
        - Tuple[Dict, List]: (kwargs, queue_list)，其中 kwargs 包含批量请求，queue_list 为对应的响应队列
        
        示例：
        >>> # 内部调用，通常在后台线程中自动执行
        >>> result = self._fetch_infer_requests()
        >>> if result:
        ...     kwargs, queue_list = result
        """
        while not self._queue.empty():  # 循环处理队列中的所有待处理请求
            infer_request, kwargs, queue = self._queue.get()  # 从队列取出一个请求（包含请求对象、参数字典、响应队列）
            template = kwargs['template']  # 提取模板对象
            info = hashlib.sha256(pickle.dumps((kwargs['request_config'], template  # 计算配置哈希（用于分组）
                                                and template.template_meta))).hexdigest()  # 使用 SHA256 哈希，基于请求配置和模板元数据
            if info not in self._task_pool:  # 若该配置哈希不在任务池中
                self._task_pool[info] = kwargs, []  # 创建新条目（存储配置和请求列表）
            self._task_pool[info][1].append((infer_request, queue))  # 将请求和队列添加到对应配置的任务列表
        if len(self._task_pool) == 0:  # 若任务池为空（无待处理请求）
            return  # 返回 None
        key, (kwargs, data) = next(iter(self._task_pool.items()))  # 取出任务池中的第一个条目（FIFO）
        max_batch_size = self.max_batch_size  # 获取最大批处理大小
        if max_batch_size <= 0:  # 若设置为 0 或负数（表示无限制）
            max_batch_size = len(data)  # 处理所有请求
        data, remain_data = data[:max_batch_size], data[max_batch_size:]  # 分割为当前批次和剩余数据
        if remain_data:  # 若有剩余数据
            self._task_pool[key] = kwargs, remain_data  # 更新任务池，保留剩余数据
        else:  # 若无剩余数据
            self._task_pool.pop(key)  # 从任务池移除该配置条目
        kwargs = kwargs.copy()  # 拷贝参数字典，避免修改原始数据
        kwargs['infer_requests'] = [d[0] for d in data]  # 提取批量请求对象列表
        queue_list = [d[1] for d in data]  # 提取对应的响应队列列表
        return kwargs, queue_list  # 返回批量请求参数和队列列表

    def _infer_worker(self):  # 后台推理工作线程（内部方法）
        """函数功能：
        后台推理工作线程的主循环，持续从任务池获取请求并执行推理。
        处理流式和非流式两种推理模式，并将结果通过异步队列返回给调用方。
        
        参数：
        - 无（在线程中运行）
        
        返回值：
        - None（无限循环，直到进程终止）
        
        示例：
        >>> # 内部调用，在后台线程中运行
        >>> self._infer_worker()  # 无限循环
        """
        while True:  # 无限循环，持续处理任务
            time.sleep(0.01)  # 短暂睡眠（10ms），避免 CPU 占用过高
            item = self._fetch_infer_requests()  # 尝试获取待处理的批量请求
            if item is not None:  # 若有待处理请求
                kwargs, queue_list = item  # 解包为参数字典和队列列表
                request_config = kwargs['request_config']  # 提取请求配置
                res_list_or_gen = self._infer(**kwargs)  # 执行推理，返回结果列表或生成器
                if request_config.stream:  # 若是流式推理
                    finished = False  # 初始化完成标志
                    while not finished:  # 循环直到流式推理完成
                        try:  # 尝试获取下一批流式结果
                            res_list = next(res_list_or_gen)  # 从生成器获取下一批结果
                        except StopIteration:  # 若生成器耗尽
                            finished = True  # 标记为完成
                            res_list = [None] * len(queue_list)  # 发送 None 列表，通知调用方流式结束
                        for (queue, loop), res in zip(queue_list, res_list):  # 遍历队列和结果
                            asyncio.run_coroutine_threadsafe(queue.put(res), loop)  # 在对应事件循环中将结果放入异步队列
                else:  # 若是非流式推理
                    for (queue, loop), res in zip(queue_list, res_list_or_gen):  # 遍历队列和结果
                        asyncio.run_coroutine_threadsafe(queue.put(res), loop)  # 在对应事件循环中将结果放入异步队列

    def _add_adapter(self, adapter_path: str, adapter_name: Optional[str] = None) -> None:  # 添加 LoRA/Swift 适配器（内部方法）
        """函数功能：
        加载并添加 LoRA/Swift 适配器到模型。
        
        参数：
        - adapter_path (str): 适配器路径（本地路径或 HuggingFace ID）
        - adapter_name (Optional[str]): 适配器名称，None 则自动生成
        
        返回值：
        - None
        
        示例：
        >>> # 内部调用，通常在 __init__ 或动态添加时执行
        >>> self._add_adapter('path/to/lora', 'lora1')
        """
        self.model = Swift.from_pretrained(self.model, adapter_path, adapter_name)  # 使用 Swift 库加载适配器并更新模型

    @classmethod  # 声明为类方法
    def from_model_template(cls, model, template=None, *, max_batch_size: int = 1):  # 从已加载的模型和模板创建引擎实例（工厂方法）
        """函数功能：
        从已加载的模型对象和模板创建 PtEngine 实例（跳过模型加载步骤）。
        
        参数：
        - model: 已加载的 PyTorch 模型对象
        - template (Optional[Template]): 对话模板
        - max_batch_size (int): 最大批处理大小
        
        返回值：
        - PtEngine: 引擎实例
        
        示例：
        >>> model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-7B-Chat')
        >>> template = get_template('qwen')
        >>> engine = PtEngine.from_model_template(model, template, max_batch_size=8)
        """
        self = super().__new__(cls)  # 创建新实例（不调用 __init__）
        self.model = model  # 设置模型
        self.processor = template.processor  # 从模板获取分词器
        self.max_batch_size = max_batch_size  # 设置批处理大小
        self._post_init(template)  # 调用后初始化
        return self  # 返回实例

    def _prepare_generation_config(self, request_config: RequestConfig) -> _GenerationConfig:  # 准备生成配置（内部方法）
        """函数功能：
        根据请求配置和默认配置，构造生成配置对象。
        
        参数：
        - request_config (RequestConfig): 请求配置
        
        返回值：
        - _GenerationConfig: 生成配置对象
        
        示例：
        >>> # 内部调用
        >>> gen_config = self._prepare_generation_config(req_config)
        """
        generation_config = prepare_generation_config(self.generation_config, request_config, self.tokenizer)  # 调用工具函数准备配置
        generation_config.return_dict_in_generate = True  # 设置生成时返回字典格式
        if request_config.logprobs:  # 若需要返回对数概率
            generation_config.output_logits = True  # 设置输出 logits
        generation_config.num_return_sequences = request_config.n  # 设置返回序列数（生成多个候选）
        return _GenerationConfig(**generation_config.to_dict())  # 返回增强版生成配置

    def _add_stop_words(self, generation_config: _GenerationConfig, request_config: RequestConfig,  # 添加停止词到生成配置（内部方法）
                        template_meta: TemplateMeta) -> None:  # 模板元数据
        """函数功能：
        将请求配置和模板配置中的停止词合并，并设置到生成配置中。
        
        参数：
        - generation_config (_GenerationConfig): 待修改的生成配置对象
        - request_config (RequestConfig): 请求配置
        - template_meta (TemplateMeta): 模板元数据
        
        返回值：
        - None（直接修改 generation_config）
        
        示例：
        >>> # 内部调用
        >>> self._add_stop_words(gen_config, req_config, template_meta)
        """
        stop_words = (request_config.stop or []) + template_meta.stop_words  # 合并停止词列表（请求配置 + 模板配置）
        generation_config.stop_words = self._get_stop_words(stop_words)  # 调用父类方法处理停止词（去重、规范化），并设置到生成配置

    @staticmethod  # 声明为静态方法
    def preprocess_logits(batched_logits: Optional[List[torch.Tensor]], batched_generate_ids: torch.Tensor,  # 预处理 logits 为对数概率（静态方法）
                          top_logprobs: Optional[int]):  # 需要返回的 top-k 对数概率数量
        """函数功能：
        将原始 logits 张量转换为对数概率字典列表。
        
        参数：
        - batched_logits: 原始 logits 列表（每个 token 对应一个张量）
        - batched_generate_ids: 生成的 token ID 张量 [batch_size, seq_len]
        - top_logprobs: 需要返回的 top-k 数量
        
        返回值：
        - Optional[List[List[Dict]]]: 批量对数概率列表，或 None
        
        示例：
        >>> logprobs = PtEngine.preprocess_logits(logits, gen_ids, top_k=5)
        """
        top_logprobs = top_logprobs or 1  # 若未指定则默认为 1
        batch_size = batched_generate_ids.shape[0]  # 获取批大小
        if batched_logits is None:  # 若 logits 为 None
            return None  # 返回 None
        batched_logprobs = []  # 初始化批量对数概率列表
        for i in range(batch_size):  # 遍历每个批次
            logprobs_list = []  # 初始化该批次的对数概率列表
            generate_ids = batched_generate_ids[i]  # 获取该批次的生成 ID
            for j, logits in enumerate(batched_logits):  # 遍历每个 token 的 logits
                token = generate_ids[j].item()  # 获取实际生成的 token ID
                logprobs = torch.log_softmax(logits[i], -1)  # 计算对数概率（log softmax）
                tokens = [token] + logprobs.argsort(descending=True, dim=-1)[:top_logprobs].tolist()  # 获取 top-k token ID（包含实际 token）
                logprobs_list.append({token: logprobs[token].item() for token in tokens})  # 构造字典（token -> logprob）
            batched_logprobs.append(logprobs_list)  # 添加到批量列表
        return batched_logprobs  # 返回批量对数概率

    @staticmethod  # 声明为静态方法
    def _update_batched_logprobs(batched_logprobs: List[torch.Tensor], logits_streamer: Optional[LogitsStreamer],  # 更新批量对数概率（静态方法，用于流式推理）
                                 generate_ids: torch.Tensor, top_logprobs: int) -> None:  # 生成 ID 和 top-k 数量
        """函数功能：
        在流式推理中更新批量对数概率，从 logits_streamer 获取新的 logits 并追加。
        
        参数：
        - batched_logprobs: 待更新的批量对数概率列表
        - logits_streamer: logits 流式输出器
        - generate_ids: 生成的 token ID 张量
        - top_logprobs: top-k 数量
        
        返回值：
        - None（直接修改 batched_logprobs）
        
        示例：
        >>> # 内部调用，流式推理中使用
        >>> PtEngine._update_batched_logprobs(logprobs, streamer, ids, 5)
        """
        seq_len = generate_ids.shape[1] - len(batched_logprobs[0])  # 计算新生成的序列长度（当前长度 - 已处理长度）
        if logits_streamer is None or seq_len == 0:  # 若没有 logits streamer 或无新生成 token
            return  # 直接返回

        res = []  # 初始化新 logits 列表
        for i in range(seq_len):  # 遍历新生成的每个 token
            res.append(logits_streamer.queue.get())  # 从队列获取对应的 logits
        new_batched_logprobs = PtEngine.preprocess_logits(res, generate_ids[:, -seq_len:], top_logprobs)  # 预处理新 logits
        for logprobs, new_logprobs in zip(batched_logprobs, new_batched_logprobs):  # 遍历每个批次
            logprobs += new_logprobs  # 追加新对数概率到列表

    def _infer_stream(self, template: Template, inputs: Dict[str, Any], *, generation_config: GenerationConfig,  # 流式推理方法（内部方法）
                      adapter_request: Optional[AdapterRequest], request_config: RequestConfig,  # 适配器请求和请求配置
                      **kwargs) -> Iterator[List[Optional[ChatCompletionStreamResponse]]]:  # 返回流式响应列表的迭代器
        """函数功能：
        执行流式推理，在后台线程生成 token，并逐步产出流式响应。
        适用于需要实时显示生成结果的场景。
        
        参数：见上方签名注释。
        
        返回值：
        - Iterator[List[Optional[ChatCompletionStreamResponse]]]: 流式响应列表的迭代器
        
        示例：
        >>> # 内部调用，通常由 _infer 调用
        >>> for res_list in self._infer_stream(template, inputs, ...):
        ...     for res in res_list:
        ...         if res:
        ...             print(res.choices[0].delta.content, end='')
        """

        if generation_config.num_beams != 1:  # 若使用束搜索（beam search）
            error_msg = 'Streaming generation does not support beam search.'  # 定义错误信息
            raise ValueError(error_msg)  # 抛出异常（流式生成不支持束搜索）
        streamer = TokensIteratorStreamer()  # 创建 token 流式输出器（用于在生成时逐步产出 token）
        generate_kwargs = {  # 初始化生成参数字典
            'generation_config': generation_config,  # 生成配置
            'streamer': streamer,  # 流式输出器
            **inputs,  # 展开输入（input_ids、attention_mask 等）
        }
        adapter_names = self._get_adapter_names(adapter_request)  # 获取适配器名称列表
        if adapter_names is not None:  # 若存在适配器
            generate_kwargs['adapter_names'] = adapter_names  # 添加适配器名称到生成参数
        num_prompt_tokens = self._get_num_tokens(inputs)  # 计算 prompt 的 token 数量

        logits_streamer = None  # 初始化 logits 流式输出器为 None
        if generation_config.output_logits:  # 若需要输出 logits（用于计算对数概率）
            generate_kwargs['logits_processor'] = LogitsProcessorList([LogitsStreamer()])  # 添加 logits 处理器到生成参数

        def _model_generate(**kwargs):  # 定义模型生成的内部函数（在后台线程中运行）
            """内部函数：在后台线程中执行模型生成。"""
            if is_torch_npu_available():  # 若检测到华为 NPU
                torch.npu.set_device(self.model.device)  # 设置当前 NPU 设备（确保线程中的设备一致）
            template.generate(self.model, **kwargs)  # 调用模板的 generate 方法执行生成

        generate_kwargs = template.prepare_generate_kwargs(generate_kwargs, model=self.model)  # 调用模板方法准备生成参数（可能添加多模态数据等）
        thread = Thread(target=_model_generate, kwargs=generate_kwargs)  # 创建后台线程（目标函数为 _model_generate）
        thread.start()  # 启动后台线程（开始生成）
        batch_size = inputs['attention_mask'].shape[0]  # 从 attention_mask 获取批大小
        all_is_finished = False  # 初始化全部完成标志为 False
        is_finished = [False] * batch_size  # 初始化每个样本的完成标志列表
        infer_streamers = [InferStreamer(template) for _ in range(batch_size)]  # 为每个样本创建流式输出器（处理增量 token 的打印）
        request_id_list = [f'chatcmpl-{random_uuid()}' for _ in range(batch_size)]  # 为每个样本生成唯一的请求 ID
        token_idxs = [0] * batch_size  # 初始化每个样本的 token 索引（用于跟踪已处理的 token 数量）

        raw_batched_generate_ids = None  # or torch.Tensor: [batch_size, seq_len]  # 初始化原始生成 ID 张量为 None（后续会累积）
        batched_logprobs = [[] for _ in range(batch_size)]  # 初始化每个样本的对数概率列表
        while not all_is_finished:  # 循环直到所有样本完成生成
            try:  # 尝试从流式输出器获取下一批 token
                batched_tokens = next(streamer)  # 从流式输出器获取下一批生成的 token（可能是一维或二维）
                if batched_tokens.ndim == 1:  # 若 token 张量是一维的（单个 token）
                    batched_tokens = batched_tokens[:, None]  # 扩展为二维张量 [batch_size, 1]

                raw_batched_generate_ids = torch.concat(  # 将新生成的 token 拼接到累积张量
                    [batched_tokens]  # 若累积张量为 None，则直接使用新 token
                    if raw_batched_generate_ids is None else [raw_batched_generate_ids, batched_tokens],  # 否则拼接到累积张量末尾
                    dim=1)  # 在序列维度（dim=1）上拼接
            except StopIteration:  # 若流式输出器耗尽（生成完成）
                all_is_finished = True  # 标记全部完成

            batched_generate_ids = template.get_generate_ids(raw_batched_generate_ids, num_prompt_tokens)  # 调用模板方法提取纯生成部分（去除 prompt）
            self._update_batched_logprobs(batched_logprobs, logits_streamer, batched_generate_ids,  # 更新批量对数概率（若启用）
                                          request_config.top_logprobs)  # 传入 top_logprobs 参数

            res = []  # 初始化本次迭代的响应列表
            for i in range(batched_generate_ids.shape[0]):  # 遍历批次中的每个样本
                if is_finished[i]:  # 若该样本已完成
                    res.append(None)  # 添加 None 到响应列表
                    continue  # 跳过该样本
                generate_ids = batched_generate_ids[i]  # 获取该样本的生成 ID 张量

                # ignore pad_token  # 注释：忽略填充 token
                masks = generate_ids != self.tokenizer.pad_token_id  # 创建掩码（非填充 token 为 True）
                generate_ids = generate_ids[masks].tolist()  # 过滤掉填充 token 并转为列表
                logprobs_list = None  # 初始化对数概率列表为 None
                if batched_logprobs[i]:  # 若该样本有对数概率数据
                    logprobs_list = [logprobs for m, logprobs in zip(masks, batched_logprobs[i]) if m.item()]  # 根据掩码过滤对数概率

                is_finished[i] = (  # 判断该样本是否完成生成
                    all_is_finished or is_finished[i]  # 全部完成 或 已标记完成
                    or len(generate_ids) > 0 and generate_ids[-1] == self.tokenizer.pad_token_id)  # 或 最后一个 token 是填充 token
                delta_text = infer_streamers[i].get_printable_text(generate_ids, is_finished[i])  # 使用流式输出器提取增量可打印文本
                if not delta_text and not is_finished[i]:  # 若无新增文本且未完成
                    res.append(None)  # 添加 None（跳过本次响应）
                    continue  # 跳过后续处理
                logprobs = self._get_logprobs(logprobs_list, generate_ids[token_idxs[i]:], request_config.top_logprobs)  # 获取新增 token 的对数概率
                token_idxs[i] = len(generate_ids)  # 更新该样本的 token 索引为当前总长度

                usage_info = self._get_usage_info(num_prompt_tokens, len(generate_ids))  # 构造用量信息
                toolcall = None  # 初始化工具调用为 None
                if is_finished[i]:  # 若该样本已完成
                    toolcall = self._get_toolcall(template.decode(generate_ids), template)  # 解码完整输出并提取工具调用信息
                finish_reason = self._get_finish_reason(generation_config.max_new_tokens, usage_info.completion_tokens,  # 获取完成原因
                                                        is_finished[i])  # 传入完成标志

                choices = [  # 构造选项列表
                    ChatCompletionResponseStreamChoice(  # 创建流式响应选项对象
                        index=0,  # 选项索引（单个候选时为 0）
                        delta=DeltaMessage(role='assistant', content=delta_text, tool_calls=toolcall),  # 增量消息（角色、增量文本、工具调用）
                        finish_reason=finish_reason,  # 完成原因
                        logprobs=logprobs)  # 对数概率信息
                ]
                res.append(  # 添加到响应列表
                    ChatCompletionStreamResponse(  # 创建流式响应对象
                        model=self.model_name, choices=choices, usage=usage_info, id=request_id_list[i]))  # 包含模型名、选项、用量、请求 ID
            if any(res):  # 若响应列表中有非 None 元素
                yield res  # 产出响应列表给调用方

    def _get_adapter_names(self, adapter_request: Optional[AdapterRequest]) -> Optional[List[str]]:  # 获取适配器名称列表（内部方法）
        """函数功能：
        根据适配器请求获取适配器名称列表，支持动态加载新适配器。
        
        参数：
        - adapter_request (Optional[AdapterRequest]): 适配器请求对象
        
        返回值：
        - Optional[List[str]]: 适配器名称列表，或 None
        
        示例：
        >>> # 内部调用
        >>> names = self._get_adapter_names(adapter_req)
        """
        if adapter_request is None:  # 若未提供适配器请求
            if self._adapters_pool:  # 若适配器池不为空（有已加载的适配器）
                return ['__base__']  # 返回基础模型标识（表示不使用适配器，使用原始模型）
            return  # 返回 None
        adapter_name = adapter_request.name  # 获取适配器名称
        if adapter_name not in self._adapters_pool:  # 若适配器尚未加载
            self._adapters_pool[adapter_name] = adapter_request  # 添加到适配器池
            self._add_adapter(adapter_request.path, adapter_name)  # 动态加载适配器
        return [adapter_name]  # 返回适配器名称列表

    def _infer_forward(self, template: Template, inputs: Dict[str, Any], adapter_request: Optional[AdapterRequest],  # 前向推理方法（用于非生成任务）
                       request_config: RequestConfig, **kwargs):  # 请求配置和额外参数
        """函数功能：
        执行前向推理（Forward Pass），用于非文本生成任务（序列分类、PRM、嵌入等）。
        直接调用模型的 forward 方法，无需自回归生成。
        
        参数：
        - template: 对话模板
        - inputs: 输入字典（input_ids、attention_mask 等）
        - adapter_request: 适配器请求
        - request_config: 请求配置
        - **kwargs: 额外参数
        
        返回值：
        - List[Union[ChatCompletionResponse, EmbeddingResponse]]: 响应列表
        
        示例：
        >>> # 内部调用，用于序列分类等任务
        >>> responses = self._infer_forward(template, inputs, None, req_config)
        """
        call_kwargs = {}  # 初始化模型调用参数字典
        top_logprobs = request_config.top_logprobs or 20  # 获取 top_logprobs 参数（默认 20）
        adapter_names = self._get_adapter_names(adapter_request)  # 获取适配器名称列表
        if adapter_names is not None:  # 若存在适配器
            call_kwargs['adapter_names'] = adapter_names  # 添加适配器名称到调用参数
        num_prompt_tokens = self._get_num_tokens(inputs)  # 计算 prompt 的 token 数量
        inputs.pop('labels', None)  # 移除 labels 字段（前向推理不需要）
        output = self.model(**inputs, **call_kwargs)  # 调用模型的 forward 方法（一次前向传播）
        if hasattr(output, 'logits'):  # 若输出有 logits 属性（序列分类、PRM 等任务）
            logits = output.logits  # 提取 logits 张量
        elif 'last_hidden_state' in output:  # 若输出是字典且包含 last_hidden_state（嵌入任务）
            # embeddings  # 注释：嵌入任务
            logits = output['last_hidden_state']  # 提取最后隐藏状态作为嵌入向量
        if template.task_type == 'seq_cls':  # 若任务类型为序列分类
            preds, logprobs = template.decode_seq_cls(logits, top_logprobs)  # 调用模板方法解码分类结果和对数概率
        elif template.task_type == 'prm':  # 若任务类型为 PRM（Process Reward Model）
            preds = template.decode_prm(inputs['input_ids'], logits)  # 调用模板方法解码 PRM 结果
            logprobs = [None] * len(preds)  # PRM 任务不返回对数概率
        elif template.task_type == 'embedding':  # 若任务类型为嵌入生成
            preds = logits  # 直接使用 logits 作为嵌入向量
            logprobs = [None] * len(preds)  # 嵌入任务不返回对数概率
        else:  # 若任务类型不支持
            raise ValueError(f'Unsupported task_type: {template.task_type}')  # 抛出异常

        res = []  # 初始化结果列表
        for i, pred in enumerate(preds):  # 遍历每个预测结果
            usage_info = self._get_usage_info(num_prompt_tokens, 1)  # 构造用量信息（生成 token 数为 1，因为是前向推理）
            if template.task_type == 'embedding':  # 若任务类型为嵌入生成
                res.append(  # 添加嵌入响应到结果列表
                    EmbeddingResponse(  # 创建嵌入响应对象
                        model=self.model_name,  # 模型名称
                        usage=usage_info,  # 用量信息
                        data=[EmbeddingResponseData(embedding=pred.to(torch.float32).cpu().numpy().tolist())]))  # 嵌入数据（转为 float32、CPU、NumPy、列表）
            else:  # 若任务类型为序列分类或 PRM
                choices = [  # 构造选项列表
                    ChatCompletionResponseChoice(  # 创建响应选项对象
                        index=0,  # 选项索引（单个候选时为 0）
                        message=ChatMessage(role='assistant', content=pred, tool_calls=None),  # 消息（预测结果作为内容）
                        finish_reason='stop',  # 完成原因（前向推理总是 stop）
                        logprobs=logprobs[i])  # 对数概率信息
                ]
                res.append(ChatCompletionResponse(model=self.model_name, choices=choices, usage=usage_info))  # 添加聊天补全响应到结果列表
        return res  # 返回结果列表

    def _infer_full(self, template: Template, inputs: Dict[str, Any], *, generation_config: GenerationConfig,  # 非流式完整推理方法（内部方法）
                    adapter_request: Optional[AdapterRequest], request_config: RequestConfig,  # 适配器请求和请求配置
                    template_inputs) -> List[ChatCompletionResponse]:  # 返回聊天补全响应列表
        """函数功能：
        执行非流式完整推理，一次性生成所有 token 后返回完整响应。
        适用于不需要实时显示的场景（如批处理）。
        
        参数：见上方签名注释。
        
        返回值：
        - List[ChatCompletionResponse]: 聊天补全响应列表
        
        示例：
        >>> # 内部调用，通常由 _infer 调用
        >>> responses = self._infer_full(template, inputs, ...)
        """
        # bos_token TODO: encoder-decoder  # 注释：BOS token 处理待实现（编码器-解码器模型）
        generate_kwargs = {'generation_config': generation_config, **inputs}  # 初始化生成参数字典（合并生成配置和输入）
        adapter_names = self._get_adapter_names(adapter_request)  # 获取适配器名称列表
        if adapter_names is not None:  # 若存在适配器
            generate_kwargs['adapter_names'] = adapter_names  # 添加适配器名称到生成参数
        num_prompt_tokens = self._get_num_tokens(inputs)  # 计算 prompt 的 token 数量
        generate_kwargs = template.prepare_generate_kwargs(generate_kwargs, model=self.model)  # 调用模板方法准备生成参数（可能添加多模态数据等）
        output = dict(template.generate(self.model, **generate_kwargs))  # 调用模板的 generate 方法执行完整生成，并转为字典
        output.pop('past_key_values', None)  # 移除 past_key_values（KV 缓存，不需要返回）
        batched_generate_ids = output['sequences']  # 提取生成的序列张量（包含 prompt + 生成部分）
        batched_generate_ids = template.get_generate_ids(batched_generate_ids, num_prompt_tokens)  # 调用模板方法提取纯生成部分（去除 prompt）
        template.debug_logger({'generate_ids': batched_generate_ids})  # debug  # 记录调试日志（生成的 token ID）
        batched_logprobs = self.preprocess_logits(  # 预处理 logits 为对数概率
            output.get('logits'), batched_generate_ids, request_config.top_logprobs)  # 传入 logits、生成 ID 和 top_logprobs 参数

        res = []  # 初始化结果列表
        num_return_sequences = generation_config.num_return_sequences  # 获取每个样本需要返回的序列数（n 参数）
        for i in range(inputs['attention_mask'].shape[0]):  # 遍历批次中的每个样本
            choices = []  # 初始化该样本的选项列表
            usage_info = self._get_usage_info(num_prompt_tokens, 0)  # 初始化用量信息（生成 token 数为 0，后续会更新）
            for j in range(num_return_sequences):  # 遍历该样本的每个候选序列
                batched_index = i * num_return_sequences + j  # 计算在批量张量中的索引（批大小 * 候选数）
                generate_ids = batched_generate_ids[batched_index]  # 获取该候选的生成 ID 张量

                # ignore pad_token  # 注释：忽略填充 token
                masks = generate_ids != self.tokenizer.pad_token_id  # 创建掩码（非填充 token 为 True）
                generate_ids = generate_ids[masks].tolist()  # 过滤掉填充 token 并转为列表
                logprobs_list = None  # 初始化对数概率列表为 None
                if batched_logprobs is not None:  # 若有对数概率数据
                    logprobs_list = [  # 根据掩码过滤对数概率
                        logprobs for m, logprobs in zip(masks, batched_logprobs[batched_index]) if m.item()  # 仅保留非填充 token 的对数概率
                    ]

                logprobs = self._get_logprobs(logprobs_list, generate_ids, request_config.top_logprobs)  # 获取对数概率信息
                usage_info = self._update_usage_info(usage_info, len(generate_ids))  # 更新用量信息（累加生成 token 数）
                response = template.decode(generate_ids, template_inputs=template_inputs[i])  # 使用模板解码生成 ID 为文本（传入模板输入用于解码）
                finish_reason = self._get_finish_reason(generation_config.max_new_tokens, len(generate_ids), True)  # 获取完成原因（True 表示已完成）
                toolcall = self._get_toolcall(response, template)  # 从解码文本中提取工具调用信息
                token_ids = template.skip_stop_tokens(generate_ids) if request_config.return_details else None  # 若需要返回详细信息，则跳过停止 token 并返回 token_ids
                choices.append(  # 添加到选项列表
                    ChatCompletionResponseChoice(  # 创建响应选项对象
                        index=j,  # 候选索引
                        message=ChatMessage(role='assistant', content=response, tool_calls=toolcall),  # 完整消息（角色、内容、工具调用）
                        finish_reason=finish_reason,  # 完成原因
                        logprobs=logprobs,  # 对数概率信息
                        token_ids=token_ids))  # token_ids（若 return_details=True）
            prompt_token_ids = None  # 初始化 prompt token_ids 为 None
            if request_config.return_details and 'input_ids' in inputs:  # 若需要返回详细信息且输入中有 input_ids
                non_pad_indices = (inputs['input_ids'][i] != self.tokenizer.pad_token_id).nonzero()  # 找到非填充 token 的索引
                if non_pad_indices.numel() > 0:  # 若存在非填充 token
                    idx = non_pad_indices.min().item()  # 获取第一个非填充 token 的索引
                    prompt_token_ids = inputs['input_ids'][i][idx:].tolist()  # 提取 prompt 的 token_ids（去除前面的填充）
            res.append(  # 添加到结果列表
                ChatCompletionResponse(  # 创建聊天补全响应对象
                    model=self.model_name, choices=choices, usage=usage_info, prompt_token_ids=prompt_token_ids))  # 包含模型名、选项、用量、prompt token_ids
        return res  # 返回结果列表

    async def infer_async(  # 异步推理入口（公开方法）
        self,  # 实例自身引用
        infer_request: InferRequest,  # 单个推理请求
        request_config: Optional[RequestConfig] = None,  # 可选的推理配置
        *,  # 仅限关键字参数分隔符
        template: Optional[Template] = None,  # 可选的对话模板
        adapter_request: Optional[AdapterRequest] = None,  # 可选的 LoRA/Swift 适配器请求
        pre_infer_hook=None,  # 可选的推理前钩子函数
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:  # 返回非流式响应或异步流式迭代器
        """函数功能：
        异步推理入口，支持单个请求的流式/非流式输出。
        内部使用后台线程和任务队列实现动态批处理。
        
        参数：见上方签名注释。
        
        返回值：
        - Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
          非流式返回 ChatCompletionResponse；流式返回 AsyncIterator。
        
        示例：
        >>> engine = PtEngine('Qwen/Qwen-7B-Chat', max_batch_size=8)
        >>> req = InferRequest(messages=[{"role": "user", "content": "hi"}])
        >>> # 非流式
        >>> response = await engine.infer_async(req)
        >>> print(response.choices[0].message.content)
        >>> # 流式
        >>> async for chunk in await engine.infer_async(req, RequestConfig(stream=True)):
        ...     print(chunk.choices[0].delta.content, end='')
        """
        if request_config is None:  # 若未提供请求配置
            request_config = RequestConfig()  # 使用默认配置
        queue = asyncio.Queue()  # 创建异步队列（用于接收推理结果）
        self._queue.put((infer_request, {  # 将请求放入任务队列（线程安全的同步队列）
            'request_config': request_config,  # 请求配置
            'template': template,  # 模板
            'adapter_request': adapter_request,  # 适配器请求
            'pre_infer_hook': pre_infer_hook  # 推理前钩子
        }, (queue, asyncio.get_event_loop())))  # 附加异步队列和事件循环（用于跨线程通信）
        await asyncio.sleep(0)  # 让出控制权，允许其他协程运行
        if self._task_thread is None:  # 若后台线程尚未启动
            self._start_infer_worker()  # 启动后台推理工作线程
        if request_config.stream:  # 若需要流式输出

            async def _gen_wrapper():  # 定义异步生成器包装函数
                """内部异步生成器：从队列中获取流式响应并产出。"""
                while True:  # 无限循环
                    item = await queue.get()  # 从异步队列获取结果
                    await asyncio.sleep(0)  # 让出控制权
                    if item is None:  # 若结果为 None（流式结束标记）
                        break  # 终止生成器
                    yield item  # 产出流式响应给调用方

            return _gen_wrapper()  # 返回异步生成器
        else:  # 若需要非流式输出
            return await queue.get()  # 等待并返回完整响应

    # Ensure `template._post_encode` has no gradient.  # 注释：确保模板的 _post_encode 方法无梯度
    @torch.inference_mode()  # 启用推理模式装饰器（禁用梯度计算，提升性能和降低显存）
    def _infer(  # 核心推理方法（内部方法）
        self,  # 实例自身引用
        infer_requests: List[InferRequest],  # 批量推理请求列表
        request_config: RequestConfig,  # 请求配置
        *,  # 仅限关键字参数分隔符
        template: Optional[Template] = None,  # 可选的对话模板
        adapter_request: Optional[AdapterRequest] = None,  # 可选的适配器请求
        pre_infer_hook=None,  # 可选的推理前钩子函数
    ) -> Union[List[ChatCompletionResponse], Iterator[List[Optional[ChatCompletionStreamResponse]]]]:  # 返回响应列表或流式响应生成器
        """函数功能：
        核心推理方法，负责编码输入、选择推理模式（流式/非流式、生成/前向）并执行推理。
        支持多种任务类型（causal_lm、seq_cls、prm、embedding）。
        
        参数：见上方签名注释。
        
        返回值：
        - Union[List[ChatCompletionResponse], Iterator[List[Optional[ChatCompletionStreamResponse]]]]:
          非流式返回响应列表；流式返回生成器。
        
        示例：
        >>> # 内部调用，通常由 infer 或 infer_async 调用
        >>> responses = self._infer(reqs, req_config)
        """
        self.model.eval()  # 设置模型为评估模式（禁用 dropout 等训练特有操作）
        request_config = deepcopy(request_config)  # 深拷贝请求配置，避免副作用
        if template is None:  # 若未提供模板
            template = self.default_template  # 使用默认模板
        if template.use_model:  # 若模板需要使用模型（某些模板需要访问模型进行特殊处理）
            template.model = self.model  # 将模型引用传递给模板

        if self.model_info.task_type == 'causal_lm':  # 若任务类型为因果语言模型（文本生成）
            template.set_mode('pt')  # 设置模板为 PyTorch 模式

        batched_inputs, error_list = self._batch_encode(  # 批量编码请求
            infer_requests, template=template, strict=getattr(self, 'strict', True))  # 传入模板和严格模式标志
        if len(batched_inputs) > 0:  # 若有成功编码的输入
            template_inputs = [inputs.pop('template_inputs') for inputs in batched_inputs]  # 从每个输入中提取 template_inputs（用于解码）
            inputs = to_device(template.data_collator(batched_inputs), self.model.device)  # 使用数据收集器整理批次，并移动到模型设备（GPU/CPU）
            template.debug_logger(inputs)  # debug  # 记录调试日志（输入数据）
            if self.model.model_meta.is_multimodal:  # 若模型支持多模态
                _, inputs = template.pre_forward_hook(self.model, None, inputs)  # 调用模板的前向钩子处理多模态输入（如图像编码）
            if self.model_info.task_type == 'causal_lm':  # 若任务类型为因果语言模型
                self.set_default_max_tokens(request_config, inputs)  # 根据输入长度设置默认的 max_tokens
                generation_config = self._prepare_generation_config(request_config)  # 准备生成配置
                self._add_stop_words(generation_config, request_config, template.template_meta)  # 添加停止词
            else:  # 若任务类型为其他（序列分类、PRM、嵌入等）
                generation_config = request_config  # 直接使用请求配置作为生成配置

            kwargs = {  # 构造推理参数字典
                'template': template,  # 模板
                'inputs': inputs,  # 输入
                'generation_config': generation_config,  # 生成配置
                'adapter_request': adapter_request,  # 适配器请求
                'request_config': request_config,  # 请求配置
                'template_inputs': template_inputs,  # 模板输入（用于解码）
            }
            if pre_infer_hook:  # 若提供了推理前钩子
                kwargs = pre_infer_hook(kwargs)  # 调用钩子修改参数
        else:  # 若无成功编码的输入（所有请求都编码失败）
            kwargs = {}  # 设置空参数字典
        if request_config.stream:  # 若需要流式输出

            def _gen_wrapper():  # 定义生成器包装函数
                """内部生成器：包装流式推理结果并添加错误列表。"""
                if len(kwargs) > 0:  # 若有有效参数
                    for res in self._infer_stream(**kwargs):  # 调用流式推理方法
                        yield self._add_error_list(res, error_list)  # 将错误列表添加到响应中并产出
                else:  # 若无有效参数（所有请求编码失败）
                    yield self._add_error_list([], error_list)  # 产出仅包含错误的空列表

            return _gen_wrapper()  # 返回生成器
        else:  # 若需要非流式输出
            if len(kwargs) > 0:  # 若有有效参数
                infer_func = self._infer_forward if template.task_type in {'seq_cls', 'prm', 'embedding'  # 根据任务类型选择推理函数
                                                                           } else self._infer_full  # 序列分类/PRM/嵌入用 _infer_forward，文本生成用 _infer_full
                res = infer_func(**kwargs)  # 调用对应的推理函数
            else:  # 若无有效参数（所有请求编码失败）
                res = []  # 设置空结果列表
            return self._add_error_list(res, error_list)  # 将错误列表添加到响应中并返回

    def infer(  # 同步推理入口（公开方法）
        self,  # 实例自身引用
        infer_requests: List[InferRequest],  # 批量推理请求列表
        request_config: Optional[RequestConfig] = None,  # 可选的统一推理配置
        metrics: Optional[List[Metric]] = None,  # 可选的指标采集器列表
        *,  # 仅限关键字参数分隔符
        template: Optional[Template] = None,  # 可选的对话模板
        use_tqdm: Optional[bool] = None,  # 是否显示进度条
        adapter_request: Optional[AdapterRequest] = None  # 可选的 LoRA/Swift 适配器请求
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:  # 返回响应列表
        """函数功能：
        同步推理入口，支持批量处理和流式/非流式输出。
        流式请求调用父类方法（使用异步包装）；非流式请求直接调用 _infer。
        
        参数：见上方签名注释。
        
        返回值：
        - List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
          与输入等长的响应列表，元素为非流式响应或流式响应迭代器。
        
        示例：
        >>> engine = PtEngine('Qwen/Qwen-7B-Chat', max_batch_size=8)
        >>> reqs = [InferRequest(messages=[{"role": "user", "content": f"hi {i}"}]) for i in range(10)]
        >>> responses = engine.infer(reqs)
        >>> for resp in responses:
        ...     print(resp.choices[0].message.content)
        """
        if request_config is None:  # 若未提供请求配置
            request_config = RequestConfig()  # 使用默认配置
        if request_config.stream:  # 若需要流式输出
            return super().infer(  # 调用父类方法（父类会自动转为异步并阻塞等待）
                infer_requests,  # 推理请求列表
                request_config,  # 请求配置
                metrics,  # 指标采集器
                template=template,  # 模板
                use_tqdm=use_tqdm,  # 进度条
                adapter_request=adapter_request)  # 适配器请求
        # Has higher stability than calling super().infer  # 注释：非流式模式下直接调用 _infer 比调用父类方法更稳定
        if use_tqdm is None:  # 若未指定是否显示进度条
            use_tqdm = not request_config.stream and len(infer_requests) > 1  # 非流式且多请求时默认显示
        prog_bar = tqdm(total=len(infer_requests), dynamic_ncols=True, disable=not use_tqdm)  # 创建进度条
        # If self.max_batch_size <= 0, then process all infer_requests at once.  # 注释：若 max_batch_size <= 0，则一次处理所有请求
        max_batch_size = self.max_batch_size  # 获取最大批处理大小
        if max_batch_size <= 0:  # 若设置为 0 或负数（表示无限制）
            max_batch_size = len(infer_requests)  # 一次处理所有请求
        res = []  # 初始化结果列表
        i = 0  # 初始化索引
        while i < len(infer_requests):  # 循环处理所有请求
            infer_requests_samples = infer_requests[i:i + max_batch_size]  # 取出一批请求
            res += self._infer(  # 执行推理并追加结果
                infer_requests_samples, request_config, template=template, adapter_request=adapter_request)  # 传递批量请求和配置
            i += max_batch_size  # 更新索引
            prog_bar.update(len(infer_requests_samples))  # 更新进度条
        prog_bar.close()  # 关闭进度条
        self._update_metrics(res, metrics)  # 更新指标
        return res  # 返回结果列表
