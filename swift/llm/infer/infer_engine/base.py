"""模块功能概述：
本模块定义推理引擎的抽象基类 `BaseInferEngine`，统一规范同步与异步两种推理接口的签名、
输入输出与行为约定，便于不同后端（本地/远程，流式/非流式）以一致方式接入。

- 定义同步批量推理方法 `infer` 与单请求异步推理方法 `infer_async`。
- 通过类型注解约束请求与响应的数据结构，提升实现与调用的可读性和可靠性。
- 该模块仅包含接口定义，不包含任何具体推理逻辑，需由子类实现。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，指明源代码的版权所有者
from abc import ABC, abstractmethod  # 从标准库 abc 导入抽象基类 ABC 与抽象方法装饰器 abstractmethod
from typing import AsyncIterator, Iterator, List, Optional, Union  # 导入常用类型注解，用于声明参数与返回值的类型

from swift.llm import InferRequest  # 导入推理请求的数据结构，描述一次对话/补全的输入
from swift.plugin import Metric  # 导入指标采集插件类型，用于收集用量与性能等指标
from ..protocol import ChatCompletionResponse, ChatCompletionStreamResponse, RequestConfig  # 导入协议层的响应/流式响应与请求配置类型


class BaseInferEngine(ABC):

    """类功能：
    `BaseInferEngine` 用于约束推理引擎实现需提供的统一接口，包含同步批量推理与异步单请求推理。
    是所有推理引擎需遵循的抽象基类，基于 ABC 无法直接实例化。

    - 继承关系：继承 `ABC`，声明为抽象基类；必须由具体引擎子类实现抽象方法。
    - 主要职责：统一对外的推理 API，使上层调用方无需关心底层后端差异。
    - 适用场景：对话补全、函数调用、流式增量解码等大模型推理任务。
    - 并发/线程安全：是否线程/协程安全由具体子类文档明确说明。
    """

    @abstractmethod  # 将方法标记为抽象方法，强制子类给出实现
    def infer(self,  # 同步推理主入口：处理一批推理请求并同步返回结果列表
              infer_requests: List[InferRequest],  # 要处理的推理请求列表，每个元素描述一次对话/补全
              request_config: Optional[RequestConfig] = None,  # 可选的统一请求配置（如采样策略、最大生成长度等）
              metrics: Optional[List[Metric]] = None,  # 可选的指标采集器列表，用于记录用量/性能等
              *,  # 仅限关键字参数分隔符，* 之后的参数必须以关键字形式传入
              use_tqdm: Optional[bool] = None,  # 是否显示进度条；None 表示由具体实现决定
              **kwargs) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:  # 返回与输入等长的结果，元素为非流式响应或流式迭代器
        """函数功能：
        同步执行一批推理请求，依次生成结果。支持非流式一次性返回，或以迭代器方式流式产出。

        参数：
        - infer_requests (List[InferRequest]): 待处理的推理请求列表。
        - request_config (Optional[RequestConfig]): 推理配置（如是否流式、温度、top_p 等）。
        - metrics (Optional[List[Metric]]): 记录/上报用量与性能指标的采集器集合。
        - use_tqdm (Optional[bool]): 是否启用进度条显示，便于长批次任务观测进度。
        - **kwargs: 其他与具体实现相关的可选扩展参数。

        返回值：
        - List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
          与输入请求列表等长的结果列表。对于非流式推理返回 `ChatCompletionResponse`；
          对于流式推理返回 `Iterator[ChatCompletionStreamResponse]`，可在调用方逐步消费。

        示例：
        >>> class MyEngine(BaseInferEngine):
        ...     def infer(self, infer_requests, request_config=None, metrics=None, *, use_tqdm=None, **kwargs):
        ...         return [ChatCompletionResponse()] * len(infer_requests)
        ...     async def infer_async(self, infer_request, request_config=None, **kwargs):
        ...         return ChatCompletionResponse()
        >>> engine = MyEngine()
        >>> reqs = [InferRequest(messages=[{"role": "user", "content": "hi"}])]
        >>> results = engine.infer(reqs, request_config=RequestConfig())
        >>> assert len(results) == 1
        """
        pass  # 抽象方法占位符：基类不提供实现，由子类覆盖

    @abstractmethod  # 将以下协程方法标记为抽象方法，强制子类实现
    async def infer_async(self,  # 异步推理主入口：处理单个请求，便于在异步框架中高并发使用
                          infer_request: InferRequest,  # 单个推理请求对象，包含对话上下文/参数等
                          request_config: Optional[RequestConfig] = None,  # 可选的请求配置（如是否流式）
                          **kwargs) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:  # 返回非流式响应或异步流式迭代器
        """函数功能：
        以协程方式处理单个推理请求。若启用流式，将返回一个异步迭代器以逐步产出增量结果。

        参数：
        - infer_request (InferRequest): 待处理的单个推理请求。
        - request_config (Optional[RequestConfig]): 推理配置（如是否开启流式输出）。
        - **kwargs: 其他与实现相关的可选扩展参数。

        返回值：
        - Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
          非流式返回 `ChatCompletionResponse`；
          流式返回 `AsyncIterator[ChatCompletionStreamResponse]`，可通过 `async for` 逐条消费。

        示例：
        >>> async def run(engine: BaseInferEngine):
        ...     req = InferRequest(messages=[{"role": "user", "content": "hello"}])
        ...     result = await engine.infer_async(req, request_config=RequestConfig(stream=False))
        ...     # 若 stream=True，可：
        ...     # async for chunk in await engine.infer_async(req, request_config=RequestConfig(stream=True)):
        ...     #     print(chunk)
        """
        pass  # 抽象方法占位符：基类不提供实现，由子类覆盖
