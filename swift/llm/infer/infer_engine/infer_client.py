"""模块功能概述：
该模块实现面向 HTTP 服务的推理客户端 `InferClient`，用于调用推理服务端点以完成
聊天补全（Chat Completions）相关请求。提供：
- 同步接口封装（复用父类），便于批量请求；
- 异步接口（单请求，支持流式/非流式），基于 `aiohttp` 访问 RESTful API；
- 模型列表查询与请求参数准备/解析工具方法。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，标注代码版权所有者
import os  # 引入 os 模块（当前未直接使用，保留以便扩展环境变量或路径相关能力）
from copy import deepcopy  # 引入 deepcopy，用于对请求配置进行深拷贝，避免副作用
from dataclasses import asdict  # 引入 asdict，将数据类实例转换为字典，便于序列化
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union  # 引入类型注解，明确参数与返回值类型

import aiohttp  # 引入 aiohttp，用于异步 HTTP 客户端请求
import json  # 引入 json，用于处理流式响应中的 JSON 文本
from dacite import from_dict  # 引入 from_dict，将字典反序列化为数据类对象
from requests.exceptions import HTTPError  # 引入 HTTPError，用于在 HTTP 出错时抛出异常

from swift.plugin import Metric  # 引入 Metric 类型，用于传递统计/用量指标采集器
from ..protocol import (ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStreamResponse, InferRequest,  # 引入协议层数据类：请求/响应/流式响应/推理请求
                        ModelList, RequestConfig)  # 引入模型列表与请求配置的数据类
from .infer_engine import InferEngine  # 引入父类 InferEngine，提供通用推理接口与工具方法


class InferClient(InferEngine):  # 定义面向服务的推理客户端，继承通用推理引擎基类

    """类功能：
    `InferClient` 负责与远端推理服务交互，完成模型列表获取与对话补全推理的调用。

    - 角色：HTTP 客户端；
    - 能力：支持同步（经父类封装）与异步（本类提供）两种推理调用；
    - 适用：对接符合 OpenAI 风格的 /v1/chat/completions 与 /v1/models 接口；
    - 线程/协程安全：同一实例在多协程场景下建议为每次请求创建新的 `ClientSession`（本类实现即如此）。
    """

    def __init__(self,  # 初始化客户端实例，设置基础连接参数与鉴权信息
                 host: str = '127.0.0.1',  # 推理服务主机名，默认本地回环地址
                 port: int = 8000,  # 推理服务端口，默认 8000
                 api_key: str = 'EMPTY',  # 鉴权所需 API Key，默认 'EMPTY'（测试用途）
                 *,  # 仅限关键字参数分隔符，后续参数必须以关键字形式传入
                 base_url: Optional[str] = None,  # 可选的基础 URL（若不指定，将由 host/port 拼装）
                 timeout: Optional[int] = 86400) -> None:  # 请求超时秒数，默认 86400（一天）
        """函数功能：
        初始化 `InferClient` 实例，配置服务地址、鉴权与超时策略。

        参数：
        - host (str): 推理服务主机名，默认 '127.0.0.1'。
        - port (int): 推理服务端口，默认 8000。
        - api_key (str): 鉴权使用的 API Key，默认 'EMPTY'。
        - base_url (Optional[str]): 完整基础 URL，若未提供，将由 host/port 自动拼接。
        - timeout (Optional[int]): 请求超时秒数，默认 86400。

        返回值：
        - None

        示例：
        >>> client = InferClient(host='127.0.0.1', port=8000, api_key='test')
        >>> assert client.base_url.endswith('/v1')
        """
        self.api_key = api_key  # 保存 API Key，用于后续请求头鉴权
        self.host = host  # 保存服务主机名
        self.port = port  # 保存服务端口
        self.timeout = timeout  # 保存超时设置（秒）
        if base_url is None:  # 若未显式提供 base_url，则根据 host/port 组装默认值
            base_url = f'http://{self.host}:{self.port}/v1'  # 组装符合 OpenAI 风格的基础 URL 前缀
        self.base_url = base_url  # 保存基础 URL，供后续接口拼接路径
        self._models = None  # 缓存可用模型列表，延迟加载，避免频繁请求

    @property  # 将方法暴露为只读属性，访问时自动计算并缓存
    def models(self):
        """函数功能：
        返回懒加载并缓存的可用模型 ID 列表，首次访问会从服务器拉取并缓存，避免重复网络请求。

        参数：
        - 无

        返回值：
        - List[str]: 模型标识符列表。

        示例：
        >>> client = InferClient()
        >>> ids = client.models  # 首次访问触发请求并缓存
        >>> assert isinstance(ids, list)
        """
        if self._models is None:  # 若尚未缓存，则从服务端获取
            models = []  # 临时列表，用于收集模型 ID
            for model in self.get_model_list().data:  # 遍历服务端返回的模型信息列表
                models.append(model.id)  # 提取模型 ID 并追加到列表
            assert len(models) > 0, f'models: {models}'  # 简单校验，确保至少存在一个模型
            self._models = models  # 将结果写入缓存，后续直接复用
        return self._models  # 返回缓存的模型 ID 列表

    def get_model_list(self) -> ModelList:  # 获取模型列表的同步封装（）
        """函数功能：
        从推理服务同步获取可用模型列表，内部转为异步并阻塞等待。

        参数：
        - 无

        返回值：
        - ModelList: 模型列表数据类对象，包含 `data` 字段等。

        示例：
        >>> client = InferClient()
        >>> ml = client.get_model_list()
        >>> assert hasattr(ml, 'data')
        """
        coro = self.get_model_list_async()  # 获取异步协程对象（不立即执行）
        return self.safe_asyncio_run(coro)  # 通过父类工具在同步环境中安全运行协程并返回结果

    def _get_request_kwargs(self) -> Dict[str, Any]:  # 组装通用的请求关键字参数（headers/timeout 等）
        request_kwargs = {}  # 初始化空的请求参数字典
        if isinstance(self.timeout, (int, float)) and self.timeout > 0:  # 若设置了有效的超时
            request_kwargs['timeout'] = self.timeout  # 写入超时参数，以秒为单位
        if self.api_key is not None:  # 若提供了 API Key
            request_kwargs['headers'] = {'Authorization': f'Bearer {self.api_key}'}  # 使用 Bearer 方案设置鉴权头
        return request_kwargs  # 返回拼装好的请求参数

    async def get_model_list_async(self) -> ModelList:  # 异步方式获取模型列表
        url = f"{self.base_url.rstrip('/')}/models"  # 拼接模型列表接口 URL，并去除末尾斜杠避免重复
        async with aiohttp.ClientSession() as session:  # 创建异步 HTTP 会话，确保用后关闭
            async with session.get(url, **self._get_request_kwargs()) as resp:  # 以 GET 方式请求模型列表
                resp_obj = await resp.json()  # 解析响应体为 JSON 字典
        return from_dict(ModelList, resp_obj)  # 将字典反序列化为 ModelList 数据类并返回

    def infer(  # 同步推理入口（签名与父类一致），此处仅透传给父类实现
            self,  # 实例自身引用
            infer_requests: List[InferRequest],  # 批量推理请求列表
            request_config: Optional[RequestConfig] = None,  # 可选的统一推理配置
            metrics: Optional[List[Metric]] = None,  # 可选的指标采集器列表
            *,  # 仅限关键字参数分隔符
            model: Optional[str] = None,  # 指定使用的模型名称，若未指定且仅有一个可用模型则自动选用
            use_tqdm: Optional[bool] = None  # 是否启用进度条显示
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:  # 返回非流式或流式结果的列表
        """函数功能：
        以同步方式执行批量推理调用。当前实现直接复用父类逻辑，仅补充了 `model` 参数透传。

        参数：
        - infer_requests (List[InferRequest]): 推理请求列表。
        - request_config (Optional[RequestConfig]): 推理配置（温度、top_p、是否流式等）。
        - metrics (Optional[List[Metric]]): 指标采集器列表。
        - model (Optional[str]): 指定推理模型名；若 None，将在异步实现中依据可用模型决定。
        - use_tqdm (Optional[bool]): 是否显示进度条。

        返回值：
        - List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
          与输入等长的响应列表，元素为非流式响应或流式响应迭代器。

        示例：
        >>> client = InferClient()
        >>> reqs = [InferRequest(messages=[{"role": "user", "content": "hi"}])]
        >>> _ = client.infer(reqs, request_config=RequestConfig())
        """
        return super().infer(infer_requests, request_config, metrics, model=model, use_tqdm=use_tqdm)  # 调用父类同步推理逻辑并返回

    @staticmethod  # 声明为静态方法，与实例状态无关
    def _prepare_request_data(model: str, infer_request: InferRequest, request_config: RequestConfig) -> Dict[str, Any]:  # 组装发送到服务端的请求体
        if not isinstance(infer_request, dict):  # 若传入的推理请求不是字典（通常为数据类实例）
            infer_request = asdict(infer_request)  # 将数据类实例转换为字典，便于后续合并
        res = asdict(ChatCompletionRequest(model, **infer_request, **asdict(request_config)))  # 构造协议层请求对象并转为字典
        # ignore empty  # 注：忽略与空请求相同的字段，以减少无意义的冗余传输
        empty_request = ChatCompletionRequest('', [])  # 构造一个字段全空的请求对象用于对比
        for k in list(res.keys()):  # 遍历当前请求字典的所有键（用 list 包裹以便迭代中修改）
            if res[k] == getattr(empty_request, k):  # 若该字段与空请求对应字段内容相同
                res.pop(k)  # 从结果中删除该字段，避免发送无用信息
        return res  # 返回最终请求字典

    @staticmethod  # 声明为静态方法，纯解析工具
    def _parse_stream_data(data: bytes) -> Optional[str]:  # 解析服务端 SSE/流式返回的一行数据，提取 JSON 串
        data = data.decode(encoding='utf-8')  # 将字节序列按 UTF-8 解码为字符串
        data = data.strip()  # 去除首尾空白字符，兼容不同服务器实现
        if len(data) == 0:  # 若为空行（可能为心跳或分隔符）
            return  # 返回 None，表示本行可忽略
        assert data.startswith('data:'), f'data: {data}'  # 断言行以 'data:' 开头，符合 SSE 数据格式
        return data[5:].strip()  # 去除前缀并再次去空白，得到纯净的 JSON 文本

    async def infer_async(  # 异步推理入口：单请求，支持流式与非流式两种返回形态
        self,  # 实例自身引用
        infer_request: InferRequest,  # 单个推理请求对象
        request_config: Optional[RequestConfig] = None,  # 可选的推理配置
        *,  # 仅限关键字参数分隔符
        model: Optional[str] = None,  # 指定模型名；若为 None 并且仅有一个可用模型，则自动选择
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:  # 返回非流式响应或异步流式响应迭代器
        """函数功能：
        以协程方式执行单条对话补全请求。若启用流式，将返回异步迭代器以逐步产出增量结果。

        参数：
        - infer_request (InferRequest): 单条推理请求。
        - request_config (Optional[RequestConfig]): 推理配置（如 `stream=True` 启用流式）。
        - model (Optional[str]): 指定模型名；未指定时，若仅有一个可选模型则自动选择，否则抛出异常。

        返回值：
        - Union[ChatCompletionResponse, AsyncIterator[ChatCompletionStreamResponse]]:
          非流式返回 `ChatCompletionResponse`；流式返回 `AsyncIterator[ChatCompletionStreamResponse]`。

        示例：
        >>> client = InferClient()
        >>> req = InferRequest(messages=[{"role": "user", "content": "hello"}])
        >>> # 非流式
        >>> resp = await client.infer_async(req, request_config=RequestConfig(stream=False))
        >>> # 流式
        >>> stream_iter = await client.infer_async(req, request_config=RequestConfig(stream=True))
        >>> async for chunk in stream_iter:
        ...     _ = chunk
        """
        request_config = deepcopy(request_config or RequestConfig())  # 拷贝配置，避免对外部对象产生副作用
        if model is None:  # 若未显式指定模型
            if len(self.models) == 1:  # 且仅有一个可用模型
                model = self.models[0]  # 自动选择该模型，简化使用
            else:  # 若存在多个可选模型
                raise ValueError(f'Please explicitly specify the model. Available models: {self.models}.')  # 明确要求调用方指定模型
        url = f"{self.base_url.rstrip('/')}/chat/completions"  # 拼接聊天补全接口 URL

        request_data = self._prepare_request_data(model, infer_request, request_config)  # 准备 POST 的 JSON 请求体
        if request_config.stream:  # 若启用流式返回

            async def _gen_stream() -> AsyncIterator[ChatCompletionStreamResponse]:  # 定义内部异步生成器，逐条产出增量响应
                async with aiohttp.ClientSession() as session:  # 为本次请求创建独立会话
                    async with session.post(url, json=request_data, **self._get_request_kwargs()) as resp:  # 以 POST 提交请求
                        async for data in resp.content:  # 异步迭代响应内容的每个数据块（SSE 行）
                            data = self._parse_stream_data(data)  # 解析数据块为纯 JSON 字符串或 None
                            if data == '[DONE]':  # 若标记为流式结束
                                break  # 终止生成器
                            if data is not None:  # 过滤空行
                                resp_obj = json.loads(data)  # 将 JSON 字符串解析为字典
                                if resp_obj['object'] == 'error':  # 若服务端返回错误对象
                                    raise HTTPError(resp_obj['message'])  # 抛出 HTTPError 以中断流程
                                yield from_dict(ChatCompletionStreamResponse, resp_obj)  # 反序列化并向上游产出流式响应对象

            return _gen_stream()  # 返回异步生成器给调用方以消费
        else:  # 非流式路径：一次性返回完整响应
            async with aiohttp.ClientSession() as session:  # 创建会话
                async with session.post(url, json=request_data, **self._get_request_kwargs()) as resp:  # 提交请求
                    resp_obj = await resp.json()  # 解析完整 JSON 响应体
                    if resp_obj['object'] == 'error':  # 若为错误对象
                        raise HTTPError(resp_obj['message'])  # 抛出异常提示调用方
                    return from_dict(ChatCompletionResponse, resp_obj)  # 反序列化并返回最终响应对象
