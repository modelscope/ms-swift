"""模块功能概述：
该模块实现通用推理引擎 `InferEngine`，统一组织与编排推理流程：
- 初始化模型与模板上下文；
- 批量/流式/异步任务调度与进度管理；
- 停止词与停止 Token 解析；
- Token 计数、最大生成长度设定与用量统计；
- 指标采集与错误处理；
- 与协议层（聊天补全/流式补全）的数据结构对接。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，标注源代码权属

import asyncio  # 协程与事件循环支持，用于异步执行与调度
import concurrent.futures  # 线程池执行器，用于并发预处理/编码
import os  # 操作系统相关工具，用于获取 CPU 数量等
from queue import Queue  # 线程安全队列，用于线程间通信
from threading import Thread  # 线程类，用于在后台线程中执行任务
from typing import Any, Dict, Iterator, List, Optional, Union  # 类型注解，约束参数与返回值类型，提高可读性

from tqdm import tqdm  # 进度条显示库，用于批量任务的可视化进度

from swift.llm import InferRequest, ProcessorMixin, get_template  # 推理请求类型、处理器混入、模板工厂方法
from swift.llm.template import Template  # 模板类，封装提示词/系统消息编码规则
from swift.llm.utils import get_ckpt_dir  # 工具函数，用于解析模型检查点目录
from swift.plugin import Metric  # 指标采集插件接口，记录用量/性能等
from swift.utils import get_logger  # 日志获取工具
from ..protocol import (ChatCompletionMessageToolCall, ChatCompletionResponse, ChatCompletionStreamResponse,  # 协议层数据结构
                        RequestConfig, UsageInfo)  # 请求配置与用量统计结构体
from .base import BaseInferEngine  # 推理引擎抽象基类，统一对外接口

logger = get_logger()  # 获取当前模块的日志记录器实例


class InferEngine(BaseInferEngine, ProcessorMixin):  # 具体推理引擎实现，继承抽象基类与处理器混入

    """类功能：
    `InferEngine` 负责将模型处理器、模板系统与协议层接口串联，提供批量/流式推理能力。

    - 继承：`BaseInferEngine`（统一接口）、`ProcessorMixin`（提供 tokenizer/processor 等）
    - 职责：初始化模板、调度异步任务、管理进度与指标、处理停止条件和用量统计
    - 场景：对话补全、函数调用（ToolCall）、流式增量解码等
    """

    def _post_init(self, template=None):
        """函数功能：
        初始化后处理：结合处理器初始化引擎上下文，创建/绑定模型信息与默认模板，准备适配器池。

        参数：
        - template (Optional[Template]): 外部传入的模板；若为空则自动构建默认模板。

        返回值：
        - None

        示例：
        >>> engine = InferEngine()
        >>> engine._post_init()
        """
        processor = self.processor  # 读取处理器实例，供模板初始化与编码/解码使用
        self.model_info = processor.model_info  # 模型信息（目录、名称、长度上限、任务类型等）
        self.model_meta = processor.model_meta  # 模型元信息（模板名等）
        self.model_dir = self.model_info.model_dir  # 模型所在目录路径
        self.model_name = self.model_info.model_name  # 模型名称
        self.max_model_len = self.model_info.max_model_len  # 模型支持的最大上下文长度
        self.task_type = self.model_info.task_type  # 任务类型（chat/instruct/等）
        self.config = self.model_info.config  # 其他配置项
        if template is None:  # 若未提供模板，则根据检查点与元信息创建默认模板
            ckpt_dir = get_ckpt_dir(self.model_dir, getattr(self, 'adapters', None))  # 根据 adapters 推断检查点目录
            logger.info('Create the default_template for the infer_engine')  # 记录创建默认模板的日志
            if ckpt_dir:  # 若存在检查点目录
                from swift.llm import BaseArguments  # 延迟导入以避免循环依赖
                args = BaseArguments.from_pretrained(ckpt_dir)  # 从检查点加载参数
                self.default_template = args.get_template(self.processor)  # 基于参数与处理器创建模板
            else:  # 若不存在检查点目录
                self.default_template = get_template(self.model_meta.template, self.processor)  # 按模板名创建模板
        else:  # 外部已提供模板
            self.default_template = template  # 直接赋值为默认模板
            self.default_template.init_processor(self.processor)  # 用当前处理器初始化模板（绑定 tokenizer 等）

        self._adapters_pool = {}  # 适配器缓存池（如 LoRA/Adapter），按需加载与复用

    def _get_stop_words(self, stop_words: List[Union[str, List[int], None]]) -> List[str]:
        """函数功能：
        将传入的停止词（字符串/单词 ID 列表/None）统一转为去重后的字符串列表。

        参数：
        - stop_words (List[Union[str, List[int], None]]): 待处理的停止词集合。

        返回值：
        - List[str]: 去重后的字符串停止词列表。

        示例：
        >>> self._get_stop_words(['\n\n', [13]])
        ['\n\n', '\r']  # 示例仅作形式展示，取决于 tokenizer 实现
        """
        stop: List[str] = []  # 结果列表
        for stop_word in stop_words:  # 遍历输入停止词
            if stop_word is None:  # 忽略空值
                continue  # 进入下一项
            elif isinstance(stop_word, list):  # 若为 token id 列表
                stop_word = self.tokenizer.decode(stop_word)  # 解码为字符串
            assert isinstance(stop_word, str)  # 校验类型
            if stop_word not in stop:  # 去重
                stop.append(stop_word)  # 追加结果
        return stop  # 返回字符串停止词

    def _get_stop_token_ids(self, stop_words: List[Union[str, List[int], None]]) -> List[int]:
        """函数功能：
        获取单 token 停止词的 id 列表，多 token 停止词将被忽略。

        参数：
        - stop_words (List[Union[str, List[int], None]]): 停止词，支持 str/int/list 等形式。

        返回值：
        - List[int]: 单 token 的停止 id 列表（去重）。

        示例：
        >>> self._get_stop_token_ids(['\n', 2, [13]])
        [<id_of_\n>, 2, 13]
        """
        stop_token_ids: List[int] = []  # 结果容器
        for stop_word in stop_words:  # 遍历每个停止词
            if stop_word is None:  # 忽略空值
                continue  # 下一项
            if isinstance(stop_word, str):  # 文本停止词
                stop_word = self.tokenizer.encode(stop_word, add_special_tokens=False)  # 编码为 token id 列表
            if isinstance(stop_word, list):  # 列表形式
                if len(stop_word) != 1:  # 仅接受单 token
                    continue  # 多 token 忽略
                else:
                    stop_token = stop_word[0]  # 取唯一 id
            elif isinstance(stop_word, int):  # 直接为 id
                stop_token = stop_word  # 使用该 id
            assert isinstance(stop_token, int)  # 确保为整数 id
            if stop_token not in stop_token_ids:  # 去重
                stop_token_ids.append(stop_token)  # 收录
        return stop_token_ids  # 返回 id 列表

    def async_iter_to_iter(self, async_iter, prog_bar, metrics) -> Iterator:
        """函数功能：
        将异步流式结果包装为同步迭代器。在后台线程中消费异步迭代器，将产出通过队列同步地提供给调用方，以便在同步环境中按流式消费。

        参数：
        - async_iter: 一个返回 AsyncIterator 的 awaitable 对象。
        - prog_bar: 进度条对象，用于展示任务进度。
        - metrics: 指标采集器列表。

        返回值：
        - Iterator: 同步可迭代对象，逐项产出流式片段。

        示例：
        >>> it = self.async_iter_to_iter(coro, tqdm(total=1), metrics=None)
        >>> for chunk in it: ...
        """
        queue = Queue()  # 用于从后台线程向主线程传递数据

        async def _run_async_iter():  # 后台线程中运行的协程，负责消费异步迭代器
            try:  # 捕获异常，按 strict 策略决定是否抛出
                async for item in await async_iter:  # 等待获得异步迭代器并逐项异步迭代
                    queue.put(item)  # 将每个片段放入队列
            except Exception as e:  # 发生异常
                if getattr(self, 'strict', True):  # 严格模式下直接抛出
                    raise  # 抛出异常
                queue.put(e)  # 非严格模式：将异常对象入队
            else:  # 正常完成
                queue.put(None)  # 放入结束标记

        try:  # 获取或创建事件循环
            loop = asyncio.get_event_loop()  # 获取当前线程事件循环
        except RuntimeError:  # 当前线程无事件循环
            loop = asyncio.new_event_loop()  # 创建新的事件循环
            asyncio.set_event_loop(loop)  # 设为当前线程事件循环
        thread = Thread(target=lambda: loop.run_until_complete(_run_async_iter()))  # 后台线程执行协程直到完成
        thread.start()  # 启动线程
        pre_output = None  # 记录上一条输出，用于在结束时更新指标
        while True:  # 主线程同步消费队列
            output = queue.get()  # 取一个元素（阻塞）
            if output is None or isinstance(output, Exception):  # 结束或异常
                prog_bar.update()  # 更新进度
                self._update_metrics(pre_output, metrics)  # 用最后一次有效输出更新指标
                return  # 结束迭代
            pre_output = output  # 更新上一条输出
            yield output  # 向调用方产出数据

    @staticmethod
    async def batch_run(tasks):
        """函数功能：
        异步并发执行一组协程任务列表并聚合结果。

        参数：
        - tasks: 协程对象列表。

        返回值：
        - List[Any]: 每个任务对应的结果列表。

        示例：
        >>> await InferEngine.batch_run([coro1, coro2])
        """
        return await asyncio.gather(*tasks)  # 并发执行并收集结果

    def _batch_infer_stream(
        self,
        tasks,
        stream: bool = True,
        use_tqdm: bool = True,
        metrics: Optional[List[Metric]] = None
    ) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
        """函数功能：
        对一组推理任务进行批量统一调度，按流式/非流式返回不同结构：
        - 流式：返回同步可迭代器列表；
        - 非流式：阻塞等待并返回结果列表；
        同时更新进度条与指标。

        参数：
        - tasks: 由 `infer_async` 构造的协程任务列表。
        - stream (bool): 是否以流式形式返回。
        - use_tqdm (bool): 是否显示进度条。
        - metrics (Optional[List[Metric]]): 指标采集器列表。

        返回值：
        - List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
          与任务等长的结果（或迭代器）列表。

        示例：
        >>> self._batch_infer_stream(tasks, stream=True, use_tqdm=True)
        """
        prog_bar = tqdm(total=len(tasks), dynamic_ncols=True, disable=not use_tqdm)  # 创建进度条
        if stream:  # 流式：包装为同步可迭代器
            return [self.async_iter_to_iter(task, prog_bar, metrics) for task in tasks]  # 返回迭代器列表
        else:  # 非流式：阻塞并收集结果

            async def _new_run(task):  # 单任务包装：异常转结果、更新进度与指标
                try:
                    res = await task  # 等待结果
                except Exception as e:  # 出错
                    if getattr(self, 'strict', True):  # 严格模式抛出
                        raise
                    res = e  # 非严格模式返回异常对象
                prog_bar.update()  # 更新进度
                self._update_metrics(res, metrics)  # 更新指标
                return res  # 返回该任务结果

            new_tasks = [_new_run(task) for task in tasks]  # 构造包裹后的任务列表
            try:
                loop = asyncio.get_event_loop()  # 获取事件循环
            except RuntimeError:
                loop = asyncio.new_event_loop()  # 新建事件循环
                asyncio.set_event_loop(loop)  # 绑定到当前线程
            return loop.run_until_complete(self.batch_run(new_tasks))  # 同步运行直到完成

    @staticmethod
    def _get_usage_info(num_prompt_tokens: int, num_generated_tokens: int) -> UsageInfo:
        """函数功能：
        根据提示词与生成 token 数，构造一次请求的用量统计对象。

        参数：
        - num_prompt_tokens (int): 提示词 token 数。
        - num_generated_tokens (int): 生成 token 数。

        返回值：
        - UsageInfo: 用量统计信息。
        """
        return UsageInfo(  # 返回用量统计数据类
            prompt_tokens=num_prompt_tokens,  # 提示词数量
            completion_tokens=num_generated_tokens,  # 生成数量
            total_tokens=num_prompt_tokens + num_generated_tokens,  # 总计数量
        )

    @staticmethod
    def _update_usage_info(origin_use_info: UsageInfo, num_generated_tokens: int) -> UsageInfo:
        """函数功能：
        在已有用量统计的基础上，累加新增生成 token 数并返回新对象。

        参数：
        - origin_use_info (UsageInfo): 原始用量统计。
        - num_generated_tokens (int): 新增生成数量。

        返回值：
        - UsageInfo: 更新后的用量统计。
        """
        return UsageInfo(  # 返回新的用量统计
            prompt_tokens=origin_use_info.prompt_tokens,  # 保留原值
            completion_tokens=origin_use_info.completion_tokens + num_generated_tokens,  # 生成量累加
            total_tokens=origin_use_info.total_tokens + num_generated_tokens,  # 总量累加
        )

    @staticmethod
    def _update_metrics(result, metrics: Optional[List[Metric]] = None):
        """函数功能：
        将结果上报给各指标采集器：遍历结果（或结果列表），对每项有效响应调用各采集器的 `update` 方法。

        参数：
        - result: 单个响应或响应列表。
        - metrics (Optional[List[Metric]]): 采集器列表；为空时不做任何事。

        返回值：
        - Any: 原样返回入参 `result`。
        """
        if metrics is None:  # 未提供采集器
            return result  # 直接返回
        result_origin = result  # 保存原始对象
        if not isinstance(result, (list, tuple)):  # 统一转为列表以便遍历
            result = [result]
        for response in result:  # 遍历每项响应
            if response is None or isinstance(response, Exception):  # 跳过空/异常
                continue
            for metric in metrics:  # 遍历采集器
                metric.update(response)  # 上报响应
        return result_origin  # 返回原值

    def infer(self,
              infer_requests: List[InferRequest],
              request_config: Optional[RequestConfig] = None,
              metrics: Optional[List[Metric]] = None,
              *,
              use_tqdm: Optional[bool] = None,
              **kwargs) -> List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
        """函数功能：
        使用同步方式执行批量推理：为每个请求创建异步任务，按配置以流式/非流式方式返回结果。

        参数：
        - infer_requests (List[InferRequest]): 请求列表。
        - request_config (Optional[RequestConfig]): 推理配置；缺省时创建默认配置。
        - metrics (Optional[List[Metric]]): 指标采集器列表。
        - use_tqdm (Optional[bool]): 是否显示进度条；None 时自动推断。
        - **kwargs: 透传给 `infer_async` 的其他参数（例如模型选择等）。

        返回值：
        - List[Union[ChatCompletionResponse, Iterator[ChatCompletionStreamResponse]]]:
          与输入等长的响应或流式迭代器列表。

        示例：
        >>> reqs = [InferRequest(messages=[{"role": "user", "content": "hi"}])]
        >>> self.infer(reqs, RequestConfig(stream=False))
        """
        if request_config is None:  # 若未提供配置
            request_config = RequestConfig()  # 使用默认配置
        tasks = [self.infer_async(infer_request, request_config, **kwargs) for infer_request in infer_requests]  # 构造协程任务
        if use_tqdm is None:  # 未指定是否显示进度条
            use_tqdm = not request_config.stream and len(infer_requests) > 1  # 非流式且批量>1 则默认展示
        return self._batch_infer_stream(tasks, request_config.stream, use_tqdm, metrics)  # 统一调度并返回

    @staticmethod
    def _get_toolcall(response: str, template: Template) -> Optional[List[ChatCompletionMessageToolCall]]:
        """函数功能：
        解析函数调用：使用模板的代理组件从模型输出文本中解析工具/函数调用。

        参数：
        - response (str): 模型输出文本。
        - template (Template): 当前使用的模板对象。

        返回值：
        - Optional[List[ChatCompletionMessageToolCall]]: 解析到的函数调用列表，或 None。
        """
        try:
            functions = template.agent_template.get_toolcall(response)  # 使用代理模板解析函数调用
        except Exception:  # 解析失败
            functions = None  # 返回空
        if functions:  # 成功解析
            return [ChatCompletionMessageToolCall(function=function) for function in functions]  # 包装为协议对象列表
    
    @staticmethod
    def _get_num_tokens(inputs: Dict[str, Any]) -> int:
        """函数功能：
        根据 `inputs` 中的 `input_ids` 或 `inputs_embeds` 推断序列长度（输入 token 数）。

        参数：
        - inputs (Dict[str, Any]): 编码后的输入字典。

        返回值：
        - int: 输入序列长度。
        """
        if 'input_ids' in inputs:  # 1d or 2d  # 存在输入 id
            input_ids = inputs['input_ids']  # 取出 input_ids
            if isinstance(input_ids, list):  # 列表形式
                return len(input_ids)  # 列表长度即 token 数
            else:  # 数组/张量形式
                return input_ids.shape[-1]  # 最后一维即序列长度
        elif 'inputs_embeds' in inputs:  # 2d or 3d  # 嵌入形式
            return inputs['inputs_embeds'].shape[-2]  # 倒数第二维为序列维
        raise ValueError(f'Unable to retrieve input_ids and inputs_embeds. inputs: {inputs}')  # 无法推断则报错

    def set_default_max_tokens(self, request_config: RequestConfig, inputs: Dict[str, Any]) -> None:
        """函数功能：
        自动设置/修正 max_tokens：依据模型最大长度与当前输入长度，为请求设置合理的 `max_tokens` 上限，或修正过大的上限。

        参数：
        - request_config (RequestConfig): 推理配置对象，将被就地修改。
        - inputs (Dict[str, Any]): 模型编码后的输入字典。

        返回值：
        - None
        """
        max_model_len = self.max_model_len  # 模型支持的最大上下文长度
        assert isinstance(inputs, dict)  # 类型校验
        # The num_tokens takes the maximum value from inputs_list.  # 说明：num_tokens 即输入长度
        num_tokens = self._get_num_tokens(inputs)  # 计算当前输入长度
        max_tokens = request_config.max_tokens  # 读取已设置的生成上限
        if max_model_len is None:  # 未提供最大长度
            max_model_len = 8192  # 使用默认值 8192
            logger.warning(  # 告警提示
                'The current model is unable to retrieve `max_model_len`. It is set to the default value of 8192.')
        max_max_tokens = max_model_len - num_tokens  # 可用生成上限 = 模型上限 - 输入长度
        if max_tokens is None:  # 若未指定
            request_config.max_tokens = max_max_tokens  # 直接设置为可用上限
        elif max_max_tokens < request_config.max_tokens:  # 若指定的超出可用范围
            logger.warning(f'max_model_len({max_model_len}) - num_tokens({num_tokens}) < max_tokens({max_tokens}). '
                           f'Setting max_tokens: {max_model_len - num_tokens}')  # 发出告警
            request_config.max_tokens = max_max_tokens  # 下调为可用上限

    def _get_logprobs(self,
                      logprobs_list: Optional[List[Dict[int, float]]],
                      token_ids: List[int],
                      top_logprobs: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """函数功能：构造对数概率明细（可选 Top-K）
        将每步生成的 token 对数概率映射与实际生成序列对齐，构造可读明细；
        可选地提供前 K 高概率候选。

        参数：
        - logprobs_list (Optional[List[Dict[int, float]]]): 每步的 id->logprob 映射列表。
        - token_ids (List[int]): 实际生成的 token id 序列。
        - top_logprobs (Optional[int]): 返回每步的 Top-K 候选数量；None 表示不返回候选。

        返回值：
        - Optional[Dict[str, Any]]: 包含 `content` 的结构或 None。
        """
        if logprobs_list is None or len(token_ids) == 0:  # 无概率或未生成
            return None  # 返回空
        if len(token_ids) > 0:  # 存在生成
            logprobs_list = logprobs_list[-len(token_ids):]  # 截取对应长度的后缀
        res = []  # 结果列表
        for logprobs, token_id in zip(logprobs_list, token_ids):  # 对齐遍历
            token = self.tokenizer.decode(token_id)  # 将 id 解码为文本
            _res = {'token': token, 'logprob': logprobs[token_id], 'bytes': list(token.encode('utf8'))}  # 基础项
            if top_logprobs is not None:  # 需要 Top-K
                # NOTE:
                # sorted(logprobs) 会得到字典 logprobs 的键列表（dict 迭代返回键）。
                # key=lambda k: -logprobs[k] 指定排序依据是 -logprobs[k]（注意是负号），因此按照 logprobs[k] 的降序排序（因为把值取负再按升序等价于按原值降序）。
                logprobs = {k: logprobs[k] for k in sorted(logprobs, key=lambda k: -logprobs[k])[:top_logprobs]}  # 选前 K
                res_top_logprobs = []  # 候选列表
                for k, logprob in logprobs.items():  # 遍历候选
                    if logprob == float('-inf'):  # 过滤无效概率
                        continue
                    token = self.tokenizer.decode(k)  # 解码候选 id
                    res_top_logprobs.append({'token': token, 'logprob': logprob, 'bytes': list(token.encode('utf8'))})  # 记录
                _res['top_logprobs'] = res_top_logprobs  # 写入候选
            res.append(_res)  # 写入当前步
        return {'content': res}  # 返回结构

    @staticmethod
    def _get_finish_reason(max_tokens: int, completion_tokens: int, is_finished: bool):
        """函数功能：推断停止原因
        根据完成标记与生成数量，判断停止原因（长度/停止词/未完成）。

        参数：
        - max_tokens (int): 生成上限。
        - completion_tokens (int): 已生成数量。
        - is_finished (bool): 逻辑层是否判定完成。

        返回值：
        - Optional[str]: 停止原因，取值 {'length','stop',None}。
        """
        if is_finished:  # 逻辑已完成
            if completion_tokens >= max_tokens:  # 触及上限
                finish_reason = 'length'  # 因长度停止
            else:
                finish_reason = 'stop'  # 因停止条件停止
        else:
            finish_reason = None  # 尚未完成
        return finish_reason  # 返回判定

    @staticmethod
    def thread_run(target, args=(), kwargs=None):
        """函数功能：
        启动一个后台线程执行 `target(*args, **kwargs)`，并以队列返回结果或异常。

        参数：
        - target: 可调用对象。
        - args (tuple): 位置参数。
        - kwargs (Optional[dict]): 关键字参数。

        返回值：
        - Any: 执行结果；若发生异常将在当前线程中重新抛出。
        """
        kwargs = kwargs or {}  # 规范化关键字参数

        def func(target, queue, args, kwargs):  # 在线程中执行的包装函数
            try:
                queue.put(target(*args, **kwargs))  # 执行并放入结果
            except Exception as e:  # 捕获异常
                queue.put(e)  # 放入异常对象

        queue = Queue()  # 创建队列传递结果
        thread = Thread(target=func, args=(target, queue, args, kwargs))  # 构造线程
        thread.start()  # 启动线程
        thread.join()  # 等待完成
        result = queue.get()  # 取回结果或异常
        if isinstance(result, Exception):  # 若为异常
            raise result  # 在当前线程抛出
        return result  # 返回正常结果

    @staticmethod
    def safe_asyncio_run(coro):
        """函数功能：在同步环境中安全执行协程
        使用新线程调用 `asyncio.run(coro)`，避免与现有事件循环冲突。

        参数：
        - coro: 协程对象。

        返回值：
        - Any: 协程返回值。
        """

        def asyncio_run(core):  # 实际执行函数
            return asyncio.run(core)  # 运行协程并返回结果

        return InferEngine.thread_run(asyncio_run, args=(coro, ))  # 在线程中执行以规避事件循环冲突

    @staticmethod
    def _batch_encode(infer_requests: List[InferRequest], template: Template, strict: bool):
        """函数功能：并发编码请求
        使用线程池并发执行模板编码，返回成功结果与错误列表。

        参数：
        - infer_requests (List[InferRequest]): 推理请求列表。
        - template (Template): 编码所用模板。
        - strict (bool): 严格模式；为 True 时遇错即抛出，否则记录错误并继续。

        返回值：
        - Tuple[List[Any], List[Tuple[int, Exception]]]: (成功结果列表, 错误列表)。
        """
        max_workers = max(min(32, os.cpu_count(), len(infer_requests)), 1)  # 线程数：不超过 32 与请求数，至少 1
        error_list = []  # 记录 (索引, 异常)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  # 构造线程池
            futures = [  # 提交编码任务
                executor.submit(template.encode, infer_request, return_template_inputs=True)
                for infer_request in infer_requests
            ]
            concurrent.futures.wait(futures)  # 等待全部完成
            batched_inputs = []  # 成功结果
            for i, future in enumerate(futures):  # 遍历结果
                try:
                    batched_inputs.append(future.result())  # 读取结果
                except Exception as e:  # 捕获异常
                    if strict:  # 严格模式：直接抛出
                        raise
                    error_list.append((i, e))  # 记录错误位置与异常
                    continue  # 继续处理
        return batched_inputs, error_list  # 返回成功结果与错误列表

    @staticmethod
    def _add_error_list(outputs, error_list):
        """函数功能：将错误按原始索引插回输出
        按 (索引, 异常) 将错误对象插回到对应输出位置，保证与输入顺序对齐。

        参数：
        - outputs: 成功结果列表（可被原地修改）。
        - error_list: [(index, Exception)] 错误信息列表。

        返回值：
        - List[Any]: 插入异常后的输出列表。
        """
        for i, error in error_list:  # 遍历错误项
            outputs.insert(i, error)  # 在对应索引插入异常占位
        return outputs  # 返回合并列表
