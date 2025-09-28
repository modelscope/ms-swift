"""
模块功能概述：
    本脚本实现 `swift rollout` 的服务与执行逻辑，专用于 GRPO 强化学习训练阶段的外部采样。
    基于 vLLM 推理后端扩展了权重同步与多进程数据并行（DP），并通过 FastAPI 暴露 HTTP 接口。

核心能力：
    - 多进程 worker（同步/异步）：子进程通过管道接收父进程的 RPC 指令，执行推理与权重同步相关方法。
    - FastAPI 服务：`SwiftRolloutDeploy` 提供健康检查、初始化通信器、更新命名参数、重置 prefix cache、
      查询引擎类型、批量推理等端点。
    - 便捷上下文管理器：`run_rollout` 可在子进程中临时拉起服务并在退出时安全终止，便于测试/集成。
    - 兼容性补丁：兼容 TRL 对 `update_named_param` 的 dtype 注解变更（torch.dtype → str）。

使用限制：
    - 仅用于 GRPO 训练期间的采样（rollout）；通用推理/部署请使用 `swift infer` 或 `swift deploy`。

示例：
    >>> from swift.llm.infer.rollout import run_rollout
    >>> from swift.llm import RolloutArguments
    >>> args = RolloutArguments(model='your-model', port=8000)
    >>> with run_rollout(args, return_url=True) as base_url:
    ...     print(base_url)  # http://127.0.0.1:8000/v1
"""

# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明：阿里巴巴及其附属公司保留所有权利
# Code partially sourced from Hugging Face TRL  # 说明：部分代码来源于 Hugging Face TRL 项目

import asyncio  # 导入 asyncio：用于事件循环与异步协程
import inspect  # 导入 inspect：用于函数签名检查与字段过滤
import multiprocessing  # 导入 multiprocessing：用于多进程管理
import os  # 导入 os：用于环境变量设置与路径操作
import time  # 导入 time：用于简单的轮询等待与计时
from contextlib import asynccontextmanager, contextmanager  # 导入上下文管理器装饰器（同步与异步）
from dataclasses import asdict  # 导入 asdict：将 dataclass 转为字典
from functools import wraps  # 导入 wraps：装饰器工具，保留函数元信息
from itertools import chain  # 导入 chain：将嵌套列表拍平
from multiprocessing import Pipe, Process  # 导入 Pipe/Process：进程间通信与子进程创建
from multiprocessing.connection import Connection  # 导入 Connection：Pipe 连接的类型注解
from typing import Dict, List, Optional, Union, get_type_hints  # 导入类型注解工具

import torch  # 导入 torch：用于 dtype 类型与兼容性处理
import uvicorn  # 导入 uvicorn：ASGI 服务器，用于启动 FastAPI 应用
from aiohttp import ClientConnectorError  # 导入 ClientConnectorError：用于捕获客户端连接异常
from fastapi import FastAPI  # 导入 FastAPI：Web 服务框架
from trl.scripts.vllm_serve import WeightSyncWorkerExtension  # 导入 TRL 扩展：vLLM 权重同步扩展点

from swift.llm import RolloutArguments, SwiftPipeline  # 导入 RolloutArguments/SwiftPipeline：参数/管道基类
from swift.llm.template.template_inputs import RolloutInferRequest  # 导入 RolloutInferRequest：rollout 推理请求体
from swift.utils import get_logger  # 导入 get_logger：统一日志器
from .infer_engine import GRPOVllmEngine, InferClient  # 导入 GRPOVllmEngine/InferClient：引擎与客户端
from .protocol import InitCommunicatorRequest, RequestConfig, UpdateWeightsRequest  # 导入协议请求体

try:  # 可选依赖导入尝试：这些模块仅在 vLLM/TRL 环境下存在
    from vllm.utils import get_open_port  # 选择可用端口：用于 DP 主端口分配
    from trl.scripts.vllm_serve import chunk_list  # 将列表平均切分：用于请求分片

except ImportError:  # 当可选依赖缺失时捕获导入错误
    pass  # 先忽略，实际使用到相关功能时会报错提示缺依赖

"""
英文说明（保留）：
This module defines the execution logic for `swift rollout`.
It adds weight synchronization logic based on `vLLMEngine`.

Usage:
    swift rollout \
        --model xxx \
        --vllm_tensor_parallel_size xxx \
        --vllm_data_parallel_size xxx \
        --vllm_use_async_engine true/false \
        --use_gym_env true/false \
        --other_vllm_arguments

Note:
- Rollout is intended solely for GRPO training sampling.
- For inference or deployment, please use the `swift infer` or `swift deploy` commands.
"""

logger = get_logger()  # 模块级日志器：用于记录服务运行状态、异常与调试信息


def safe_set_start_method():
    """函数功能：安全设置多进程启动方式为 'spawn'（若当前未设置）。
    示例：
        >>> safe_set_start_method()  # 若未设置，则设置为 'spawn'；若已设置则不做处理
    """
    if multiprocessing.get_start_method(allow_none=True) is None:  # 判断：若当前未设置多进程启动方式
        multiprocessing.set_start_method('spawn')  # 设置启动方式为 'spawn'，确保在 Windows/macOS 的兼容性


def llm_worker(args: RolloutArguments, data_parallel_rank: int, master_port: int, connection: Connection) -> None:
    """函数功能：同步 worker 进程入口；配置 vLLM DP 环境，构造引擎，循环接收并执行父进程 RPC 指令。

    入参：
        args (RolloutArguments): rollout 启动参数对象。
        data_parallel_rank (int): 当前数据并行 worker 的 rank（从 0 开始）。
        master_port (int): DP 主端口，用于进程间通信。
        connection (Connection): 与父进程通信的管道连接。

    返回值：
        None

    示例：
        >>> from multiprocessing import Pipe, Process
        >>> parent_conn, child_conn = Pipe()
        >>> p = Process(target=llm_worker, args=(args, 0, 29500, child_conn))
        >>> p.start(); parent_conn.recv()  # {'status': 'ready'}
        >>> parent_conn.send({'type': 'shutdown'})
    """
    args._import_external_plugins()  # 导入外部插件（如需），保证引擎实例化前已注册扩展
    os.environ['VLLM_DP_RANK'] = str(data_parallel_rank)  # 设置 vLLM DP rank：标识当前数据并行进程序号
    os.environ['VLLM_DP_RANK_LOCAL'] = str(data_parallel_rank)  # 设置本地 DP rank：与 DP_RANK 一致
    os.environ['VLLM_DP_SIZE'] = str(args.vllm_data_parallel_size)  # 设置 DP 总大小：用于组网
    os.environ['VLLM_DP_MASTER_PORT'] = str(master_port)  # 设置 DP 主端口：用于通信器初始化
    engine = SwiftRolloutDeploy.get_infer_engine(args, template=args.get_template(None))  # 构造 rollout 推理引擎
    connection.send({'status': 'ready'})  # 通过管道向父进程发送 ready 信号，表明可接受命令

    while True:  # 进入 worker 主循环：持续接收并处理父进程指令
        try:  # 尝试接收父进程命令
            command = connection.recv()  # 阻塞等待父进程通过管道发送的命令字典
        except KeyboardInterrupt:  # 捕获中断信号，用于优雅退出
            engine.engine.collective_rpc(method='close_communicator')  # 退出前关闭通信器，清理资源
            break  # 跳出循环，结束进程

        if command['type'] in ['call', 'fire_and_forget']:  # 如果是调用类命令（需返回/无需返回）
            method_name = command['method']  # 读取目标方法名
            args, kwargs = command.get('args', ()), command.get('kwargs', {})  # 读取位置参数与关键字参数
            method = getattr(engine, method_name, None) or getattr(engine.engine, method_name, None)  # 从封装或底层引擎解析方法
            result = method(*args, **kwargs)  # 调用目标方法并获取结果
            if command['type'] == 'call':  # 若需要返回结果
                connection.send(result)  # 将结果通过管道回传给父进程
        elif command['type'] == 'shutdown':  # 若接收到关闭指令
            break  # 结束主循环并退出进程


async def async_llm_worker(args: RolloutArguments, data_parallel_rank: int, master_port: int,
                           connection: Connection) -> None:
    """函数功能：异步 worker 进程入口；以协程方式接收并执行父进程命令（适配 vLLM 异步引擎）。

    入参：
        args (RolloutArguments): rollout 启动参数对象。
        data_parallel_rank (int): 数据并行 rank（签名保持一致，实际未直接使用）。
        master_port (int): DP 主端口（签名保持一致，实际未直接使用）。
        connection (Connection): 与父进程通信的管道连接。

    返回值：
        None

    示例：
        >>> # 一般由 llm_worker_entry 包装后作为子进程入口被调用
        >>> # await async_llm_worker(args, 0, 29500, conn)
    """
    engine = SwiftRolloutDeploy.get_infer_engine(args)  # 构造异步推理引擎（封装 vLLM Async 引擎）
    connection.send({'status': 'ready'})  # 通过管道向父进程发送 ready 信号

    loop = asyncio.get_running_loop()  # 获取当前事件循环
    while True:  # 异步主循环：持续处理父进程命令
        try:  # 通过线程池执行阻塞的 connection.recv，避免阻塞事件循环
            command = await loop.run_in_executor(None, connection.recv)  # 在线程池中阻塞等待命令
        except KeyboardInterrupt:  # 捕获中断信号
            await engine.engine.collective_rpc(method='close_communicator')  # 异步关闭通信器
            break  # 退出主循环

        if command['type'] in ['call', 'fire_and_forget']:  # 处理调用型命令
            import traceback  # 延迟导入 traceback：仅用于异常打印
            method_name = command['method']  # 目标方法名
            args, kwargs = command.get('args', ()), command.get('kwargs', {})  # 参数与关键字参数
            method = getattr(engine, method_name, None) or getattr(engine.engine, method_name, None)  # 方法解析
            try:  # 执行异步方法并捕获异常
                result = await method(*args, **kwargs)  # 等待方法执行结果
            except Exception:  # 出现异常时记录错误日志
                logger.error(f'Method execution failed: {method_name}\n{traceback.format_exc()}')  # 打印堆栈
                result = None  # 返回空结果，避免父进程阻塞

            if command['type'] == 'call':  # 若需要返回
                connection.send(result)  # 回传执行结果
        elif command['type'] == 'shutdown':  # 收到关闭指令
            break  # 退出循环


def llm_worker_entry(*args, **kwargs):
    """函数功能：异步 worker 的进程入口；导入插件后运行协程版 worker。

    入参：
        *args: 位置参数（第一个参数应为 RolloutArguments）。
        **kwargs: 关键字参数（透传给 `async_llm_worker`）。

    返回值：
        None

    示例：
        >>> from multiprocessing import Process
        >>> p = Process(target=llm_worker_entry, args=(args, 0, 29500, conn))
        >>> p.start()
    """
    rollout_args: RolloutArguments = args[0]  # 从参数中取出 RolloutArguments 对象
    rollout_args._import_external_plugins()  # 导入外部插件，保证引擎初始化前完成
    asyncio.run(async_llm_worker(*args, **kwargs))  # 运行异步 worker 协程入口


class SwiftRolloutDeploy(SwiftPipeline):
    """类功能：GRPO 采样服务管道；注册 HTTP 路由、拉起并管理 DP workers，处理推理与权重同步。

    关键职责：
        - 封装 FastAPI 服务的生命周期，负责创建/管理数据并行子进程（worker）。
        - 暴露 rollout 相关的 HTTP 端点，并协调与 vLLM 引擎的交互（推理与权重同步）。

    重要属性：
        - args (RolloutArguments): 解析后的启动参数对象。
        - use_gym_env (bool): 是否启用 Gym 环境以支持多轮/环境交互式采样。
        - use_async_engine (bool): 是否使用 vLLM 异步引擎（AsyncLLMEngine）。
        - num_connections (int): worker 数量/连接数（异步=1；同步=DP 大小）。
        - master_port (int): 数据并行通信主端口。
        - connections (List[Connection]): 父进程端与各子进程通信的管道集合。
        - processes (List[Process]): 子进程句柄集合。
    """
    args_class = RolloutArguments  # 指定参数类：用于基类解析 CLI/对象参数
    args: args_class  # 类型标注：便于 IDE/类型检查识别实例属性

    def _register_rl_rollout_app(self):
        """函数功能：注册 rollout 相关的 HTTP 路由到 FastAPI 应用。

        示例：
            >>> deploy = SwiftRolloutDeploy(args)
            >>> # 初始化完成后，即可访问 /health/ 等端点
        """
        self.app.get('/health/')(self.health)  # 注册 GET /health/：健康检查
        self.app.get('/get_world_size/')(self.get_world_size)  # 注册 GET /get_world_size/：查询 world_size
        self.app.post('/init_communicator/')(self.init_communicator)  # 注册 POST /init_communicator/：初始化通信器
        self.app.post('/update_named_param/')(self.update_named_param)  # 注册 POST /update_named_param/：更新命名参数
        self.app.post('/reset_prefix_cache/')(self.reset_prefix_cache)  # 注册 POST /reset_prefix_cache/：重置前缀缓存
        self.app.post('/close_communicator/')(self.close_communicator)  # 注册 POST /close_communicator/：关闭通信器
        self.app.post('/infer/', response_model=None)(self.infer)  # 注册 POST /infer/：批量推理
        self.app.post('/get_engine_type/')(self.get_engine_type)  # 注册 POST /get_engine_type/：查询引擎类型

    def __init__(self, args: Optional[Union[List[str], RolloutArguments]] = None):
        """函数功能：初始化服务；解析参数、创建应用、启动 worker 进程。

        入参：
            args (Optional[Union[List[str], RolloutArguments]]): 命令行样式参数或参数对象。

        返回值：
            None

        示例：
            >>> deploy = SwiftRolloutDeploy(["--model", "your-model"])  # 解析 CLI 参数
            >>> deploy.run()  # 启动服务
        """
        super().__init__(args)  # 调用基类初始化：完成参数解析与基本属性设置
        self.use_gym_env = self.args.use_gym_env  # 标记：是否启用 Gym 环境
        self.use_async_engine = self.args.vllm_use_async_engine  # 标记：是否使用异步引擎
        self.num_connections = 1 if self.use_async_engine else self.args.vllm_data_parallel_size  # 计算连接/worker 数量
        safe_set_start_method()  # 确保多进程启动方式设置为 spawn，兼容各平台
        self.app = FastAPI(lifespan=self.lifespan)  # 创建 FastAPI 应用并绑定生命周期管理器
        self._register_rl_rollout_app()  # 注册 HTTP 路由
        self.master_port = get_open_port()  # 选择 DP 主端口
        self.connections = []  # 初始化父进程端连接列表，用于与子进程通信
        self.processes = []  # 初始化子进程句柄列表
        self._start_data_parallel_workers()  # 根据配置启动 DP worker 子进程

    def _start_data_parallel_workers(self):
        """函数功能：根据配置启动数据并行 worker 进程并建立通信管道。

        示例：
            >>> self._start_data_parallel_workers()  # 内部调用，一般无需手动触发
        """
        for data_parallel_rank in range(self.num_connections):  # 遍历 DP rank：为每个 rank 启动一个子进程
            parent_conn, child_conn = Pipe()  # 创建父/子进程通信管道
            worker_func = llm_worker_entry if self.use_async_engine else llm_worker  # 根据引擎类型选择入口函数
            process = Process(target=worker_func, args=(self.args, data_parallel_rank, self.master_port, child_conn))  # 构造子进程
            process.start()  # 启动子进程
            self.connections.append(parent_conn)  # 保存父端连接以便后续下发命令与接收结果
            self.processes.append(process)  # 保存子进程句柄用于生命周期管理

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """函数功能：应用生命周期管理；等待 worker 就绪，服务结束时清理子进程。

        入参：
            app (FastAPI): 当前 FastAPI 应用实例（由框架注入）。

        返回值：
            异步上下文管理器：在 `yield` 前进行启动同步，`yield` 后进行资源清理。

        示例：
            >>> # 由 FastAPI/uvicorn 自动调用，无需手动使用
        """
        ready_connections = set()  # 使用集合记录已发送 ready 信号的连接

        while len(ready_connections) < self.num_connections:  # 阻塞等待，直到所有 worker 就绪
            for connection in self.connections:  # 轮询所有父端连接
                msg = connection.recv()  # 阻塞接收子进程消息
                if isinstance(msg, dict) and msg.get('status') == 'ready':  # 如果收到就绪信号
                    ready_connections.add(connection)  # 将该连接标记为就绪

        yield  # 应用开始对外提供服务

        for process in self.processes:  # 服务结束：等待所有子进程退出
            process.join(timeout=10)  # 最长等待 10 秒
            if process.is_alive():  # 若仍未退出
                logger.warning(f'Process {process} is still alive after 10 seconds, attempting to terminate...')  # 记录警告
                process.terminate()  # 强制终止子进程
                process.join()  # 再次等待，确保子进程结束

    @staticmethod
    def get_infer_engine(args: RolloutArguments, template=None, **kwargs):
        """函数功能：根据 rollout 参数构造并返回 GRPO vLLM 引擎实例。

        入参：
            args (RolloutArguments): 推理与引擎的核心配置。
            template (Any, optional): 模板对象/名称，用于对话模板化。
            **kwargs: 额外覆盖参数（如 infer_backend、engine_kwargs 等）。

        返回值：
            GRPOVllmEngine: 已配置完成的 vLLM 引擎封装实例。

        示例：
            >>> engine = SwiftRolloutDeploy.get_infer_engine(args, template=args.get_template(None))
        """
        kwargs.update({  # 汇总并补充引擎初始化所需的关键参数
            'model_id_or_path': args.model,  # 模型 ID 或本地路径
            'model_type': args.model_type,  # 模型类型：影响模板/分支逻辑
            'revision': args.model_revision,  # 模型版本/修订号
            'torch_dtype': args.torch_dtype,  # Torch 精度类型（如 float16/float32）
            'template': template,  # 对话模板
            'use_async_engine': args.vllm_use_async_engine,  # 是否使用异步引擎
            'multi_turn_scheduler': args.multi_turn_scheduler,  # 多轮调度器名称
            'max_turns': args.max_turns,  # 多轮对话最大轮数
            'use_gym_env': args.use_gym_env,  # 是否启用 Gym 环境
            'gym_env': args.gym_env,  # Gym 环境名称/配置
            'context_manager': args.context_manager,  # 上下文管理器
        })
        infer_backend = kwargs.pop('infer_backend', None) or args.infer_backend  # 读取/覆盖推理后端
        if infer_backend != 'vllm':  # rollout 仅支持 vLLM 后端
            infer_backend = 'vllm'  # 强制设置为 vLLM
            logger.info('Currently, rollout only supports the vLLM backend. Set vLLM backend')  # 记录提示信息
        kwargs.update(args.get_vllm_engine_kwargs())  # 合并 vLLM 特定的引擎参数（如并行相关）
        engine_kwargs = kwargs.get('engine_kwargs', {})  # 取出/初始化 engine_kwargs 字典
        engine_kwargs.update({'worker_extension_cls': 'trl.scripts.vllm_serve.WeightSyncWorkerExtension'})  # 注入权重同步扩展
        if args.vllm_use_async_engine and args.vllm_data_parallel_size > 1:  # 异步引擎下可设置 DP 大小
            engine_kwargs['data_parallel_size'] = args.vllm_data_parallel_size  # 设置数据并行大小
        kwargs['engine_kwargs'] = engine_kwargs  # 写回 engine_kwargs 到 kwargs

        return GRPOVllmEngine(**kwargs)  # 构造并返回 vLLM 引擎封装实例

    async def health(self):
        """函数功能：健康检查端点，返回服务运行状态。

        入参：
            无

        返回值：
            dict: 形如 {"status": "ok"} 的健康状态。

        示例：
            >>> # HTTP GET /health/ -> {"status": "ok"}
        """
        return {'status': 'ok'}  # 固定返回健康状态

    async def get_world_size(self):
        """函数功能：返回 world_size 字典（tp_size * dp_size）。

        入参：
            无

        返回值：
            dict: 形如 {"world_size": int} 的字典。
        """
        return {'world_size': self.args.vllm_tensor_parallel_size * self.args.vllm_data_parallel_size}  # 计算 world_size

    async def init_communicator(self, request: InitCommunicatorRequest):
        """函数功能：初始化通信器，用于客户端与多 worker 进行权重同步。

        入参：
            request (InitCommunicatorRequest): 包含 host/port/world_size 与可选 client_device_uuid 的请求体。

        返回值：
            dict: 接收初始化请求的确认消息。
        """
        world_size = self.args.vllm_tensor_parallel_size * self.args.vllm_data_parallel_size + 1  # +1 表示包含客户端

        # The function init_communicator is called this way: init_communicator(host, port, world_size)
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc(method="init_communicator", args=(host, port, world_size))
        kwargs = {  # 组装 collective_rpc 的方法与参数
            'method':
            'init_communicator',  # 调用的集体方法名
            'args': (request.host, request.port, world_size, *(() if request.client_device_uuid is None else
                                                               (request.client_device_uuid, )))  # 参数元组
        }
        for connection in self.connections:  # 广播到所有 worker 进程
            connection.send({'type': 'fire_and_forget', 'method': 'collective_rpc', 'kwargs': kwargs})  # 下发异步指令

        return {'message': 'Request received, initializing communicator'}  # 返回确认消息

    async def update_named_param(self, request: UpdateWeightsRequest):
        """函数功能：更新命名权重张量的元信息（客户端随后应广播具体权重数据）。

        入参：
            request (UpdateWeightsRequest): 包含 name/dtype/shape 的权重描述。

        返回值：
            dict: 接收更新请求的确认消息。
        """
        # The function update_named_param is called this way: update_named_param("name", torch.float32, (10, 10))
        # So with collective_rpc we need to call it this way:
        # llm.collective_rpc("update_named_param", args=("name", torch.float32, (10, 10)))
        kwargs = {'method': 'update_named_param', 'args': (request.name, request.dtype, tuple(request.shape))}  # 组装参数
        for connection in self.connections:  # 广播至所有 worker
            connection.send({'type': 'fire_and_forget', 'method': 'collective_rpc', 'kwargs': kwargs})  # 下发异步更新

        return {'message': 'Request received, updating named parameter'}  # 返回确认消息

    async def reset_prefix_cache(self):
        """函数功能：重置模型的 prefix cache，并汇总各 worker 执行结果。

        返回值：
            dict: 包含是否全部成功的状态消息。
        """
        for connection in self.connections:  # 依次向各 worker 发送重置指令
            connection.send({'type': 'call', 'method': 'reset_prefix_cache'})  # 发送需要返回结果的调用
        all_outputs = [connection.recv() for connection in self.connections]  # 收集所有 worker 的响应布尔值
        success = all(output for output in all_outputs)  # 聚合：全部为 True 则表示成功
        return {'message': 'Request received, resetting prefix cache status: ' + str(success)}  # 返回状态汇总

    async def get_engine_type(self):
        """函数功能：返回当前引擎类型（AsyncLLMEngine/LLMEngine）与是否使用 Gym 环境标记。

        返回值：
            dict: 包含 engine_type 与可选 gym_env 字段的字典。
        """
        if self.use_async_engine:  # 如果使用异步引擎
            if self.use_gym_env:  # 且启用了 Gym 环境
                return {'engine_type': 'AsyncLLMEngine', 'gym_env': True}  # 返回包含 gym_env 的标记
            return {'engine_type': 'AsyncLLMEngine'}  # 仅返回异步引擎标记
        else:  # 否则为同步引擎
            return {'engine_type': 'LLMEngine'}  # 返回同步引擎标记

    async def close_communicator(self):
        """函数功能：关闭权重同步通信器并清理资源。

        返回值：
            dict: 接收关闭请求的确认消息。
        """
        kwargs = {'method': 'close_communicator'}  # 组装关闭通信器的 RPC 参数
        for connection in self.connections:  # 广播关闭指令
            connection.send({'type': 'fire_and_forget', 'method': 'collective_rpc', 'kwargs': kwargs})  # 无需返回
        return {'message': 'Request received, closing communicator'}  # 返回确认消息

    async def infer(
        self,  # self：实例自身引用
        infer_requests: List[Union[Dict, RolloutInferRequest]],  # infer_requests：推理请求列表（字典或结构体）
        request_config: Optional[RequestConfig] = None,  # request_config：生成配置（可选）
        *,  # 仅限关键字参数分隔符
        use_tqdm: Optional[bool] = None,  # use_tqdm：是否显示进度（部分实现支持）
    ):
        """函数功能：将推理请求按连接数分片，下发至多个 worker 并汇总返回。

        入参：
            infer_requests (List[Union[Dict, RolloutInferRequest]]): 推理请求集合。
            request_config (Optional[RequestConfig]): 生成配置；若提供则透传至引擎。
            use_tqdm (Optional[bool]): 是否显示 tqdm 进度（仅部分实现支持）。

        返回值：
            List: 扁平化后的各 worker 输出集合。

        示例：
            >>> # HTTP POST /infer/ 由客户端发起，此处给出方法级含义说明
        """
        chunked_infer_requests = chunk_list(infer_requests, self.num_connections)  # 按连接数切分请求列表

        for i, (connection, requests) in enumerate(zip(self.connections, chunked_infer_requests)):  # 遍历分片并分发
            if not requests:  # 当某些 worker 分片为空时
                requests = [RolloutInferRequest(messages=[{'role': 'user', 'content': '<placeholder>'}])]  # 构造占位请求
            if request_config and request_config.seed is not None:  # 当提供随机种子时，为不同 worker 扰动种子
                request_config.seed += i * len(requests)  # 按分片索引和请求数叠加偏移
            kwargs = {'infer_requests': requests, 'request_config': request_config, 'use_tqdm': use_tqdm}  # 组装参数
            method = 'async_infer' if self.use_async_engine else 'infer'  # 选择调用方法名（异步/同步）
            connection.send({'type': 'call', 'method': method, 'kwargs': kwargs})  # 发送调用至对应 worker

        all_outputs = [connection.recv() for connection in self.connections]  # 收集各 worker 返回的结果列表
        all_outputs = [output for output, requests in zip(all_outputs, chunked_infer_requests) if requests]  # 过滤空分片
        all_outputs = list(chain.from_iterable(all_outputs))  # 将嵌套列表拍平为单层列表

        return all_outputs  # 返回合并后的推理结果

    def run(self):
        """函数功能：使用 uvicorn 启动 FastAPI 服务（阻塞运行）。

        示例：
            >>> SwiftRolloutDeploy(args).run()
        """
        args = self.args  # 读取启动参数对象
        uvicorn.run(self.app, host=args.host, port=args.port, log_level=args.log_level)  # 启动 uvicorn 服务


def rollout_main(args: Optional[Union[List[str], RolloutArguments]] = None) -> None:
    """函数功能：命令行主入口；构建部署对象并执行主流程。
    示例：
        >>> rollout_main(["--model", "your-model"])  # 从 CLI 样式参数启动
    """
    SwiftRolloutDeploy(args).main()  # 调用管道主流程：启动服务与事件循环


def is_accessible(port: int):
    """函数功能：检查 rollout 服务是否已经在指定端口上可访问。

    入参：
        port (int): 目标端口号。

    返回值：
        bool: True 表示端口可达且服务响应；False 表示不可达。

    示例：
        >>> is_accessible(8000)
        False
    """
    infer_client = InferClient(port=port)  # 构造客户端以访问服务端 /v1 接口
    try:  # 尝试请求模型列表以判断服务是否可达
        infer_client.get_model_list()  # 请求 /models 等接口
    except ClientConnectorError:  # 捕获连接异常（端口未开放或服务未启动）
        return False  # 返回不可达
    return True  # 返回可达


@contextmanager
def run_rollout(args: RolloutArguments, return_url: bool = False):
    """函数功能：在子进程中临时启动 rollout 服务，并在退出时安全终止。

    入参：
        args (RolloutArguments): 启动参数对象（或可被 dataclass 解析的同构对象）。
        return_url (bool): 若为 True 返回 base URL（'http://127.0.0.1:port/v1'），否则返回端口号。

    返回值：
        ContextManager: with 语句上下文中可获取端口或 URL。

    示例：
        >>> from swift.llm import RolloutArguments
        >>> with run_rollout(RolloutArguments(model='your-model'), return_url=True) as url:
        ...     print(url)
    """
    if isinstance(args, RolloutArguments) and args.__class__.__name__ == 'RolloutArguments':  # 若已是标准参数对象
        deploy_args = args  # 直接使用现有对象
    else:  # 否则从类似结构对象中筛选出合法字段构造参数对象
        args_dict = asdict(args)  # 将对象转为字典，便于过滤无关键
        parameters = inspect.signature(RolloutArguments).parameters  # 读取 RolloutArguments 的参数签名
        for k in list(args_dict.keys()):  # 遍历输入字典的键
            if k not in parameters or args_dict[k] is None:  # 移除不存在于参数签名或值为 None 的字段
                args_dict.pop(k)  # 删除无效键
        deploy_args = RolloutArguments(**args_dict)  # 构造规范的参数对象

    mp = multiprocessing.get_context('spawn')  # 以 'spawn' 启动方式获取多进程上下文
    process = mp.Process(target=rollout_main, args=(deploy_args, ))  # 构造子进程以启动 rollout_main
    process.start()  # 启动子进程，异步拉起服务
    try:  # 在上下文中等待服务可达
        while not is_accessible(deploy_args.port):  # 轮询直到端口可访问
            time.sleep(1)  # 每秒检查一次，避免忙等
        yield f'http://127.0.0.1:{deploy_args.port}/v1' if return_url else deploy_args.port  # 按需返回 URL 或端口
    finally:  # 上下文退出：终止子进程并记录日志
        process.terminate()  # 结束子进程
        logger.info('The deployment process has been terminated.')  # 记录已终止信息


# https://github.com/huggingface/trl/pull/3690  # 引用 PR：说明变更背景
# 兼容补丁：TRL 不同版本对 dtype 注解不同（<=0.19 为 torch.dtype，>0.19 为 str）。
old_update_named_param = WeightSyncWorkerExtension.update_named_param  # 记录原方法引用
dtype_annotation = get_type_hints(old_update_named_param).get('dtype')  # 读取 dtype 注解类型

if not hasattr(WeightSyncWorkerExtension, 'old_update_named_param') and dtype_annotation == torch.dtype:  # 仅当需要时打补丁

    @wraps(old_update_named_param)
    def patched_update_named_param(self, name, dtype, shape) -> None:
        """函数功能：兼容 TRL 对 dtype 注解的变更，支持传入 str 或 torch.dtype。

        入参：
            name (str): 命名参数名称。
            dtype (Union[str, torch.dtype]): 数据类型（字符串或 torch.dtype 对象）。
            shape (tuple): 张量形状。

        返回值：
            None
        """
        if isinstance(dtype, str):  # 若传入字符串类型
            dtype = getattr(torch, dtype.split('.')[-1])  # 解析字符串末尾的类型名映射到 torch.dtype
        return old_update_named_param(self, name, dtype, shape)  # 调用原方法完成更新

    WeightSyncWorkerExtension.update_named_param = patched_update_named_param  # 将扩展方法替换为兼容版本
    WeightSyncWorkerExtension.old_update_named_param = old_update_named_param  # 记录原方法以避免重复打补丁
