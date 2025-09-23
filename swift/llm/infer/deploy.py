"""
模块功能概述：
本模块提供LLM推理服务的部署实现，支持OpenAI风格API（/v1/chat/completions 等）：
- SwiftDeploy: 基于FastAPI的服务管道，注册路由、检查请求、执行业务推理与日志统计
- deploy_main/run_deploy: 提供部署入口与上下文管理器，便于在评测中临时拉起服务
- is_accessible: 简易健康检查，判断端口上的服务是否可用
"""

# 版权信息：阿里巴巴及其附属公司保留所有权利
# Copyright (c) Alibaba, Inc. and its affiliates.
# 导入异步IO：用于后台统计协程与异步推理接口
import asyncio
# 反射/签名工具：用于从参数对象筛取DeployArguments合法字段
import inspect
# 多进程模块：用于spawn子进程启动独立部署服务
import multiprocessing
# 时间模块：用于等待端口可用与节流
import time
# 上下文管理器装饰器：用于包装run_deploy等with上下文
from contextlib import contextmanager
# dataclasses工具：asdict将数据类实例转字典
from dataclasses import asdict
# HTTP状态码枚举：构造错误响应时使用
from http import HTTPStatus
# 线程类：用于后台统计线程
from threading import Thread
# 类型注解：用于标注参数与返回值类型
from typing import List, Optional, Union

# JSON库：用于SSE数据序列化
import json
# Uvicorn启动ASGI服务
import uvicorn
# aiohttp错误类型：用于客户端健康检查时识别连接失败
from aiohttp import ClientConnectorError
# FastAPI框架与请求类型
from fastapi import FastAPI, Request
# FastAPI响应类型：JSON与流式响应（SSE）
from fastapi.responses import JSONResponse, StreamingResponse

# 导入推理/部署所需的协议与参数
from swift.llm import AdapterRequest, DeployArguments
from swift.llm.infer.protocol import EmbeddingRequest, MultiModalRequestMixin
# 推理统计插件：聚合吞吐、延迟等指标
from swift.plugin import InferStats
# 工具：JsonlWriter用于写入推理日志；get_logger初始化日志器
from swift.utils import JsonlWriter, get_logger
# 推理基类：提供infer_async等能力
from .infer import SwiftInfer
# 推理客户端：用于健康检查（获取模型列表）
from .infer_engine import InferClient
# 协议数据模型：请求/响应模型列表
from .protocol import ChatCompletionRequest, CompletionRequest, Model, ModelList

# 初始化模块级日志器
logger = get_logger()


# 部署服务类：继承SwiftInfer，封装路由注册、请求校验与推理流程
class SwiftDeploy(SwiftInfer):
    """
    类功能：
        封装FastAPI服务部署逻辑，注册OpenAI风格的聊天/补全/嵌入接口，
        提供API Key校验、模型可用性检查、流式输出、统计记录与JSONL日志。

    关键属性：
        args_class: 参数类类型，固定为DeployArguments。
        args: 部署参数实例（由基类解析）。
        infer_engine.strict: 严格模式，编码错误直接抛出，便于定位问题。
        infer_stats: 推理统计聚合器。
        app: FastAPI应用实例，注册路由与生命周期。
    """

    # 指定该管道的参数类
    args_class = DeployArguments
    # 类型标注：实例属性args为上述类型
    args: args_class

    # 注册FastAPI路由：模型列表、聊天补全、纯补全、嵌入
    def _register_app(self):
        """
        函数功能：
            将内部处理函数绑定到FastAPI应用的对应路由，提供/v1开头的OpenAI风格接口。
        """
        # GET接口：返回可用模型列表
        self.app.get('/v1/models')(self.get_available_models)
        # POST接口：聊天补全
        self.app.post('/v1/chat/completions')(self.create_chat_completion)
        # POST接口：纯补全
        self.app.post('/v1/completions')(self.create_completion)
        # POST接口：文本嵌入
        self.app.post('/v1/embeddings')(self.create_embedding)

    # 构造函数：初始化基类、严格模式、统计器与FastAPI应用
    def __init__(self, args: Optional[Union[List[str], DeployArguments]] = None) -> None:
        """
        函数功能：
            初始化部署管道，开启严格模式，创建统计器与FastAPI应用，并注册路由。

        入参：
            args (Optional[Union[List[str], DeployArguments]]): 参数列表或参数对象。
        """
        # 调用基类构造与初始化（解析参数、构建模板/引擎等）
        super().__init__(args)

        # 将推理引擎设置为严格模式，错误立刻暴露
        self.infer_engine.strict = True
        # 构建推理统计器，用于周期性汇总吞吐/延迟等
        self.infer_stats = InferStats()
        # 创建FastAPI应用并设置生命周期回调
        self.app = FastAPI(lifespan=self.lifespan)
        # 绑定所有路由到应用
        self._register_app()

    # 后台统计协程：按间隔聚合并重置统计
    async def _log_stats_hook(self):
        """
        函数功能：
            后台周期性任务：按log_interval间隔聚合推理统计并重置计数。
        """
        # 无限循环，直到进程结束或事件循环关闭
        while True:
            # 等待指定的日志周期
            await asyncio.sleep(self.args.log_interval)
            # 计算并打印全局统计
            self._compute_infer_stats()
            # 重置统计计数器，为下一周期做准备
            self.infer_stats.reset()

    # 聚合并打印统计
    def _compute_infer_stats(self):
        """
        函数功能：
            计算全局推理统计并格式化数值精度后输出到日志。
        """
        # 计算得到字典形式的统计指标
        global_stats = self.infer_stats.compute()
        # 四舍五入到8位小数，便于阅读
        for k, v in global_stats.items():
            global_stats[k] = round(v, 8)
        # 打印到日志
        logger.info(global_stats)

    # FastAPI生命周期：启动统计线程/退出前补采样
    def lifespan(self, app: FastAPI):
        """
        函数功能：
            FastAPI应用的生命周期管理：
            - 启动时，如设置了log_interval则拉起后台线程运行统计协程
            - 退出时，若启用统计则补一次聚合输出
        """
        # 便捷引用参数
        args = self.args
        # 若开启了统计，则在后台线程运行异步统计协程
        if args.log_interval > 0:
            thread = Thread(target=lambda: asyncio.run(self._log_stats_hook()), daemon=True)
            thread.start()
        try:
            # 交还控制权给FastAPI，允许应用运行
            yield
        finally:
            # 退出应用前，如启用统计则补打一版汇总
            if args.log_interval > 0:
                self._compute_infer_stats()

    # 获取服务端提供的模型列表（含适配器映射名）
    def _get_model_list(self):
        """
        函数功能：
            组合主模型名与可热切换的适配器别名，形成对外展示的模型ID列表。
        """
        # 读取参数对象
        args = self.args
        # 主模型名：优先served_model_name，否则使用model_suffix
        model_list = [args.served_model_name or args.model_suffix]
        # 若定义了适配器映射，追加其键名作为额外模型ID
        if args.adapter_mapping:
            model_list += [name for name in args.adapter_mapping.keys()]
        # 返回可用列表
        return model_list

    # 接口：返回可用模型列表
    async def get_available_models(self):
        """
        函数功能：
            以OpenAI风格返回可用模型列表（ModelList）。
        """
        # 读取内部模型列表
        model_list = self._get_model_list()
        # 将每个模型ID包装为Model对象，设置owned_by字段
        data = [Model(id=model_id, owned_by=self.args.owned_by) for model_id in model_list]
        # 返回ModelList
        return ModelList(data=data)

    # 校验请求中的模型名是否在可用列表
    async def _check_model(self, request: ChatCompletionRequest) -> Optional[str]:
        """
        函数功能：
            校验请求中携带的model字段是否在服务端可用模型列表中。

        返回值：
            Optional[str]: 错误消息；若合法返回None。
        """
        # 获取可用模型列表
        available_models = await self.get_available_models()
        model_list = [model.id for model in available_models.data]
        # 若请求模型不在列表，返回提示消息
        if request.model not in model_list:
            return f'`{request.model}` is not in the model_list: `{model_list}`.'

    # 校验API Key，若配置了api_key则必须匹配
    def _check_api_key(self, raw_request: Request) -> Optional[str]:
        """
        函数功能：
            从HTTP头Authorization读取Bearer Token并与服务端api_key比对。

        返回值：
            Optional[str]: 错误消息；若合法返回None。
        """
        # 读取服务端配置的API Key
        api_key = self.args.api_key
        # 若未配置API Key，则无需校验
        if api_key is None:
            return
        # 从请求头中取Authorization字段
        authorization = dict(raw_request.headers).get('authorization')
        # 统一错误消息
        error_msg = 'API key error'
        # 头缺失或不是Bearer开头则错误
        if authorization is None or not authorization.startswith('Bearer '):
            return error_msg
        # 去掉前缀取token
        request_api_key = authorization[7:]
        # 与服务端配置比对
        if request_api_key != api_key:
            return error_msg

    # 校验top_logprobs上限，避免服务端负载过大
    def _check_max_logprobs(self, request):
        """
        函数功能：
            若请求的top_logprobs超过服务端允许的最大值，则返回错误信息。
        """
        # 便捷引用参数
        args = self.args
        # 仅当top_logprobs为int且超过上限时拦截
        if isinstance(request.top_logprobs, int) and request.top_logprobs > args.max_logprobs:
            return (f'The value of top_logprobs({request.top_logprobs}) is greater than '
                    f'the server\'s max_logprobs({args.max_logprobs}).')

    # 构造统一错误响应
    @staticmethod
    def create_error_response(status_code: Union[int, str, HTTPStatus], message: str) -> JSONResponse:
        """
        函数功能：
            返回OpenAI风格的错误JSON对象与对应HTTP状态码。
        """
        # 规范化状态码为int
        status_code = int(status_code)
        # 构造JSON响应
        return JSONResponse({'message': message, 'object': 'error'}, status_code)

    # 统一后处理：更新统计、写日志、SSE拼接等
    def _post_process(self, request_info, response, return_cmpl_response: bool = False):
        """
        函数功能：
            处理推理返回：
            - 对多模态图片内容做base64内嵌
            - 根据是否流式追加或覆盖response文本
            - 结束时更新统计/写jsonl/按需转为兼容的CompletionResponse
        """
        # 便捷引用参数
        args = self.args

        # 遍历所有choices，识别message.content为列表/元组（多模态）时处理图片
        for i in range(len(response.choices)):
            if not hasattr(response.choices[i], 'message') or not isinstance(response.choices[i].message.content,
                                                                             (tuple, list)):
                continue
            for j, content in enumerate(response.choices[i].message.content):
                if content['type'] == 'image':
                    # 将图片对象转base64，并以data URI写回
                    b64_image = MultiModalRequestMixin.to_base64(content['image'])
                    response.choices[i].message.content[j]['image'] = f'data:image/jpg;base64,{b64_image}'

        # 判断是否所有choice都给出finish_reason（表示完成）
        is_finished = all(response.choices[i].finish_reason for i in range(len(response.choices)))
        # 流式响应：增量拼接delta内容
        if 'stream' in response.__class__.__name__.lower():
            request_info['response'] += response.choices[0].delta.content
        else:
            # 非流式：直接取完整消息内容
            request_info['response'] = response.choices[0].message.content
        # 需要转换为兼容的CompletionResponse时，做类型转换
        if return_cmpl_response:
            response = response.to_cmpl_response()
        # 若已完成，则记录统计/写日志/按需打印详细信息
        if is_finished:
            if args.log_interval > 0:
                self.infer_stats.update(response)
            if self.jsonl_writer:
                self.jsonl_writer.append(request_info)
            if self.args.verbose:
                logger.info(request_info)
        # 返回处理后的响应对象
        return response

    # 将默认请求配置合并到当前请求配置（仅当默认不为空、当前为空时生效）
    def _set_request_config(self, request_config) -> None:
        """
        函数功能：
            使用args.get_request_config()提供的默认配置填充request_config中为空的字段。
        """
        # 读取默认请求配置
        default_request_config = self.args.get_request_config()
        # 若未提供默认配置，则不作处理
        if default_request_config is None:
            return
        # 遍历当前配置的各字段，若为空则用默认值覆盖
        for key, val in asdict(request_config).items():
            default_val = getattr(default_request_config, key)
            if default_val is not None and (val is None or isinstance(val, (list, tuple)) and len(val) == 0):
                setattr(request_config, key, default_val)

    # 主接口：聊天补全，支持SSE流式与非流式
    async def create_chat_completion(self,
                                     request: ChatCompletionRequest,
                                     raw_request: Request,
                                     *,
                                     return_cmpl_response: bool = False):
        """
        函数功能：
            接收聊天补全请求，进行模型与API Key校验，解析生成配置，调用异步推理，
            根据是否流式返回SSE或JSON响应。
        """
        # 便捷引用参数
        args = self.args
        # 组合各种检查：模型可用性、API Key与top_logprobs上限
        error_msg = (await self._check_model(request) or self._check_api_key(raw_request)
                     or self._check_max_logprobs(request))
        # 若存在错误，直接返回400错误响应
        if error_msg:
            return self.create_error_response(HTTPStatus.BAD_REQUEST, error_msg)
        # 复制推理关键字参数，避免污染全局
        infer_kwargs = self.infer_kwargs.copy()
        # 若根据model名称存在适配器映射，则构造AdapterRequest以热切换
        adapter_path = args.adapter_mapping.get(request.model)
        if adapter_path:
            infer_kwargs['adapter_request'] = AdapterRequest(request.model, adapter_path)

        # 解析请求为内部的InferRequest与RequestConfig
        infer_request, request_config = request.parse()
        # 将默认配置合并到请求配置中
        self._set_request_config(request_config)
        # 初始化日志记录对象：累计响应文本与可打印的请求信息
        request_info = {'response': '', 'infer_request': infer_request.to_printable()}

        # 在推理前拦截生成配置，存入request_info便于审计
        def pre_infer_hook(kwargs):
            request_info['generation_config'] = kwargs['generation_config']
            return kwargs

        # 将前置钩子加入推理参数
        infer_kwargs['pre_infer_hook'] = pre_infer_hook
        try:
            # 调用异步推理，可能返回生成器（流式）或完整响应对象
            res_or_gen = await self.infer_async(infer_request, request_config, template=self.template, **infer_kwargs)
        except Exception as e:
            # 捕获异常并打印堆栈，返回400错误
            import traceback
            logger.info(traceback.format_exc())
            return self.create_error_response(HTTPStatus.BAD_REQUEST, str(e))
        # 若开启流式，则将生成器包装为SSE响应
        if request_config.stream:

            async def _gen_wrapper():
                # 逐条读取流式增量，做后处理并序列化为SSE数据帧
                async for res in res_or_gen:
                    res = self._post_process(request_info, res, return_cmpl_response)
                    yield f'data: {json.dumps(asdict(res), ensure_ascii=False)}\n\n'
                # 发送结束帧
                yield 'data: [DONE]\n\n'

            # 返回text/event-stream类型的StreamingResponse
            return StreamingResponse(_gen_wrapper(), media_type='text/event-stream')
        # 非流式响应：若为ChatCompletionResponse实例，做统一后处理
        elif hasattr(res_or_gen, 'choices'):
            # instance of ChatCompletionResponse
            return self._post_process(request_info, res_or_gen, return_cmpl_response)
        else:
            # 其他情况（如EmbeddingResponse）直接返回
            return res_or_gen

    # 纯补全接口：将CompletionRequest转换为ChatCompletionRequest并复用实现
    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        """
        函数功能：
            将补全请求转换为聊天补全请求，复用create_chat_completion处理逻辑。
        """
        # 由协议层提供的转换方法生成ChatCompletionRequest
        chat_request = ChatCompletionRequest.from_cmpl_request(request)
        # 调用主实现，并指定返回兼容Completion的响应结构
        return await self.create_chat_completion(chat_request, raw_request, return_cmpl_response=True)

    # 嵌入接口：与上类似，将EmbeddingRequest转换为ChatCompletionRequest
    async def create_embedding(self, request: EmbeddingRequest, raw_request: Request):
        """
        函数功能：
            将嵌入请求转换为聊天补全请求，复用create_chat_completion处理逻辑以统一校验与错误处理。
        """
        # 由协议层提供的转换方法生成ChatCompletionRequest
        chat_request = ChatCompletionRequest.from_cmpl_request(request)
        # 调用主实现，获取兼容的Completion响应
        return await self.create_chat_completion(chat_request, raw_request, return_cmpl_response=True)

    # 启动服务：配置JSONL日志、打印模型列表并启动Uvicorn
    def run(self):
        """
            启动FastAPI服务：
            - 初始化JsonlWriter（如指定result_path）
            - 打印可用模型列表
            - 调用uvicorn.run绑定端口与证书等参数
        """
        # 便捷引用参数
        args = self.args
        # 如指定结果路径，则创建JSONL写入器
        self.jsonl_writer = JsonlWriter(args.result_path) if args.result_path else None
        # 打印当前服务可用的模型列表
        logger.info(f'model_list: {self._get_model_list()}')
        # 启动ASGI服务，传入主机、端口、SSL证书与日志级别
        uvicorn.run(
            self.app,
            host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            log_level=args.log_level)


# 部署入口：以管道形式启动服务
def deploy_main(args: Optional[Union[List[str], DeployArguments]] = None) -> None:
    """
    函数功能：
        部署命令入口。构造SwiftDeploy并执行其main流程。
    """
    # 创建部署管道并执行
    SwiftDeploy(args).main()


# 简易可用性检查：尝试拉取模型列表，失败则不可用
def is_accessible(port: int):
    """
    函数功能：
        尝试通过InferClient连接到指定端口的服务并获取模型列表，以此判断端口是否可用。

    返回值：
        bool: True可访问，False不可访问。
    """
    # 创建推理客户端，指向指定端口
    infer_client = InferClient(port=port)
    try:
        # 调用接口，若成功则说明服务可用
        infer_client.get_model_list()
    except ClientConnectorError:
        # 捕获连接错误，返回不可用
        return False
    # 正常返回可用
    return True


# 上下文管理器：在子进程中临时启动服务，退出时自动终止
@contextmanager
def run_deploy(args: DeployArguments, return_url: bool = False):
    """
    函数功能：
        在独立子进程中启动FastAPI部署服务，并在端口可用后yield端口或完整URL；
        上下文退出时，终止子进程并记录日志。

    入参：
        args (DeployArguments): 部署参数对象或兼容的数据类/命名元组。
        return_url (bool): 为True时返回完整URL（含/v1），否则返回端口号。

    返回值：
        上下文管理器：yield端口号或URL。
    """
    # 如果已经是DeployArguments实例，则直接使用；否则从对象中提取字段构造
    if isinstance(args, DeployArguments) and args.__class__.__name__ == 'DeployArguments':
        deploy_args = args
    else:
        # 转为字典，便于筛除无效或None字段
        args_dict = asdict(args)
        parameters = inspect.signature(DeployArguments).parameters
        for k in list(args_dict.keys()):
            if k not in parameters or args_dict[k] is None:
                args_dict.pop(k)
        # 仅使用有效字段构造DeployArguments
        deploy_args = DeployArguments(**args_dict)

    # 使用spawn上下文创建进程，避免fork带来的句柄继承问题
    mp = multiprocessing.get_context('spawn')
    # 启动子进程运行deploy_main，并传入构造好的参数
    process = mp.Process(target=deploy_main, args=(deploy_args, ))
    process.start()
    try:
        # 等待直到端口可用（健康检查通过）
        while not is_accessible(deploy_args.port):
            time.sleep(1)
        # 产出URL或端口号，供调用方使用
        yield f'http://127.0.0.1:{deploy_args.port}/v1' if return_url else deploy_args.port
    finally:
        # 退出上下文时终止子进程，确保资源释放
        process.terminate()
        logger.info('The deployment process has been terminated.')
