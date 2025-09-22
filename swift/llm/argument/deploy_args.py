"""模块说明：
该模块定义了用于模型部署与 rollout 的参数数据类。
- DeployArguments：继承自 InferArguments，扩展部署的网络/鉴权/日志/适配器等配置与初始化。
- RolloutArguments：继承自 DeployArguments，针对 GRPO/多轮等场景，提供异步引擎与设备校验逻辑。
通过集中化的配置与校验，保证服务在启动阶段即可发现配置与资源问题。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass  # 数据类装饰器，用于简化参数类定义
from typing import Literal, Optional  # 类型注解：字面量与可选类型

from swift.llm import safe_snapshot_download  # 安全地下载/缓存适配器或权重快照
from swift.utils import find_free_port, get_device_count, get_logger  # 工具方法：找空闲端口/获取设备数/获取日志器
from .base_args import BaseArguments  # 基础参数类，提供通用初始化能力
from .infer_args import InferArguments  # 推理参数基类，部署参数在其基础上扩展

logger = get_logger()  # 初始化模块级日志器，用于统一打印


@dataclass  # 数据类装饰器，自动生成 __init__/__repr__ 等方法
class DeployArguments(InferArguments):  # 部署参数类，继承推理参数基类
    """
    类说明：部署参数数据类，继承自 `InferArguments`，用于定义启动推理服务/部署所需的配置项。

    主要职责：
    - 配置服务监听、SSL 与鉴权。
    - 统一服务标识、日志级别与统计输出间隔。
    - 解析/下载适配器，并规范输出目录与流式设置。

    属性：
        host: 服务绑定地址。
        port: 服务端口（若被占用，将自动寻找空闲端口）。
        api_key: 调用鉴权用的 API Key；None 表示不启用鉴权。
        ssl_keyfile: SSL 私钥文件路径。
        ssl_certfile: SSL 证书文件路径。
        owned_by: 服务归属标识。
        served_model_name: 对外暴露的模型服务名。
        verbose: 是否打印请求信息。
        log_interval: 全局统计打印间隔（秒）。
        log_level: 日志级别（'critical'/'error'/'warning'/'info'/'debug'/'trace'）。
        max_logprobs: 返回的 logprobs 最大数量。
        vllm_use_async_engine: 是否启用 vLLM 异步引擎。
    """
    host: str = '0.0.0.0'  # 服务监听地址，默认对所有网卡开放
    port: int = 8000  # 服务监听端口，启动时会校验并可自动替换为空闲端口
    api_key: Optional[str] = None  # 可选鉴权 Token；None 表示不启用鉴权
    ssl_keyfile: Optional[str] = None  # 可选 SSL 私钥路径
    ssl_certfile: Optional[str] = None  # 可选 SSL 证书路径

    owned_by: str = 'swift'  # 服务归属方标识
    served_model_name: Optional[str] = None  # 对外暴露的模型服务名
    verbose: bool = True  # 是否打印请求级别日志
    log_interval: int = 20  # 全局统计信息打印间隔（秒）
    log_level: Literal['critical', 'error', 'warning', 'info', 'debug', 'trace'] = 'info'  # 运行时日志级别

    max_logprobs: int = 20  # 返回 token 级 logprobs 的最大个数
    vllm_use_async_engine: bool = True  # 是否启用 vLLM 异步引擎（部署默认启用）

    def __post_init__(self):  # 数据类初始化后的钩子
        """
        函数说明：在数据类完成初始化后，继续执行部署相关的二次初始化逻辑。

        示例：
            >>> args = DeployArguments(port=8000)
            >>> isinstance(args.port, int)
            True
        """
        super().__post_init__()  # 调用父类初始化，确保通用推理参数就绪
        self.port = find_free_port(self.port)  # 若端口被占用则选择一个空闲端口

    def _init_adapters(self):  # 初始化并下载/规范化适配器配置
        """
        函数说明：规范化 `self.adapters` 的格式，支持 "name=path" 语法，并通过
        `safe_snapshot_download` 下载/缓存适配器；构建具名映射与匿名列表。

        参数：
            self: 当前实例。

        返回：
            None（就地更新 `self.adapters` 与 `self.adapter_mapping`）。

        示例：
            >>> args = DeployArguments()
            >>> args.adapters = ['lora1=/path/to/loraA', '/path/to/loraB']
            >>> args._init_adapters()
            >>> isinstance(args.adapters, list)
            True
        """
        if isinstance(self.adapters, str):  # 若仅给定单个适配器路径，统一转为列表
            self.adapters = [self.adapters]  # 规范为列表以便统一处理
        self.adapter_mapping = {}  # 具名适配器：名称到本地路径的映射
        adapters = []  # 匿名适配器路径列表
        for i, adapter in enumerate(self.adapters):  # 遍历用户提供的每个适配器
            adapter_path = adapter.split('=')  # 支持 "name=path" 语法
            if len(adapter_path) == 1:  # 未提供名称时，使用 None 表示匿名适配器
                adapter_path = (None, adapter_path[0])  # 规范为二元组 (name, path)
            adapter_name, adapter_path = adapter_path  # 拆解出名称与路径
            adapter_path = safe_snapshot_download(adapter_path, use_hf=self.use_hf, hub_token=self.hub_token)  # 下载/缓存权重
            if adapter_name is None:  # 匿名适配器：按顺序追加
                adapters.append(adapter_path)  # 保存到匿名列表
            else:  # 具名适配器：记录映射
                self.adapter_mapping[adapter_name] = adapter_path  # 以名称索引路径
        self.adapters = adapters  # 回写规范化后的适配器列表

    def _init_ckpt_dir(self, adapters=None):  # 初始化 checkpoint 目录，保持签名兼容
        """
        函数说明：基于所有适配器路径，初始化/返回 checkpoint 目录。

        参数：
            self: 当前实例。
            adapters: 兼容形参（未使用），保留以匹配父类/外部调用签名。

        返回：
            任意：父类 `_init_ckpt_dir` 的返回结果（通常为路径）。

        示例：
            >>> args = DeployArguments()
            >>> _ = args._init_ckpt_dir()
        """
        return super()._init_ckpt_dir(self.adapters + list(self.adapter_mapping.values()))  # 传入全部适配器路径

    def _init_stream(self):  # 初始化流式输出设置
        """
        函数说明：初始化流式（stream）相关配置，直接复用 `BaseArguments` 的实现。

        参数：
            self: 当前实例。

        返回：
            任意：基类 `_init_stream` 的返回结果。

        示例：
            >>> args = DeployArguments()
            >>> _ = args._init_stream()
        """
        return BaseArguments._init_stream(self)  # 复用基类实现

    def _init_eval_human(self):  # 预留的人类评估初始化（部署默认不启用）
        """
        函数说明：占位方法。部署场景下不需要人类评估的初始化逻辑。

        参数：
            self: 当前实例。

        返回：
            None

        示例：
            >>> DeployArguments()._init_eval_human()
        """
        pass  # 无操作，占位

    def _init_result_path(self, folder_name: str) -> None:  # 初始化结果输出目录名
        """
        函数说明：将推理阶段默认的输出目录名 `infer_result` 替换为部署阶段的 `deploy_result`，
        然后调用父类完成路径初始化。

        参数：
            folder_name: 请求初始化的目录名称。

        返回：
            None

        示例：
            >>> args = DeployArguments()
            >>> args._init_result_path('infer_result')
        """
        if folder_name == 'infer_result':  # 若为推理默认目录名
            folder_name = 'deploy_result'  # 替换为部署目录名
        return super()._init_result_path(folder_name)  # 调用父类完成目录创建与记录


@dataclass  # 数据类装饰器，自动生成 __init__/__repr__ 等
class RolloutArguments(DeployArguments):  # Rollout 参数类，继承部署参数
    """
    类说明：面向 GRPO/多轮对话等 rollout 场景的参数类，扩展/校验与并发相关的配置。

    要点：
    - 可按需启用 vLLM 异步引擎以提升并发吞吐。
    - 提供多轮调度器、Gym 环境等可选集成，并在初始化阶段进行参数与资源校验。

    属性：
        vllm_use_async_engine: 若未显式指定，将按场景决定默认值。
        use_gym_env: 是否使用 Gym 环境。
        multi_turn_scheduler: 多轮对话调度策略名称。
        max_turns: 最大对话轮数。
        gym_env: Gym 环境 ID（如 'CartPole-v1'）。
        context_manager: 上下文管理器名称。
    """
    vllm_use_async_engine: Optional[bool] = None  # rollout 阶段是否使用异步引擎；None 表示初始化时再决定
    use_gym_env: Optional[bool] = None  # 是否启用 Gym 环境进行交互式评测/训练
    # only for GRPO rollout with AsyncEngine, see details in swift/plugin/multi_turn
    multi_turn_scheduler: Optional[str] = None  # 多轮对话调度器名称
    max_turns: Optional[int] = None  # 多轮对话的最大轮数限制

    # GYM env
    gym_env: Optional[str] = None  # Gym 环境标识（如 'CartPole-v1'）
    context_manager: Optional[str] = None  # 上下文管理器名称

    def __post_init__(self):  # 初始化钩子：检查依赖、设置默认值并校验资源
        """
        函数说明：数据类初始化完成后，执行依赖检查、默认引擎设置与参数/设备校验。

        参数：
            self: 当前实例。

        返回：
            None

        示例：
            >>> args = RolloutArguments(vllm_data_parallel_size=1, vllm_tensor_parallel_size=1)
            >>> # 通常由 dataclass 自动调用
        """
        self._check_trl_version()  # 校验 trl 版本是否满足需求
        super().__post_init__()  # 继承父类初始化（含端口探测与适配器初始化）
        self._set_default_engine_type()  # 根据场景设置异步引擎默认值
        self._check_args()  # 校验不被支持的参数组合
        self._check_device_count()  # 校验设备数量是否满足 DP*TP 需求

    def _check_trl_version(self):  # 检查 trl 版本是否包含所需扩展
        """
        函数说明：通过尝试导入 `WeightSyncWorkerExtension` 来确认 trl 版本是否满足要求。

        参数：
            self: 当前实例。

        返回：
            None；若版本不满足，将抛出 ImportError。

        示例：
            >>> RolloutArguments()._check_trl_version()
        """
        try:  # 尝试导入以判断依赖是否就绪
            from trl.scripts.vllm_serve import WeightSyncWorkerExtension  # 依赖存在性检查
        except ImportError as e:  # 依赖缺失时捕获导入错误
            raise ImportError("Could not import 'WeightSyncWorkerExtension' from 'trl.scripts.vllm_serve'. "
                              "Please upgrade your 'trl' package by 'pip install -U trl'") from e  # 指导用户升级

    def _set_default_engine_type(self):  # 设置异步引擎默认启用策略
        """
        函数说明：若用户未显式指定是否使用异步引擎，则根据是否为多轮/Gym 场景决定默认值。

        参数：
            self: 当前实例。

        返回：
            None

        示例：
            >>> args = RolloutArguments(multi_turn_scheduler='fifo')
            >>> args._set_default_engine_type()
            >>> args.vllm_use_async_engine
            True
        """
        if self.vllm_use_async_engine is None:  # 未显式指定时按场景设置
            if self.multi_turn_scheduler or self.use_gym_env:  # 多轮/Gym 场景默认开启异步引擎
                self.vllm_use_async_engine = True  # 启用异步引擎
            else:  # 非上述场景默认关闭
                self.vllm_use_async_engine = False  # 关闭异步引擎

    def _check_args(self):  # 校验不被支持的参数组合
        """
        函数说明：Rollout 阶段不支持流水线并行（pipeline parallel），若开启则报错。

        参数：
            self: 当前实例。

        返回：
            None；若配置不合法，将抛出 ValueError。

        示例：
            >>> args = RolloutArguments()
            >>> args.vllm_pipeline_parallel_size = 2
            >>> # args._check_args()  # 将抛出 ValueError
        """
        if self.vllm_pipeline_parallel_size > 1:  # rollout 不支持流水线并行
            # 抛出明确错误提示，指导设置为 1
            raise ValueError('RolloutArguments does not support pipeline parallelism, '
                             'please set vllm_pipeline_parallel_size to 1.')

    def _check_device_count(self):  # 校验设备数量是否满足 DP*TP 需求
        """
        函数说明：检查本机可见设备数量与参数配置的匹配情况，并在不匹配时给出错误/告警。

        参数：
            self: 当前实例。

        返回：
            None；若设备不足，将抛出 ValueError；若设备超出需求，将给出告警。

        示例：
            >>> args = RolloutArguments(vllm_data_parallel_size=1, vllm_tensor_parallel_size=1)
            >>> args._check_device_count()
        """
        local_device_count = get_device_count()  # 本机可见 GPU 数量
        required_device_count = self.vllm_data_parallel_size * self.vllm_tensor_parallel_size  # 需求设备数 = DP * TP

        if local_device_count < required_device_count:  # 本机资源不足
            msg = (f'Error: local_device_count ({local_device_count}) must be greater than or equal to '  # 错误信息：设备不足
                   f'the product of vllm_data_parallel_size ({self.vllm_data_parallel_size}) and '  # 继续描述不足原因
                   f'vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}). '  # 给出计算公式
                   f'Current required_device_count = {required_device_count}.')  # 给出所需设备数
            raise ValueError(msg)  # 抛出异常中断启动

        if local_device_count > required_device_count:  # 设备多于需求，输出一次性警告
            logger.warning_once(  # 仅提示一次，避免刷屏
                f'local_device_count ({local_device_count}) is greater than required_device_count ({required_device_count}). '  # noqa
                f'Only the first {required_device_count} devices will be utilized for rollout. '  # 告知仅使用前 N 张卡
                f'To fully utilize resources, set vllm_tensor_parallel_size * vllm_data_parallel_size = device_count. '  # noqa
                f'device_count: {local_device_count}, '  # 打印本机设备数
                f'vllm_tensor_parallel_size: {self.vllm_tensor_parallel_size}, '  # 打印 TP 配置
                f'vllm_data_parallel_size: {self.vllm_data_parallel_size}, '  # 打印 DP 配置
                f'required_device_count: {required_device_count}.')  # 打印需求设备数
