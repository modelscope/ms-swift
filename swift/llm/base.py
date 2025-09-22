# Copyright (c) Alibaba, Inc. and its affiliates.
"""
swift.llm.base 模块：定义 SwiftPipeline 抽象基类。

功能：
- 统一解析命令行/传入参数并注入到 self.args
- 根据 args.seed 与 args.rank 设置全局随机种子，保证分布式场景下不同 rank 的可复现性
- 在阿里云 DSW + Gradio 环境下自动设置 GRADIO_ROOT_PATH，确保 WebUI 反向代理路径正确
- 提供 main() 统一入口，打印开始/结束时间与版本信息，并调用子类实现的 run()
"""
import datetime as dt  # 时间相关，用于打印运行起止时间
import os  # 读取/设置环境变量（兼容 DSW + Gradio）
from abc import ABC, abstractmethod  # 抽象基类与抽象方法定义
from typing import List, Optional, Union  # 类型注解：列表、可选、联合

import swift  # 读取包版本等信息
from swift.utils import get_logger, parse_args, seed_everything  # 日志、参数解析、全局随机种子
from .argument import BaseArguments  # Pipeline 基础参数定义
from .utils import ProcessorMixin  # 处理流程混入

logger = get_logger()  # 模块级日志记录器


class SwiftPipeline(ABC, ProcessorMixin):
    """
    抽象 Pipeline 基类。

    职责:
    - 解析/注入参数到 self.args
    - 设置随机种子（考虑分布式 rank）
    - 兼容 DSW + Gradio 环境
    - 提供 main() 入口并调用子类 run()

    继承:
    - ABC: 要求子类实现 run()
    - ProcessorMixin: 提供通用处理能力
    """

    args_class = BaseArguments  # 指定参数类，供解析与类型提示使用

    def __init__(self, args: Optional[Union[List[str], args_class]] = None):
        """
        初始化 Pipeline 并完成通用准备工作。

        参数:
            args: None | List[str] | args_class 实例。None/列表将被解析为 args_class。
        """
        self.args = self._parse_args(args)  # 解析参数
        args = self.args  # 简化引用
        if hasattr(args, 'seed'):  # 支持设置随机种子
            seed = args.seed + max(getattr(args, 'rank', -1), 0)  # 多卡按 rank 偏移
            seed_everything(seed)  # 设定全局随机种子
        logger.info(f'args: {args}')  # 打印参数
        self._compat_dsw_gradio(args)  # 处理 DSW + Gradio 兼容

    def _parse_args(self, args: Optional[Union[List[str], args_class]] = None) -> args_class:
        """
        解析输入参数为 args_class 实例。

        参数:
            args: None / List[str] / args_class 实例

        返回:
            args_class: 解析后的参数对象
        """
        if isinstance(args, self.args_class):  # 已是目标类型
            return args  # 直接返回
        assert self.args_class is not None  # 防御性检查
        args, remaining_argv = parse_args(self.args_class, args)  # 解析参数
        if len(remaining_argv) > 0:  # 存在未消费参数
            if getattr(args, 'ignore_args_error', False):  # 可忽略
                logger.warning(f'remaining_argv: {remaining_argv}')  # 告警
            else:
                raise ValueError(f'remaining_argv: {remaining_argv}')  # 抛错
        return args  # 返回

    @staticmethod
    def _compat_dsw_gradio(args) -> None:
        """
        在阿里云 DSW 环境下为 Gradio 设置 GRADIO_ROOT_PATH 以适配反向代理路径。

        参数:
            args: WebUIArguments/AppArguments 或其他参数对象
        """
        from swift.llm import WebUIArguments, AppArguments  # 延迟导入避免循环依赖
        if (isinstance(args, (WebUIArguments, AppArguments)) and 'JUPYTER_NAME' in os.environ
                and 'dsw-' in os.environ['JUPYTER_NAME'] and 'GRADIO_ROOT_PATH' not in os.environ):
            os.environ['GRADIO_ROOT_PATH'] = f"/{os.environ['JUPYTER_NAME']}/proxy/{args.server_port}"  # 设置根路径

    def main(self):
        """
        打印起止时间与版本信息，执行 run() 并返回结果。

        返回:
            Any: 子类 run() 的返回值
        """
        logger.info(f'Start time of running main: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')  # 开始时间
        logger.info(f'swift.__version__: {swift.__version__}')  # 当前 swift 版本
        result = self.run()  # 执行业务逻辑
        logger.info(f'End time of running main: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}')  # 结束时间
        return result  # 返回结果

    @abstractmethod
    def run(self):
        """
        子类必须实现的核心逻辑。

        返回:
            Any: 子类自定义返回类型
        """
        pass  # 抽象方法，无默认实现
