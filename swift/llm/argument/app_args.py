"""
模块说明：
    本模块定义了 WebUI/部署一体化应用的参数数据类 `AppArguments`，用于驱动
    在线推理服务（含流式输出、多模态开关、语言偏好等）以及与底层部署/网页
    UI 参数的对接。

    主要能力：
    - 通过 `_init_torch_dtype` 在直连远程服务（base_url）与本地加载模型之间做初始化分流；
    - 在 `__post_init__` 中补全服务端口、模板系统提示、多模态标志等应用级参数。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Literal, Optional  # 类型注解：限定可选值与可空类型

from swift.utils import find_free_port, get_logger  # 实用函数：查找可用端口与获取日志器
from ..model import get_matched_model_meta  # 根据模型标识匹配模型元信息
from ..template import get_template_meta  # 根据模板标识获取模板元数据
from .deploy_args import DeployArguments  # 部署相关参数基类
from .webui_args import WebUIArguments  # WebUI 参数基类

logger = get_logger()  # 初始化模块级日志器


@dataclass
class AppArguments(WebUIArguments, DeployArguments):
    """应用层参数数据类。

    该类整合了 WebUI 与部署相关的配置项，补充了用于运行在线应用的通用选项，
    如语言、日志详尽程度、流式输出开关等；并在初始化流程中依据是否直连远程
    服务（base_url）决定是否需要本地模型 dtype 初始化。

    字段:
        base_url: 远程已部署推理服务的基础地址；若设置则不会在本地初始化模型 dtype。
        studio_title: WebUI 页面标题（可选）。
        is_multimodal: 是否为多模态应用；若未指定，将从模型元信息推断，并最终回退为 False。
        lang: 界面/输出语言，'en' 或 'zh'。
        verbose: 是否输出详细日志。
        stream: 是否启用流式输出（服务端推送）。
    """
    base_url: Optional[str] = None
    studio_title: Optional[str] = None
    is_multimodal: Optional[bool] = None

    lang: Literal['en', 'zh'] = 'en'  # 界面/输出语言偏好
    verbose: bool = False  # 是否打印更详细的日志
    stream: bool = True  # 是否启用流式输出

    def _init_torch_dtype(self) -> None:
        """根据是否直连远程服务决定 dtype 初始化策略。

        行为:
            - 若设置了 `base_url`，表示直连远程已部署服务：
              - 通过 `get_matched_model_meta` 获取本地可用的模型元信息用于 UI/逻辑；
              - 直接返回，跳过本地模型 dtype 初始化；
            - 否则，调用父类逻辑进行 dtype 初始化（可能包含混合精度推导等）。

        返回:
            None
        """
        if self.base_url:
            self.model_meta = get_matched_model_meta(self.model)  # 直连远程服务时仍获取模型元信息供 UI/流程使用
            return  # 跳过本地 dtype 初始化
        super()._init_torch_dtype()

    def __post_init__(self):
        """实例化完成后的应用级初始化。

        步骤:
            1) 调用父类的 `__post_init__`，完成通用参数初始化；
            2) 绑定可用的 `server_port`（若冲突则自动寻找新端口）；
            3) 若存在 `model_meta`：
               - 在 `system` 未显式指定时，从模板元数据中填充默认 system；
               - 在 `is_multimodal` 未显式指定时，从模型元信息推断多模态标志；
            4) 仍未明确 `is_multimodal` 时，回退为 False。

        返回:
            None
        """
        super().__post_init__()
        self.server_port = find_free_port(self.server_port)  # 确保服务端口可用
        if self.model_meta:
            if self.system is None:
                self.system = get_template_meta(self.model_meta.template).default_system  # 从模板元数据填充默认 system
            if self.is_multimodal is None:
                self.is_multimodal = self.model_meta.is_multimodal  # 从模型元信息推断是否多模态
        if self.is_multimodal is None:
            self.is_multimodal = False  # 最终回退，默认为非多模态
