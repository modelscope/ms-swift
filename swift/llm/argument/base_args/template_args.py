"""
模块说明：
    本模块定义了模板相关的参数数据类 `TemplateArguments`，用于描述对话/生成模板
    的选择与行为策略（如系统提示、最大长度、截断方式、是否使用聊天模板、
    模板后端等）。

    通过 `__post_init__` 在实例化后自动完成：
    - 若未显式指定 template，则回退到 `model_meta` 提供的模板；
    - 默认启用聊天模板 `use_chat_template=True`；
    - 支持将 system/response_prefix 中的 "\n" 转义为换行，或从 .txt 文件加载 system；
    - 若未指定截断策略，设置为 'delete'（内部会在使用时映射为 'raise'）。

    `get_template_kwargs` 会据此组装交给模板渲染/数据处理的关键参数字典，
    并结合是否为训练场景来确定是否移除未使用列等行为。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field  # 导入 dataclass/field，用于数据类与字段默认值/元信息
from typing import Literal, Optional  # 导入类型注解工具，限定可选值与可空类型

from swift.llm import TEMPLATE_MAPPING  # 导入可用模板映射，用于提示/校验 template 选项
from swift.utils import get_logger  # 导入日志器工厂，便于记录初始化信息

logger = get_logger()  # 初始化模块级日志器


@dataclass
class TemplateArguments:
    """模板参数数据类。

    用于承载模板相关的配置，并在实例化后完成必要的解析与默认值设定，
    以支撑训练/推理/部署阶段的模板渲染与数据处理。

    参数:
        template: 模板类型；若为 None，则使用 `model_meta` 提供的默认模板。
        system: 覆盖模板内默认的系统提示字符串；支持传入 .txt 文件路径。
        max_length: 模板/输入的最大长度，用于截断或检查。
        truncation_strategy: 截断策略，可取 'delete'/'left'/'right'/None。
        max_pixels: 图像像素上限（多模态场景）。
        agent_template: 代理/Agent 模板标识（可选）。
        norm_bbox: 归一化边框方案，'norm1000' 或 'none'。
        use_chat_template: 是否使用聊天模板而非默认生成模板。
        padding_free: 是否启用 padding-free 训练推理优化。
        padding_side: 填充方向，'left' 或 'right'。
        loss_scale: 训练损失缩放策略，'default' 表示仅计算 assistant 的损失。
        sequence_parallel_size: 序列并行的规模，用于长序列/并行优化。
        response_prefix: 响应前缀（推理/部署时前置内容），支持 "\n" 转换为换行。
        template_backend: 模板后端，使用 'swift' 或 'jinja'。
    """
    template: Optional[str] = field(
        default=None, metadata={'help': f'template choices: {list(TEMPLATE_MAPPING.keys())}'})  # 模板类型，带帮助信息以提示可选项
    system: Optional[str] = None  # 覆盖模板中的默认 system 提示（可为文本或 .txt 文件路径）
    max_length: Optional[int] = None  # 模板/输入的最大长度（用于截断/限制）

    truncation_strategy: Literal['delete', 'left', 'right', None] = None  # 截断策略，None 时后续设为 'delete'
    max_pixels: Optional[int] = None  # 图像像素上限（多模态）
    agent_template: Optional[str] = None  # 代理/Agent 模板标识
    norm_bbox: Literal['norm1000', 'none', None] = None  # 边框归一化策略
    use_chat_template: Optional[bool] = None  # 是否使用聊天模板，None 时默认 True
    # train
    padding_free: bool = False  # 训练阶段是否启用 padding-free 优化
    padding_side: Literal['left', 'right'] = 'right'  # batch_size>=2 时填充方向
    loss_scale: str = 'default'  # 损失缩放策略，'default' 表示仅计算 assistant 的损失
    sequence_parallel_size: int = 1  # 序列并行规模
    # infer/deploy
    response_prefix: Optional[str] = None  # 推理/部署时为响应添加的前缀（支持 \n 转换为换行）
    template_backend: Literal['swift', 'jinja'] = 'swift'  # 模板渲染后端（swift/jinja）

    def __post_init__(self):
        """实例化完成后的初始化与字段规范化。

        行为:
            - 若未显式指定 `template` 且存在 `model_meta`，则采用其默认模板；
            - 若 `use_chat_template` 未指定，则默认置为 True；
            - 若提供 `system`，允许从 .txt 文件加载或将字符串中的 "\\n" 转换为换行；
            - 若提供 `response_prefix`，同样转换 "\\n" 为换行；
            - 若未指定截断策略，则设为 'delete'（在使用时会映射为 'raise'）。
        """
        if self.template is None and hasattr(self, 'model_meta'):
            self.template = self.model_meta.template  # 无显式 template 时，从模型元信息回退
        if self.use_chat_template is None:
            self.use_chat_template = True  # 默认启用聊天模板
        if self.system is not None:
            if self.system.endswith('.txt'):
                assert os.path.isfile(self.system), f'self.system: {self.system}'  # 校验文件存在
                with open(self.system, 'r') as f:  # 从 .txt 文件读取 system 文本
                    self.system = f.read()  # 覆盖为文件内容
            else:
                self.system = self.system.replace('\\n', '\n')  # 将转义的 \n 替换为实际换行
        if self.response_prefix is not None:
            self.response_prefix = self.response_prefix.replace('\\n', '\n')  # 同样转换响应前缀中的换行转义
        if self.truncation_strategy is None:
            self.truncation_strategy = 'delete'  # 默认截断策略（后续在 kwargs 中映射为 'raise'）

    def get_template_kwargs(self):
        """组装模板渲染/数据处理所需的关键参数字典。

        逻辑:
            - 将 'delete' 截断策略内部映射为 'raise'；
            - 根据是否为标准训练或 RLHF GRPO 模式，决定是否移除未使用列；
            - 返回模板渲染组件所需的完整参数集合。

        返回:
            dict: 传递给模板/数据管线的关键配置。
        """
        from ..train_args import TrainArguments
        truncation_strategy = self.truncation_strategy  # 读取当前截断策略
        if truncation_strategy == 'delete':
            truncation_strategy = 'raise'  # 对外使用时将 'delete' 映射为 'raise'
        remove_unused_columns = self.remove_unused_columns  # 默认值来自 DataArguments
        if not isinstance(self, TrainArguments) or hasattr(self, 'rlhf_type') and self.rlhf_type == 'grpo':
            remove_unused_columns = True  # 非标准训练或 GRPO 训练时，强制移除未使用列
        return {
            'default_system': self.system,  # 覆盖模板默认 system 的内容
            'max_length': self.max_length,  # 输入/模板最大长度
            'truncation_strategy': truncation_strategy,  # 截断策略（已完成映射）
            'max_pixels': self.max_pixels,  # 图像像素上限
            'agent_template': self.agent_template,  # Agent 模板标识
            'norm_bbox': self.norm_bbox,  # 边框归一化策略
            'use_chat_template': self.use_chat_template,  # 是否使用聊天模板
            'remove_unused_columns': remove_unused_columns,  # 是否移除未使用列
            # train
            'padding_free': self.padding_free,  # padding-free 优化开关
            'padding_side': self.padding_side,  # 填充方向
            'loss_scale': self.loss_scale,  # 损失缩放策略
            'sequence_parallel_size': self.sequence_parallel_size,  # 序列并行规模
            # infer/deploy
            'response_prefix': self.response_prefix,  # 响应前缀（支持换行）
            'template_backend': self.template_backend,  # 模板后端（swift/jinja）
        }
