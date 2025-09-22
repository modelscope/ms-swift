"""模块说明：
该模块定义了模型合并（merge）与导出流程中通用的合并相关参数数据类。
- MergeArguments：用于控制是否合并 LoRA、是否使用 safetensors、安全序列化，以及单分片文件的最大体积等。
通过集中化的参数定义，便于在推理/导出/部署等阶段共享一致的合并行为。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass  # 数据类装饰器，简化参数类字段定义

from swift.utils import get_logger  # 获取日志器工厂函数

logger = get_logger()  # 初始化模块级日志器，用于统一打印


@dataclass  # 数据类装饰器，自动生成 __init__/__repr__ 等方法
class MergeArguments:
    """
    类说明：合并参数数据类，统一配置模型合并相关的开关与限制。

    属性：
        merge_lora: 是否合并 LoRA 权重到基座模型中，默认 False 表示不合并。
        safe_serialization: 是否使用 safetensors 格式进行安全序列化，默认 True。
        max_shard_size: 单个分片文件的最大体积限制（字符串格式，如 '5GB'）。
    """
    merge_lora: bool = False  # 是否执行 LoRA 合并（True 表示将 LoRA 权重写回基模型）
    safe_serialization: bool = True  # 是否使用 safetensors 进行安全序列化存储
    max_shard_size: str = '5GB'  # 单分片文件最大体积，超过时会进行分片
