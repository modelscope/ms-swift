"""模块文档注释：
该脚本定义了用于数据集相关配置与处理的参数数据类 `DataArguments`，
负责描述训练/验证数据的来源、划分与下载方式、缓存与打乱策略、
列映射、模型元信息（名称/作者）及自定义数据集信息注册等能力。
主要功能包括：
- 参数声明：覆盖数据集 ID/路径、校验/划分、流式/并行、下载与缓存控制；
- 兼容与初始化：解析列映射、根据条件自动关闭拆分比例、注册自定义数据集信息；
- 映射导出：提供 `get_dataset_kwargs` 以统一导出数据管线所需关键参数。
"""

# 版权声明：本文件版权归 Alibaba 及其关联方所有
# Copyright (c) Alibaba, Inc. and its affiliates.

# 导入 dataclass 相关工具：用于声明数据类与字段默认值/元信息
from dataclasses import dataclass, field
# 导入类型注解：List 列表类型、Literal 枚举字面量、Optional 可选、Union 联合类型
from typing import List, Literal, Optional, Union

# 从 swift.llm 导入数据集映射与注册函数：用于提示可用数据集与注册自定义数据集信息
from swift.llm import DATASET_MAPPING, register_dataset_info
# 导入通用工具：日志记录器与 JSON 字符串解析为字典
from swift.utils import get_logger, json_parse_to_dict

# 创建模块级日志记录器，用于输出信息/调试日志
logger = get_logger()


# 使用 dataclass 声明数据参数容器类
@dataclass
class DataArguments:
    """数据参数容器。

    该数据类承载数据加载与预处理相关的配置项（训练/验证数据、流式下载、并行数、
    下载模式、列映射、缓存/打乱、模型元信息、自定义数据集信息等），并在初始化时
    进行必要的参数规范化与自动化调整（如根据条件关闭数据集拆分比例），同时提供
    将关键数据管线参数导出的便捷方法。

    参数：
        dataset (List[str]): 训练数据集的标识/路径/目录列表。
        val_dataset (List[str]): 验证数据集的标识/路径/目录列表。
        split_dataset_ratio (float): 若 `val_dataset` 为空时用于拆分验证集的比例；否则置 0。
        data_seed (Optional[int]): 数据随机种子（用于打乱与划分）。
        dataset_num_proc (int): 数据加载与预处理的并行进程数。
        streaming (bool): 是否启用数据集流式读取。
        download_mode (Literal): 数据下载模式（force_redownload 或 reuse_dataset_if_exists）。
        columns (Optional[Union[dict,str]]): 数据集列名映射配置（可为 JSON 字符串）。
        model_name (Optional[List[str]]): 模型的中英文名称。
        model_author (Optional[List[str]]): 模型作者的中英文名称。
        custom_dataset_info (List[str]): 自定义 dataset_info.json 的路径列表。
    """
    # 训练数据集：支持 dataset_id / 数据集路径 / 数据集目录
    dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    # 验证数据集：同上，若非空将覆盖自动拆分策略
    val_dataset: List[str] = field(
        default_factory=list, metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    # 训练集-验证集拆分比例：当 val_dataset 非空或 streaming+比例>0 时将被置 0
    split_dataset_ratio: float = 0.

    # 数据随机种子：用于打乱与划分
    data_seed: int = 42
    # 数据并行进程数
    dataset_num_proc: int = 1
    # 是否从缓存文件加载（若存在）
    load_from_cache_file: bool = True
    # 是否对训练集进行打乱
    dataset_shuffle: bool = True
    # 是否对验证集进行打乱
    val_dataset_shuffle: bool = False
    # 是否启用流式数据集
    streaming: bool = False
    # 多数据集交错采样概率（与 streaming 等配合使用）
    interleave_prob: Optional[List[float]] = None
    # 数据耗尽策略：first_exhausted 或 all_exhausted
    stopping_strategy: Literal['first_exhausted', 'all_exhausted'] = 'first_exhausted'
    # 打乱缓冲区大小
    shuffle_buffer_size: int = 1000

    # 下载模式：强制重下或复用已存在数据集
    download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists'
    # 列映射配置（dict 或 JSON 字符串）；用于手工指定列名映射
    columns: Optional[Union[dict, str]] = None
    # 严格模式（可用于列/格式校验等）
    strict: bool = False
    # 是否移除未使用列（以减少 I/O 与内存）
    remove_unused_columns: bool = True
    # 模型的中英文名称（例如 ['小黄','Xiao Huang']）
    model_name: Optional[List[str]] = field(default=None, metadata={'help': "e.g. ['小黄', 'Xiao Huang']"})
    # 模型作者的中英文名称（例如 ['魔搭','ModelScope']）
    model_author: Optional[List[str]] = field(default=None, metadata={'help': "e.g. ['魔搭', 'ModelScope']"})

    # 自定义数据集信息文件路径列表（.json），用于注册到数据集系统
    custom_dataset_info: List[str] = field(default_factory=list)  # .json

    # 初始化自定义数据集信息注册
    def _init_custom_dataset_info(self):
        """注册自定义的 dataset_info.json 到数据集系统。"""
        # 若传入为字符串则包装为列表，统一处理
        if isinstance(self.custom_dataset_info, str):
            self.custom_dataset_info = [self.custom_dataset_info]
        # 逐个路径执行注册
        for path in self.custom_dataset_info:
            register_dataset_info(path)

    # dataclass 初始化后置钩子：参数规范化与自动化调整
    def __post_init__(self):
        """在实例化后规范化参数与执行自动化调整：

        - 解析列映射配置为字典；
        - 当存在显式验证集或流式模式下设置了拆分比例时，强制关闭拆分比例；
        - 注册自定义数据集信息。
        """
        # 将 columns 从可能的 JSON 字符串解析为 dict
        self.columns = json_parse_to_dict(self.columns)
        # 若存在显式验证集，或处于流式模式且设置了拆分比例，则强制关闭拆分（置 0）
        if len(self.val_dataset) > 0 or self.streaming and self.split_dataset_ratio > 0:
            # 关闭拆分比例
            self.split_dataset_ratio = 0.
            # 记录触发关闭拆分的原因
            if len(self.val_dataset) > 0:
                msg = 'len(args.val_dataset) > 0'
            else:
                msg = 'args.streaming is True'
            # 输出到日志，说明调整原因与结果
            logger.info(f'Because {msg}, setting split_dataset_ratio: {self.split_dataset_ratio}')
        # 执行自定义数据集信息注册
        self._init_custom_dataset_info()

    # 导出数据管线所需关键参数字典
    def get_dataset_kwargs(self):
        """返回数据加载/预处理所需的关键参数映射。

        返回：
            dict: 包含数据随机种子、并行数、缓存/流式设置、下载模式、列映射、
                  模型元信息、严格模式与移除未使用列等配置项。
        """
        # 以标准字典形式返回关键参数
        return {
            # 数据随机种子
            'seed': self.data_seed,
            # 并行进程数
            'num_proc': self.dataset_num_proc,
            # 是否从缓存文件加载
            'load_from_cache_file': self.load_from_cache_file,
            # 是否启用流式读取
            'streaming': self.streaming,
            # 多数据集交错概率
            'interleave_prob': self.interleave_prob,
            # 数据耗尽策略
            'stopping_strategy': self.stopping_strategy,
            # 打乱缓冲区大小
            'shuffle_buffer_size': self.shuffle_buffer_size,
            # 是否使用 HF（由上层 BaseArguments 提供）
            'use_hf': self.use_hf,
            # Hub 访问令牌（由上层 BaseArguments 提供）
            'hub_token': self.hub_token,
            # 下载模式
            'download_mode': self.download_mode,
            # 列映射
            'columns': self.columns,
            # 严格模式
            'strict': self.strict,
            # 模型名称（中英）
            'model_name': self.model_name,
            # 模型作者（中英）
            'model_author': self.model_author,
            # 是否移除未使用列
            'remove_unused_columns': self.remove_unused_columns,
        }
