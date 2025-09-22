"""模块说明：
该模块定义了采样（sampling）流程的参数类与初始化逻辑。
- SamplingArguments：继承自 BaseArguments，支持纯采样、MCTS、蒸馏等多种采样策略，
  并可在本地 PT、lmdeploy、vLLM 或远端 client 模式下运行；同时规范输出目录/文件、引擎参数解析等。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
import dataclasses  # dataclasses 工具模块，提供 field/default_factory 等
import os  # 操作系统相关：路径/存在性/目录
from dataclasses import dataclass  # 数据类装饰器，简化参数定义
from datetime import datetime  # 获取当前时间用于生成文件名
from typing import List, Literal, Optional  # 类型注解：列表/字面量/可选

import json  # JSON 解析，用于解析引擎参数字符串

from swift.llm import BaseArguments  # 基础参数类，提供通用初始化与模型信息
from swift.utils import get_logger  # 获取日志器

logger = get_logger()  # 初始化模块级日志器


@dataclass  # 数据类装饰器
class SamplingArguments(BaseArguments):  # 采样参数类，继承基础参数
    """
    类说明：采样/搜索流程的参数类，支持多种采样策略与推理引擎，并负责输出文件命名与引擎参数解析。

    主要职责：
    - 定义采样策略（纯采样/MCTS/蒸馏）与引擎类型（pt/lmdeploy/vllm/no/client）。
    - 管理采样输出目录与文件名（自动生成时间戳文件）。
    - 解析引擎参数字符串为字典，便于传递给实际引擎。
    - 提供与 RM（PRM/ORM）配合的辅助阈值配置。

    属性（部分）：
        prm_model/orm_model: 过程/结果奖励模型标识。
        sampler_type: 采样类型：'sample'/'mcts'/'distill'。
        sampler_engine: 推理引擎类型：'pt'/'lmdeploy'/'vllm'/'no'/'client'。
        output_dir: 采样输出目录。
        output_file: 输出文件名（无路径），None 时自动生成为 时间戳.jsonl。
        resume/override_exist_file: 是否续跑/是否覆盖已有文件。
        num_return_sequences: 每个样本返回几条候选。
        num_sampling_per_gpu_batch_size/num_sampling_per_gpu_batches: 每 GPU 的采样批大小与批次数。
        n_best_to_keep: 保留最优候选数。
        data_range: 采样数据范围控制（索引 list）。
        temperature: 生成温度。
        prm_threshold/easy_query_threshold: PRM 阈值与“简单询问”阈值。
        engine_kwargs: 引擎关键字参数（字符串或字典）。
        cache_files: 复用缓存文件列表。
        rollout_depth/rollout_start_depth/max_iterations: MCTS 深度/起始深度/最大迭代数。
        process_reward_rate/exploration_rate: MCTS 处理奖励比例/探索率。
        api_key/base_url: client 引擎远端访问凭据与地址。
    """
    # rm models
    prm_model: Optional[str] = None  # 过程奖励模型（Process RM）标识
    orm_model: Optional[str] = None  # 结果奖励模型（Outcome RM）标识

    # sampler settings
    # sample/mcts/dvts/xxx
    sampler_type: Literal['sample', 'mcts', 'distill'] = 'sample'  # 采样策略类型
    sampler_engine: Literal['pt', 'lmdeploy', 'vllm', 'no', 'client'] = 'pt'  # 推理引擎
    output_dir: str = 'sample_output'  # 输出目录
    output_file: Optional[str] = None  # 输出文件名（仅文件名，不含路径）
    resume: bool = False  # 是否从已有进度续跑
    override_exist_file: bool = False  # 是否覆盖已存在的输出文件
    num_return_sequences: int = 64  # 每条输入返回的候选数量
    num_sampling_per_gpu_batch_size: int = 1  # 每 GPU 采样批大小
    num_sampling_per_gpu_batches: Optional[int] = None  # 每 GPU 采样批次数（None 表示按数据量推断）
    n_best_to_keep: int = 5  # 保留 top-N 候选
    data_range: List[int] = dataclasses.field(default_factory=list)  # 采样数据索引范围

    # generate settings
    temperature: float = 1.0  # 生成温度，越大越随机
    prm_threshold: float = 0.0  # PRM 阈值，用于筛选候选
    easy_query_threshold: Optional[float] = None  # 简单问题阈值（可用于跳过复杂推理）

    # engine settings
    engine_kwargs: Optional[str] = None  # 引擎额外参数（JSON 字符串或 None）

    # Vanilla
    cache_files: List[str] = dataclasses.field(default_factory=list)  # 可选缓存文件列表

    # MCTS
    rollout_depth: int = 5  # 最深搜索深度
    rollout_start_depth: int = 3  # 开始 rollout 的深度
    max_iterations: int = 100  # 最大迭代次数
    process_reward_rate: float = 0.0  # 过程奖励比例
    exploration_rate: float = 0.5  # 探索率（MCTS）
    api_key: str = 'EMPTY'  # 远端 client 调用的 API Key
    base_url: str = 'https://dashscope.aliyuncs.com/compatible-mode/v1'  # 远端 client 调用的基础 URL

    def _init_model_info(self):  # 初始化模型元信息
        """
        函数说明：当引擎为 client 时，仅设置 task_type；否则复用基类的模型信息初始化逻辑。

        示例：
            >>> args = SamplingArguments(sampler_engine='client')
            >>> args._init_model_info()
            >>> args.task_type
            'causal_lm'
        """
        if self.sampler_engine != 'client':  # 服务端引擎：交给基类处理
            return super()._init_model_info()  # 返回父类初始化结果
        self.task_type = 'causal_lm'  # client 模式下固定为 causal_lm
        return  # 显式返回 None

    def __post_init__(self):  # 数据类初始化后的钩子：处理文件名/引擎参数/基类初始化/系统消息
        """
        函数说明：
        - 自动生成无路径的时间戳输出文件名（当未显式提供时）。
        - 校验 `output_file` 不包含路径分隔符，仅接受文件名前缀。
        - 解析 `engine_kwargs` 为字典，便于下游引擎调用。
        - 设置 padding_side 与构建系统消息列表。

        示例：
            >>> args = SamplingArguments()
            >>> args.__post_init__()  # 通常由 dataclass 自动调用
        """
        if self.output_file is None:  # 未提供输出文件名
            now = datetime.now()  # 当前时间
            formatted_time = now.strftime('%Y-%m-%d-%H-%M-%S')  # 格式化时间戳
            self.output_file = formatted_time + '.jsonl'  # 生成时间戳文件名
            logger.info(f'Setting output_file to {self.output_file}')  # 记录生成的文件名
        else:  # 已提供输出文件名
            if '/' in self.output_file or '\\' in self.output_file:  # 不允许包含路径分隔符
                raise ValueError(f'Please use a string prefix without directory to '
                                 f'`--output_file` but now is: {self.output_file}')  # 抛出错误
        self.padding_side = 'left'  # 采样阶段默认左侧 padding
        if self.engine_kwargs is not None:  # 若提供了引擎参数字符串
            print(self.engine_kwargs)  # 打印原始字符串（调试用途）
            self.engine_kwargs = json.loads(self.engine_kwargs)  # 解析为字典
        else:  # 未提供
            self.engine_kwargs = {}  # 使用空字典

        super().__post_init__()  # 调用基类初始化，完成通用设置

        if self.system is not None:  # 若传入系统消息文本
            self.system_message = [{  # 构建统一的消息结构
                'role': 'system',  # 角色标识为 system
                'content': self.system,  # 消息内容为传入的 system 字符串
            }]
        else:  # 未传入系统消息文本
            self.system_message = []  # 使用空列表
