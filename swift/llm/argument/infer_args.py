"""模块说明：
该模块定义了模型推理（inference）相关的参数数据类与初始化逻辑。
- LmdeployArguments：lmdeploy 推理引擎的配置项与入参构造。
- SglangArguments：SGLang 推理引擎的配置项与入参构造。
- InferArguments：整合 MergeArguments/Lmdeploy/Sglang/Vllm/BaseArguments 的推理参数总控，
  负责推理结果路径、流式输出策略、分布式初始化等准备工作。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt  # 导入 datetime 模块并命名为 dt，用于生成时间戳
import os  # 导入 os，用于路径操作与目录创建
from dataclasses import dataclass  # 导入 dataclass 装饰器，简化数据类定义
from typing import Literal, Optional  # 导入类型注解工具：字面量与可选类型

import torch.distributed as dist  # 导入 PyTorch 分布式通信模块

from swift.trainers import VllmArguments  # 导入 vLLM 推理相关参数基类
from swift.utils import get_logger, init_process_group, is_dist  # 导入日志器/进程组初始化/分布式判断
from .base_args import BaseArguments, to_abspath  # 导入基础参数类与路径绝对化函数
from .merge_args import MergeArguments  # 导入合并/训练相关的参数基类

logger = get_logger()  # 初始化模块级日志器，用于统一打印


@dataclass  # 数据类装饰器，自动生成 __init__/__repr__ 等方法
class LmdeployArguments:  # lmdeploy 引擎参数类
    """
    类说明：lmdeploy 推理引擎的配置项容器，提供构造引擎入参的辅助方法。

    属性：
        lmdeploy_tp: 张量并行（TP）大小。
        lmdeploy_session_len: 会话最大长度。
        lmdeploy_cache_max_entry_count: KV 缓存最大占比/条目数比例。
        lmdeploy_quant_policy: 量化策略（如 4、8）。
        lmdeploy_vision_batch_size: 视觉分支的最大 batch 大小。
    """

    # lmdeploy
    lmdeploy_tp: int = 1  # 张量并行大小（TP）
    lmdeploy_session_len: Optional[int] = None  # 会话长度上限
    lmdeploy_cache_max_entry_count: float = 0.8  # KV 缓存最大比例
    lmdeploy_quant_policy: int = 0  # 量化策略，例如 4 或 8
    lmdeploy_vision_batch_size: int = 1  # 视觉推理的最大批大小（VisionConfig 中的 max_batch_size）

    def get_lmdeploy_engine_kwargs(self):  # 构造 lmdeploy 引擎入参字典
        """
        函数说明：将当前实例属性整合为 lmdeploy 引擎可接受的关键字参数字典。

        返回：
            Dict：lmdeploy 引擎初始化所需的关键字参数。

        示例：
            >>> args = LmdeployArguments(lmdeploy_tp=2)
            >>> kwargs = args.get_lmdeploy_engine_kwargs()
            >>> kwargs['tp']
            2
        """
        kwargs = {  # 构造引擎入参字典
            'tp': self.lmdeploy_tp,  # 张量并行规模
            'session_len': self.lmdeploy_session_len,  # 会话长度
            'cache_max_entry_count': self.lmdeploy_cache_max_entry_count,  # 缓存最大占比/条目
            'quant_policy': self.lmdeploy_quant_policy,  # 量化策略
            'vision_batch_size': self.lmdeploy_vision_batch_size  # 视觉 batch 大小
        }
        if dist.is_initialized():  # 若已在分布式环境中
            kwargs.update({'devices': [dist.get_rank()]})  # 指定使用当前 rank 对应的设备
        return kwargs  # 返回参数字典


@dataclass  # 数据类装饰器
class SglangArguments:  # SGLang 引擎参数类
    """
    类说明：SGLang 推理引擎的配置项容器，提供构造引擎入参的辅助方法。

    属性：
        sglang_tp_size: 张量并行大小。
        sglang_pp_size: 流水线并行大小。
        sglang_dp_size: 数据并行大小。
        sglang_ep_size: 专家并行大小（Expert Parallel）。
        sglang_enable_ep_moe: 是否启用 EP-MoE。
        sglang_mem_fraction_static: 静态内存占比配置。
        sglang_context_length: 上下文长度。
        sglang_disable_cuda_graph: 是否禁用 CUDA Graph。
        sglang_quantization: 量化策略配置。
        sglang_kv_cache_dtype: KV 缓存数据类型（'auto' 表示自动选择）。
        sglang_enable_dp_attention: 是否启用 DP Attention。
        sglang_disable_custom_all_reduce: 是否禁用自定义 all-reduce。
    """
    sglang_tp_size: int = 1  # 张量并行大小
    sglang_pp_size: int = 1  # 流水线并行大小
    sglang_dp_size: int = 1  # 数据并行大小
    sglang_ep_size: int = 1  # 专家并行大小
    sglang_enable_ep_moe: bool = False  # 是否开启 EP-MoE
    sglang_mem_fraction_static: Optional[float] = None  # 静态内存占比
    sglang_context_length: Optional[int] = None  # 上下文长度
    sglang_disable_cuda_graph: bool = False  # 是否禁用 CUDA Graph
    sglang_quantization: Optional[str] = None  # 量化策略字符串
    sglang_kv_cache_dtype: str = 'auto'  # KV 缓存数据类型
    sglang_enable_dp_attention: bool = False  # 是否启用 DP Attention
    sglang_disable_custom_all_reduce: bool = True  # 是否禁用自定义 all-reduce

    def get_sglang_engine_kwargs(self):  # 构造 SGLang 引擎入参字典
        """
        函数说明：将当前实例属性整合为 SGLang 引擎可接受的关键字参数字典。

        返回：
            Dict：SGLang 引擎初始化所需的关键字参数。

        示例：
            >>> args = SglangArguments(sglang_tp_size=2)
            >>> kwargs = args.get_sglang_engine_kwargs()
            >>> kwargs['tp_size']
            2
        """
        kwargs = {  # 构造引擎入参字典
            'tp_size': self.sglang_tp_size,  # 张量并行
            'pp_size': self.sglang_pp_size,  # 流水线并行
            'dp_size': self.sglang_dp_size,  # 数据并行
            'ep_size': self.sglang_ep_size,  # 专家并行
            'enable_ep_moe': self.sglang_enable_ep_moe,  # 是否启用 EP-MoE
            'mem_fraction_static': self.sglang_mem_fraction_static,  # 静态内存占比
            'context_length': self.sglang_context_length,  # 上下文长度
            'disable_cuda_graph': self.sglang_disable_cuda_graph,  # 是否禁用 CUDA Graph
            'quantization': self.sglang_quantization,  # 量化策略
            'kv_cache_dtype': self.sglang_kv_cache_dtype,  # KV 缓存数据类型
            'enable_dp_attention': self.sglang_enable_dp_attention,  # 是否启用 DP Attention
            'disable_custom_all_reduce': self.sglang_disable_custom_all_reduce,  # 是否禁用自定义 all-reduce
        }
        if self.task_type == 'embedding':  # 若任务类型为 embedding
            kwargs['task_type'] = 'embedding'  # 显式声明任务类型
        return kwargs  # 返回参数字典


@dataclass  # 数据类装饰器
class InferArguments(MergeArguments, LmdeployArguments, SglangArguments, VllmArguments, BaseArguments):  # 推理参数总控类
    """
    类说明：推理参数数据类，聚合 `MergeArguments`、`LmdeployArguments`、`SglangArguments`、`VllmArguments`、
    `BaseArguments` 的能力，用于统一管理推理所需的配置与初始化流程。

    属性（部分）：
        infer_backend: 推理后端选择，支持 'vllm'/'pt'/'sglang'/'lmdeploy'。
        result_path: 推理结果文件路径；None 表示自动生成时间戳文件。
        write_batch_size: 结果写入批大小。
        metric: 评测/验证指标（'acc' 或 'rouge'）。
        max_batch_size: 仅对 pt 引擎有效的最大 batch 大小。
        val_dataset_sample: 验证数据抽样数量。
    """
    infer_backend: Literal['vllm', 'pt', 'sglang', 'lmdeploy'] = 'pt'  # 推理后端，默认使用原生 PyTorch

    result_path: Optional[str] = None  # 推理结果文件路径；None 表示自动生成
    write_batch_size: int = 1000  # 写入结果的批大小
    metric: Literal['acc', 'rouge'] = None  # 指定度量指标
    # for pt engine
    max_batch_size: int = 1  # 原生 pt 引擎的最大 batch

    # only for inference
    val_dataset_sample: Optional[int] = None  # 验证集抽样数量

    def _get_result_path(self, folder_name: str) -> str:  # 生成带时间戳的结果文件路径
        """
        函数说明：在 checkpoint 目录或默认结果目录下，创建子目录并生成时间戳命名的 jsonl 文件路径。

        参数：
            folder_name: 子目录名，用于区分不同阶段（如 'infer_result'）。

        返回：
            str：结果文件的绝对路径。

        示例：
            >>> args = InferArguments()
            >>> path = args._get_result_path('infer_result')
            >>> path.endswith('.jsonl')
            True
        """
        result_dir = self.ckpt_dir or f'result/{self.model_suffix}'  # 使用已有 ckpt 目录或默认结果目录
        os.makedirs(result_dir, exist_ok=True)  # 确保目录存在
        result_dir = to_abspath(os.path.join(result_dir, folder_name))  # 拼接子目录并绝对化
        os.makedirs(result_dir, exist_ok=True)  # 确保子目录存在
        time = dt.datetime.now().strftime('%Y%m%d-%H%M%S')  # 生成时间戳，精确到秒
        return os.path.join(result_dir, f'{time}.jsonl')  # 返回时间戳命名的结果文件路径

    def _init_result_path(self, folder_name: str) -> None:  # 初始化结果文件路径
        """
        函数说明：若用户未指定 `result_path`，则自动生成时间戳结果文件路径并记录日志。

        参数：
            folder_name: 子目录名，用于区分不同阶段（如 'infer_result'）。

        示例：
            >>> args = InferArguments()
            >>> args._init_result_path('infer_result')
        """
        if self.result_path is not None:  # 若已指定结果路径
            self.result_path = to_abspath(self.result_path)  # 统一绝对化
            return  # 提前返回
        self.result_path = self._get_result_path(folder_name)  # 自动生成时间戳路径
        logger.info(f'args.result_path: {self.result_path}')  # 记录结果路径

    def _init_stream(self):  # 初始化流式输出策略
        """
        函数说明：根据是否有人类交互评估与束搜索配置，决定是否启用流式输出。

        示例：
            >>> args = InferArguments()
            >>> args._init_eval_human(); args._init_stream()
        """
        self.eval_human = not (self.dataset and self.split_dataset_ratio > 0 or self.val_dataset)  # 是否为交互评估
        if self.stream is None:  # 若未用户指定，则跟随交互评估开关
            self.stream = self.eval_human  # 同步设置
        if self.stream and self.num_beams != 1:  # 流式与多束搜索冲突
            self.stream = False  # 关闭流式
            logger.info('Setting args.stream: False')  # 记录关闭原因

    def _init_ddp(self):  # 初始化分布式（DDP）环境
        """
        函数说明：在分布式场景下进行必要的检查与初始化，包括禁止交互/流式、初始化设备与进程组。

        示例：
            >>> args = InferArguments()
            >>> args._init_ddp()
        """
        if not is_dist():  # 若非分布式环境
            return  # 直接返回
        eval_human = getattr(self, 'eval_human', False)  # 获取交互评估开关，默认 False
        assert not eval_human and not self.stream, (  # 分布式下不支持交互界面与流式输出
            'In DDP scenarios, interactive interfaces and streaming output are not supported.'
            f'args.eval_human: {eval_human}, args.stream: {self.stream}')
        self._init_device()  # 初始化当前进程的设备（由基类提供）
        init_process_group(backend=self.ddp_backend, timeout=self.ddp_timeout)  # 初始化分布式进程组

    def __post_init__(self) -> None:  # 数据类初始化后的钩子
        """
        函数说明：在数据类完成初始化后，依次初始化基础参数、vLLM 参数、结果路径、交互评估与 DDP 环境。

        示例：
            >>> args = InferArguments()
            >>> # 通常由 dataclass 自动调用
        """
        BaseArguments.__post_init__(self)  # 初始化基础通用参数
        VllmArguments.__post_init__(self)  # 初始化 vLLM 相关参数
        self._init_result_path('infer_result')  # 初始化推理结果路径
        self._init_eval_human()  # 初始化是否为人类交互评估
        self._init_ddp()  # 如需则初始化分布式环境

    def _init_eval_human(self):  # 初始化人类交互评估开关
        """
        函数说明：当既无训练数据集也无验证数据集时，默认开启人类交互评估模式。

        示例：
            >>> args = InferArguments()
            >>> args._init_eval_human()
        """
        if len(self.dataset) == 0 and len(self.val_dataset) == 0:  # 无训练与验证数据
            eval_human = True  # 启用交互评估
        else:  # 其它情况
            eval_human = False  # 禁用交互评估
        self.eval_human = eval_human  # 写回配置
        logger.info(f'Setting args.eval_human: {self.eval_human}')  # 记录配置
