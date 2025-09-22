"""模块说明：
该模块定义了模型导出（export）相关的参数数据类与初始化逻辑。
- ExportArguments：继承自 `MergeArguments` 与 `BaseArguments`，用于控制量化、格式转换（HF/MC0RE/Ollama/缓存数据集）、
  以及推送到模型仓库（Hub）等导出流程的行为。
模块通过集中化的参数与校验逻辑，帮助在导出阶段自动生成输出目录、配置量化/并行、初始化分布式环境等准备工作。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
import os  # 操作系统相关：路径拼接、目录创建与存在性检查
from dataclasses import dataclass, field  # 数据类装饰器与字段工厂，简化参数定义与默认值
from typing import List, Literal, Optional  # 类型注解工具：列表/字面量/可选

import torch  # PyTorch 主包，用于张量与 dtype 设置
import torch.distributed as dist  # PyTorch 分布式通信模块

from swift.utils import get_logger, init_process_group, set_default_ddp_config  # 工具函数：日志、初始化分布式、默认 DDP 配置
from .base_args import BaseArguments, to_abspath  # 基础参数类与路径绝对化工具
from .merge_args import MergeArguments  # 合并/量化相关的参数基类

logger = get_logger()  # 初始化模块级日志器，用于统一打印


@dataclass  # 数据类装饰器，自动生成 __init__/__repr__ 等方法
class ExportArguments(MergeArguments, BaseArguments):  # 导出参数类，继承合并参数与基础参数
    """
    类说明：模型导出参数数据类，继承自 `MergeArguments` 与 `BaseArguments`，用于控制量化、
    导出格式（HF/MCore/Ollama/缓存数据集）、以及推送至模型仓库（Hub）等流程。

    主要职责：
    - 自动推断与创建导出输出目录，并进行存在性校验。
    - 约束量化配置（方法、比特数、样本/序列长度/批大小等）。
    - 在需要时初始化 PyTorch 分布式环境（DDP）。
    - 支持导出到不同生态（HuggingFace、Megatron-Core、Ollama、缓存数据集）。

    属性（部分）：
        output_dir: 导出结果保存目录；None 则按规则自动推断生成。
        quant_n_samples: 量化使用的样本数。
        max_length: 量化的序列长度上限。
        quant_batch_size: 量化时的批大小（-1 表示自动/None）。
        to_ollama: 是否导出为 Ollama 格式。
        push_to_hub: 是否将导出结果推送到模型仓库。
        hub_model_id: 推送至仓库时使用的模型仓库标识。
        hub_private_repo: 是否将仓库设为私有。
        commit_message: 推送时的提交说明。
        to_peft_format: 是否导出为 PEFT 格式（目前暂不使用）。
    """
    output_dir: Optional[str] = None  # 导出输出目录；None 表示自动推断

    # awq/gptq
    quant_method: Literal['awq', 'gptq', 'bnb', 'fp8'] = None  # 量化方法选择
    quant_n_samples: int = 256  # 量化样本数
    max_length: int = 2048  # 量化序列长度上限
    quant_batch_size: int = 1  # 量化批大小（-1 表示自动）
    group_size: int = 128  # 分组大小（部分量化方法使用）

    # cached_dataset
    to_cached_dataset: bool = False  # 是否导出为缓存数据集格式

    # ollama
    to_ollama: bool = False  # 是否导出为 Ollama 模型格式

    # megatron
    to_mcore: bool = False  # 是否导出为 Megatron-Core(MCore) 格式
    to_hf: bool = False  # 是否导出为 HuggingFace Transformers 格式
    mcore_model: Optional[str] = None  # MCore 模型路径/标识
    mcore_adapters: List[str] = field(default_factory=list)  # MCore 适配器列表
    thread_count: Optional[int] = None  # 转换/导出时的线程数量限制
    test_convert_precision: bool = False  # 是否进行转换精度的测试

    # push to ms hub
    push_to_hub: bool = False  # 是否推送到模型仓库（Hub）
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None  # 仓库标识：'user/repo' 或 'repo'
    hub_private_repo: bool = False  # 仓库是否为私有
    commit_message: str = 'update files'  # 推送时的提交说明
    # compat
    to_peft_format: bool = False  # 兼容参数：是否导出为 PEFT 格式（暂不使用）
    exist_ok: bool = False  # 若输出目录存在，是否允许覆盖/复用

    def _init_output_dir(self):  # 推断并初始化导出输出目录
        """
        函数说明：根据当前导出配置（量化/格式转换等）自动推断输出目录名，并进行绝对化与存在性检查。

        示例：
            >>> args = ExportArguments(to_ollama=True)
            >>> args.ckpt_dir = './ckpts/modelA'
            >>> args.model_suffix = 'modelA'
            >>> args._init_output_dir()
            >>> assert args.output_dir.endswith('modelA-ollama')
        """
        if self.output_dir is None:  # 若未显式指定输出目录，则自动推断
            ckpt_dir = self.ckpt_dir or f'./{self.model_suffix}'  # 以 ckpt_dir 或模型后缀为基准目录
            ckpt_dir, ckpt_name = os.path.split(ckpt_dir)  # 拆分为目录与名称部分
            if self.to_peft_format:  # 若导出为 PEFT 格式
                suffix = 'peft'  # 使用 peft 作为后缀
            elif self.quant_method:  # 若启用量化
                suffix = f'{self.quant_method}'  # 以量化方法作为后缀
                if self.quant_bits is not None:  # 若显式指定量化比特数（来自继承的参数）
                    suffix += f'-int{self.quant_bits}'  # 在后缀中追加比特数
            elif self.to_ollama:  # 若导出为 Ollama 格式
                suffix = 'ollama'  # 使用 ollama 作为后缀
            elif self.merge_lora:  # 若进行了 LoRA 权重合并
                suffix = 'merged'  # 使用 merged 作为后缀
            elif self.to_mcore:  # 若导出为 Megatron-Core 格式
                suffix = 'mcore'  # 使用 mcore 作为后缀
            elif self.to_hf:  # 若导出为 HuggingFace 格式
                suffix = 'hf'  # 使用 hf 作为后缀
            elif self.to_cached_dataset:  # 若导出为缓存数据集
                suffix = 'cached_dataset'  # 使用 cached_dataset 作为后缀
            else:  # 若以上条件均不满足，则无需初始化输出目录
                return  # 提前返回

            self.output_dir = os.path.join(ckpt_dir, f'{ckpt_name}-{suffix}')  # 组合生成输出目录路径

        self.output_dir = to_abspath(self.output_dir)  # 统一转为绝对路径
        if not self.exist_ok and os.path.exists(self.output_dir):  # 若不允许已存在且目录已存在
            raise FileExistsError(f'args.output_dir: `{self.output_dir}` already exists.')  # 抛出已存在错误
        logger.info(f'args.output_dir: `{self.output_dir}`')  # 记录最终输出目录

    def __post_init__(self):  # 数据类初始化后的钩子：量化/分布式/输出目录等初始化
        """
        函数说明：在数据类完成初始化后，继续执行导出相关的二次初始化逻辑，包括量化参数校验、
        分布式初始化、路径规范化以及必要的前置条件检查等。

        示例：
            >>> args = ExportArguments(to_hf=True)
            >>> args.mcore_model = './some_model_dir'
            >>> args.__post_init__()  # 通常由 dataclass 自动调用
        """
        if self.quant_batch_size == -1:  # -1 表示自动/不限制，转为 None 以便下游逻辑统一处理
            self.quant_batch_size = None  # 设置为 None
        if isinstance(self.mcore_adapters, str):  # 兼容单字符串输入
            self.mcore_adapters = [self.mcore_adapters]  # 统一为列表形式
        if self.quant_bits and self.quant_method is None:  # 指定了量化比特但未指定方法
            raise ValueError('Please specify the quantization method using `--quant_method awq/gptq/bnb`.')  # 抛出错误
        if self.quant_method and self.quant_bits is None and self.quant_method != 'fp8':  # 指定量化方法但未给量化比特（fp8 例外）
            raise ValueError('Please specify `--quant_bits`.')  # 抛出错误提示
        if self.quant_method in {'gptq', 'awq'} and self.torch_dtype is None:  # GPTQ/AWQ 默认使用 fp16
            self.torch_dtype = torch.float16  # 设置默认 dtype 为 float16
        if self.to_mcore or self.to_hf:  # 导出为 MCore/HF 时需要准备分布式环境
            self.mcore_model = to_abspath(self.mcore_model, check_path_exist=True)  # 规范并校验模型路径
            if not dist.is_initialized():  # 若分布式组未初始化
                set_default_ddp_config()  # 设置默认 DDP 环境变量/参数
                init_process_group(backend=self.ddp_backend, timeout=self.ddp_timeout)  # 初始化进程组

        BaseArguments.__post_init__(self)  # 调用基础参数的初始化逻辑
        self._init_output_dir()  # 初始化输出目录
        if self.quant_method in {'gptq', 'awq'} and len(self.dataset) == 0:  # 量化需要提供量化数据集
            raise ValueError(f'self.dataset: {self.dataset}, Please input the quant dataset.')  # 抛出缺少数据集的错误
        if self.to_cached_dataset:  # 导出为缓存数据集格式的限制
            if self.packing:  # packing 仅在训练时处理
                raise ValueError('Packing will be handled during training; here we only perform tokenization '
                                 'in advance, so you do not need to set up packing separately.')  # 抛出错误提示
            assert not self.streaming and not self.lazy_tokenize, 'not supported'  # 不支持流式与懒惰分词
