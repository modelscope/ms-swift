"""
模块说明：
    本模块定义了用于大语言模型（LLM）与相关下游任务的参数数据类 `ModelArguments`，
    负责组织与初始化模型加载/推理/训练所需的关键配置，包括：
    - 模型标识、修订版本、任务类型与标签数
    - 数据类型（precision）、注意力实现、RoPE 扩展与最大序列长度
    - 设备映射与显存上限、多进程/分布式训练兼容
    - 本地仓库路径、量化配置与特殊 token 管理

    通过 `__post_init__` 中的初始化流程，类会在构造完成后自动解析用户输入、
    拉取模型元信息并据此补全/校验参数，确保后续模型构建的一致性与可用性。
"""

# Copyright (c) Alibaba, Inc. and its affiliates.
# 导入 ast，用于安全地将字符串解析为 Python 字面量对象（如 dict/list/tuple 等）
import ast
# 导入数学库，用于上取整等数学运算
import math
# 导入 os，用于文件/路径存在性判断与文件读写
import os
# 从 dataclasses 导入 dataclass 与 field，用于定义数据类与字段默认行为
from dataclasses import dataclass, field
# 从 typing 导入常用类型注解构件
from typing import Any, Dict, List, Literal, Optional, Union

# 导入 json，用于 JSON 文本处理（偶尔与 ast 解析互补）
import json
# 导入 torch，用于张量与数据类型（dtype）处理
import torch
# 从 transformers 工具集中导入 MPS 可用性检测，用于苹果芯片环境判断
from transformers.utils import is_torch_mps_available

# 从 swift.llm 导入模型映射、配置工厂与模型信息获取工具
from swift.llm import MODEL_MAPPING, HfConfigFactory, get_model_info_meta, get_model_name
# 从 swift.utils 导入分布式设置、日志器与 JSON 解析工具
from swift.utils import get_dist_setting, get_logger, json_parse_to_dict

# 初始化模块级日志器
logger = get_logger()


# 使用 dataclass 自动生成 __init__/__repr__/__eq__ 等方法
@dataclass
class ModelArguments:
    """模型参数数据类。

    用于承载与模型加载、训练与推理相关的核心配置，并在实例化后自动完成：
    - 设备映射与显存上限解析（含多进程/分布式环境兼容）
    - 模型数据类型（dtype）与混合精度策略推导
    - 模型元信息查询与派生参数补全（如 `task_type`、`num_labels`、`max_model_len` 等）
    - RoPE 扩展配置与特殊 token 文件加载

    参数:
        model: 模型标识（Hub 上的 model_id）或本地模型路径。
        model_type: 模型族/类型（用于选择对应的适配逻辑），可选。
        model_revision: 模型修订版本（如 Git 分支/commit/tag），可选。
        task_type: 任务类型，包含 'causal_lm'/'seq_cls'/'embedding'/'reranker'/'generative_reranker'。
        torch_dtype: 模型参数 dtype，支持 'bfloat16'/'float16'/'float32'/None。
        attn_impl: 注意力实现后端，支持 'flash_attn'/'sdpa'/'eager' 等，None 表示自动选择。
        new_special_tokens: 新增的特殊 token 列表，或包含 token 的文本文件路径。
        num_labels: 分类任务的类别数（由模型元信息或用户指定）。
        problem_type: 分类问题类型，'regression'/'single_label_classification'/'multi_label_classification'。
        rope_scaling: RoPE 扩展配置，可为字符串类型（'linear'/'dynamic'/'yarn'）或 JSON/dict。
        device_map: 设备映射配置，字符串/JSON 或字典；支持多进程偏移对齐。
        sh: 每个设备的最大显存限制，字符串/JSON 或字典；支持多进程键偏移。
        max_model_len: 目标最大模型序列长度，用于与 RoPE 扩展配合校验。
        local_repo_path: 需要从 GitHub 获取自定义模型代码时，指定本地仓库路径。
        init_strategy: 未初始化参数的初始化策略，
            可选 'zero'/'uniform'/'normal'/'xavier_uniform'/'xavier_normal'/'kaiming_uniform'/'kaiming_normal'/'orthogonal'。
    """
    # 模型标识或路径（必填），用于定位模型
    model: Optional[str] = None  # model id or model path
    # 模型族类型，可用于选择特定适配逻辑；提供帮助信息以提示可选项
    model_type: Optional[str] = field(
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    # 模型修订版本（分支/commit/tag），可选
    model_revision: Optional[str] = None
    # 任务类型，由模型元信息推断或外部指定
    task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'reranker', 'generative_reranker'] = None

    # 模型参数 dtype，可由配置与环境共同决定
    torch_dtype: Literal['bfloat16', 'float16', 'float32', None] = None
    # flash_attn: It will automatically convert names based on the model.
    # None: It will be automatically selected between sdpa and eager.
    # 'flash_attn', 'sdpa', 'eager', 'flex_attention', 'flash_attention_2', 'flash_attention_3'
    # 注意力实现后端选择（可自动匹配不同模型的命名）
    attn_impl: Optional[str] = None
    # 新增特殊 token 列表或文件
    new_special_tokens: List[str] = field(default_factory=list)

    # 分类任务的标签数
    num_labels: Optional[int] = None
    # 分类问题类型：回归/单标签/多标签
    problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] = None
    # RoPE 扩展配置（字符串/JSON/dict），与 max_model_len 联动
    rope_scaling: Optional[str] = None
    # 设备映射配置（字符串/JSON/dict），支持 "module_name": device_id 的细粒度映射
    device_map: Optional[Union[dict, str]] = None
    # 每个设备的最大显存限制（字符串/JSON/dict）
    max_memory: Optional[Union[dict, str]] = None
    # 目标最大序列长度（若未显式指定，可由 RoPE 配置推导）
    max_model_len: Optional[int] = None
    # When some model code needs to be downloaded from GitHub,
    # this parameter specifies the path to the locally downloaded repository.
    # 需要从 GitHub 获取自定义模型代码时，指定本地仓库路径
    local_repo_path: Optional[str] = None
    # 未初始化参数的初始化策略
    init_strategy: Literal['zero', 'uniform', 'normal', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
                           'kaiming_normal', 'orthogonal'] = None

    def _init_device_map(self):
        """初始化设备映射配置。

        行为:
            - 若 `device_map` 为字符串/JSON，解析为字典或原始类型。
            - 在多进程/分布式环境（local_world_size>1）下，对整型设备号按 `local_rank` 偏移，避免冲突。

        返回:
            None
        """
        # 若用户提供了 device_map，则尝试解析为 Python 对象（字典/字符串/None）
        if self.device_map:
            # 宽松解析（支持非严格 JSON 格式），便于命令行传参
            self.device_map: Union[str, Dict[str, Any], None] = json_parse_to_dict(self.device_map, strict=False)
        # 兼容多进程（mp）与分布式（ddp）场景，获取当前进程的本地 rank 与并行规模
        _, local_rank, _, local_world_size = get_dist_setting()
        # 当为多进程且 device_map 为 dict，且非主进程（local_rank>0）时，对设备号做偏移
        if local_world_size > 1 and isinstance(self.device_map, dict) and local_rank > 0:
            # 遍历模块到设备的映射表
            for k, v in self.device_map.items():
                # 若映射值为整型设备号，则加上 local_rank 做偏移
                if isinstance(v, int):
                    self.device_map[k] += local_rank

    def _init_max_memory(self):
        """初始化每个设备的最大显存限制配置。

        行为:
            - 若 `max_memory` 为字符串，优先尝试用 `ast.literal_eval` 解析（安全）；
            - 再使用 `json_parse_to_dict` 做进一步宽松解析；
            - 在多进程场景下，若键为整型设备号，则对键做 `+local_rank` 偏移以区分进程。

        返回:
            None
        """
        # 若用户以字符串传入 max_memory，尝试转为 Python 字面量对象（如 dict）
        if isinstance(self.max_memory, str):
            try:
                # 使用 ast.literal_eval 进行安全解析（不执行任意代码）
                self.max_memory = ast.literal_eval(self.max_memory)
            except Exception:
                # 解析失败则保持原值，进入下一步宽松解析
                pass
        # 使用统一的宽松 JSON 解析工具，兼容 JSON 字符串/字典/None
        self.max_memory = json_parse_to_dict(self.max_memory)
        # 兼容多进程/分布式环境
        _, local_rank, _, local_world_size = get_dist_setting()
        # 当为多进程且为 dict，且非主进程时，对整型键（设备号）做偏移
        if local_world_size > 1 and isinstance(self.max_memory, dict) and local_rank > 0:
            # 遍历所有键，若为整型则变换为加上 local_rank 的新键
            for k in list(self.max_memory.keys()):
                if isinstance(k, int):
                    self.max_memory[k + local_rank] = self.max_memory.pop(k)

    def _init_torch_dtype(self) -> None:
        """初始化或推导模型的 `torch_dtype`，并在训练场景下推导混合精度。

        行为:
            - 将入参 `torch_dtype` 从字符串规范化为 `torch.dtype`；
            - 依据模型元信息补全最终 dtype；
            - 若当前实例也属于训练参数（TrainArguments），进一步推导混合精度开关。

        返回:
            None
        """
        # 延迟导入 TrainArguments，避免循环依赖
        from swift.llm import TrainArguments

        # 将字符串/枚举形式的 dtype 统一转换为 torch.dtype（如 'bfloat16' -> torch.bfloat16）
        self.torch_dtype: Optional[torch.dtype] = HfConfigFactory.to_torch_dtype(self.torch_dtype)
        # 结合模型元信息，得到最终的 dtype（若未明确指定）
        self.torch_dtype: torch.dtype = self._init_model_info()
        # 若当前对象同时具备训练参数特征，则初始化混合精度配置
        if isinstance(self, TrainArguments):
            self._init_mixed_precision()

    def _init_mixed_precision(self):
        """根据运行环境与 `torch_dtype` 推导 fp16/bf16 混合精度开关。

        规则:
            - 若为 Apple MPS 环境，则同时关闭 fp16/bf16；
            - 若 dtype 为 float16/float32，则启用 fp16（仅当需要）；
            - 若 dtype 为 bfloat16，则启用 bf16；
            - 否则抛出异常以提示非法的 dtype。

        返回:
            None
        """
        # 若为 macOS MPS 环境，则统一关闭混合精度开关
        if is_torch_mps_available():
            fp16, bf16 = False, False
        # 若 dtype 为 float16/float32，则偏向使用 fp16
        elif self.torch_dtype in {torch.float16, torch.float32}:
            fp16, bf16 = True, False
        # 若 dtype 为 bfloat16，则启用 bf16
        elif self.torch_dtype == torch.bfloat16:
            fp16, bf16 = False, True
        else:
            # 其他 dtype 视为非法配置
            raise ValueError(f'args.torch_dtype: {self.torch_dtype}')
        # 若外部未显式设置 fp16，则采用推导值
        if self.fp16 is None:
            self.fp16 = fp16
        # 若外部未显式设置 bf16，则采用推导值
        if self.bf16 is None:
            self.bf16 = bf16

    def _init_rope_scaling(self):
        """根据 `rope_scaling` 与 `max_model_len` 初始化 RoPE 位置编码扩展配置。

        行为:
            - 若用户提供 `rope_scaling`，解析并规范化为字典结构；
            - 否则从 `model_info` 读取默认的 RoPE 配置，并去除可能残留的 `factor`；
            - 计算原始最大长度 `origin_max_model_len`，并据此推导/校验 `factor` 与 `max_model_len`；
            - 最终回写 `self.rope_scaling` 与 `self.max_model_len`，并打印日志。

        返回:
            None
        """
        # 若用户显式提供 rope_scaling，则进行解析与校验
        if self.rope_scaling:
            # 宽松解析（支持字符串/JSON），得到字典或字符串
            rope_scaling: dict = json_parse_to_dict(self.rope_scaling, strict=False)
            if isinstance(rope_scaling, str):
                # 仅支持三种类型：'linear'/'dynamic'/'yarn'
                assert rope_scaling in ['linear', 'dynamic', 'yarn']
                # 将字符串规格化为字典形式，便于后续处理
                rope_scaling = {'type': rope_scaling}
        else:
            # 未提供则回退到模型默认 RoPE 配置
            rope_scaling = self.model_info.rope_scaling
            # reset the factor
            # 移除默认配置中的 factor，确保后续按需求重算
            rope_scaling.pop('factor', None)

        # get origin_max_model_len
        # 优先从 rope_scaling 中读取模型原始最大位置编码长度
        if rope_scaling and 'original_max_position_embeddings' in rope_scaling:
            origin_max_model_len = rope_scaling['original_max_position_embeddings']
        # 否则回退到 model_info.rope_scaling 中的同名字段
        elif self.model_info.rope_scaling and 'original_max_position_embeddings' in self.model_info.rope_scaling:
            origin_max_model_len = self.model_info.rope_scaling['original_max_position_embeddings']
        else:
            # 再次回退到模型的默认 max_model_len
            origin_max_model_len = self.model_info.max_model_len
        # 若仍无法确定，则属于模型配置不完整的严重问题
        assert origin_max_model_len is not None, '`origin_max_model_len` from model config is not set'

        # 若未指定 factor，则依据目标 max_model_len 与原始长度推导 factor
        if 'factor' not in rope_scaling:
            assert self.max_model_len is not None, '`max_model_len` or `rope_scaling_factor` is not set'
            rope_scaling['factor'] = max(float(math.ceil(self.max_model_len / origin_max_model_len)), 1.0)
        # 计算 RoPE 扩展后的可支持最大长度
        rope_model_len = int(origin_max_model_len * rope_scaling['factor'])
        # 若外部未给定 max_model_len，则直接采用 rope 扩展后的长度
        if self.max_model_len is None:
            self.max_model_len = rope_model_len
        else:
            # 否则要求 rope 配置的上限不小于外部声明的 max_model_len
            assert self.max_model_len <= rope_model_len, (
                f'rope config ({rope_model_len} = {rope_scaling["factor"]} * '
                f'{origin_max_model_len}) should be bigger than max_model_len '
                f'from command line ({self.max_model_len})')
        # 回写标准化的 rope_scaling，并打印关键信息
        self.rope_scaling = rope_scaling
        logger.info(f'Setting args.rope_scaling: {rope_scaling}')
        logger.info(f'Setting args.max_model_len: {self.max_model_len}')

    def _init_model_info(self) -> torch.dtype:
        """拉取模型元信息并据此补全若干参数，返回推荐的 `torch_dtype`。

        行为:
            - 调用 `get_model_info_meta` 获取 `model_info` 与 `model_meta`；
            - 回填 `task_type`、`num_labels`、`model_dir`、`model_type`；
            - 若存在 RoPE 相关需求则执行 `_init_rope_scaling`；
            - 返回模型推荐的 `torch_dtype`。

        返回:
            torch.dtype: 模型推荐的数据类型。
        """
        # 获取模型信息与元信息，用于填充后续配置
        self.model_info, self.model_meta = get_model_info_meta(**self.get_model_kwargs())
        # 从模型信息中回填任务类型
        self.task_type = self.model_info.task_type
        # 从模型信息中回填分类标签数
        self.num_labels = self.model_info.num_labels

        # 模型目录（本地路径或缓存路径）
        self.model_dir = self.model_info.model_dir
        # 模型族类型
        self.model_type = self.model_info.model_type
        # 若用户指定了 rope_scaling 或模型自带 rope_scaling 且显式给定了 max_model_len，则执行 rope 初始化
        if self.rope_scaling or self.model_info.rope_scaling and self.max_model_len is not None:
            self._init_rope_scaling()
        # 返回模型推荐的数据类型
        return self.model_info.torch_dtype

    def _init_new_special_tokens(self):
        """解析并展开 `new_special_tokens`。

        行为:
            - 若传入为字符串，则包装为单元素列表；
            - 遍历列表，对以 .txt 结尾的项按文件路径读取并按空白分割为多个 token；
            - 汇总得到最终的特殊 token 列表。

        返回:
            None
        """
        # 若传入为单个字符串（可能是 token 或文件路径），先转为列表统一处理
        if isinstance(self.new_special_tokens, str):
            self.new_special_tokens = [self.new_special_tokens]
        # 累积展开后的 token 列表
        new_special_tokens = []
        # 遍历用户传入的条目
        for token in self.new_special_tokens:
            # 若为文件路径（.txt），则读取文件并按空白切分为多个 token
            if token.endswith('.txt'):
                assert os.path.isfile(token), f'special_tokens_path: {token}'
                with open(token, 'r') as f:
                    text = f.read()
                new_special_tokens += text.split()
            else:
                # 否则认为其本身就是一个 token
                new_special_tokens.append(token)
        # 回写最终的新特殊 token 列表
        self.new_special_tokens = new_special_tokens

    def __post_init__(self):
        """数据类实例化完成后的自动初始化流程。

        步骤:
            1) 校验 `model` 必填；
            2) 解析/展开 `new_special_tokens`；
            3) 提取模型名后缀 `model_suffix`；
            4) 初始化设备映射与显存限制；
            5) 初始化 dtype（并在训练场景下推导混合精度）。

        返回:
            None
        """
        # `model` 为必填参数，若缺失则立即报错
        if self.model is None:
            raise ValueError(f'Please set --model <model_id_or_path>`, model: {self.model}')
        # 初始化/展开新特殊 token
        self._init_new_special_tokens()
        # 提取模型名后缀（通常用于输出路径/日志标识）
        self.model_suffix = get_model_name(self.model)
        # 初始化设备映射（含分布式兼容）
        self._init_device_map()
        # 初始化显存限制（含分布式键偏移）
        self._init_max_memory()
        # 初始化 dtype 与混合精度（若适用）
        self._init_torch_dtype()

    def get_model_kwargs(self):
        """组装用于构建模型/配置工厂的关键参数字典。

        返回:
            Dict[str, Any]: 供 `get_model_info_meta` 或模型构建使用的关键参数集合。
        """
        # 返回用于模型信息拉取与构建的关键参数集合
        return {
            # 模型标识或本地路径
            'model_id_or_path': self.model,
            # 目标 dtype（可能来自用户、也可能来自模型推荐）
            'torch_dtype': self.torch_dtype,
            # 模型族类型（可能由元信息回填）
            'model_type': self.model_type,
            # 模型修订版本
            'revision': self.model_revision,
            # 是否优先使用 HuggingFace Hub 生态
            'use_hf': self.use_hf,
            # 访问私有模型仓库需要的 Hub token
            'hub_token': self.hub_token,
            # 自定义模型代码的本地仓库路径
            'local_repo_path': self.local_repo_path,
            # 设备映射配置（字典或字符串）
            'device_map': self.device_map,
            # 每设备显存上限配置
            'max_memory': self.max_memory,
            # 量化相关配置（若外部实现提供）
            'quantization_config': self.get_quantization_config(),
            # 注意力实现后端
            'attn_impl': self.attn_impl,
            # 新增的特殊 token 列表
            'new_special_tokens': self.new_special_tokens,
            # RoPE 扩展配置
            'rope_scaling': self.rope_scaling,
            # 目标最大序列长度
            'max_model_len': self.max_model_len,
            # 任务类型
            'task_type': self.task_type,
            # 分类标签数
            'num_labels': self.num_labels,
            # 分类问题类型
            'problem_type': self.problem_type,
            # 参数初始化策略
            'init_strategy': self.init_strategy,
        }
