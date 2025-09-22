"""
模块说明：
    本模块定义了量化相关的参数数据类 `QuantizeArguments`，用于描述并构造
    不同量化方案（BitsAndBytes、HQQ、EETQ、Quanto、FP8）所需的配置信息。

    通过 `get_quantization_config` 方法，根据用户指定的量化方法与位宽等参数，
    动态生成并返回对应的 Transformers 量化配置对象；
    同时提供 `get_modules_to_not_convert` 用于返回不参与量化/转换的模块列表；
    在 `__post_init__` 中根据 `torch_dtype` 自动推导 bnb 4bit 的计算精度，
    并将其规范化为 `torch.dtype` 类型以便下游使用。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.

# 从 dataclasses 导入 dataclass，用于定义数据类并自动生成构造/表示等方法
from dataclasses import dataclass
# 从 typing 导入 Literal/Optional，用于类型限定与可选参数的类型注解
from typing import Literal, Optional

import torch  # 导入 PyTorch，用于 dtype 与张量相关的类型/常量

from swift.llm import HfConfigFactory  # 导入工厂工具，将字符串 dtype 转换为 torch.dtype


# 使用 dataclass 自动生成 __init__/__repr__/__eq__ 等方法
@dataclass
class QuantizeArguments:
    """量化参数数据类。

    用于承载模型量化相关配置，并根据不同量化后端构造对应的 Transformers
    量化配置对象。

    参数:
        quant_method: 量化方法，支持 'bnb'、'hqq'、'eetq'、'quanto'、'fp8'。
        quant_bits: 量化位宽，依据方法不同支持 1/2/3/4/8 及 'float8'。
        hqq_axis: HQQ 量化的轴方向（可选）。
        bnb_4bit_compute_dtype: BitsAndBytes 4bit 的计算精度（float16/bfloat16/float32/None）。
        bnb_4bit_quant_type: BitsAndBytes 4bit 的量化类型（'fp4' 或 'nf4'）。
        bnb_4bit_use_double_quant: BitsAndBytes 4bit 是否启用双重量化。
        bnb_4bit_quant_storage: BitsAndBytes 4bit 的权重量化存储精度（如 'float32'）。
    """
    # awq, gptq, and aqlm need to be pre-quantized models.
    #   It can be detected automatically, without the need to pass in.
    # while bnb, hqq, and eetq can be quantized during SFT using the original models.
    # 量化方法选择：bnb/hqq/eetq/quanto/fp8（awq/gptq/aqlm 通常要求预量化模型）
    quant_method: Literal['bnb', 'hqq', 'eetq', 'quanto', 'fp8'] = None
    # bnb: 4,8; hqq: 1,2,3,4,8'; eetq: 8
    # awq: 4; gptq: 2,3,4,8
    # 量化位宽；quanto 还支持 'float8'
    quant_bits: Literal[1, 2, 3, 4, 8, 'float8'] = None
    # hqq
    # HQQ 量化轴方向
    hqq_axis: Optional[int] = None
    # bnb
    # BitsAndBytes 4bit 的计算精度（若未指定，将在 __post_init__ 中按 torch_dtype 推导）
    bnb_4bit_compute_dtype: Literal['float16', 'bfloat16', 'float32', None] = None
    # BitsAndBytes 4bit 的量化类型，默认 'nf4'
    bnb_4bit_quant_type: Literal['fp4', 'nf4'] = 'nf4'
    # BitsAndBytes 4bit 是否使用双重量化，默认启用
    bnb_4bit_use_double_quant: bool = True
    # BitsAndBytes 4bit 权重量化的存储精度（可选）
    bnb_4bit_quant_storage: Optional[str] = None

    def get_quantization_config(self):
        """根据当前参数构造并返回 Transformers 的量化配置对象。

        行为:
            - None/'awq'/'gptq' 直接返回 None（此处不负责其配置）；
            - 'bnb' 返回 BitsAndBytesConfig（支持 4/8bit 加载与若干 4bit 参数）；
            - 'fp8' 返回 FineGrainedFP8Config（细粒度 FP8 配置）；
            - 'hqq' 返回 HqqConfig；
            - 'quanto' 返回 QuantoConfig（支持 int2/int4/int8/float8 权重）；
            - 'eetq' 返回 EetqConfig。

        返回:
            对应的 Transformers 量化配置对象或 None。
        """
        # 若未设置量化方法，或为 awq/gptq（通常需预量化），则不在此返回配置
        if self.quant_method is None or self.quant_method in {'awq', 'gptq'}:
            return None
        # 校验量化方法合法性
        assert self.quant_method in {'bnb', 'hqq', 'eetq', 'quanto', 'fp8'}
        # 除 FP8 外，其余量化方法需要指定量化位宽
        if self.quant_method != 'fp8' and self.quant_bits is None:
            raise ValueError(f'Please set the quant_bits. args.quant_bits: {self.quant_bits}')
        # BitsAndBytes（bnb）分支
        if self.quant_method == 'bnb':
            # 根据位宽决定 4bit/8bit 的加载方式
            if self.quant_bits == 4:
                load_in_4bit, load_in_8bit = True, False
            elif self.quant_bits == 8:
                load_in_4bit, load_in_8bit = False, True
            else:
                raise ValueError(f'bnb not support quant_bits: {self.quant_bits}')

            # 延迟导入，避免无关路径下的依赖开销
            from transformers import BitsAndBytesConfig
            # 需要跳过 int8 转换的模块（例如视觉塔/对齐器等）
            llm_int8_skip_modules = self.get_modules_to_not_convert()
            # 构造 bnb 量化配置对象
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                bnb_4bit_quant_storage=self.bnb_4bit_quant_storage,
                llm_int8_skip_modules=llm_int8_skip_modules)
        # 细粒度 FP8 分支
        elif self.quant_method == 'fp8':
            from transformers import FineGrainedFP8Config
            # 获取不转换模块列表
            modules_to_not_convert = self.get_modules_to_not_convert()
            # 构造细粒度 FP8 配置
            quantization_config = FineGrainedFP8Config(modules_to_not_convert=modules_to_not_convert)
        # HQQ 分支
        elif self.quant_method == 'hqq':
            from transformers import HqqConfig
            # 通过位宽和量化轴构造 HqqConfig
            quantization_config = HqqConfig(nbits=self.quant_bits, axis=self.hqq_axis)
        # Quanto 分支
        elif self.quant_method == 'quanto':
            from transformers import QuantoConfig
            # 将位宽映射到 Quanto 所需的权重类型字符串
            if self.quant_bits == 8:
                weights = 'int8'
            elif self.quant_bits == 'float8':
                weights = 'float8'
            elif self.quant_bits == 4:
                weights = 'int4'
            elif self.quant_bits == 2:
                weights = 'int2'
            else:
                raise ValueError('quanto quantization only support quant bits 2/4/8/float8')
            # 构造 QuantoConfig
            quantization_config = QuantoConfig(weights=weights)
        # EETQ 分支
        else:  # 'eetq'
            from transformers import EetqConfig
            # 直接通过位宽拼接 'int{bits}' 构造配置
            quantization_config = EetqConfig(f'int{self.quant_bits}')

        # 返回最终的量化配置对象
        return quantization_config

    def get_modules_to_not_convert(self):
        """返回在量化/转换过程中需要跳过的模块名称列表。

        依据 `model_meta` 与 `model_info` 的结构，收集以下模块：
            - MoE 门控相关：'mlp.gate'、'mlp.shared_expert_gate'
            - 视觉塔与对齐器：来自 `model_arch.vision_tower` 与 `model_arch.aligner`
            - 若最终非空，则追加输出头：'lm_head'

        返回:
            list[str] | None: 模块名列表；若为空则返回 None。
        """
        # 若缺少模型元信息/信息对象，则无法判定需要跳过的模块
        if not hasattr(self, 'model_meta') or not hasattr(self, 'model_info'):
            return None
        # 读取模型架构信息
        model_arch = self.model_meta.model_arch
        # 初始化待返回的模块名列表
        res = []
        # 若为 MoE 架构，跳过门控相关模块
        if self.model_info.is_moe_model:
            res += ['mlp.gate', 'mlp.shared_expert_gate']
        # 从模型架构中收集视觉塔与对齐器模块
        if model_arch is not None:
            for key in ['vision_tower', 'aligner']:
                value = getattr(model_arch, key, None)
                if value:
                    res += value
        # 若仍为空，则返回 None
        if not res:
            return None
        # 存在其它需要跳过的模块时，额外跳过输出头
        res.append('lm_head')
        # 返回最终结果
        return res

    def __post_init__(self):
        """在数据类实例化完成后进行字段规范化与默认值设定。

        行为:
            - 若未显式指定 `bnb_4bit_compute_dtype`，则依据 `torch_dtype` 推导默认值；
            - 使用 `HfConfigFactory.to_torch_dtype` 统一转换为 `torch.dtype` 类型。

        返回:
            None
        """
        # 若用户未设置 bnb 4bit 计算精度，则根据整体 dtype 推导
        if self.bnb_4bit_compute_dtype is None:
            if self.torch_dtype in {torch.float16, torch.float32}:
                self.bnb_4bit_compute_dtype = torch.float32
            elif self.torch_dtype == torch.bfloat16:
                self.bnb_4bit_compute_dtype = torch.bfloat16
        # 统一将 bnb_4bit_compute_dtype 转换为 torch.dtype
        self.bnb_4bit_compute_dtype: torch.dtype = HfConfigFactory.to_torch_dtype(self.bnb_4bit_compute_dtype)
