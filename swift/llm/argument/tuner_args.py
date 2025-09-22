"""
模块功能
-------
本模块集中定义与多种参数高效微调方法（如 LoRA、BOFT、Vera、Adapter、Galore、AdaLoRA、LLaMAPro、LISA、ReFT 等）
相关的超参数数据类 `TunerArguments`，并提供必要的初始化逻辑以便在不同模型结构与训练类型下
自动配置冻结/可训练参数集。

典型用法
-------
1. 通过命令行或配置文件实例化 `TunerArguments` 并与训练主参数合并；
2. 在 `__post_init__` 中会自动规范布尔型字符串、处理 target_regex 别名以及多模态全参训练的参数冻结策略；
3. 训练器据此确定可训练模块与优化器细节。

说明：本文件中的每一行代码均附有中文行内注释，帮助快速理解其作用与用途。
"""

# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，表明代码归属与授权信息
from dataclasses import dataclass, field  # 引入 dataclass 装饰器与字段工厂，便于声明参数数据类
from typing import List, Literal, Optional  # 类型注解：列表、字面量限定与可选类型

from transformers.utils import strtobool  # 将字符串形式的布尔值转换为布尔类型的工具函数

from swift.utils import get_logger  # 引入日志工具，获取模块级 logger

logger = get_logger()  # 初始化模块级日志记录器


@dataclass  # 数据类装饰器：自动生成初始化等方法
class TunerArguments:
    """
    类说明
    -----
    面向多种高效微调（PEFT）方法的统一参数集合，覆盖 LoRA/GA、BOFT、Vera、Adapter、Galore、
    AdaLoRA、LLaMAPro、LISA、ReFT 等。该类仅负责承载与轻量初始化参数，不直接执行训练。

    关键说明
    -------
    - target_modules/target_regex: 控制哪些模块参与微调；
    - 各方法前缀参数（如 lora_*、boft_*、vera_* 等）用于细化各自算法行为；
    - __post_init__ 会规范布尔字符串、处理 regex 别名与多模态全参训练下的冻结策略。

    示例
    ---
    >>> args = TunerArguments(target_modules=['all-linear'], lora_rank=8)
    >>> # 创建后交由上层 Trainer/构建器使用
    """
    # full
    freeze_parameters: List[str] = field(default_factory=list)  # 指定需冻结的精确参数名列表
    freeze_parameters_regex: Optional[str] = None  # 指定通过正则匹配需冻结的参数名模式
    freeze_parameters_ratio: float = 0.  # 0 ~ 1  # 按比例冻结参数，用于大模型选择性冻结
    trainable_parameters: List[str] = field(default_factory=list)  # 额外声明可训练的参数名列表
    trainable_parameters_regex: Optional[str] = None  # 通过正则指定额外可训练的参数名模式
    # lora or full
    freeze_llm: bool = False  # 多模态模型中是否冻结语言模型（LLM）部分
    freeze_vit: bool = True  # 多模态模型中是否冻结视觉编码器（ViT）部分
    freeze_aligner: bool = True  # 多模态模型中是否冻结对齐器（aligner）模块
    # tuners
    target_modules: List[str] = field(default_factory=lambda: ['all-linear'])  # 指定微调目标模块名；默认全部线性层
    target_regex: Optional[str] = None  # 以正则表达式形式指定微调目标模块
    # e.g. ['wte', 'ln_1', 'ln_2', 'ln_f', 'lm_head']
    modules_to_save: List[str] = field(default_factory=list)  # 训练结束后需要额外保存的模块名列表

    # lora
    lora_rank: int = 8  # LoRA 的秩（低秩分解的维度）
    lora_alpha: int = 32  # LoRA 的缩放系数 alpha
    lora_dropout: float = 0.05  # LoRA 的 dropout 比例
    lora_bias: Literal['none', 'all'] = 'none'  # LoRA 是否对 bias 进行适配
    lora_dtype: Literal['float16', 'bfloat16', 'float32', None] = None  # LoRA 训练使用的数据类型
    lorap_lr_ratio: Optional[float] = None  # LoRA+ 的学习率比例因子（相对主干学习率）
    use_rslora: bool = False  # 是否启用 RSLora 变体
    use_dora: bool = False  # 是否启用 DoRA 变体
    # Lora: Literal['gaussian', 'pissa', 'pissa_niter_[number of iters]', 'olora', 'loftq', 'true', 'false', 'lora-ga']
    lora_ga_batch_size: int = 2  # LoRA-GA 初始化时用于估计梯度的 batch 大小
    lora_ga_iters: int = 2  # LoRA-GA 初始化的迭代次数
    lora_ga_max_length: int = 1024  # LoRA-GA 初始化时输入序列的最大长度
    lora_ga_direction: str = 'ArB2r'  # LoRA-GA 初始化的方向策略
    lora_ga_scale: str = 'stable'  # LoRA-GA 初始化的缩放策略
    lora_ga_stable_gamma: int = 16  # LoRA-GA 在 stable 策略下使用的 gamma 参数

    # Bone: Literal['bat', 'true', 'false']
    init_weights: str = 'true'  # 是否初始化支持的微调器权重；使用字符串以兼容命令行解析

    # fourierft
    fourier_n_frequency: int = 2000  # FourierFT 的频率数
    fourier_scaling: float = 300.0  # FourierFT 的缩放系数

    # BOFT
    boft_block_size: int = 4  # BOFT 的块大小
    boft_block_num: int = 0  # BOFT 的块数量
    boft_n_butterfly_factor: int = 1  # BOFT 的 butterfly 因子
    boft_dropout: float = 0.0  # BOFT 的 dropout 比例

    # Vera
    vera_rank: int = 256  # Vera 的秩
    vera_projection_prng_key: int = 0  # Vera 投影的随机数种子
    vera_dropout: float = 0.0  # Vera 的 dropout 比例
    vera_d_initial: float = 0.1  # Vera 的初始 D 值

    # adapter
    adapter_act: str = 'gelu'  # Adapter 的激活函数
    adapter_length: int = 128  # Adapter 的长度（中间维度）

    # galore
    use_galore: bool = False  # 是否启用 Galore（低秩梯度投影）
    galore_target_modules: Optional[List[str]] = None  # Galore 目标模块列表
    galore_rank: int = 128  # Galore 的秩
    galore_update_proj_gap: int = 50  # Galore 投影更新间隔（步数）
    galore_scale: float = 1.0  # Galore 缩放系数
    galore_proj_type: str = 'std'  # Galore 投影类型
    galore_optim_per_parameter: bool = False  # 是否对每个参数使用独立优化配置
    galore_with_embedding: bool = False  # 是否包含 embedding 层
    galore_quantization: bool = False  # 是否启用 Q-Galore（量化）
    galore_proj_quant: bool = False  # 是否对投影矩阵进行量化
    galore_proj_bits: int = 4  # 投影量化的位数
    galore_proj_group_size: int = 256  # 量化分组大小
    galore_cos_threshold: float = 0.4  # 投影量化的余弦相似度阈值
    galore_gamma_proj: int = 2  # 投影量化的 gamma 参数
    galore_queue_size: int = 5  # 量化队列大小

    # adalora
    adalora_target_r: int = 8  # AdaLoRA 的目标秩
    adalora_init_r: int = 12  # AdaLoRA 的初始秩
    adalora_tinit: int = 0  # AdaLoRA 的初始 T 超参
    adalora_tfinal: int = 0  # AdaLoRA 的最终 T 超参
    adalora_deltaT: int = 1  # AdaLoRA 的增量 T
    adalora_beta1: float = 0.85  # AdaLoRA 的 Beta1
    adalora_beta2: float = 0.85  # AdaLoRA 的 Beta2
    adalora_orth_reg_weight: float = 0.5  # AdaLoRA 的正交正则权重

    # llamapro
    llamapro_num_new_blocks: int = 4  # LLaMAPro 新增块数量
    llamapro_num_groups: Optional[int] = None  # LLaMAPro 组数量（可选）

    # lisa
    lisa_activated_layers: int = 0  # LISA 激活的层数
    lisa_step_interval: int = 20  # LISA 激活步间隔

    # reft
    reft_layer_key: Optional[str] = None  # ReFT 层的键标识
    reft_layers: Optional[List[int]] = None  # 参与 ReFT 的层索引列表
    reft_rank: int = 4  # ReFT 的秩
    reft_intervention_type: Literal['NoreftIntervention', 'LoreftIntervention', 'ConsreftIntervention',
                                    'LobireftIntervention', 'DireftIntervention',
                                    'NodireftIntervention'] = 'LoreftIntervention'  # ReFT 的干预类型
    reft_args: Optional[str] = None  # ReFT 的额外参数字符串（JSON/CLI 形式）

    def __post_init__(self):
        """
        dataclass 初始化后钩子：完成若干参数的规范化与派生设置。

        功能
        ----
        - 将字符串形式的布尔值 `init_weights` 规范为布尔；
        - 根据多模态与训练类型，自动冻结/解冻特定模块；
        - 若提供 `target_regex`，则将其作为微调目标的别名赋值到 `target_modules`。

        示例
        ----
        >>> args = TunerArguments(init_weights='false', target_regex='.*q_proj.*')
        >>> args.__post_init__()  # 规范化 init_weights，并用正则别名覆盖 target_modules
        """
        if isinstance(self.init_weights, str) and self.init_weights.lower() in {'true', 'false'}:  # 兼容 'true'/'false' 字符串
            self.init_weights = bool(strtobool(self.init_weights))  # 使用工具规范为布尔类型
        self._init_multimodal_full()  # 依据模型结构与训练模式，自动设置冻结/可训练参数
        if self.target_regex:  # 若提供了正则表达式形式的目标
            self.target_modules = self.target_regex  # 将正则直接覆盖到 target_modules（与上游保持兼容）

    def _init_multimodal_full(self):
        """
        在多模态全参训练场景下，根据开关决定冻结哪些子模块并记录日志。

        功能
        ----
        - 仅当 is_multimodal 为真且提供了 model_arch 且 train_type 为 'full' 才生效；
        - 根据 freeze_llm/freeze_vit/freeze_aligner 将对应子模块加入冻结列表；
        - 若未冻结 aligner，则将其加入可训练列表；
        - generator 子模块默认加入冻结列表；
        - 最终打印冻结与可训练参数集合，便于审计。

        示例
        ----
        >>> # 假设 model_meta 与 model_arch 已由上游注入
        >>> args = TunerArguments(freeze_llm=True, freeze_vit=False, freeze_aligner=True)
        >>> args.train_type = 'full'
        >>> args.model_meta = type('M', (), {'is_multimodal': True, 'model_arch': type('A', (), {
        ...     'language_model': ['lm.*'], 'vision_tower': ['vit.*'], 'aligner': ['al.*'], 'generator': ['gen.*']})})()
        >>> args._init_multimodal_full()
        """
        model_arch = self.model_meta.model_arch  # 取得模型结构信息（包含各子模块的参数命名集合）
        if not self.model_meta.is_multimodal or not model_arch or self.train_type != 'full':  # 仅在多模态+全参训练时生效
            return  # 条件不满足直接返回
        if self.freeze_llm:  # 需要冻结语言模型部分
            self.freeze_parameters += model_arch.language_model  # 将 LLM 相关参数加入冻结集合
        if self.freeze_vit:  # 需要冻结视觉编码器部分
            self.freeze_parameters += model_arch.vision_tower  # 将 ViT 相关参数加入冻结集合
        if self.freeze_aligner:  # 冻结对齐器
            self.freeze_parameters += model_arch.aligner  # 将 aligner 相关参数加入冻结集合
        else:  # 不冻结对齐器
            self.trainable_parameters += model_arch.aligner  # 将 aligner 加入可训练集合
        self.freeze_parameters += model_arch.generator  # 默认冻结 generator 相关参数
        if self.freeze_parameters:  # 若存在冻结参数
            logger.info(f'freeze_parameters: {self.freeze_parameters}')  # 打印冻结参数集合
        if self.trainable_parameters:  # 若存在额外可训练参数
            logger.info(f'additional trainable_parameters: {self.trainable_parameters}')  # 打印可训练参数集合
