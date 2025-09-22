"""
模块功能
-------
本模块集中定义 LLM 训练相关的参数类及其初始化逻辑，封装了以下能力：

- 继承与覆盖 transformers 的 `Seq2SeqTrainingArguments`，对评估/保存策略、学习率等默认行为做定制；
- 集成插件生态（如损失函数映射、优化器工厂）与自定义 `Trainer` 获取训练参数；
- 提供 SwanLab 可选集成（实验记录与消息通知）；
- 支持 DeepSpeed 配置（含预设映射、ZeRO++/AutoTP 动态注入）；
- 处理在 PAI 平台上的兼容逻辑（日志目录、版本追加策略）；
- 统一准备输出目录、日志目录与运行名等元信息。

典型用法
-------
1. 直接通过命令行或配置文件实例化 `TrainArguments`，其内部的 `__post_init__` 会自动完成设备/分布式、
   DeepSpeed、评估策略、输出路径、SwanLab 等初始化；
2. 之后将 `TrainArguments.training_args` 交给 `TrainerFactory` 创建的训练器使用。

注意：本文件中的每一行代码都配有中文注释，便于快速理解每一处行为与目的。
"""

# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，标识归属与许可范围
import os  # 引入标准库 os，用于路径拼接与目录操作
from dataclasses import dataclass, field  # 引入 dataclass 装饰器与字段工厂，用于声明参数数据类
from typing import Literal, Optional  # 类型注解：限定字面量取值与可选类型

from transformers import Seq2SeqTrainingArguments  # 引入 HuggingFace 的序列到序列训练参数基类
from transformers.utils.versions import require_version  # 版本检测工具，用于确保可选依赖存在

from swift.plugin import LOSS_MAPPING  # 插件侧暴露的损失函数名称到实现的映射字典
from swift.trainers import TrainerFactory  # 训练器工厂，用于生成底层训练参数与 Trainer
from swift.trainers.arguments import TrainArgumentsMixin  # 训练参数混入类，提供通用训练参数行为
from swift.utils import (add_version_to_work_dir, get_device_count, get_logger, get_pai_tensorboard_dir, is_master,  # 工具方法集合
                         is_mp, is_pai_training_job, is_swanlab_available, json_parse_to_dict)  # 分布式/平台/配置解析辅助
from .base_args import BaseArguments, to_abspath  # 本包定义的基础参数与绝对路径转换工具
from .tuner_args import TunerArguments  # 微调相关参数定义

logger = get_logger()  # 获取模块级日志记录器，用于输出信息与告警


@dataclass  # 使用数据类简化参数定义与初始化
class Seq2SeqTrainingOverrideArguments(TrainArgumentsMixin, Seq2SeqTrainingArguments):
    """
    类说明
    -----
    在 transformers 的 `Seq2SeqTrainingArguments` 基础上覆盖/补充若干默认参数与便捷初始化逻辑，
    以便与 ms-swift 的训练生态配合使用（如保存/评估策略联动、指标方向自动推断等）。

    继承关系
    -------
    - TrainArgumentsMixin: 提供通用训练参数的混入能力。
    - Seq2SeqTrainingArguments: HuggingFace 标准训练参数。

    主要属性
    -------
    - output_dir: 训练输出目录，若未指定则基于 `model_suffix` 自动生成。
    - learning_rate: 学习率，按训练类型给出合理默认值。
    - eval_strategy: 评估策略，支持 'no'/'steps'/'epoch'，未设置时与保存策略对齐。
    - fp16/bf16: 半精度/混合精度开关，交由上层配置启用。

    示例
    ---
    >>> args = Seq2SeqTrainingOverrideArguments(output_dir=None, eval_strategy=None)
    >>> # args.__post_init__ 会在继承链中被调用，自动补齐 output_dir 与 eval 策略
    """
    output_dir: Optional[str] = None  # 训练产物保存的根目录；None 时自动推导
    learning_rate: Optional[float] = None  # 学习率；为空时按 train_type 设置默认值
    eval_strategy: Optional[str] = None  # steps, epoch  # 评估策略；与保存策略对齐或显式指定
    fp16: Optional[bool] = None  # 是否使用 FP16 训练
    bf16: Optional[bool] = None  # 是否使用 BF16 训练

    def _init_output_dir(self):
        """\
        初始化输出目录。

        示例
        ----
        >>> args.output_dir = None
        >>> args.model_suffix = 'qwen2'
        >>> args._init_output_dir()  # output_dir 将被设置为绝对路径 'output/qwen2'
        """
        if self.output_dir is None:  # 若未显式指定输出目录
            self.output_dir = f'output/{self.model_suffix}'  # 使用模型后缀拼接默认输出目录
        self.output_dir = to_abspath(self.output_dir)  # 统一转换为绝对路径，避免相对路径引发歧义

    def _init_eval_strategy(self):
        """
        初始化评估策略：
        - 未设置时与保存策略一致；
        - 当不评估时（'no'）禁用 eval_steps 并强制不划分验证集；
        - 当按步评估且未指定 eval_steps 时，沿用 save_steps。

        示例
        ----
        >>> args.save_strategy = 'steps'; args.save_steps = 100
        >>> args.eval_strategy = None; args.eval_steps = None
        >>> args._init_eval_strategy(); assert args.eval_steps == 100
        """
        if self.eval_strategy is None:  # 若未设置评估策略
            self.eval_strategy = self.save_strategy  # 与保存策略保持一致以简化配置
        if self.eval_strategy == 'no':  # 不进行评估
            self.eval_steps = None  # 关闭按步评估
            if self.split_dataset_ratio > 0:  # 若先前启用了划分验证集
                self.split_dataset_ratio = 0.  # 关闭数据集切分，避免产生无用验证集
                logger.info(f'Setting args.split_dataset_ratio: {self.split_dataset_ratio}')  # 记录自动调整
        elif self.eval_strategy == 'steps' and self.eval_steps is None:  # 按步评估但未设置步数
            self.eval_steps = self.save_steps  # 复用保存步数以保持节奏一致
        self.evaluation_strategy = self.eval_strategy  # 写回 transformers 期望的字段名

    def _init_metric_for_best_model(self):
        """
        初始化用于选择最佳模型的指标：
        - 生成式任务默认使用 'rouge-l'；
        - 其他任务默认使用 'loss'（越低越好）。

        示例
        ----
        >>> args.predict_with_generate = True
        >>> args.metric_for_best_model = None
        >>> args._init_metric_for_best_model(); assert args.metric_for_best_model == 'rouge-l'
        """
        if self.metric_for_best_model is None:  # 仅在未设置时推断默认指标
            self.metric_for_best_model = 'rouge-l' if self.predict_with_generate else 'loss'  # 基于是否生成式推断

    def __post_init__(self):
        """
        dataclass 初始化后钩子：补齐目录/指标与默认学习率，并联动评估策略。

        参数/返回
        --------
        无；该方法通过副作用修改实例属性。

        示例
        ----
        >>> args = Seq2SeqTrainingOverrideArguments(output_dir=None, learning_rate=None, eval_strategy=None)
        >>> args.train_type = 'full'
        >>> args.__post_init__(); assert args.learning_rate == 1e-5
        """
        self._init_output_dir()  # 规范化与准备输出目录
        self._init_metric_for_best_model()  # 设置用于选择最佳模型的指标
        if self.greater_is_better is None and self.metric_for_best_model is not None:  # 若未指定比较方向但已有指标
            self.greater_is_better = 'loss' not in self.metric_for_best_model  # 非 loss 指标通常是越大越好

        if self.learning_rate is None:  # 若未手动设置学习率
            if self.train_type == 'full':  # 全量训练默认更小学习率
                self.learning_rate = 1e-5  # 设定全量训练默认学习率
            else:  # 其他（如 LoRA/Adapter）训练可用稍大学习率
                self.learning_rate = 1e-4  # 设定微调默认学习率
        self._init_eval_strategy()  # 最后基于保存策略完善评估策略


@dataclass  # 作为纯参数承载体的数据类
class SwanlabArguments:
    """
    类说明
    -----
    封装与 SwanLab 集成所需的可选参数，包括鉴权、项目/工作空间、实验名、以及可选的飞书通知配置。

    主要属性
    -------
    - swanlab_token: 登录 token，用于非交互式鉴权。
    - swanlab_project: 项目名。
    - swanlab_workspace: 工作空间。
    - swanlab_exp_name: 实验名，默认回退为 `output_dir`。
    - swanlab_lark_webhook_url/swanlab_lark_secret: 飞书群机器人通知配置。
    - swanlab_mode: 运行模式：'cloud' 或 'local'。

    示例
    ---
    >>> args = SwanlabArguments(swanlab_project='demo', swanlab_mode='cloud')
    """

    swanlab_token: Optional[str] = None  # SwanLab 登录 token，用于自动化登录
    swanlab_project: Optional[str] = None  # SwanLab 项目名称
    swanlab_workspace: Optional[str] = None  # SwanLab 工作空间
    swanlab_exp_name: Optional[str] = None  # 实验名称；默认使用 output_dir
    swanlab_lark_webhook_url: Optional[str] = None  # 飞书通知机器人 webhook（可选）
    swanlab_lark_secret: Optional[str] = None  # 飞书机器人签名 secret（可选）
    swanlab_mode: Literal['cloud', 'local'] = 'cloud'  # 运行模式，默认为云端
    
    def _init_swanlab(self):
        """
        初始化 SwanLab 集成：检查可用性、准备实验名、完成登录与回调注册。

        参数/返回
        --------
        无；该方法通过副作用与第三方库交互。

        示例
        ----
        >>> args.report_to = ['swanlab']
        >>> args._init_swanlab()  # 完成 SwanLab 初始化（若已安装）
        """
        if not is_swanlab_available():  # 若未安装 SwanLab 包则直接报错提示安装
            raise ValueError('You are using swanlab as `report_to`, please install swanlab by ' '`pip install swanlab`')  # 明确安装指引
        if not self.swanlab_exp_name:  # 若未显式设置实验名
            self.swanlab_exp_name = self.output_dir  # 默认使用输出目录作为实验名
        from importlib import import_module  # 动态导入，避免静态检查对可选依赖报未解析警告
        INTEGRATION_TO_CALLBACK = import_module('transformers.integrations').INTEGRATION_TO_CALLBACK  # 动态获取回调注册表
        swanlab = import_module('swanlab')  # 动态导入 swanlab 主包
        SwanLabCallback = import_module('swanlab.integration.transformers').SwanLabCallback  # 动态获取 SwanLabCallback 类型
        if self.swanlab_token:  # 提供了 token 则进行无头登录
            swanlab.login(self.swanlab_token)  # 执行登录

        if self.swanlab_lark_webhook_url is not None:  # 配置了飞书通知
            LarkCallback = import_module('swanlab.plugin.notification').LarkCallback  # 动态获取飞书通知回调
            lark_callback = LarkCallback(  # 构造飞书回调对象
                webhook_url=self.swanlab_lark_webhook_url,  # 指定 webhook
                secret=self.swanlab_lark_secret,  # 指定签名 secret（可选）
            )
            swanlab.register_callbacks([lark_callback])  # 在 SwanLab 中注册该回调

        INTEGRATION_TO_CALLBACK['swanlab'] = SwanLabCallback(  # 将 'swanlab' 注册为 transformers 可识别的回调
            project=self.swanlab_project,  # SwanLab 项目
            workspace=self.swanlab_workspace,  # SwanLab 工作空间
            experiment_name=self.swanlab_exp_name,  # 实验名称
            config={'UPPERFRAME': '🐦‍⬛ms-swift'},  # 附加配置，标注上层框架来源
            mode=self.swanlab_mode,  # 运行模式（云/本地）
        )


@dataclass  # 汇总训练所需全部参数的数据类
class TrainArguments(SwanlabArguments, TunerArguments, BaseArguments, Seq2SeqTrainingOverrideArguments):
    """
    类说明
    -----
    汇集基础参数、调优参数与序列到序列训练参数的统一入口，完成从数据类到底层训练参数的桥接，
    并在初始化阶段执行一系列与平台、DeepSpeed、设备、评估策略、日志目录等相关的准备工作。

    继承关系
    -------
    - SwanlabArguments: SwanLab 集成参数。
    - TunerArguments: 微调相关参数。
    - BaseArguments: 通用基础参数（设备、数据集等）。
    - Seq2SeqTrainingOverrideArguments: 覆盖的 HF 训练参数默认逻辑。

    关键字段
    -------
    - add_version: 是否在输出目录追加版本标识（时间戳等）。
    - loss_type/metric: 插件生态中的损失与评估指标名称。
    - max_new_tokens/temperature: 推理相关的辅助参数（在训练脚本中亦可传入以便统一）。
    - zero_hpz_partition_size/deepspeed_autotp_size: DeepSpeed 相关动态注入配置。

    示例
    ---
    >>> args = TrainArguments(dataset=['/path/to/ds'], cached_dataset=[], output_dir='output/run')
    >>> # args.__post_init__ 将自动完成 DeepSpeed/设备/评估/日志等准备
    """
    add_version: bool = True  # 是否给 output_dir 动态追加版本后缀
    create_checkpoint_symlink: bool = False  # 是否创建 checkpoint 的符号链接，便于定位最新权重

    # plugin
    loss_type: Optional[str] = field(default=None, metadata={'help': f'loss_func choices: {list(LOSS_MAPPING.keys())}'})  # 自定义损失函数类型名称
    metric: Optional[str] = None  # 评估指标名称（插件侧定义）

    # extra
    max_new_tokens: int = 64  # 推理阶段默认生成的最大新标记数
    temperature: float = 0.  # 采样温度，默认为贪心（0）
    load_args: bool = False  # 是否从磁盘加载历史参数（上层可用）

    max_new_tokens: int = 64
    temperature: float = 0.
    load_args: bool = False

    # zero++
    zero_hpz_partition_size: Optional[int] = None  # ZeRO++ 分区大小，存在时注入到 DeepSpeed 配置

    # auto_tp
    deepspeed_autotp_size: Optional[int] = None  # 自动张量并行大小（AutoTP），存在时注入到 DeepSpeed 配置

    def __post_init__(self) -> None:
        """
        dataclass 初始化后钩子：串联基础/覆盖/调优参数的后初始化流程，并完成：
        - 功能检查与安全约束；
        - 设备、平台与 DeepSpeed 配置；
        - 训练参数生成与日志目录准备；
        - SwanLab 集成。

        示例
        ----
        >>> args = TrainArguments(dataset=['ds'], cached_dataset=[])
        >>> args.__post_init__()  # 内部副作用式初始化
        """
        if self.padding_free or self.packing:  # 若启用 padding_free 或样本打包
            if self.packing:  # packing 与 padding_free 互斥，优先设置 packing
                feature = 'packing'  # 记录启用的功能名
                self.padding_free = False  # packing 时显式关闭 padding_free
            else:
                feature = 'padding_free'  # 仅启用 padding_free
            if self.attn_impl not in {'flash_attn', 'flash_attention_2', 'flash_attention_3'}:  # 这两者需要 FlashAttention 支持
                raise ValueError(f'The "{feature}" feature requires a flash attention implementation. '  # 若未满足则报错提示
                                 'Please use one of: "flash_attn", "flash_attention_2", "flash_attention_3".')
        if self.resume_from_checkpoint:  # 配置了从 checkpoint 恢复
            self.resume_from_checkpoint = to_abspath(self.resume_from_checkpoint, True)  # 规范化为绝对路径（可不存在）
            # The non-resume_only_model will have its weights loaded in the trainer.  # 说明：非 resume_only_model 时权重由 trainer 处理
            if self.resume_only_model:  # 若仅恢复模型权重而非训练状态
                if self.train_type == 'full':  # 全量训练直接将模型路径赋给 model
                    self.model = self.resume_from_checkpoint  # 设置待加载的基础模型路径
                else:  # 适配参数化训练（如 LoRA）
                    self.adapters = [self.resume_from_checkpoint]  # 以适配器路径列表形式传递
        BaseArguments.__post_init__(self)  # 先初始化基础参数（设备、seed、数据相关等）
        Seq2SeqTrainingOverrideArguments.__post_init__(self)  # 再初始化覆盖的 HF 训练参数逻辑
        TunerArguments.__post_init__(self)  # 最后初始化调优相关参数

        if self.optimizer is None:  # 若未显式选择优化器
            if self.lorap_lr_ratio:  # 指定了 LoRA+ 学习率比例时，使用 lorap 优化器
                self.optimizer = 'lorap'  # 选择 lorap
            elif self.use_galore:  # 启用 GaLore 低秩优化时
                self.optimizer = 'galore'  # 选择 galore

        if len(self.dataset) == 0 and len(self.cached_dataset) == 0:  # 未提供任何训练数据
            raise ValueError(f'self.dataset: {self.dataset}, self.cached_dataset: {self.cached_dataset}. '  # 直接报错提示必须提供数据
                             'Please input the training dataset.')

        self._handle_pai_compat()  # 针对 PAI 训练作业做兼容（日志目录/版本控制）

        self._init_deepspeed()  # 初始化与解析 DeepSpeed 配置
        self._init_device()  # 初始化设备/分布式配置（由基类提供）

        if getattr(self, 'accelerator_config', None) is None:  # 若未配置加速器参数
            self.accelerator_config = {'dispatch_batches': False}  # 设置默认加速器行为（不拆批调度）
        if self.split_dataset_ratio == 0 and not self.val_dataset and not self.eval_dataset:  # 完全无验证集
            self.eval_strategy = 'no'  # 明确不进行评估
        self.training_args = TrainerFactory.get_training_args(self)  # 通过工厂基于本实例生成 HF 的 TrainingArguments
        self.training_args.remove_unused_columns = False  # 保留数据集中未被使用的列，便于自定义 collator
        self._add_version()  # 处理输出目录版本后缀与日志目录

        if 'swanlab' in self.report_to:  # 若上层要求上报到 SwanLab
            self._init_swanlab()  # 完成 SwanLab 初始化

    def _init_deepspeed(self):
        """
        初始化 DeepSpeed：
        - 支持简写名称到预设 JSON 的映射；
        - 将字符串/路径配置解析为字典；
        - 动态注入 ZeRO++ 与 AutoTP 配置项；
        - 做好与 device_map 的互斥校验。

        示例
        ----
        >>> args.deepspeed = 'zero2'  # 将映射到内置配置文件
        >>> args._init_deepspeed()
        """
        if self.deepspeed:  # 仅在启用 DeepSpeed 时处理
            require_version('deepspeed')  # 确认已安装 DeepSpeed 包
            if is_mp():  # 若启用了 device_map（MP）则与 DeepSpeed 冲突
                raise ValueError('DeepSpeed is not compatible with `device_map`. '  # 抛出明确错误与环境信息
                                 f'n_gpu: {get_device_count()}, '
                                 f'local_world_size: {self.local_world_size}.')

            ds_config_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ds_config'))  # 预设配置存放目录
            deepspeed_mapping = {  # 将友好名称映射为 JSON 文件名
                name: f'{name}.json'
                for name in ['zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload']
            }
            for ds_name, ds_config in deepspeed_mapping.items():  # 若用户传入的是上述简写名称
                if self.deepspeed == ds_name:  # 命中简写
                    self.deepspeed = os.path.join(ds_config_folder, ds_config)  # 拼接出实际 JSON 路径
                    break  # 完成映射

            self.deepspeed = json_parse_to_dict(self.deepspeed)  # 将路径/字符串解析为 dict（或保持原 dict）
            if self.zero_hpz_partition_size is not None:  # 若指定 ZeRO++ 分区大小
                assert 'zero_optimization' in self.deepspeed  # 断言配置包含 zero_optimization 节点
                self.deepspeed['zero_optimization']['zero_hpz_partition_size'] = self.zero_hpz_partition_size  # 注入参数
                logger.warn('If `zero_hpz_partition_size`(ZeRO++) causes grad_norm NaN, please'  # 给出潜在数值不稳定的提示
                            ' try `--torch_dtype float16`')
            if self.deepspeed_autotp_size is not None:  # 若启用 AutoTP
                assert self.deepspeed is not None, (  # 需要已经启用 DeepSpeed
                    'To use `deepspeed_autotp_size`, you need to additionally set the `--deepspeed` argument.')
                self.deepspeed['tensor_parallel'] = {'autotp_size': self.deepspeed_autotp_size}  # 注入张量并行配置
                self.deepspeed['zero_optimization']['gather_16bit_weights_on_model_save'] = True  # 保存时聚合 16bit 权重
            logger.info(f'Using deepspeed: {self.deepspeed}')  # 记录最终使用的 DeepSpeed 配置

    def _handle_pai_compat(self) -> None:
        """
        处理在阿里云 PAI 训练作业环境下的兼容逻辑：
        - 若检测到 PAI 环境，为 logging_dir 赋默认的 PAI TensorBoard 路径；
        - 关闭输出目录版本追加，避免路径管理复杂化。

        示例
        ----
        >>> if is_pai_training_job():
        ...     args._handle_pai_compat()
        """
        if not is_pai_training_job():  # 非 PAI 环境则直接返回
            return  # 保持本地/其他平台默认行为

        logger.info('Handle pai compat...')  # 记录开始处理 PAI 兼容
        pai_tensorboard_dir = get_pai_tensorboard_dir()  # 获取 PAI 环境默认的 TensorBoard 目录
        if self.logging_dir is None and pai_tensorboard_dir is not None:  # 未指定 logging_dir 且 PAI 提供了默认目录
            self.logging_dir = pai_tensorboard_dir  # 使用 PAI 的路径
            logger.info(f'Setting args.logging_dir: {self.logging_dir}')  # 记录变更
        self.add_version = False  # PAI 环境下通常不追加版本后缀
        logger.info(f'Setting args.add_version: {self.add_version}')  # 记录变更

    def _add_version(self):
        """
        准备输出与日志目录：
        - 需要时给 `output_dir` 追加版本信息；
        - 统一设置 `logging_dir`，并确保目录创建；
        - 将路径同步回 `training_args` 供 Trainer 使用。

        示例
        ----
        >>> args.output_dir = 'output/run'
        >>> args._add_version()  # 最终会创建目录并同步到 training_args
        """
        if self.add_version:  # 允许为输出目录追加版本（时间戳/增量号）
            self.output_dir = add_version_to_work_dir(self.output_dir)  # 生成带版本的输出目录
            logger.info(f'output_dir: {self.output_dir}')  # 记录最终输出目录

        if self.logging_dir is None:  # 若未指定日志目录
            self.logging_dir = f'{self.output_dir}/runs'  # 默认放在输出目录下的 runs 子目录

        self.logging_dir = to_abspath(self.logging_dir)  # 规范化日志目录为绝对路径
        if is_master():  # 仅主进程创建目录，避免并发冲突
            os.makedirs(self.output_dir, exist_ok=True)  # 确保输出目录存在

        if self.run_name is None:  # 若未指定 run_name
            self.run_name = self.output_dir  # 默认使用输出目录作为运行名

        self.training_args.output_dir = self.output_dir  # 同步输出目录到 HF TrainingArguments
        self.training_args.run_name = self.run_name  # 同步运行名
        self.training_args.logging_dir = self.logging_dir  # 同步日志目录
