"""模块说明：
该模块定义了用于 RLHF（基于人类反馈的强化学习）与 GRPO 等算法训练的参数数据类与初始化逻辑。
- RewardModelArguments：奖励模型相关参数。
- TeacherModelArguments：教师模型（蒸馏/对比参考）相关参数。
- PPOArguments：PPO 算法相关超参数。
- GRPOArguments：GRPO 算法相关超参数与配置。
- RLHFArguments：聚合上述参数与训练基类的总控参数，包含一系列初始化与校验函数。
通过集中化配置，保障不同 RLHF 训练范式下的行为一致与可控。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
import os  # 操作系统工具：路径拼接/存在性判断等
from dataclasses import dataclass, field  # 数据类装饰器与字段默认工厂
from typing import Any, Dict, List, Literal, Optional  # 类型注解：任意/映射/列表/字面量/可选

from swift.llm import MODEL_MAPPING  # 可用模型类型映射，用于提示/校验 model_type
from swift.trainers import GRPOArgumentsMixin, RLHFArgumentsMixin  # 训练器侧的参数混入类
from swift.utils import get_current_device, get_logger, is_master, is_mp, set_default_ddp_config  # 工具函数
from .train_args import TrainArguments  # 通用训练参数

logger = get_logger()  # 初始化模块级日志器


@dataclass  # 数据类装饰器，自动生成 __init__/__repr__ 等
class RewardModelArguments:
    """
    类说明：奖励模型（RM）相关参数集合，用于指定奖励模型与其适配器及类型信息。

    属性：
        reward_model: 奖励模型权重/ID 列表。
        reward_adapters: 奖励模型使用的适配器列表。
        reward_model_type: 奖励模型类型列表，参考 `MODEL_MAPPING` 提供的可选项。
        reward_model_revision: 奖励模型版本/修订列表。
    """
    reward_model: Optional[List[str]] = None  # 奖励模型 ID/路径列表
    reward_adapters: List[str] = field(default_factory=list)  # 奖励模型适配器列表
    reward_model_type: Optional[List[str]] = field(  # 奖励模型类型列表
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    reward_model_revision: Optional[List[str]] = None  # 奖励模型版本/修订列表


@dataclass  # 数据类装饰器
class TeacherModelArguments:
    """
    类说明：教师模型相关参数集合，用于蒸馏/对比训练场景指定教师模型与适配器等信息。

    属性：
        teacher_model: 教师模型权重/ID。
        teacher_adapters: 教师模型使用的适配器列表。
        teacher_model_type: 教师模型类型列表，参考 `MODEL_MAPPING`。
        teacher_model_revision: 教师模型版本/修订列表。
    """
    teacher_model: Optional[str] = None  # 教师模型 ID/路径
    teacher_adapters: List[str] = field(default_factory=list)  # 教师模型适配器列表
    teacher_model_type: Optional[List[str]] = field(  # 教师模型类型列表
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    teacher_model_revision: Optional[List[str]] = None  # 教师模型版本/修订列表


@dataclass  # 数据类装饰器
class PPOArguments:
    """
    类说明：PPO 算法相关的超参数定义。

    属性（部分）：
        num_ppo_epochs: 每个更新回合的 PPO 迭代次数。
        whiten_rewards: 是否对回报做白化处理。
        kl_coef: KL 惩罚系数。
        cliprange: 策略裁剪范围（策略）。
        vf_coef: 值函数损失系数。
        cliprange_value: 值函数裁剪范围。
        gamma: 折扣因子。
        lam: GAE 的 lambda。
        num_mini_batches: 每回合的 mini-batch 数。
        local_rollout_forward_batch_size: rollout 前向批大小（本地）。
        num_sample_generations: 每步生成样本次数。
        response_length: 兼容字段，建议使用 max_completion_length。
        missing_eos_penalty: 缺失 EOS 的惩罚系数。
    """
    num_ppo_epochs: int = 4  # 每回合 PPO 训练轮数
    whiten_rewards: bool = False  # 是否对回报进行白化
    kl_coef: float = 0.05  # KL 惩罚系数
    cliprange: float = 0.2  # 策略裁剪范围
    vf_coef: float = 0.1  # 值函数损失权重
    cliprange_value: float = 0.2  # 值函数裁剪范围
    gamma: float = 1.0  # 折扣因子
    lam: float = 0.95  # GAE 的 lambda 系数

    num_mini_batches: int = 1  # mini-batch 数量
    local_rollout_forward_batch_size: int = 64  # rollout 前向批大小
    num_sample_generations: int = 10  # 每步的生成次数
    response_length: Optional[int] = None  # 兼容字段，建议使用 max_completion_length
    missing_eos_penalty: Optional[float] = None  # 未生成 EOS 的惩罚


@dataclass  # 数据类装饰器
class GRPOArguments(GRPOArgumentsMixin):
    """
    类说明：GRPO 算法相关的参数定义，继承自训练器侧的 GRPOArgumentsMixin。

    属性（部分）：
        num_generations: 每个样本的生成次数（论文中的 G）。
        reward_funcs: 奖励函数名称列表。
        reward_weights: 奖励函数权重列表。
        log_completions: 是否记录生成的输出。
        use_vllm: 是否使用 vLLM 引擎进行生成。
        num_iterations: 多步迭代次数。
        truncation_strategy: 截断策略，可选 'delete'/'left'/'right'/None。
    """
    num_generations: int = 8  # 生成次数（G）
    reward_funcs: List[str] = field(default_factory=list)  # 奖励函数列表
    reward_weights: List[float] = None  # 奖励权重列表
    log_completions: bool = False  # 是否记录生成输出

    # vLLM in GRPO
    use_vllm: bool = False  # 是否启用 vLLM

    # multi step
    num_iterations: int = 1  # 多步训练迭代次数

    truncation_strategy: Literal['delete', 'left', 'right', None] = None  # 截断策略


@dataclass  # 数据类装饰器
class RLHFArguments(TeacherModelArguments, GRPOArguments, PPOArguments, RewardModelArguments, RLHFArgumentsMixin,
                    TrainArguments):
    """
    类说明：RLHF/GRPO 训练的综合参数类，聚合教师/奖励模型、PPO、GRPO 与训练通用参数，
    并提供若干初始化与校验方法以保障不同算法范式下的配置正确。

    关键属性（部分）：
        rlhf_type: 训练范式类型，'dpo'/'orpo'/'simpo'/'kto'/'cpo'/'rm'/'ppo'/'grpo'/'gkd'。
        ref_model: 参考模型（用于某些范式的对比/约束）。
        beta/label_smoothing/rpo_alpha/cpo_alpha/simpo_gamma: 不同范式的超参数。
        max_completion_length: 最大生成长度（token）。
        temperature: 采样温度（PPO/GRPO/GKD）。
        lmbda/seq_kd: GKD 相关配置。
        max_new_tokens: 兼容字段，建议使用 max_completion_length。
    """
    rlhf_type: Literal['dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm', 'ppo', 'grpo', 'gkd'] = 'dpo'
    ref_model: Optional[str] = None  # 参考模型 ID/路径
    ref_model_type: Optional[str] = field(  # 参考模型类型
        default=None, metadata={'help': f'model_type choices: {list(MODEL_MAPPING.keys())}'})
    ref_model_revision: Optional[str] = None  # 参考模型版本/修订

    beta: Optional[float] = None  # 不同范式中的 beta 参数
    label_smoothing: float = 0  # 标签平滑系数
    max_completion_length: int = 512  # 最大生成长度（token）
    loss_scale: Optional[str] = None  # 损失缩放策略，如 'last_round'
    # DPO
    rpo_alpha: float = 1.  # RPO 的 alpha 参数
    loss_type: Optional[List[str]] = None  # 损失类型（或列表）
    loss_weights: Optional[List[float]] = None  # 多损失的权重列表
    # CPO
    cpo_alpha: float = 1.  # CPO 的 alpha 参数
    # SimPO
    simpo_gamma: float = 1  # SimPO 的 gamma 参数
    # KTO
    desirable_weight: float = 1.0  # KTO：正向样本权重
    undesirable_weight: float = 1.0  # KTO：负向样本权重
    # PPO/GRPO/GKD
    temperature: float = 0.9  # 采样温度（PPO/GRPO/GKD）
    # RM
    center_rewards_coefficient: Optional[float] = None  # 奖励中心化系数（RM）
    # GKD
    lmbda: float = 0.5  # GKD：损失系数
    seq_kd: bool = False  # GKD：是否进行序列级蒸馏
    # compat
    max_new_tokens: Optional[int] = None  # 兼容字段，建议使用 max_completion_length

    def _prepare_training_args(self, training_args: Dict[str, Any]) -> None:  # 准备 Trainer 的入参
        """
        函数说明：根据 rlhf_type 调整训练器的关键参数。

        入参：
            training_args: 训练器的参数字典，将在此被修改。

        示例：
            >>> args = RLHFArguments(rlhf_type='ppo')
            >>> td = {}
            >>> args._prepare_training_args(td)
            >>> 'world_size' in td
            True
        """
        if self.rlhf_type == 'ppo':  # PPO 需要设定并行规模
            training_args['world_size'] = self.global_world_size  # 设置 world_size 为全局并行度

    def __post_init__(self):  # 数据类初始化后的钩子：集中初始化与校验
        """
        函数说明：在参数类完成初始化后，依次执行损失类型处理、兼容提醒、不同范式初始化、
        默认值设置、外部 vLLM 初始化、父类初始化与一系列安全性检查。

        示例：
            >>> args = RLHFArguments()
            >>> # 通常由 dataclass 自动调用
        """
        self._process_loss_type()  # 处理损失类型与权重格式
        self._deprecated_warning()  # 打印/处理废弃参数
        self._init_grpo()  # 初始化 GRPO 特定配置
        self._init_rm()  # 初始化 RM 特定配置
        self._init_simpo()  # 将 simpo 规范化为 cpo 等
        self._init_max_completion_length()  # 规范最大生成长度
        self._init_padding_side()  # 根据范式设置 padding 侧
        self._set_default()  # 设置默认超参数
        self._init_external_vllm()  # 初始化外部 vLLM 客户端（若启用）
        GRPOArguments.__post_init__(self)  # 调用 GRPO mixin 的后初始化
        TrainArguments.__post_init__(self)  # 调用通用训练参数的后初始化
        self._check_padding_free_sp()  # 校验 padding free 与并行限制
        self._check_grpo()  # 校验 GRPO 的外部依赖与配置
        self._external_vllm_warning()  # 外部 vLLM 的配置提醒

        if self.loss_scale is None:  # 若未明确设置损失缩放策略
            if self.rlhf_type == 'orpo' and not self.model_meta.is_multimodal:  # 单模态 ORPO 默认策略
                # Avoid padding labels during the model's forward pass in multimodal models.
                # Some multimodal models do not expand the image pad token.
                self.loss_scale = 'default'  # 设为 default
            elif self.rlhf_type == 'grpo':  # GRPO 的特殊逻辑
                if self.loss_scale is None:  # 再次判断，避免覆盖
                    if self.multi_turn_scheduler:  # 多轮调度器开启时使用 default
                        self.loss_scale = 'default'
                    else:  # 否则使用 last_round
                        self.loss_scale = 'last_round'
            else:  # 其它范式统一设置为 last_round
                self.loss_scale = 'last_round'
        if self.rlhf_type == 'grpo' and self.beta == 0.0:  # GRPO 且 beta=0 时不需要 ref_model
            self.ref_model = None  # 清空参考模型
        elif self.rlhf_type in ['dpo', 'kto', 'ppo', 'grpo'] and self.train_type == 'full':  # 全参训练需要对齐参考模型信息
            self.ref_model = self.ref_model or self.model  # 缺省取当前模型
            self.ref_model_type = self.ref_model_type or self.model_type  # 缺省对齐模型类型
            self.ref_model_revision = self.ref_model_revision or self.model_revision  # 缺省对齐修订版本
        elif self.ref_model is not None:  # 其它不需要参考模型的范式若传入则报错
            raise ValueError('CPO/ORPO or LoRA training does not require a ref_model to be passed in.')  # 抛出错误

    def _process_loss_type(self):  # 处理损失类型配置
        """
        函数说明：规范化 `loss_type` 的数据结构，并在多损失场景校验权重数量与版本依赖。

        示例：
            >>> args = RLHFArguments(rlhf_type='dpo', loss_type=['sigmoid', 'logsigmoid'], loss_weights=[0.5, 0.5])
            >>> args._process_loss_type()
        """
        if self.loss_type is None:  # 未指定损失类型则直接返回
            return  # 无需处理

        if isinstance(self.loss_type, list):  # 若为列表，检查数量与范式匹配
            num_loss_types = len(self.loss_type)  # 统计损失类型个数
            if num_loss_types > 1:  # 多损失仅支持 DPO
                assert self.rlhf_type == 'dpo', (f'Multiple loss types ({self.loss_type}) are only supported for DPO. '
                                                 f'Current rlhf_type: {self.rlhf_type}.')
                from trl.trainer.dpo_config import DPOConfig  # 动态导入以兼容版本
                assert 'loss_weights' in DPOConfig.__dict__, (
                    'Multiple loss types requires trl >= 0.20, please install trl `pip install -U trl`')

        if hasattr(self.loss_type, '__len__') and len(self.loss_type) == 1:  # 单元素列表转为标量
            self.loss_type = self.loss_type[0]  # 解包

        # Validate loss_type
        if self.loss_weights is not None:  # 如给定权重则校验长度一致
            assert self.rlhf_type == 'dpo'  # 权重仅适用于 DPO 多损失
            loss_types = self.loss_type if isinstance(self.loss_type, list) else [self.loss_type]  # 统一为列表
            if len(self.loss_weights) != len(loss_types):  # 长度不一致时报错
                raise ValueError(f'Length of loss_weights list ({self.loss_weights}) must match number of loss types '
                                 f'({loss_types}).')

    def _init_grpo(self):  # 初始化 GRPO 相关设置
        """
        函数说明：仅在 rlhf_type 为 grpo 时生效，设置 GRPO 的数据/并行/生成等相关限制与默认值。
        """
        if self.rlhf_type == 'grpo':  # 仅 GRPO 执行
            if self.cached_dataset:  # GRPO 不支持 cached_dataset
                raise ValueError('cached_dataset is not supported for GRPO.')  # 抛出错误
            if self.use_vllm:  # 使用 vLLM 时设置默认 DDP 配置
                set_default_ddp_config()  # 设置默认分布式配置
            if self.async_generate or not self.use_vllm:  # 异步生成或非 vLLM 时关闭休眠以提升吞吐
                self.sleep_level = 0  # 关闭休眠
            self.remove_unused_columns = False  # 训练数据不移除未使用列（GRPO 需要）
            logger.info(f'Setting args.remove_unused_columns: {self.remove_unused_columns}')  # 记录设置
            if self.truncation_strategy is None:  # 未指定则默认左截断
                self.truncation_strategy = 'left'  # 设为 left
            assert self.truncation_strategy in ['left', 'delete'], (
                "GRPO requires `truncation_strategy 'left' or 'delete'`, "
                f"Current value: `truncation_strategy='{self.truncation_strategy}'`.")  # noqa
            if self.beta is None:  # 未设置则给定论文推荐默认值
                self.beta = 0.04  # https://arxiv.org/abs/2402.03300
            if self.async_generate:  # 异步生成模式的提醒
                logger.info('Using async mode. This is a approximate version which '
                            'will use the old weights to generate responses to accelerate. '
                            'This will ignore the `CLIP` of advantages, if you found the training '
                            'is unstable, you may consider using --async_generate false.')
            if 'soft_overlong' in self.reward_funcs:  # 使用 soft_overlong 奖励时的参数约束
                assert self.soft_cache_length is not None, \
                    'The soft_cache_length must be set when using soft overlong rewards.'
                if self.soft_max_length is None:  # 若未设置 soft_max_length，则等于 max_completion_length
                    self.soft_max_length = self.max_completion_length  # 同步长度
                    logger.info(f'Auto-configured soft_max_length = max_completion_length {self.max_completion_length}')
            if self.use_vllm:  # vLLM 模式设置：server/colocate
                # set vllm mode
                if self.vllm_server_host is not None:  # 指定了服务端 host
                    if self.vllm_mode != 'server':  # 模式与配置不一致则调整
                        self.vllm_mode = 'server'  # 切换为 server
                        logger.warning('set vllm_mode to `server` since vllm_server_host is provided')  # 提醒
                else:  # 未指定服务端 host
                    if self.vllm_mode != 'colocate':  # 模式不一致则调整
                        self.vllm_mode = 'colocate'  # 切换为 colocate
                        logger.warning('set vllm_mode to `colocate` since vllm_server_host is not provided')  # 提醒

    def _init_padding_side(self):  # 根据范式设置 padding 侧
        """
        函数说明：对部分范式（PPO/GKD）强制设置 tokenizer 的 padding 侧为 left。
        """
        if self.rlhf_type in {'ppo', 'gkd'}:  # 指定范式
            self.padding_side = 'left'  # 设置左侧 padding
            # TODO: streaming, MLLM  # 保留待办

    def _init_max_completion_length(self):  # 规范最大生成长度
        """
        函数说明：将 response_length/max_new_tokens/max_completion_length 对齐为统一值，
        优先级依次为 response_length > max_new_tokens > max_completion_length。
        """
        max_completion_length = self.response_length or self.max_new_tokens or self.max_completion_length  # 选取优先值
        self.max_completion_length = self.max_new_tokens = self.response_length = max_completion_length  # 三者统一

    def _init_metric_for_best_model(self):  # 初始化最佳模型度量
        """
        函数说明：除 PPO/GRPO 外，沿用父类逻辑；GRPO 默认以 reward 作为最佳模型度量。
        """
        if self.rlhf_type not in {'ppo', 'grpo'}:  # 非 PPO/GRPO
            super()._init_metric_for_best_model()  # 走父类逻辑
        elif self.rlhf_type == 'grpo' and self.metric_for_best_model is None:  # GRPO 且未设置
            self.metric_for_best_model = 'reward'  # 默认以 reward 为准

    def _init_simpo(self):  # SimPO 兼容处理
        """
        函数说明：若指定为 simpo，则内部转换为 cpo，并设置默认损失与 beta。
        """
        if self.rlhf_type != 'simpo':  # 仅 simpo 执行
            return  # 直接返回

        self.rlhf_type = 'cpo'  # 转换为 cpo
        if self.loss_type is None:  # 若未设置损失
            self.loss_type = 'simpo'  # 设置为 simpo
        if self.beta is None:  # 若未设置 beta
            self.beta = 2.  # 设置为 2

    def _init_rm(self):  # RM 范式初始化
        """
        函数说明：在 RM 范式下设置任务类型与标签数。
        """
        if self.rlhf_type == 'rm':  # 仅 RM 执行
            self.task_type = 'seq_cls'  # 序列分类任务
            self.num_labels = 1  # 单回归标签

    def _init_external_vllm(self):  # 初始化外部 vLLM 客户端
        """
        函数说明：在 GRPO 且配置了 vLLM server 时，初始化客户端并只在主进程构建通信器。
        """
        if self.rlhf_type != 'grpo' or self.vllm_server_host is None:  # 仅 GRPO 且提供了 server_host 才执行
            return  # 直接返回
        from swift.trainers.rlhf_trainer.vllm_client import VLLMClient  # 延迟导入客户端类
        if is_master():  # 仅主进程构建客户端
            self.vllm_client = VLLMClient(  # 构建客户端实例
                base_urls=self.vllm_server_base_url,  # 基地址列表
                hosts=self.vllm_server_host,  # 主机列表
                server_ports=self.vllm_server_port,  # 端口列表
                connection_timeout=self.vllm_server_timeout)  # 连接超时
            self.vllm_client.init_communicator(device=get_current_device())  # 初始化通信器（绑定当前设备）

    def _set_default(self):  # 设置默认超参数
        """
        函数说明：根据 rlhf_type 设置 beta、loss_type 与梯度累计步数等默认值。
        """
        if self.beta is None:  # 未设置 beta
            if self.rlhf_type == 'gkd':  # GKD 默认 0.5
                self.beta = 0.5  # 设置为 0.5
            else:  # 其它范式默认 0.1
                self.beta = 0.1  # 设置为 0.1
        if self.loss_type is None:  # 未设置损失类型
            if self.rlhf_type in ['dpo', 'cpo']:  # DPO/CPO 默认 sigmoid
                self.loss_type = 'sigmoid'  # else None
            elif self.rlhf_type in ['kto']:  # KTO 默认 kto
                self.loss_type = 'kto'  # 设置为 kto
            elif self.rlhf_type == 'grpo':  # GRPO 默认 grpo
                self.loss_type = 'grpo'  # 设置为 grpo
        if self.gradient_accumulation_steps is None:  # 未设置梯度累计步数
            if self.rlhf_type == 'grpo':  # GRPO 默认 1
                self.gradient_accumulation_steps = 1  # 设置为 1
                logger.info('Setting default gradient_accumulation_steps to 1 for GRPO.')  # 记录日志

    def _check_grpo(self):  # 校验 GRPO 依赖与配置
        """
        函数说明：检查 trl 版本、vLLM 兼容性、Liger kernel 设置以及若干约束条件，必要时抛出异常或更正配置。
        """
        if self.rlhf_type != 'grpo':  # 非 GRPO 直接返回
            return  # 无需检查
        from packaging import version  # 版本解析工具

        import trl  # 导入 trl 以检查版本
        trl_version = version.parse(trl.__version__)  # 解析 trl 版本
        assert trl_version >= version.parse('0.17'), ('Your current version of `trl` is outdated. '
                                                      'Please update it by running: pip install -U trl')  # 版本下限
        if is_mp() and self.use_vllm:  # device_map 与 vLLM 不兼容
            raise ValueError('GRPO with vLLM is not compatible with `device_map`. '
                             'Please set NPROC_PER_NODE equal to num_processes.')
        if self.use_liger_kernel:  # 使用 liger kernel 的额外约束
            assert trl_version >= version.parse('0.18')  # 需要 trl>=0.18
            if self.delta is not None:  # 不支持 two-sided GRPO loss
                raise ValueError('Liger loss does not support two-sided GRPO loss yet.')
            if self.sequence_parallel_size > 1:  # 不支持序列并行
                raise ValueError('Liger loss does not support sequence parallel yet.')
            if self.padding_free:  # 不支持 padding free
                raise ValueError('Liger loss does not support padding free yet.')
            if self.top_entropy_quantile < 1.0:  # 不支持 entropy mask
                raise ValueError('Liger loss does not support entropy mask yet.')
            if self.log_entropy:  # 不支持记录 entropy
                raise ValueError('Liger loss does not support log entropy yet.')
            if self.importance_sampling_level != 'token':  # 仅支持 token 级重要性采样
                raise ValueError('Liger loss currently only support token-level importance sampling'
                                 'Please set `importance_sampling_level` to `token`')
            from trl.import_utils import is_liger_kernel_available  # 检查 liger kernel 是否可用
            assert is_liger_kernel_available(), (
                'Please install/update liger-kernel by running: pip install -U liger-kernel')  # 安装提醒
        if self.vllm_mode == 'server':  # server 模式必须提供 host
            assert not self.use_vllm or self.vllm_server_host is not None  # 断言 host 存在

        if self.async_generate:  # 异步生成的前置条件
            assert self.vllm_mode == 'server', 'async generate require vllm_mode == server, '
            'please deploy vLLM server by `swift rollout` and assign with `vllm_server_host` '
            'for more infomations, please check '
            'https://swift.readthedocs.io/en/latest/Instruction/GRPO/getstarted/GRPO.html'

        if not self.use_vllm and self.vllm_tensor_parallel_size != 1:  # 非 vLLM 下 TP 大小强制为 1
            self.vllm_tensor_parallel_size = 1  # 更正 TP 大小
            logger.warning('set vllm_tensor_parallel_size to 1 since use_vllm false')  # 警告提醒

        if self.async_generate and self.multi_turn_scheduler is not None:  # 异步生成不支持多轮调度
            raise NotImplementedError('Currently, async_generate is not supported with multi-turn functionality.')

        if self.generation_batch_size or self.steps_per_generation:  # 需要 trl>=0.18 的字段
            from trl.trainer.grpo_config import GRPOConfig  # 导入以做字段存在性检查
            assert 'generation_batch_size' in GRPOConfig.__dict__, (
                'generation_batch_size or steps_per_generation needs trl >= 0.18, '
                'please install trl `pip install trl>=0.18')  # 版本提醒

    def _external_vllm_warning(self):  # 外部 vLLM 配置提醒
        """
        函数说明：当使用外部 vLLM server 时，对无效配置进行提示，避免误解。
        """
        if self.rlhf_type != 'grpo' or not self.vllm_server_host:  # 非 GRPO 或未配置 server_host
            return  # 直接返回

        if self.vllm_max_model_len is not None:  # 指定了无效的本地配置项
            logger.warning(  # 打印提醒
                "Configuration conflict: 'vllm_max_model_len=%s' is ignored for external vLLM. "
                'Please specify it when launching the inference service: '
                '`swift rollout --vllm_max_model_len <value>`', self.vllm_max_model_len)

    def _deprecated_warning(self):  # 废弃参数提醒
        """
        函数说明：针对 GRPO 场景，对已废弃参数进行警告并做向后兼容处理。
        """
        if self.rlhf_type != 'grpo':  # 仅 GRPO 检查
            return  # 返回

        if self.multi_turn_func:  # multi_turn_func 已废弃
            logger.warning("The parameter 'multi_turn_func' has been deprecated and will be removed in version 3.7. "
                           "Please use 'multi_turn_scheduler' instead")  # 提醒使用新参数

            self.multi_turn_scheduler = self.multi_turn_func  # 兼容赋值

        if self.gc_collect_after_offload:  # gc_collect_after_offload 已废弃
            logger.warning(
                "The parameter 'gc_collect_after_offload' has been deprecated and will be removed in version 3.7. ")  # 打印提醒

    def _check_padding_free_sp(self):  # 检查 padding free 与 sequence parallel 支持性
        """
        函数说明：对 padding_free 与 sequence_parallel_size 的范式支持进行约束，不支持时抛出异常提示。
        """
        if self.padding_free:  # 开启 padding free 时
            supported_types = ['grpo', 'dpo', 'gkd']  # 支持的范式
            if self.rlhf_type not in supported_types:  # 若当前范式不支持
                raise NotImplementedError(f"The current rlhf_type '{self.rlhf_type}' does not support padding_free. "
                                          'Please set --padding_free to false.')  # 抛出异常
        if self.sequence_parallel_size > 1:  # 启用序列并行时
            supported_types = ['grpo', 'dpo']  # 支持的范式
            if self.rlhf_type not in supported_types:  # 不支持则报错
                raise NotImplementedError(
                    f"The current rlhf_type '{self.rlhf_type}' does not support sequence_parallel. "
                    'Please set --sequence_parallel_size to 1.')  # 抛出异常
