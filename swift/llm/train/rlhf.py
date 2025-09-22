"""
脚本用途:
- 定义 RLHF（基于人类反馈的强化学习）训练流水线 `SwiftRLHF`，在 `SwiftSft` 基础上扩展以支持 KTO/GKD/PPO/GRPO 等模式。
- 负责多模型（参考/价值/奖励/教师）准备、模板模式切换、数据集适配与训练器参数注入。

主要功能:
- 自动解析模型任务类型与标签数，用于分类等下游任务配置。
- 按 RLHF 类型准备 ref/reward/value/teacher 模型，并按需冻结或适配。
- 为 GRPO/PPO 等模式设置模板与停止 token，准备奖励模板与函数。
- 覆盖数据集准备与训练器参数注入逻辑。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
import os  # 标准库：文件与路径操作
from typing import List, Optional, Union  # 类型提示：列表、可选、联合类型

from swift.llm import safe_snapshot_download  # 项目内：安全快照下载，解析并缓存模型文件
from swift.utils import get_logger, get_model_parameter_info  # 通用工具：日志与模型参数统计
from ..argument import BaseArguments, RLHFArguments  # 参数定义：基础参数与 RLHF 专用参数
from ..model import HfConfigFactory  # 模型配置工厂：便捷设置 HF 配置属性
from .kto import prepare_kto_dataset  # 数据集适配：KTO 模式的数据预处理
from .sft import SwiftSft  # 基类：SFT 训练流水线

logger = get_logger()  # 初始化模块级日志记录器


class SwiftRLHF(SwiftSft):
    """RLHF 训练流水线。

    作用:
        - 扩展 SFT 流水线以支持 RLHF 相关训练模式（kto/gkd/ppo/grpo 等）。
        - 统一准备参考模型（ref）、奖励模型（reward）、价值模型（value）与教师模型（teacher），
          并在不同模式下决定其加载、冻结、并行与模板设置等行为。

    属性:
        args_class: 指定该类使用的参数类型（`RLHFArguments`）。
        args: 运行时注入的训练参数实例，类型为 `args_class`。
    """
    args_class = RLHFArguments  # 指定参数类型为 RLHFArguments
    args: args_class  # 类型注解：实例属性 args 的类型为 args_class

    @staticmethod
    def _get_model_task_type(model_dir):
        """根据模型目录推断任务类型与标签数。

        参数:
            model_dir: 模型目录（可包含 `args.json` 或 HF 配置）。

        返回:
            (task_type, num_labels): 任务类型与标签数，若无法判断任务类型则返回 (None, num_labels)。
        """
        task_type = None  # 初始化任务类型为空
        num_labels = None  # 初始化标签数为空
        if os.path.exists(os.path.join(model_dir, 'args.json')):  # 若存在保存的训练参数
            model_args = BaseArguments.from_pretrained(model_dir)  # 加载参数文件
            if hasattr(model_args, 'task_type'):  # 若参数中包含任务类型
                task_type = model_args.task_type  # 读取任务类型
            if hasattr(model_args, 'num_labels'):  # 若参数中包含标签数
                num_labels = model_args.num_labels  # 读取标签数
            if task_type == 'seq_cls' and num_labels is None:  # 序列分类但未提供标签数
                num_labels = 1  # 回退为 1（如回归/二分类特例）
        else:  # 无 args.json 时，从 HF 配置推断
            from transformers import AutoConfig  # 延迟导入，避免不必要依赖
            model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)  # 读取模型配置
            if hasattr(model_config, 'architectures') and model_config.architectures:  # 根据结构名推断
                if any('sequenceclassification' in arch.lower() for arch in model_config.architectures):  # 包含分类架构
                    task_type = 'seq_cls'  # 设置为序列分类
                    num_labels = getattr(model_config, 'num_labels', None) or 1  # 优先取配置，否则回退为 1

            if task_type is None:  # 若仍未确定任务类型
                if hasattr(model_config, 'num_labels'):  # 尝试仅从标签数推断
                    num_labels = model_config.num_labels  # 读取标签数
                    # PretrainedConfig default num_labels = 2
                    if num_labels == 1:  # 若标签数为 1，推断为序列分类（如回归）
                        task_type = 'seq_cls'
        return task_type, num_labels  # 返回推断结果

    def _prepare_single_model(self, key, origin_key, model_type, model_revision):
        """准备单个模型（ref/reward/value/teacher）及其处理器与适配器。

        参数:
            key: 目标模型键名（'ref'/'reward'/'value'/'teacher'）。
            origin_key: 调用来源键名，用于判定冻结策略；为空时回退为 key。
            model_type: 模型类型（用于工厂方法）。
            model_revision: 模型版本/修订号。

        返回:
            (model, processor): 准备好的模型与处理器；若未提供模型路径则返回 None。
        """
        from swift.llm.infer.utils import prepare_adapter  # 延迟导入：准备适配器（如 LoRA/Adapter）
        args = self.args  # 读取训练参数
        origin_key = origin_key or key  # 规范化来源键名
        model_id_or_path = getattr(args, f'{key}_model')  # 读取模型路径或 ID
        if model_id_or_path is None:  # 未配置则跳过
            return  # 返回 None
        if isinstance(model_id_or_path, list):  # PPO 的 value 可能为列表
            # value model in PPO
            model_id_or_path = model_id_or_path[0]  # 仅取第一个路径

        model_dir = safe_snapshot_download(  # 解析模型快照目录（可跳过权重下载）
            model_id_or_path=model_id_or_path,
            revision=model_revision,
            download_model=False,
            use_hf=args.use_hf,
            hub_token=args.hub_token,
        )
        task_type, num_labels = self._get_model_task_type(model_dir)  # 推断任务类型与标签数
        model, processor = args.get_model_processor(  # 构建模型与处理器
            model=model_id_or_path,
            model_type=model_type,
            model_revision=model_revision,
            task_type=task_type,
            num_labels=num_labels)

        adapters = args.adapters if key == 'ref' else args.reward_adapters  # 选择适配器配置来源
        model = prepare_adapter(args, model, adapters)  # 为模型装配适配器
        if origin_key in {'ref', 'reward', 'teacher'}:  # 非训练主模型：仅推理使用
            if self.args.sequence_parallel_size > 1:  # 序列并行场景需准备模型
                from swift.trainers.sequence_parallel import sequence_parallel  # 延迟导入并行工具
                sequence_parallel.prepare_model(model, processor)  # 按并行策略准备模型与处理器
            model.requires_grad_(False).eval()  # 冻结参数并切换到评估模式
        else:  # 训练用 value 模型
            model = self.prepare_model(args, model, task_type=task_type)  # 走训练前准备流程（混入 Tuner）
            logger.info(f'value_model: {model}')  # 打印 value 模型摘要
            model_parameter_info = get_model_parameter_info(model)  # 统计参数信息
            self.train_msg['value_model_parameter_info'] = model_parameter_info  # 记录到训练消息
            logger.info(f'value_model_parameter_info: {model_parameter_info}')  # 输出统计信息

        HfConfigFactory.set_model_config_attr(model, 'use_cache', False)  # 关闭生成缓存以避免显存/行为问题
        return model, processor  # 返回模型与处理器

    def _prepare_model_tokenizer(self):
        """准备模型与处理器，覆盖以加载 RLHF 相关的多个模型。"""
        if self.args.sequence_parallel_size > 1:  # 若开启序列并行
            # Duplicate calling is allowd to promise this function will
            # be called before model initializing.
            from swift.trainers.sequence_parallel import sequence_parallel  # 延迟导入并行工具
            sequence_parallel.init_sequence_parallel(self.args.sequence_parallel_size)  # 初始化序列并行环境
        # prepare ref/reward/value model
        args = self.args  # 读取训练参数
        # Handle ref and value models
        for key in ['ref', 'value', 'teacher']:  # 处理三类模型
            setattr(self, f'{key}_model', None)  # 先清空对应属性
            if key == 'ref' and args.rlhf_type == 'gkd':  # GKD 模式下无需 ref
                continue  # 跳过准备
            if key == 'value' and args.rlhf_type != 'ppo':  # 非 PPO 下无需 value
                continue  # 跳过准备
            if key == 'teacher' and args.rlhf_type != 'gkd':  # 非 GKD 下无需 teacher
                continue  # 跳过准备
            model_key = 'reward' if key == 'value' else key  # value 使用 reward 的配置键
            model_type = getattr(args, f'{model_key}_model_type')  # 读取模型类型
            model_revision = getattr(args, f'{model_key}_model_revision')  # 读取模型修订
            if key == 'value':  # value 的类型/修订可能为列表
                model_type = model_type[0] if model_type else None  # 取首个元素或 None
                model_revision = model_revision[0] if model_revision else None  # 取首个元素或 None

            result = self._prepare_single_model(model_key, key, model_type, model_revision)  # 准备单模型
            if result is not None:  # 若成功返回
                model, _ = result  # 忽略处理器（此处仅存放模型）
                setattr(self, f'{key}_model', model)  # 绑定到实例属性

        # Handle reward model(s)
        self.reward_model = None  # 初始化奖励模型容器
        if hasattr(args, 'reward_model') and args.reward_model is not None:  # 若配置了奖励模型
            rms = args.reward_model if isinstance(args.reward_model, list) else [args.reward_model]  # 统一为列表
            num_rms = len(rms)  # 奖励模型数量
            rm_types = args.reward_model_type if args.reward_model_type else [None] * num_rms  # 类型列表
            rm_revisions = args.reward_model_revision if args.reward_model_revision else [None] * num_rms  # 修订列表
            assert len(rms) == len(rm_types) == len(rm_revisions)  # 保证一一对应

            self.reward_model = []  # 收集奖励模型
            if args.rlhf_type == 'grpo':  # GRPO 需要奖励模板
                self.reward_template = []  # 收集奖励模板
            for reward_model_path, rm_type, rm_revision in zip(rms, rm_types, rm_revisions):  # 逐个准备奖励模型
                args.reward_model = reward_model_path  # Temporarily set for prepare_single_model  # 临时写入以复用准备逻辑
                result = self._prepare_single_model('reward', None, rm_type, rm_revision)  # 准备 reward 模型
                if result is not None:  # 成功则收集
                    model, processor = result  # 同时拿到处理器
                    self.reward_model.append(model)  # 追加奖励模型

                    if args.rlhf_type == 'grpo':  # GRPO 需要单独的奖励模板
                        reward_template = self.args.get_template(processor, processor.model_meta.template)  # 基于处理器模板构建
                        if reward_template.use_model:  # 若模板需要绑定模型
                            reward_template.model = model  # 注入模型
                        self.reward_template.append(reward_template)  # 收集模板
                args.reward_model = rms  # Restore original value  # 恢复原始列表配置
                if args.rlhf_type != 'grpo' and self.reward_model:  # 非 GRPO 限制奖励模型数量
                    assert len(self.reward_model) <= 1  # 最多 1 个
                    self.reward_model = self.reward_model[0]  # 若仅 1 个则解包为单对象

        super()._prepare_model_tokenizer()  # 继续父类逻辑（加载主模型与生成配置等）

    def _prepare_template(self) -> None:
        """准备模板并根据 RLHF 类型切换模板模式。"""
        args = self.args  # 读取训练参数
        super()._prepare_template()  # 先按父类逻辑创建模板
        model_mapping = {'kto': 'kto', 'gkd': 'gkd', 'ppo': 'pt', 'grpo': 'pt'}  # RLHF 类型到模板模式映射
        self.template.set_mode(model_mapping.get(args.rlhf_type, 'rlhf'))  # 切换模板模式

        if args.rlhf_type == 'ppo':  # PPO 需要设置停止 token
            args.training_args.stop_token_id = self.template.template_meta.stop_token_id  # 从模板元数据读取

    def _get_dataset(self):
        """获取并按 RLHF 类型适配训练/验证数据集。"""
        args = self.args  # 读取训练参数
        train_dataset, val_dataset = super()._get_dataset()  # 先用父类加载/切分数据集
        if args.rlhf_type == 'kto':  # KTO 模式需要特殊的数据格式
            train_dataset, val_dataset = prepare_kto_dataset(args, train_dataset, val_dataset)  # 适配为 KTO 所需结构
        return train_dataset, val_dataset  # 返回数据集

    def _get_trainer_kwargs(self):
        """为训练器构造关键字参数，注入 RLHF 相关模型与资源。"""
        trainer_kwargs = {}  # 初始化参数字典
        for key in ['ref', 'reward', 'value', 'teacher']:  # 遍历四类模型
            key = f'{key}_model'  # 属性名（如 ref_model）
            model = getattr(self, key, None)  # 读取实例属性
            if model or self.args.rlhf_type == 'ppo' and key != 'teacher_model':  # PPO 可能需要 None 占位
                trainer_kwargs[key] = model  # 注入模型
        if hasattr(self, 'reward_template'):  # GRPO 可能需要奖励模板列表
            trainer_kwargs['reward_template'] = self.reward_template  # 注入奖励模板
        if self.args.rlhf_type == 'grpo':  # GRPO 需注入奖励函数与 vLLM 客户端
            trainer_kwargs['reward_funcs'] = self.args.reward_funcs  # 奖励函数集合
            trainer_kwargs['vllm_client'] = self.args.vllm_client  # vLLM 客户端用于批量推理
        return trainer_kwargs  # 返回训练器关键字参数


def rlhf_main(args: Optional[Union[List[str], RLHFArguments]] = None):
    """RLHF 训练入口。

    参数:
        args: 训练参数，可为命令行参数列表（List[str]）或 `RLHFArguments` 实例；
              为空时从环境或默认配置中解析。

    返回:
        任意类型。通常为训练主流程的返回值（由 `SwiftSft.main` 定义）。
    """
    return SwiftRLHF(args).main()  # 基于传入参数构建 SwiftRLHF 实例并执行主流程
