"""
脚本用途:
- 定义并组织多种参数高效微调（PEFT）与适配器策略（LoRA/AdaLoRA/LongLoRA/Unsloth/LoRA-GA/LLaMAPro/Adapter/Vera/BOFT/FourierFT/ReFT/Bone），
  提供目标模块解析、需保存模块列表、Vera 特定目标模块筛选以及 Liger Kernel 应用等能力。
- 通过 `TunerMixin` 提供统一的模型准备入口，按 `args.train_type` 与后端选择（swift/peft/unsloth/extra_tuners）
  对模型进行冻结/解冻、适配器注入、断点恢复、GaLore 优化、序列并行准备等操作。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect  # 标准库：检查函数签名等反射能力
import os  # 标准库：文件与路径操作
from typing import List, Union  # 类型提示：列表与联合类型

import torch  # 第三方：PyTorch 深度学习框架
import torch.nn as nn  # 第三方：神经网络模块
import transformers  # 第三方：Hugging Face Transformers
from packaging import version  # 第三方：版本比较工具
from transformers import TrainingArguments  # 第三方：训练参数定义

from swift.llm import TrainArguments, deep_getattr  # 项目内：训练参数与安全属性访问工具
from swift.plugin import Tuner, extra_tuners  # 插件：自定义 Tuner 与额外 tuner 集合
from swift.tuners import Swift  # 项目内：适配器注入入口（prepare_model/from_pretrained）
from swift.utils import activate_parameters, find_all_linears, find_embedding, find_norm, freeze_parameters, get_logger  # 工具函数

logger = get_logger()  # 初始化模块级日志记录器


def apply_liger(model_type: str):
    """按模型类型应用 Liger Kernel 加速实现。

    参数:
        model_type: 模型类型标识（见 `swift.llm.ModelType`）。
    """
    from liger_kernel.transformers import (apply_liger_kernel_to_llama, apply_liger_kernel_to_mistral,  # 延迟导入以避免无需求时带来依赖
                                           apply_liger_kernel_to_mixtral, apply_liger_kernel_to_gemma,
                                           apply_liger_kernel_to_qwen2, apply_liger_kernel_to_qwen3,
                                           apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl,
                                           apply_liger_kernel_to_phi3, apply_liger_kernel_to_mllama)
    from swift.llm import ModelType  # 导入模型类型枚举
    if model_type in (ModelType.llama, ModelType.llama3, ModelType.llama3_1, ModelType.llama3_2):  # LLaMA 系列
        apply_liger_kernel_to_llama()  # 应用 Liger 内核到 LLaMA
    elif model_type in (ModelType.mistral):  # Mistral
        apply_liger_kernel_to_mistral()
    elif model_type in (ModelType.mixtral):  # Mixtral
        apply_liger_kernel_to_mixtral()
    elif model_type in (ModelType.gemma, ModelType.gemma2):  # Gemma/Gemma2
        apply_liger_kernel_to_gemma()
    elif model_type in (ModelType.qwen2, ModelType.qwen2_5):  # Qwen2 系列
        apply_liger_kernel_to_qwen2()
    elif model_type in (ModelType.qwen3):  # Qwen3
        apply_liger_kernel_to_qwen3()
    elif model_type in (ModelType.phi3):  # Phi3
        apply_liger_kernel_to_phi3()
    elif model_type in (ModelType.llama3_2_vision):  # LLaMA3.2 Vision
        apply_liger_kernel_to_mllama()
    elif model_type in (ModelType.qwen2_vl):  # Qwen2-VL
        apply_liger_kernel_to_qwen2_vl()
    elif model_type in (ModelType.qwen2_5_vl):  # Qwen2.5-VL
        apply_liger_kernel_to_qwen2_5_vl()
    else:  # 未支持类型
        raise ValueError(f'Unsupported liger model_type: {model_type}')  # 抛出错误提示


def get_multimodal_target_regex(
    model,
    *,
    freeze_llm: bool = False,
    freeze_vit: bool = True,
    freeze_aligner: bool = True,
    include_embedding: bool = False,
) -> str:
    """为多模态模型生成目标模块的正则表达式，便于按需冻结/训练不同子模块。

    参数:
        model: 多模态模型实例，要求含 `model_meta.model_arch` 描述。
        freeze_llm: 是否冻结语言模型部分。
        freeze_vit: 是否冻结视觉塔部分。
        freeze_aligner: 是否冻结对齐模块部分。
        include_embedding: 是否将 Embedding 层包含进目标模块。

    返回:
        正则表达式字符串，用于匹配需要训练的模块路径。
    """
    model_arch = model.model_meta.model_arch  # 读取模型结构元信息
    modules = []  # 待训练的顶层模块路径集合
    if not freeze_llm:  # 需要训练 LLM
        modules += model_arch.language_model  # 加入语言模型子模块路径列表
    if not freeze_vit:  # 需要训练视觉塔
        modules += model_arch.vision_tower  # 加入视觉子模块路径
    if not freeze_aligner:  # 需要训练对齐器
        modules += model_arch.aligner  # 加入对齐器子模块路径
    assert len(modules) > 0, f'modules: {modules}'  # 至少应包含一个训练目标模块

    extra_layers = []  # 额外考虑的层类型
    if include_embedding:  # 需要包含 Embedding 层
        extra_layers.append(nn.Embedding)  # 纳入 Embedding 类型
    res = []  # 收集各模块的匹配正则
    for module in modules:  # 遍历顶层模块路径
        rejected_modules = []  # 需要排除的模块列表
        if not freeze_vit:  # 如果视觉塔参与训练
            for aligner in model_arch.aligner:  # 对齐器模块路径
                if aligner.startswith(f'{module}.'):  # 若对齐器位于该顶层模块之下
                    rejected_modules.append(aligner)  # 加入排除列表，避免重复训练

        sub_module = deep_getattr(model, module)  # 获取子模块对象
        target_modules = find_all_linears(sub_module, model_arch, extra_layers)  # 查找线性层（含额外层）
        target_modules = [tm for tm in target_modules if tm]  # 过滤空字符串
        target_pattern = rf'.*\.({"|".join(target_modules)})' if target_modules else ''  # 目标线性层匹配
        rejected_pattern = rf'(?!({"|".join(rejected_modules)}))' if rejected_modules else ''  # 排除匹配
        res.append(rf'{rejected_pattern}{module}{target_pattern}')  # 组合完整匹配片段

    return rf'^({"|".join(res)})$'  # 汇总为最终正则表达式


def get_target_modules(args, model) -> Union[str, List[str]]:
    """将占位符 `all-linear`/`all-embedding` 展开为实际模块名称或返回原始字符串。

    参数:
        args: 训练参数，包含目标模块配置与多模态冻结策略。
        model: 当前模型实例，用于扫描线性层/嵌入层。

    返回:
        目标模块名称列表或正则表达式字符串。
    """
    model_meta = model.model_meta  # 模型元信息
    if isinstance(args.target_modules, str):  # 若直接给定了正则/字符串
        return args.target_modules  # 原样返回
    target_modules = args.target_modules.copy()  # 复制一份以免修改原配置
    if 'all-linear' in target_modules:  # 需要展开所有线性层
        if model_meta.is_multimodal:  # 多模态模型走正则生成流程
            return get_multimodal_target_regex(  # 返回匹配多模态线性层的正则
                model,
                freeze_llm=args.freeze_llm,
                freeze_vit=args.freeze_vit,
                freeze_aligner=args.freeze_aligner,
                include_embedding='all-embedding' in target_modules)
        else:  # 单模态模型直接扫描线性层
            target_modules.remove('all-linear')  # 移除占位符
            target_modules += find_all_linears(model)  # 追加所有线性层名称
    if 'all-embedding' in target_modules:  # 需要包含所有 Embedding 层
        target_modules.remove('all-embedding')  # 移除占位符
        target_modules += find_embedding(model)  # 追加嵌入层名称
    return target_modules  # 返回展开后的目标模块列表


def get_modules_to_save(args, model, task_type=None):
    """展开需保存模块配置，支持 all-embedding/all-norm，并在分类任务下增加 `v_head`。

    参数:
        args: 训练参数，包含 `modules_to_save`。
        model: 当前模型，用于扫描嵌入与归一化层。
        task_type: 可选任务类型，序列分类时追加 `v_head`。

    返回:
        展开的需保存模块名称列表。
    """
    modules_to_save = args.modules_to_save.copy()  # 复制配置以免副作用
    if 'all-embedding' in args.modules_to_save:  # 展开所有嵌入层
        modules_to_save.remove('all-embedding')  # 移除占位符
        modules_to_save += find_embedding(model)  # 追加嵌入层名称
    if 'all-norm' in args.modules_to_save:  # 展开所有归一化层
        modules_to_save.remove('all-norm')  # 移除占位符
        modules_to_save += find_norm(model)  # 追加归一化层名称
    if task_type and task_type.lower() == 'seq_cls':  # 序列分类时需要保存 v_head（分类头）
        modules_to_save.append('v_head')  # 追加分类头
    return modules_to_save  # 返回最终列表


def get_vera_target_modules(model, config):
    """Vera 特定目标模块筛选：保证线性层形状一致。

    说明:
        Vera 要求所有目标线性层的形状一致，若不一致，则以包含 'v' 的目标名称为基准筛选同形状模块。
    """
    target_modules = config.target_modules  # 读取配置中的目标模块
    modules_dict = {
        name: module.weight.shape  # 记录线性层的权重形状
        for name, module in model.named_modules()  # 遍历所有命名子模块
        if isinstance(module, torch.nn.Linear) and any([t in name for t in target_modules])
    }  # only Linear for now  # 仅考虑 Linear 层
    if len(set(modules_dict.values())) > 1:  # 若存在不止一种形状
        v = [t for t in target_modules if 'v' in t]  # 查找包含 'v' 的目标模块名片段
        if not v:  # 未找到时提示用户显式传入
            raise ValueError('Please manually pass in `vera_target_modules`, do not use `all-linear`,'
                             'because Vera need all target linears to be the same size.')
        v = v[0]  # 取第一个匹配项
        shape = [shape for name, shape in modules_dict.items() if v in name][0]  # 基准形状
        names = [_name for _name, _shape in modules_dict.items() if _shape == shape]  # 所有同形状层名
        config.target_modules = [t for t in target_modules if any([t in name for name in names])]  # 过滤目标模块
    return config  # 返回更新后的配置


def prepare_adapter(args: TrainArguments, model, *, template=None, train_dataset=None, task_type=None):
    """根据训练类型与后端，对模型注入适配器或调优配置，并返回可训练模型。

    参数:
        args: 训练参数（`TrainArguments`）。
        model: 待微调的模型实例。
        template: 可选模板对象（部分方法需要，如 LoRA-GA 需要 data_collator）。
        train_dataset: 可选训练数据集（如 LoRA-GA 需要数据）。
        task_type: 可选任务类型（覆盖 args.task_type）。

    返回:
        注入适配器/调优策略后的模型实例。
    """
    from swift.tuners import (AdaLoraConfig, AdapterConfig, BOFTConfig, LLaMAProConfig, LongLoRAModelType, LoraConfig,  # 延迟导入避免无用依赖
                              LoRAConfig, ReftConfig, Swift, VeraConfig)
    task_type = (task_type or args.task_type).upper()  # 规范化任务类型为大写
    target_modules = get_target_modules(args, model)  # 解析目标模块
    modules_to_save = get_modules_to_save(args, model, task_type)  # 解析需保存模块
    lora_kwargs = {  # LoRA 通用参数
        'r': args.lora_rank,
        'target_modules': target_modules,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'bias': args.lora_bias,
        'modules_to_save': modules_to_save,
        'use_rslora': args.use_rslora,
        'use_dora': args.use_dora,
        'lorap_lr_ratio': args.lorap_lr_ratio,
        'init_lora_weights': args.init_weights,
    }

    if args.train_type in ('lora', 'longlora'):  # LoRA/LongLoRA 分支
        if args.use_swift_lora:  # 使用 swift 自带 LoRA 实现
            lora_config = LoRAConfig(lora_dtype=args.lora_dtype, **lora_kwargs)  # 构建 LoRA 配置
            model = Swift.prepare_model(model, lora_config)  # 注入 LoRA
            logger.info(f'lora_config: {lora_config}')  # 打印配置
        elif args.tuner_backend == 'peft':  # 使用 peft 后端
            if task_type == 'EMBEDDING':  # peft 的 task_type 兼容性处理
                task_type = None
            elif task_type == 'RERANKER':
                task_type = 'SEQ_CLS'
            elif task_type == 'GENERATIVE_RERANKER':
                task_type = 'CAUSAL_LM'
            lora_config = LoraConfig(task_type=task_type, lora_dtype=args.lora_dtype, **lora_kwargs)  # peft 配置
            if args.init_weights == 'lora-ga':  # LoRA-GA 特殊初始化
                try:
                    import lora_ga  # 仅在需要时导入
                except ImportError as e:  # 未安装时引导用户
                    error_message = """
                    Since 'LoRA-GA' is not implemented by PEFT, you will need to install it directly from GitHub.
                    Command: 'pip install git+https://github.com/lxline/LoRA-GA.git'.
                    """
                    logger.info(error_message)  # 打印提示信息
                    raise RuntimeError(error_message) from e  # 抛出运行时错误
                model = lora_ga.entrypoint.get_lora_ga_model(  # 通过 lora-ga 构造模型
                    model=model,
                    data_collator=template.data_collator,
                    dataset=train_dataset,
                    batch_size=args.lora_ga_batch_size,
                    num_iters=args.lora_ga_iters,
                    max_length=args.lora_ga_max_length,
                    direction=args.lora_ga_direction,
                    dtype=args.lora_dtype,
                    scale=args.lora_ga_scale,
                    stable_gamma=args.lora_ga_stable_gamma,
                )
            else:  # 常规 peft LoRA
                model = Swift.prepare_model(model, lora_config)  # 注入 LoRA
            logger.info(f'lora_config: {lora_config}')  # 打印配置
        elif args.tuner_backend == 'unsloth':  # 使用 Unsloth 加速后端
            if args.resume_from_checkpoint is None:  # 不从断点恢复时，交由 unsloth 构建
                if args.model_meta.is_multimodal:  # 多模态模型
                    from unsloth import FastVisionModel as UnslothModel  # 视觉模型
                else:
                    from unsloth import FastLanguageModel as UnslothModel  # 语言模型
                assert args.train_type == 'lora', 'Unsloth does not support LongLoRA'  # Unsloth 不支持 LongLoRA
                lora_kwargs.pop('lorap_lr_ratio')  # Unsloth 接口不需要该参数
                model = UnslothModel.get_peft_model(  # 构建带 LoRA 的模型
                    model,
                    use_gradient_checkpointing='unsloth',
                    max_seq_length=args.max_length or 2048,  # 2048 is the default value of unsloth
                    **lora_kwargs,
                )
                logger.info(f'unsloth_config: {lora_kwargs}')  # 打印配置
        if args.train_type == 'longlora':  # LongLoRA 额外处理
            assert LongLoRAModelType.LLAMA in args.model_type  # 仅支持 LLaMA 系列
            assert version.parse(transformers.__version__) >= version.parse('4.39.3')  # 版本要求
            from swift.tuners.longlora.llama import replace_llama_attn  # 导入替换注意力实现
            replace_llama_attn(model)  # 替换注意力模块
            model.config.group_size_ratio = 0.25  # 设置分组比例
    elif args.train_type == 'adalora':  # AdaLoRA 分支
        lora_kwargs.pop('lorap_lr_ratio', None)  # 移除不适用参数
        lora_kwargs['rank_pattern'] = None  # 由算法自动学习 rank pattern
        from swift.plugin.optimizer import calculate_max_steps  # 计算总步数
        adalora_config = AdaLoraConfig(  # 构建 AdaLoRA 配置
            task_type=task_type,
            **lora_kwargs,
            target_r=args.adalora_target_r,
            init_r=args.adalora_init_r,
            tinit=args.adalora_tinit,
            tfinal=args.adalora_tfinal,
            deltaT=args.adalora_deltaT,
            beta1=args.adalora_beta1,
            beta2=args.adalora_beta2,
            orth_reg_weight=args.adalora_orth_reg_weight,
            total_step=calculate_max_steps(args.training_args, train_dataset),
        )
        model = Swift.prepare_model(model, adalora_config)  # 注入 AdaLoRA
        logger.info(f'adalora_config: {adalora_config}')  # 打印配置
    elif args.train_type == 'llamapro':  # LLaMAPro 分支
        llamapro_config = LLaMAProConfig(  # 构建配置
            model_type=model.model_meta.model_arch,
            num_new_blocks=args.llamapro_num_new_blocks,
            num_groups=args.llamapro_num_groups)
        model = Swift.prepare_model(model, llamapro_config)  # 注入 LLaMAPro
        logger.info(f'llamapro_config: {llamapro_config}')  # 打印配置
    elif args.train_type == 'adapter':  # Adapter 分支
        model_arch = model.model_meta.model_arch  # 读取结构信息
        mlp_key = model_arch.mlp  # 取 MLP 路径占位
        mlp_key = mlp_key.split('.{}.')[1]  # 提取实际子键名
        adapter_config = AdapterConfig(  # 构建 Adapter 配置
            dim=model.config.hidden_size,
            target_modules=[mlp_key],
            hidden_pos=0,
            adapter_length=args.adapter_length,
            act_layer=args.adapter_act)
        model = Swift.prepare_model(model, adapter_config)  # 注入 Adapter
        logger.info(f'adapter_config: {adapter_config}')  # 打印配置
    elif args.train_type == 'vera':  # Vera 分支
        vera_config = VeraConfig(  # 构建 Vera 配置
            r=args.vera_rank,
            target_modules=target_modules,
            projection_prng_key=args.vera_projection_prng_key,
            vera_dropout=args.vera_dropout,
            d_initial=args.vera_d_initial,
            modules_to_save=args.modules_to_save,
        )
        vera_config = get_vera_target_modules(model, vera_config)  # 按形状一致性筛选目标模块
        model = Swift.prepare_model(model, vera_config)  # 注入 Vera
        logger.info(f'vera_config: {vera_config}')  # 打印配置
    elif args.train_type == 'boft':  # BOFT 分支
        boft_config = BOFTConfig(  # 构建 BOFT 配置
            boft_block_size=args.boft_block_size,
            boft_block_num=args.boft_block_num,
            boft_n_butterfly_factor=args.boft_n_butterfly_factor,
            target_modules=target_modules,
            boft_dropout=args.boft_dropout,
            modules_to_save=args.modules_to_save,
        )
        model = Swift.prepare_model(model, boft_config)  # 注入 BOFT
        logger.info(f'boft_config: {boft_config}')  # 打印配置
    elif args.train_type == 'fourierft':  # FourierFT 分支
        from peft import FourierFTConfig  # 从 peft 导入
        fourier_config = FourierFTConfig(  # 构建配置
            target_modules=target_modules,
            modules_to_save=args.modules_to_save,
            n_frequency=args.fourier_n_frequency,
            scaling=args.fourier_scaling,
        )
        model = Swift.prepare_model(model, fourier_config)  # 注入 FourierFT
        logger.info(f'fourier_config: {fourier_config}')  # 打印配置
    elif args.train_type == 'reft':  # ReFT 分支
        reft_config = ReftConfig(  # 构建 ReFT 配置
            model_type=model.model_meta.model_arch,
            layer_key=args.reft_layer_key,
            r=args.reft_rank,
            layers=args.reft_layers,
            intervention_type=args.reft_intervention_type,
            args=args.reft_args,
        )
        logger.info(f'reft config: {reft_config}')  # 打印配置
        model = Swift.prepare_model(model, {'reft': reft_config})  # 注入 ReFT
    elif args.train_type == 'bone':  # Bone 分支
        # Version loosing
        from peft import BoneConfig  # 从 peft 导入 Bone 配置
        bone_config = BoneConfig(  # 构建 Bone 配置
            target_modules=target_modules,
            r=args.reft_rank,
            init_weights=args.init_weights,
        )
        logger.info(f'bone config: {bone_config}')  # 打印配置
        model = Swift.prepare_model(model, bone_config)  # 注入 Bone
    return model  # 返回准备好的模型


class TunerMixin:
    """Tuner 混入类：统一模型准备流程。

    作用:
        - 根据 `args` 中的配置选择不同调优策略与后端，实现适配器注入、冻结策略、断点恢复等。
        - 可与 `SwiftPipeline` 组合，作为训练流水线中的模型准备步骤。
    """

    @classmethod
    def prepare_model(cls, args, model, *, template=None, train_dataset=None, task_type=None):
        """根据参数准备模型，注入/恢复适配器，或执行全参训练准备。

        参数:
            args: 训练参数，决定调优策略与后端实现。
            model: 当前模型实例。
            template: 可选模板对象，用于部分策略的辅助。
            train_dataset: 可选训练数据集。
            task_type: 可选任务类型（优先于 args.task_type）。

        返回:
            已按策略准备完毕的模型实例。
        """
        if args.use_liger_kernel and 'use_liger_kernel' not in inspect.signature(TrainingArguments).parameters:  # 缺省支持时手动应用 Liger
            # Apply liger
            apply_liger(args.model_type)  # 按模型类型应用 Liger 内核

        if args.is_adapter:  # 走适配器训练路径
            if args.tuner_backend != 'unsloth' and args.train_type not in extra_tuners:  # 非 unsloth 且不在额外 tuner 中
                # Fix the name of the layer in xcomposer that contains Plora.
                # Unsloth prepares and loads lora outside this function when
                # resume_from_checkpoint, so do not disable grad here
                model.requires_grad_(False)  # 默认先关闭梯度，避免误训练
            if args.resume_from_checkpoint:  # 断点恢复路径
                if args.train_type in extra_tuners:  # 使用额外 tuner 的 from_pretrained
                    tuner: Tuner = extra_tuners[args.train_type]
                else:
                    tuner = Swift  # 默认使用 Swift 接口
                kwargs = {}  # 预留扩展参数
                model = tuner.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True, **kwargs)  # 恢复并设为可训练
            else:  # 新训练路径
                if args.train_type in extra_tuners:  # 走额外 tuner 的 prepare_model
                    tuner: Tuner = extra_tuners[args.train_type]
                    model = tuner.prepare_model(args, model)
                else:
                    model = prepare_adapter(  # 按通用适配器准备流程
                        args, model, template=template, train_dataset=train_dataset, task_type=task_type)
            # fix bug: Attempting to unscale FP16 gradients.
            #   peft: https://github.com/huggingface/peft/issues/1249
            for p in model.parameters():  # 处理半精度可训练参数
                if p.requires_grad and p.dtype == torch.float16:  # 若为 fp16
                    logger.info_once('Convert trainable parameters from fp16 to fp32.')  # 提示转换
                    p.data = p.data.to(dtype=torch.float32)  # 转换为 fp32 以避免 unscale 问题
        elif args.train_type == 'full':  # 全参训练路径
            model.train()  # 切换训练模式
            model.requires_grad_(True)  # 打开所有梯度

            freeze_parameters(model, args.freeze_parameters_ratio, args.freeze_parameters, args.freeze_parameters_regex)  # 按比例/名单/正则冻结参数
            if args.trainable_parameters or args.trainable_parameters_regex:  # 指定额外可训练参数
                activate_parameters(model, args.trainable_parameters, args.trainable_parameters_regex)  # 激活这些参数
        else:  # 未知训练类型
            raise ValueError(f'args.train_type: {args.train_type}')  # 抛出错误

        if args.use_galore:  # 启用 GaLore 优化器配置
            from swift.trainers.optimizers.galore import GaLoreConfig  # 导入 GaLore 配置
            if args.galore_target_modules is None:  # 未指定目标模块则默认线性层
                args.galore_target_modules = find_all_linears(model)  # 扫描线性层
            if args.galore_with_embedding:  # 需要包含嵌入层
                args.galore_target_modules += find_embedding(model)  # 追加嵌入层
            args.galore_config = GaLoreConfig(  # 组装 GaLore 配置
                target_modules=args.galore_target_modules,
                rank=args.galore_rank,
                update_proj_gap=args.galore_update_proj_gap,
                galore_scale=args.galore_scale,
                proj_type=args.galore_proj_type,
                optim_per_parameter=args.galore_optim_per_parameter,
                quantize=args.galore_quantization,
                proj_quant=args.galore_proj_quant,
                proj_bits=args.galore_proj_bits,
                proj_group_size=args.galore_proj_group_size,
                cos_threshold=args.galore_cos_threshold,
                gamma_proj=args.galore_gamma_proj,
                queue_size=args.galore_queue_size,
            )
            args.training_args.galore_config = args.galore_config  # 将配置挂载到训练参数

        if args.sequence_parallel_size > 1:  # 启用序列并行
            from swift.trainers.sequence_parallel import sequence_parallel  # 导入并行工具
            sequence_parallel.prepare_model(model, template.tokenizer)  # 按并行需求准备模型与分词器

        return model  # 返回已准备好的模型
