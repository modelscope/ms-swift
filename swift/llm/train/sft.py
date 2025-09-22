"""
脚本用途:
- 定义监督微调（SFT）训练流水线基类 `SwiftSft`，组织并执行从模型/模板/数据集准备到训练器创建与训练的全流程。
- 提供模块级入口 `sft_main`，支持以参数（命令行或对象）驱动训练主流程。

主要功能:
- 模型与处理器加载、推理配置准备、模板初始化与模式设置。
- 数据集加载、缓存读取、编码与打包、统计与展示。
- 训练器创建、回调管理、训练执行与状态保存/可视化。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明
import os  # 标准库：文件与路径操作
from functools import partial  # 工具：创建偏函数以封装数据整理器
from typing import List, Optional, Union  # 类型提示：列表、可选、联合类型

from datasets import Dataset as HfDataset  # 第三方：HuggingFace 数据集类型别名
from datasets import load_from_disk  # 第三方：从磁盘读取已缓存的数据集

from swift.llm.dataset.loader import DatasetLoader  # 项目内：数据集合并与工具方法
from swift.plugin import extra_callbacks, get_loss_func, get_metric  # 插件：回调、损失与评测指标工厂
from swift.trainers import TrainerFactory  # 训练器工厂：按参数选择具体 Trainer 实现
from swift.utils import append_to_jsonl, get_logger, get_model_parameter_info, is_master, plot_images, stat_array  # 通用工具
from ..argument import TrainArguments  # 训练参数数据结构
from ..base import SwiftPipeline  # 基础流水线，提供主流程框架
from ..dataset import EncodePreprocessor, IterablePackingDataset, LazyLLMDataset, PackingDataset, load_dataset  # 数据处理组件
from ..infer import prepare_generation_config  # 生成配置准备函数
from .tuner import TunerMixin  # Tuner 混入类，提供可插拔微调策略

logger = get_logger()  # 初始化模块级日志记录器


class SwiftSft(SwiftPipeline, TunerMixin):  # 定义 SFT 流水线类，继承基础流水线并混入 Tuner 能力
    """SFT 训练流水线基类

    作用:
        - 组织并执行 SFT 训练全流程：模型/处理器与生成配置准备、模板准备、数据集加载与编码、
          训练器构建与回调注册、训练执行与状态保存/可视化。
        - 通过 `TunerMixin` 支持不同微调策略（如 LoRA/Adapter），在准备阶段注入所需配置或参数。

    属性:
        args_class: 指定该类使用的参数类型（`TrainArguments`）。
        args: 运行时注入的训练参数实例，类型为 `args_class`。
        train_msg: 训练过程中的统计信息与路径汇总字典。
        template: 当前使用的对话/指令模板对象。
        callbacks: 已注册的训练回调列表。
        model, processor: 由 `args.get_model_processor` 加载得到的模型与处理器。
    """

    args_class = TrainArguments  # 指定参数类型为 TrainArguments
    args: args_class  # 类型注解：实例属性 args 的类型为 args_class

    def __init__(self, args: Optional[Union[List[str], TrainArguments]] = None) -> None:
        """初始化流水线。

        参数:
            args: 训练参数，可以是命令行参数列表或 `TrainArguments` 实例；为空时由父类解析。

        动作:
            - 调用父类初始化以解析参数。
            - 准备模型与处理器、模板与回调，为后续训练做准备。
        """
        super().__init__(args)  # 调用父类构造，解析与规范化训练参数
        self.train_msg = {}  # 存放训练过程信息（模型参数统计、检查点路径、指标等）
        self._prepare_model_tokenizer()  # 加载模型与处理器，必要时初始化并行等配置
        self._prepare_template()  # 创建并配置模板（设置为训练模式）
        self._prepare_callbacks()  # 注册训练回调（如动态层激活、适配器回调等）

    def _prepare_generation_config(self):
        """准备模型的生成配置。

        动作:
            - 备份当前生成配置。
            - 基于请求配置与 tokenizer 对生成配置进行更新（如解码策略、长度限制等）。
            - 记录最终生成配置以便排查。
        """
        args = self.args  # 读取训练参数
        self.model.origin_generation_config = self.model.generation_config  # 备份原始生成配置
        self.model.generation_config = prepare_generation_config(self.model.generation_config,  # 基于请求配置更新生成配置
                                                                 args.get_request_config(), self.tokenizer)
        logger.info(f'model.generation_config: {self.model.generation_config}')  # 打印当前生成配置

    def _prepare_model_tokenizer(self, load_model=True):
        """加载模型与处理器，并按需要初始化并行与生成配置。

        参数:
            load_model: 是否加载模型权重（部分场景下仅需初始化处理器）。
        """
        args = self.args  # 读取训练参数
        if args.sequence_parallel_size > 1:  # 若启用序列并行
            from swift.trainers.sequence_parallel import sequence_parallel  # 延迟导入以避免不必要依赖
            sequence_parallel.init_sequence_parallel(args.sequence_parallel_size)  # 初始化序列并行环境
        self.model, self.processor = args.get_model_processor(load_model=load_model)  # 加载/构建模型与处理器
        if self.model is None:  # 可能处于仅准备阶段，暂不加载模型
            return  # 无模型时提前返回
        if hasattr(self.model, 'hf_device_map'):  # 若存在设备映射信息
            logger.info(f'model.hf_device_map: {self.model.hf_device_map}')  # 打印设备分配映射

        logger.info(f'model_info: {self.model.model_info}')  # 打印模型关键信息（结构/参数等）

        self._prepare_generation_config()  # 基于参数与 tokenizer 准备生成配置

    def _prepare_template(self) -> None:
        """准备模板并设置为训练模式。"""
        template = self.args.get_template(self.processor)  # 基于处理器与参数构建模板
        template.set_mode('train')  # 指定模板当前处于训练模式
        if template.use_model:  # 若模板内部需要引用模型
            template.model = self.model  # 注入当前模型实例
        self.template = template  # 缓存模板供后续编码与训练使用

    def _get_dataset(self):
        """按参数加载训练与验证数据集，必要时单独加载验证集。"""
        # The random shuffling of the training set occurs in the dataloader of the trainer.  # 训练集的随机打散在 Trainer 的 DataLoader 中进行
        args = self.args  # 读取训练参数
        dataset_kwargs = args.get_dataset_kwargs()  # 获取数据集加载的额外参数
        train_dataset, val_dataset = load_dataset(  # 加载训练集与（可能的）验证集
            args.dataset, split_dataset_ratio=args.split_dataset_ratio, shuffle=args.dataset_shuffle, **dataset_kwargs)
        if len(args.val_dataset) > 0:  # 若用户指定独立验证集
            # Loading val dataset
            _, val_dataset = load_dataset(  # 仅加载验证集，不进行切分
                args.val_dataset, split_dataset_ratio=1.0, shuffle=args.val_dataset_shuffle, **dataset_kwargs)
            assert args.split_dataset_ratio == 0.  # 指定独立验证集时，训练集不再切分验证集
        logger.info(f'train_dataset: {train_dataset}')  # 打印训练集信息
        logger.info(f'val_dataset: {val_dataset}')  # 打印验证集信息

        return train_dataset, val_dataset  # 返回训练集和验证集

    def _get_data_collator(self):
        """构造数据整理函数（collator），按需要设置 padding 策略。"""
        args = self.args  # 读取训练参数
        template = self.template  # 模板包含编码与整理逻辑
        padding_to = args.max_length if args.train_type == 'longlora' else None  # longlora 训练需按最大长度填充
        return partial(template.data_collator, padding_to=padding_to)  # 偏函数形式返回 collator

    def _save_val_dataset(self, val_dataset):
        """将从训练集中切分得到的验证集保存为 jsonl，便于复现实验。"""
        args = self.args  # 读取训练参数
        output_dir = getattr(args, 'output_dir', None) or getattr(args, 'save')  # 计算输出目录
        if is_master() and isinstance(val_dataset, HfDataset) and not args.val_dataset:  # 仅在主进程且未指定独立验证集时保存
            os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
            val_dataset_path = os.path.join(output_dir, 'val_dataset.jsonl')  # 验证集保存路径
            append_to_jsonl(val_dataset_path, val_dataset.to_list())  # 以 jsonl 形式写入验证集
            logger.info(f'The split dataset from the training set will be saved at: {val_dataset_path}.')  # 记录保存位置

    def _get_cached_dataset(self):
        """从磁盘读取缓存数据集（train/val 两个子目录）。"""
        args = self.args  # 读取训练参数
        assert not args.streaming and not args.lazy_tokenize  # 缓存模式下不支持 streaming 与延迟分词
        train_datasets, val_datasets = [], []  # 初始化容器
        for cached_dataset in args.cached_dataset:  # 遍历每个缓存根目录
            train_path = os.path.join(cached_dataset, 'train')  # 训练集子目录
            val_path = os.path.join(cached_dataset, 'val')  # 验证集子目录
            train_datasets.append(load_from_disk(train_path))  # 载入训练集缓存
            if os.path.exists(val_path):  # 若存在验证集缓存
                val_datasets.append(load_from_disk(val_path))  # 载入验证集缓存
        return train_datasets, val_datasets  # 返回多个缓存数据集列表

    def _prepare_dataset(self):
        """准备最终用于训练/验证的数据集，包括缓存读取、加载、编码与打包。"""
        args = self.args  # 读取训练参数
        if args.cached_dataset:  # 优先使用缓存数据集
            train_datasets, val_datasets = self._get_cached_dataset()  # 从缓存载入
        else:
            train_datasets, val_datasets = [], []  # 无缓存则初始化空列表
        if args.dataset:  # 若指定了数据集来源
            train_dataset, val_dataset = self._get_dataset()  # 加载数据集
            train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)  # 编码/预处理
            train_datasets.append(train_dataset)  # 收集训练集
            val_datasets.append(val_dataset)  # 收集验证集
        train_dataset = DatasetLoader._concat_datasets(train_datasets)  # 合并多个训练集
        val_dataset = DatasetLoader._concat_datasets(val_datasets)  # 合并多个验证集
        is_grpo = hasattr(args, 'rlhf_type') and args.rlhf_type == 'grpo'  # 是否为 GRPO 场景
        predict_with_generate = getattr(args, 'predict_with_generate', False)  # 验证是否需生成式评估
        datasets = [train_dataset, val_dataset]  # 封装为列表便于迭代处理

        if is_grpo:  # GRPO 场景下直接返回
            return datasets
        template = self.template  # 获取模板对象
        for i, dataset in enumerate(datasets):  # 依次处理 train / val 数据集
            if dataset is None:  # 为空跳过
                continue
            if i == 1 and predict_with_generate:  # 生成式评估场景下跳过 val 的编码
                # val_dataset
                continue
            if (args.model_meta.is_multimodal or args.lazy_tokenize) and not args.streaming:  # 多模态或延迟分词，且非流式
                dataset = LazyLLMDataset(dataset, template.encode, strict=args.strict, random_state=args.data_seed)  # 懒加载编码
            if args.packing:  # 启用样本打包
                packing_dataset_cls = IterablePackingDataset if args.streaming else PackingDataset  # 选择打包实现
                dataset = packing_dataset_cls(  # 基于模板打包以提升吞吐
                    template,
                    dataset,
                    num_proc=args.dataset_num_proc,
                    strict=args.strict,
                    load_from_cache_file=args.load_from_cache_file)
            elif args.streaming:  # 流式场景，逐步编码
                preprocessor = EncodePreprocessor(template=template)  # 创建编码预处理器
                dataset = preprocessor(  # 对数据集进行编码处理
                    dataset,
                    num_proc=args.dataset_num_proc,
                    load_from_cache_file=args.load_from_cache_file,
                    strict=args.strict)
            datasets[i] = dataset  # 回写处理后的数据集
        self._show_dataset(*datasets)  # 显示样本示例与统计信息
        return datasets  # 返回处理完的数据集

    def run(self):
        """执行训练主流程，创建训练器并启动训练。"""
        args = self.args  # 读取训练参数
        train_dataset, val_dataset = self._prepare_dataset()  # 准备训练与验证数据集

        if args.task_type == 'seq_cls':  # 序列分类任务需指定问题类型
            args.problem_type = args.problem_type or getattr(self.model.config, 'problem_type', None)  # 从模型或参数推断
            logger.info(f'args.problem_type: {args.problem_type}')  # 打印问题类型
        args.save_args()  # 将训练参数持久化，便于复现

        data_collator = self._get_data_collator()  # 构造数据整理函数
        # Some tuners require train_dataset and data_collator for preparation: LoRA-GA
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)  # 经 TunerMixin 适配模型
        logger.info(f'model: {self.model}')  # 打印模型摘要
        model_parameter_info = get_model_parameter_info(self.model)  # 统计模型参数信息
        self.train_msg['model_parameter_info'] = model_parameter_info  # 保存统计结果
        logger.info(f'model_parameter_info: {model_parameter_info}')  # 输出参数统计

        trainer_cls = TrainerFactory.get_trainer_cls(args)  # 选择训练器实现
        trainer = trainer_cls(  # 构造训练器实例
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)  # 启动训练并返回训练结果

    def _get_trainer_kwargs(self):
        """构造训练器关键字参数：评测指标、logits 预处理与损失函数。"""
        args = self.args  # 读取训练参数
        if args.metric is not None:  # 显式指定指标
            compute_metrics, preprocess_logits_for_metrics = get_metric(args.metric)  # 获取指标计算与预处理
        elif args.predict_with_generate:  # 生成式评估
            compute_metrics, preprocess_logits_for_metrics = get_metric('nlg')  # 使用 NLG 指标
        else:  # 默认精度指标
            compute_metrics, preprocess_logits_for_metrics = get_metric('acc')  # 使用分类准确率
            compute_metrics = partial(  # 注入策略与结构信息
                compute_metrics, acc_strategy=args.acc_strategy, is_encoder_decoder=self.template.is_encoder_decoder)
        return {
            'compute_metrics': compute_metrics,  # 指标计算函数
            'preprocess_logits_for_metrics': preprocess_logits_for_metrics,  # 指标前的 logits 预处理
            'compute_loss_func': get_loss_func(args.loss_type)  # 损失函数工厂
        }

    def _save_trainer_state(self, trainer):
        """保存训练器状态并进行可视化/同步处理。"""
        training_args = trainer.args  # 训练器运行参数
        state = trainer.state  # 训练状态对象
        if hasattr(state, 'last_model_checkpoint'):  # 若存在检查点路径
            if self.args.create_checkpoint_symlink:  # 需要为最新/最佳创建符号链接
                last_checkpoint = os.path.join(self.args.output_dir, 'last')  # 最新模型链接
                best_checkpoint = os.path.join(self.args.output_dir, 'best')  # 最佳模型链接
                if is_master():  # 仅主进程创建软链接
                    os.symlink(state.last_model_checkpoint, last_checkpoint)
                    os.symlink(state.best_model_checkpoint, best_checkpoint)
                state.last_model_checkpoint = last_checkpoint  # 用链接路径替换原路径
                state.best_model_checkpoint = best_checkpoint  # 用链接路径替换原路径
        else:  # 无检查点时保证字段存在
            state.last_model_checkpoint = None  # 设置为 None
        logger.info(f'last_model_checkpoint: {state.last_model_checkpoint}')  # 打印最新检查点
        logger.info(f'best_model_checkpoint: {state.best_model_checkpoint}')  # 打印最佳检查点

        # Visualization
        if is_master():  # 仅主进程做可视化与推送
            if 'tensorboard' in training_args.report_to:  # 使用 TensorBoard 上报
                images_dir = os.path.join(training_args.output_dir, 'images')  # 图片输出目录
                logger.info(f'images_dir: {images_dir}')  # 打印目录
                plot_images(images_dir, training_args.logging_dir, ['train/loss'], 0.9)  # 绘制训练损失曲线
            if training_args.push_to_hub:  # 开启推送模型至 Hub
                trainer.push_to_hub()  # 推送模型与配置

        self.train_msg.update({  # 汇总训练信息
            'last_model_checkpoint': state.last_model_checkpoint,
            'best_model_checkpoint': state.best_model_checkpoint,
            'best_metric': state.best_metric,
            'global_step': state.global_step,
            'log_history': state.log_history,
            'memory': trainer.max_memory,
        })
        if is_master():  # 仅主进程写日志
            jsonl_path = os.path.join(training_args.output_dir, 'logging.jsonl')  # 训练日志路径
            append_to_jsonl(jsonl_path, self.train_msg, strict=False)  # 以 jsonl 追加写入
        return self.train_msg  # 返回汇总信息

    def train(self, trainer):
        """启动训练并在结束时保存状态。"""
        logging_path = os.path.join(trainer.args.output_dir, 'logging.jsonl')  # 训练日志（jsonl）路径
        logger.info(f'The logging file will be saved in: {logging_path}')  # 提示日志保存路径
        try:  # 训练主过程
            trainer.train(trainer.args.resume_from_checkpoint)  # 支持从断点恢复
        finally:  # 不论是否异常都保存状态
            res = self._save_trainer_state(trainer)  # 保存状态并返回信息
        return res  # 返回训练结果信息

    def _prepare_callbacks(self):
        """注册与准备训练回调（如动态层激活与适配器回调）。"""
        from .callback import DynamicLayerActivationCallback, TrainerAdapterCallback  # 延迟导入回调以降低开销
        args = self.args  # 读取训练参数
        callbacks = []  # 回调列表
        if args.lisa_activated_layers > 0:  # 启用 LISA 动态层激活
            assert args.train_type == 'full', 'LISA only supports full parameter training.'  # LISA 仅支持全参训练
            lisa_callback = DynamicLayerActivationCallback(
                n_layers=args.lisa_activated_layers,  # Number of layers to activate
                step_interval=args.lisa_step_interval,  # Step interval to update active layers
                model=self.model)  # 绑定模型
            lisa_callback.switch_active_layers()  # Make trainable parameters printing a correct value
            callbacks.append(lisa_callback)  # 注册 LISA 回调

        if args.is_adapter and args.train_type == 'adalora':  # 适配器 + AdaLoRA 场景
            callbacks.append(TrainerAdapterCallback(args))  # 注册适配器回调
        callbacks += extra_callbacks  # 注入额外的全局回调
        self.callbacks = callbacks  # 保存回调列表

    @staticmethod
    def _stat_dataset(dataset: Union[HfDataset, PackingDataset]):
        """统计数据集中样本 token 长度分布并返回描述字符串。"""
        if isinstance(dataset, HfDataset):  # HuggingFace 数据集
            length = dataset['length']  # 直接读取长度字段
        else:  # PackingDataset/IterablePackingDataset
            length = dataset.packed_length  # 取打包后的长度信息
        _, stat_str = stat_array(length)  # 生成统计描述字符串
        logger.info(f'Dataset Token Length: {stat_str}')  # 打印长度分布
        return stat_str  # 返回统计字符串

    def _show_dataset(self, train_dataset, val_dataset):
        """在主进程展示样本入参并记录数据集统计信息。"""
        args = self.args  # 读取训练参数
        predict_with_generate = getattr(args, 'predict_with_generate', False)  # 是否生成式评估
        if is_master():  # 主进程打印示例输入
            inputs = train_dataset[0] if hasattr(train_dataset, '__len__') else next(iter(train_dataset))  # 取一个样本
            self.template.print_inputs(inputs, tokenizer_kwargs=inputs.pop('tokenizer_kwargs', None) or {})  # 打印样本
        elif hasattr(train_dataset, '__len__'):
            # Avoid the random mismatch issue in LazyLLMDataset.
            inputs = train_dataset[0]  # 非主进程但具备长度时，取样本以避免懒加载随机错位
        if val_dataset is not None and hasattr(val_dataset, '__len__') and len(val_dataset) == 0:  # 空验证集处理
            val_dataset = None  # 置空以避免后续错误
        if not args.lazy_tokenize and not args.streaming:  # 仅在非懒加载与非流式下统计
            self.train_msg['train_dataset'] = self._stat_dataset(train_dataset)  # 记录训练集统计
            if val_dataset is not None and not predict_with_generate:  # 需要非生成式评估才统计验证集
                self.train_msg['val_dataset'] = self._stat_dataset(val_dataset)  # 记录验证集统计

    def _encode_dataset(self, train_dataset, val_dataset):
        """对训练/验证数据集进行编码预处理，必要时跳过或分批处理。"""
        template = self.template  # 获取模板对象
        args = self.args  # 读取训练参数
        self._save_val_dataset(val_dataset)  # 保存从训练集切分得到的验证集（若适用）

        is_grpo = hasattr(args, 'rlhf_type') and args.rlhf_type == 'grpo'  # GRPO 场景标记
        predict_with_generate = getattr(args, 'predict_with_generate', False)  # 是否生成式评估
        datasets = [train_dataset, val_dataset]  # 打包两个数据集便于循环处理
        if is_grpo:  # GRPO 场景不进行通用编码
            return datasets  # 直接返回

        origin_template_model = template.model  # 记录模板中的模型引用
        template.model = None  # Avoid serializing the model.  # 暂时移除模型避免序列化开销
        for i, dataset in enumerate(datasets):  # 遍历 train / val
            if dataset is None:  # 为空直接跳过
                continue
            if i == 1 and predict_with_generate:  # 生成式评估时跳过 val 编码
                # val_dataset
                continue
            if not args.lazy_tokenize and not args.streaming:  # 需要立即编码且非流式
                preprocessor = EncodePreprocessor(template=template, pre_tokenize=args.model_meta.is_multimodal)  # 构建编码预处理器
                batch_size = 100 if args.model_meta.is_multimodal else 1000  # 多模态用较小批次
                dataset = preprocessor(  # 对数据集执行编码
                    dataset,
                    num_proc=args.dataset_num_proc,
                    load_from_cache_file=args.load_from_cache_file,
                    strict=args.strict,
                    batch_size=batch_size)
            datasets[i] = dataset  # 回写处理后数据集
        template.model = origin_template_model  # 还原模板中的模型引用

        return datasets  # 返回编码后的数据集


def sft_main(args: Optional[Union[List[str], TrainArguments]] = None):
    """SFT 训练入口。

    参数:
        args: 训练参数，可为命令行参数列表（List[str]）或 `TrainArguments` 实例；
              为空时从环境或默认配置中解析。

    返回:
        任意类型。通常为训练主流程的返回值（由 `SwiftPipeline.main` 定义）。
    """
    return SwiftSft(args).main()  # 基于传入参数构建 SwiftSft 实例并执行主流程
