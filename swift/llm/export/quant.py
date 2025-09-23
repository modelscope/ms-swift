"""
模块功能概述：
本模块提供模型量化导出的核心流程与工具：
- QuantEngine: 量化执行引擎，负责构建模型与模板、准备量化数据、调度不同量化方法（AWQ/GPTQ/BnB/FP8），并保存结果；
- quantize_model: 便捷入口函数，创建引擎并执行量化。
"""

# 版权信息：阿里巴巴及其附属公司保留所有权利
# Copyright (c) Alibaba, Inc. and its affiliates.
# 导入defaultdict：用于按键自动创建列表的字典结构（专家模块收集）
from collections import defaultdict
# 导入contextmanager装饰器：用于定义with上下文（临时替换函数）
from contextlib import contextmanager
# 导入类型注解：Dict/List/Optional用于接口签名说明
from typing import Dict, List, Optional

# 导入PyTorch：torch为张量运算库
import torch
# 导入神经网络模块：用于类型判断与模型结构遍历
import torch.nn as nn
# 导入tqdm进度条：用于显示数据准备与循环进度
from tqdm import tqdm

# 从swift.llm导入量化所需工具与类型：
# - ExportArguments: 导出/量化参数定义
# - HfConfigFactory: 统一读写HF模型与配置属性
# - MaxLengthError: 模板编码超长异常
# - ProcessorMixin: 提供processor等通用能力的混入类
# - deep_getattr: 支持多层级属性获取
# - load_dataset: 加载训练/验证数据
# - prepare_model_template: 构建模型与模板
# - save_checkpoint: 保存模型与处理器到磁盘
# - to_device: 将张量/字典迁移到指定设备
from swift.llm import (ExportArguments, HfConfigFactory, MaxLengthError, ProcessorMixin, deep_getattr, load_dataset,
                       prepare_model_template, save_checkpoint, to_device)
# 从swift.utils导入日志与参数统计工具
from swift.utils import get_logger, get_model_parameter_info

# 初始化模块级日志器：统一打印量化过程信息
logger = get_logger()


# 定义量化引擎：封装不同量化方法的调度与数据准备流程
class QuantEngine(ProcessorMixin):
    """
    类功能：
        负责模型量化全过程：加载/构建模型与模板，准备量化校准数据（AWQ/GPTQ），
        执行不同量化策略，保存量化后的模型与处理器，并输出参数统计与日志。

    关键属性：
        args (ExportArguments): 量化与导出参数集合。
        model: 底层模型（可能为包装器，具体视量化方法而定）。
        template: 模板对象，提供encode、data_collator、pre_forward_hook等方法。
        processor: 与模板配套的处理器（tokenizer/processor）。
    """

    # 初始化量化引擎：准备模型、模板、processor，并关闭缓存以便训练式前向
    def __init__(self, args: ExportArguments):
        """
        函数功能：
            根据量化方法与参数构建模型和模板，设置训练模式与禁用缓存，保存运行参数。

        入参：
            args (ExportArguments): 量化与导出参数对象。

        返回值：
            None

        示例：
            >>> engine = QuantEngine(args)
        """
        # 保存入参供后续方法使用
        self.args = args
        # 额外构造参数（仅AWQ需要指定AutoAWQ类）
        kwargs = {}
        # 当选择AWQ量化时，导入相应AutoModel并传入构造参数
        if args.quant_method == 'awq':
            from awq import AutoAWQForCausalLM
            kwargs['automodel_class'] = AutoAWQForCausalLM
        # 构建模型与模板（可能因量化方法而使用不同的AutoModel）
        self.model, self.template = prepare_model_template(args, **kwargs)
        # 设置模板模式为train，便于构造数据与前向流程
        self.template.set_mode('train')
        # 关闭模型缓存以避免训练式前向中的cache干扰
        self.model.config.use_cache = False
        # 同步在HF配置对象上关闭use_cache，确保保存/加载一致
        HfConfigFactory.set_model_config_attr(self.model, 'use_cache', False)
        # 记录模板对应的processor（tokenizer/processor）
        self.processor = self.template.processor
        # 持久化当前参数设置，便于后续恢复或复现实验
        args.save_args()

    # 执行量化主流程：根据量化方法分派处理，并最终保存模型与处理器
    def quantize(self):
        """
        函数功能：
            执行量化主流程。根据quant_method选择AWQ/GPTQ/BnB/FP8路径，
            完成量化后保存模型与处理器，并打印参数统计信息。

        入参：
            无（使用self.args）。

        返回值：
            None

        示例：
            >>> QuantEngine(args).quantize()
        """
        # 便捷引用参数对象
        args = self.args
        # 若未设置量化比特且方法不是FP8，则提示必须提供量化比特
        if args.quant_bits is None and args.quant_method != 'fp8':
            raise ValueError(f'Please set the quant_bits. args.quant_bits: {args.quant_bits}')
        # AWQ 路径：将模板内部的model指向基础模型，运行AWQ量化并保存量化权重
        if args.quant_method == 'awq':
            self.template.model = self.model.model
            self.awq_model_quantize()
            self.model.save_quantized(
                args.output_dir, safetensors=args.safe_serialization, shard_size=args.max_shard_size)
        # GPTQ 路径：记录量化器对象并调用其保存方法
        elif args.quant_method == 'gptq':
            self.template.model = self.model
            gptq_quantizer = self.gptq_model_quantize()
            gptq_quantizer.save(
                self.model,
                args.output_dir,
                safe_serialization=args.safe_serialization,
                max_shard_size=args.max_shard_size)
        # BnB 或 FP8：直接按HF接口保存预训练（含量化配置）
        elif args.quant_method in {'bnb', 'fp8'}:
            self.model.save_pretrained(
                args.output_dir, safe_serialization=args.safe_serialization, max_shard_size=args.max_shard_size)
        # 未知方法：抛错提示
        else:
            raise ValueError(f'args.quant_method: {args.quant_method}')

        # 打印模型对象与参数规模统计，便于核对
        logger.info(f'model: {self.model}')
        logger.info(f'model_parameter_info: {get_model_parameter_info(self.model)}')
        # 保存处理器与附加文件到输出目录（模型已由各方法保存或量化器保存）
        save_checkpoint(
            None,
            self.processor,
            args.output_dir,
            model_dirs=[args.model_dir],
            additional_saved_files=self.model.model_meta.additional_saved_files)
        # 提示量化与保存完成
        logger.info(f'Successfully quantized the model and saved in {args.output_dir}.')

    # 准备GPTQ所需的数据集（按batch聚合并移到CPU），推理模式避免梯度
    @torch.inference_mode()
    def _prepare_gptq_dataset(self, examples: List[Dict[str, torch.LongTensor]], batch_size: int = 1, *args, **kwargs):
        """
        函数功能：
            将已编码的样本按batch组装成输入，并放置到CPU，供GPTQ量化器使用。

        入参：
            examples (List[Dict[str, torch.LongTensor]]): 编码后的样本列表。
            batch_size (int): 每批样本数，默认1。

        返回值：
            List[Dict[str, torch.Tensor]]: CPU上的输入批次列表。
        """
        # 初始化返回列表
        res = []
        # 按批遍历样本，并显示进度
        for start in tqdm(range(0, len(examples), batch_size)):
            # 取当前批次的样本切片
            batched_inputs = examples[start:start + batch_size]
            # 使用模板的data_collator组装batch，并迁移到模型所在设备
            inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)
            # 多模态模型需要在前向前做预处理钩子，可能调整inputs结构
            if self.model.model_meta.is_multimodal:
                _, inputs = self.template.pre_forward_hook(self.model, None, inputs)
            # 将输入迁移到CPU，减少显存占用
            res.append(to_device(inputs, 'cpu'))
        # 返回准备完的批次列表
        return res

    # 准备量化数据集（AWQ/GPTQ），推理模式避免梯度
    @torch.inference_mode()
    def _get_quant_dataset(self, *args, **kwargs):
        """
        函数功能：
            按量化方法（AWQ/GPTQ）构造量化校准数据：
            - 对于多模态+GPTQ：直接返回inputs字典列表（去除labels）
            - 其他情况：收集input_ids并按block size切分

        入参：
            *args, **kwargs: 与外部量化器的签名兼容（未使用）。

        返回值：
            List: 量化所需的数据批次（结构随方法不同）。
        """
        # 便捷引用参数
        args = self.args
        # 校验仅支持AWQ或GPTQ
        assert args.quant_method in {'awq', 'gptq'}
        # 便捷引用模板
        template = self.template
        # 量化样本数上限
        n_samples = args.quant_n_samples
        # 切分块大小（最大序列长度）
        block_size = args.max_length

        # 仅使用训练集：split_dataset_ratio=0表示不切验证集
        dataset = load_dataset(
            args.dataset, split_dataset_ratio=0, shuffle=args.dataset_shuffle, **args.get_dataset_kwargs())[0]
        # 打印数据集信息
        logger.info(f'quant_dataset: {dataset}')
        # 打乱数据集以提升代表性
        dataset = dataset.shuffle()

        # 用于收集样本（input_ids或inputs字典）
        samples = []
        # 已处理样本计数
        i = 0
        # 进度条显示目标为n_samples
        prog_bar = tqdm(total=n_samples, dynamic_ncols=True)
        # 是否为多模态模型（影响数据结构）
        is_multimodal = self.model.model_meta.is_multimodal
        # 遍历数据集逐条编码
        for data in dataset:
            try:
                # 调用模板encode获取输入（可能包含input_ids/labels等）
                inputs = template.encode(data)
            except MaxLengthError:
                # 超长样本跳过
                continue
            # 多模态+GPTQ：保留除labels外的inputs字典
            if is_multimodal and args.quant_method == 'gptq':
                inputs.pop('labels', None)
                samples.append(inputs)
            else:
                # 其他情况收集input_ids（可能是token id列表）
                input_ids = inputs['input_ids']
                samples += input_ids
            # 更新计数与进度
            i += 1
            prog_bar.update()
            # 达到样本上限则结束
            if i == n_samples:
                break
        # 关闭进度条
        prog_bar.close()
        # 多模态+GPTQ：直接返回inputs批次
        if is_multimodal and args.quant_method == 'gptq':
            return samples
        # 将所有token按block_size切分成块
        n_split = max(len(samples) // block_size, 1)
        # 打印切分块数量
        logger.info(f'Split into {n_split} blocks')
        # 构建返回结果列表
        res = []
        # 逐块切分构造批次
        for i in range(n_split):
            # 取第i个块的token ids
            input_ids = samples[i * block_size:(i + 1) * block_size]
            # GPTQ需要字典形式；其他量化返回张量批次（shape: [1, seq_len]）
            if args.quant_method == 'gptq':
                res.append({'input_ids': input_ids})
            else:
                # NOTE: [None]表示在张量的最前面增加一个维度（等价于 unsqueeze(0)），常用于给数据增加 batch 维度
                res.append(torch.tensor(input_ids)[None])
        # 返回量化数据批次
        return res

    # 静态上下文管理器：临时替换AWQ模型的move_embed以兼容hook设备
    @staticmethod
    @contextmanager
    def _patch_awq_move_embed(awq_model):
        """
        函数功能：
            某些情况下，存在Accelerate hook时非CPU设备迁移会失败。通过临时替换AWQ模型的
            move_embed方法，允许在非CPU时直接返回以避免冲突，退出上下文后恢复原方法。

        入参：
            awq_model: AWQ包装的模型对象，需包含move_embed方法。

        返回值：
            上下文管理器（无显式返回）。
        """
        # 备份原始move_embed引用
        _origin_move_embed = awq_model.move_embed

        # 定义替代的move_embed实现
        def _move_embed(model, device: str):
            # 若模型存在_hf_hook且目标设备非CPU，则直接返回以避免冲突
            if hasattr(model, '_hf_hook') and device != 'cpu':
                return
            # 否则调用原始实现
            _origin_move_embed(model, device)

        # 应用替换
        awq_model.move_embed = _move_embed
        try:
            # 进入上下文
            yield
        finally:
            # 退出上下文时时恢复原方法
            awq_model.move_embed = _origin_move_embed

    # 执行AWQ量化流程并处理不转换模块等配置
    def awq_model_quantize(self) -> None:
        """
        函数功能：
            使用AWQ进行权重量化，替换量化器的校准数据集获取函数，配置量化参数，
            在上下文中处理move_embed兼容性，完成量化后恢复量化器状态。

        入参：
            无（使用self.args）。

        返回值：
            None
        """
        # 延迟导入AWQ量化器与配置类（避免环境未安装时报错）
        from awq.quantize import quantizer
        from transformers import AwqConfig

        # 便捷引用参数
        args = self.args
        # 打印量化数据集来源
        logger.info(f'Quantization dataset: {args.dataset}')
        # 备份原始的校准数据集函数
        _origin_get_calib_dataset = quantizer.get_calib_dataset
        # 替换为本引擎的量化数据集生成函数
        quantizer.get_calib_dataset = self._get_quant_dataset
        # 量化配置：零点、分组大小、权重量化比特与实现版本
        quant_config = {
            'zero_point': True,
            'q_group_size': args.group_size,
            'w_bit': args.quant_bits,
            'version': 'GEMM'
        }
        # 如果是MoE模型，记录不需转换的模块，以保留门控/专家等关键结构
        if self.model.model_info.is_moe_model:
            quant_config['modules_to_not_convert'] = self.args.get_modules_to_not_convert()
        # 打印量化配置
        logger.info(f'quant_config: {quant_config}')
        logger.info('Start quantizing the model...')
        # 在临时补丁上下文中执行量化，避免move_embed兼容问题
        with self._patch_awq_move_embed(self.model):
            self.model.quantize(
                self.tokenizer, quant_config=quant_config, n_parallel_calib_samples=args.quant_batch_size)
        # 恢复原始的校准数据集函数
        quantizer.get_calib_dataset = _origin_get_calib_dataset  # recover
        # 若量化配置存在不转换模块，则确保lm_head包含在不转换列表内
        if self.model.quant_config.modules_to_not_convert:
            model_arch = args.model_meta.model_arch
            lm_head_key = getattr(model_arch, 'lm_head', None) or 'lm_head'
            if lm_head_key not in self.model.quant_config.modules_to_not_convert:
                self.model.quant_config.modules_to_not_convert.append(lm_head_key)

    # 上下文：替换GPTQ量化器的数据集获取与准备函数
    @contextmanager
    def _patch_gptq(self):
        """
        函数功能：
            将optimum.gptq的量化器数据集接口替换为本引擎实现，以提供自定义的量化数据。
            退出上下文后恢复原始函数引用。
        """
        # 延迟导入以避免非必要依赖
        from optimum.gptq import quantizer
        # 备份原始函数引用
        _get_dataset_origin = quantizer.get_dataset
        _prepare_dataset_origin = quantizer.prepare_dataset
        # 替换为本引擎的数据集函数
        quantizer.get_dataset = self._get_quant_dataset
        quantizer.prepare_dataset = self._prepare_gptq_dataset
        try:
            # 交由调用方执行量化逻辑
            yield
        finally:
            # 恢复原始引用
            quantizer.get_dataset = _get_dataset_origin
            quantizer.prepare_dataset = _prepare_dataset_origin

    # 计算应量化的block前缀名称：选择层数最多的模块列表
    @staticmethod
    def get_block_name_to_quantize(model: nn.Module) -> Optional[str]:
        """
        函数功能：
            推断应进行量化的主干block名称：
            - 若存在language_model前缀，则进入其子模块中搜索
            - 从多个ModuleList/Sequential中选择包含层数最多的一个

        入参：
            model (nn.Module): 待量化的模型对象。

        返回值：
            Optional[str]: block的层级名前缀（如"model.layers"），未找到返回None。
        """
        # 取出模型架构信息，用于判断是否存在language_model前缀
        model_arch = model.model_meta.model_arch
        # 初始化前缀为空
        prefix = ''
        # 若包含language_model字段，则进入其子模块进行后续搜索
        if hasattr(model_arch, 'language_model'):
            assert len(model_arch.language_model) == 1, f'mllm_arch.language_model: {model_arch.language_model}'
            prefix = model_arch.language_model[0]
            model = deep_getattr(model, prefix)

        # 收集候选的模块列表（较长的ModuleList/Sequential且首层非MLP以规避MoE）
        module_lists = []
        for n, m in model.named_modules():
            if (isinstance(m, (nn.ModuleList, nn.Sequential)) and len(m) >= 10
                    and 'mlp' not in m[0].__class__.__name__.lower()):  # fix moe
                module_lists.append((n, m))
        # 选择包含层数最多的模块作为主干块，并返回其全路径前缀
        if module_lists:
            module_list = max(module_lists, key=lambda x: len(x[1]))
            return f'{prefix}.{module_list[0]}'.strip('.')

    # 获取MoE block中的experts容器名称与对象
    @staticmethod
    def _get_experts(block):
        """
        函数功能：
            在给定的block内查找包含专家模块的容器（ModuleList/Sequential），并返回其名称与对象。

        入参：
            block (nn.Module): MoE结构中的一个block。

        返回值：
            Tuple[str, nn.Module]: (容器前缀名, 容器对象)。
        """
        for n, m in block.named_modules():
            if isinstance(m, (nn.ModuleList, nn.Sequential)):
                return n, m

    # 计算MoE模型中需要量化的模块分组（按专家拆分，跳过门控）
    @staticmethod
    def get_modules_in_block_to_quantize(model, block_name: str):
        """
        函数功能：
            针对MoE模型，返回指定block中需要量化的模块分组列表：
            - 按experts维度聚合同类层
            - 跳过门控相关的层（out_features为1或num_experts）

        入参：
            model: 模型对象，需包含model_info.is_moe_model属性。
            block_name (str): 目标block的层级路径前缀。

        返回值：
            Optional[List[List[str]]]: 每组待量化模块的名称列表；非MoE模型返回None。
        """
        # 非MoE模型无需处理
        if not model.model_info.is_moe_model:
            return
        # 延迟导入工具函数
        from optimum.gptq.utils import get_layers
        # 不量化门控部分：取出目标block的最后一层（通常为真正的block容器）
        block = deep_getattr(model, block_name)[-1]
        # 找到experts容器的前缀名与对象
        prefix, experts = QuantEngine._get_experts(block)
        # 专家数量
        num_experts = len(experts)

        # 展平并获取所有层字典
        layers = get_layers(block)
        # 待返回的分组列表
        res = []
        # 以后缀为键收集专家层名称
        experts = defaultdict(list)
        # 记录插入位置索引
        experts_idx = None
        # 遍历所有层，按命名判断是否属于experts容器
        for name, layer in layers.items():
            if name.startswith(prefix):
                # 提取后缀（专家索引/名称）并归类
                suffix = name.rsplit('.', 1)[-1]
                experts[suffix].append(name)
                # 记录experts应插入的索引位置
                experts_idx = len(res)
            # 非门控（out_features既不是1也不是num_experts）的层加入普通分组
            elif layer.out_features not in {1, num_experts}:
                res.append([name])
        # 在记录的位置处插入按专家聚合的分组
        res[experts_idx:experts_idx] = experts.values()
        # 返回分组结果
        return res

    # 执行GPTQ量化流程，返回量化器对象以便后续保存
    def gptq_model_quantize(self):
        """
        函数功能：
            使用Optimum GPTQ量化器对模型进行量化：
            - 计算待量化的block名称与分组
            - 替换量化器数据集接口
            - 执行量化并清理量化配置中的临时字段

        入参：
            无（使用self.args）。

        返回值：
            GPTQQuantizer: 量化器对象（已完成量化，可调用save保存权重）。
        """
        # 延迟导入量化器
        from optimum.gptq import GPTQQuantizer
        # 便捷引用参数
        args = self.args
        # 打印量化数据集来源
        logger.info(f'Quantization dataset: {args.dataset}')
        # 推断block名称与在block中的待量化模块分组
        block_name_to_quantize = self.get_block_name_to_quantize(self.model)
        modules_in_block_to_quantize = self.get_modules_in_block_to_quantize(self.model, block_name_to_quantize)
        # 打印调试信息
        logger.info(f'block_name_to_quantize: {block_name_to_quantize}')
        logger.info(f'modules_in_block_to_quantize: {modules_in_block_to_quantize}')
        # 临时替换量化器数据集接口并执行量化
        with self._patch_gptq():
            gptq_quantizer = GPTQQuantizer(
                bits=args.quant_bits,
                group_size=args.group_size,
                dataset=','.join(args.dataset),
                batch_size=args.quant_batch_size,
                block_name_to_quantize=block_name_to_quantize,
                modules_in_block_to_quantize=modules_in_block_to_quantize)
            # 将block名称加入序列化键，便于后续保存/恢复
            gptq_quantizer.serialization_keys.append('block_name_to_quantize')
            # 进入量化阶段：提示耗时较长且无进度条
            logger.info('Start quantizing the model...')
            logger.warning('The process of packing the model takes a long time and there is no progress bar. '
                           'Please be patient and wait...')
            # 执行量化（传入模型与tokenizer）
            gptq_quantizer.quantize_model(self.model, self.tokenizer)
            # 清理量化配置中的临时字段（dataset）
            self.model.config.quantization_config.pop('dataset', None)
        # 返回量化器对象以便外部保存
        return gptq_quantizer


# 便捷入口：创建量化引擎并执行量化
def quantize_model(args: ExportArguments):
    """
    函数功能：
        量化导出的便捷入口。根据入参创建QuantEngine并执行量化流程。

    入参：
        args (ExportArguments): 量化参数对象。

    返回值：
        None

    示例：
        >>> quantize_model(ExportArguments(...))
    """
    # 创建量化引擎并执行量化
    QuantEngine(args).quantize()
