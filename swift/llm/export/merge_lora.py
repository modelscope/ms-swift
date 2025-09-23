"""
模块功能概述：
本模块提供将 LoRA 适配器权重合并回基座模型并保存的工具函数：
- check_tie_word_embeddings: 在特定条件下关闭HF配置中的tie_word_embeddings，避免合并不一致。
- merge_lora: 执行LoRA权重合并与保存，并根据参数处理输出目录、量化配置与device_map。

使用示例：
>>> from swift.llm import ExportArguments
>>> args = ExportArguments(adapters=["./lora"] , output_dir="./merged")
>>> merge_lora(args)
"""

# 版权声明：阿里巴巴及其附属公司保留所有权利
# Copyright (c) Alibaba, Inc. and its affiliates.

# 导入os模块：用于判断目录是否存在与路径处理
import os

# 导入导出流程所需的工具与工厂方法：
# - ExportArguments: 导出参数定义
# - HfConfigFactory: 统一读写HF配置属性的工厂
# - prepare_model_template: 按参数准备模型与模板
# - save_checkpoint: 将模型与处理器保存到磁盘
from swift.llm import ExportArguments, HfConfigFactory, prepare_model_template, save_checkpoint
# 导入Swift调优器：用于合并并卸载LoRA权重
from swift.tuners import Swift
# 导入日志工具：统一记录信息
from swift.utils import get_logger

# 初始化模块级日志器：用于打印合并流程的关键信息
logger = get_logger()


# 校验并在必要时关闭tie_word_embeddings设置，避免Embedding权重绑定导致的不一致
def check_tie_word_embeddings(model):
    """
    函数功能：
        当模型配置`tie_word_embeddings=True`且输入/输出Embedding模块被ModulesToSaveWrapper包裹时，
        将配置置为False，避免在LoRA合并与卸载后因Embedding共享导致的权重不一致问题。

    入参：
        model: HF/Transformers模型实例，需包含`config`属性与get_input_embeddings/get_output_embeddings接口。

    返回值：
        None

    示例：
        >>> check_tie_word_embeddings(model)
    """
    # 读取模型配置对象，便于访问/修改配置属性
    config = model.config
    try:
        # 延迟导入peft的包装器类型：仅在运行时可用，避免非必要依赖
        from peft.utils import ModulesToSaveWrapper
        # 若配置未开启tie_word_embeddings，直接返回无需处理
        if not HfConfigFactory.get_config_attr(config, 'tie_word_embeddings'):
            return
        # 遍历输入与输出Embedding模块，判断是否都为ModulesToSaveWrapper包装
        for module in [model.get_input_embeddings(), model.get_output_embeddings()]:
            # 只要有一个不是包装器类型，则无需改动配置，直接返回
            if not isinstance(module, ModulesToSaveWrapper):
                return
        # 当两者均为包装器类型时，关闭tie_word_embeddings以避免绑定
        HfConfigFactory.set_config_attr(config, 'tie_word_embeddings', False)
    except Exception:
        # 容错处理：若环境缺少peft或访问异常，静默跳过不影响主流程
        pass


# 合并LoRA权重到基座模型并保存，处理量化/设备映射/输出目录等细节
def merge_lora(args: ExportArguments, device_map=None, replace_if_exists=False) -> None:
    """
    函数功能：
        将LoRA适配器权重合并回原模型（若量化开启则暂时关闭以在原模型上合并），随后保存到输出目录。
        支持外部传入device_map，并在目录已存在时根据replace_if_exists决定是否跳过保存。

    入参：
        args (ExportArguments): 导出参数对象，包含adapters、output_dir、max_shard_size等信息。
        device_map (Any|None): 可选的设备映射字典/字符串，用于控制权重加载/推理设备分布。
        replace_if_exists (bool): 若目标目录已存在，是否强制替换保存；默认False为跳过保存。

    返回值：
        None

    示例：
        >>> merge_lora(ExportArguments(adapters=["./lora"], output_dir="./merged"), replace_if_exists=True)
    """
    # 若设置了强制替换，先记录日志以便排查
    if replace_if_exists:
        logger.info(f'replace_if_exists: {replace_if_exists}')
    # 计算输出目录：优先使用args.output_dir，否则以第一个adapter为基名添加"-merged"
    output_dir = getattr(args, 'output_dir', None) or f'{args.adapters[0]}-merged'
    # 若目录已存在且不允许替换，则提示并跳过保存流程
    if os.path.exists(output_dir) and not replace_if_exists:
        logger.info(f'The weight directory for the merged LoRA already exists in {output_dir}, '
                    'skipping the saving process.')
    else:
        # 如果模型处于量化状态，合并应在原始未量化模型上进行（避免合并到量化权重）
        # 参考：https://github.com/huggingface/peft/issues/2321
        # 显式关闭量化方法，确保随后加载的是未量化模型
        args.quant_method = None
        # 备份原始device_map，合并完成后恢复
        origin_device_map = args.device_map
        # 使用外部传入的device_map（若提供），否则沿用原配置
        args.device_map = device_map or args.device_map
        # 记录用于合并的device_map，便于调试
        logger.info(f'merge_device_map: {device_map}')
        # 按参数准备模型与模板（包含处理器等），用于后续合并与保存
        model, template = prepare_model_template(args)
        # 打印合并开始日志
        logger.info('Merge LoRA...')
        # 在必要时关闭tie_word_embeddings，避免Embedding共享影响权重一致性
        check_tie_word_embeddings(model)
        # 调用Swift工具将LoRA权重合并回原模型，并卸载LoRA结构
        Swift.merge_and_unload(model)
        # Swift合并后，真实的基础模型位于model.model属性中，取出供保存使用
        model = model.model
        # 打印保存开始日志
        logger.info('Saving merged weights...')

        # 将合并后的模型与处理器保存到指定目录；支持安全序列化与分片大小配置，并记录额外文件
        save_checkpoint(
            model,
            template.processor,
            output_dir,
            safe_serialization=args.safe_serialization,
            model_dirs=args.adapters,
            max_shard_size=args.max_shard_size,
            additional_saved_files=model.model_meta.additional_saved_files)
        # 打印保存完成日志
        logger.info(f'Successfully merged LoRA and saved in {output_dir}.')
        # 恢复原始device_map，避免影响后续流程
        args.device_map = origin_device_map

    # 更新args以指向合并后的模型目录，并清空adapters，方便后续导出链路复用
    args.model = output_dir
    args.model_dir = output_dir
    args.adapters = []
