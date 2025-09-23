"""
模块功能概述：
本模块统一管理多种导出/转换路径的调度入口：
- to_peft_format: 将 Swift 适配器转换成 PEFT 目录格式
- merge_lora: 将 LoRA 适配器权重合并进基座模型
- quant_method: 对模型进行量化导出
- to_ollama: 导出为 Ollama 可用的模型格式
- to_cached_dataset: 导出编码后的缓存数据集到磁盘
- to_mcore / to_hf: Megatron-core 与 HF 权重互转
- push_to_hub: 推送模型目录到 Model Hub
"""

# 版权声明：阿里巴巴及其附属公司保留所有权利
# Copyright (c) Alibaba, Inc. and its affiliates.

# 导入类型注解：List/Optional/Union 用于方法签名说明
from typing import List, Optional, Union

# 导入导出参数类与基础管道：用于承载导出配置并复用管道主流程
from swift.llm import ExportArguments, SwiftPipeline
# 导入转换工具：将 Swift 适配器目录转换为 PEFT 目录格式
from swift.tuners import swift_to_peft_format
# 导入日志工具：统一日志记录
from swift.utils import get_logger
# 导入子功能：导出缓存数据集
from .cached_dataset import export_cached_dataset
# 导入子功能：合并 LoRA 权重
from .merge_lora import merge_lora
# 导入子功能：导出为 Ollama 模型
from .ollama import export_to_ollama
# 导入子功能：模型量化
from .quant import quantize_model

# 初始化模块级日志器：用于记录导出流程信息
logger = get_logger()


# 定义导出主管道：根据参数选择不同导出/转换路径
class SwiftExport(SwiftPipeline):
    """
    类功能：
        负责将导出相关的多种分支能力汇聚在一个管道中，根据命令行/配置参数决定执行流程：
        - 适配器格式转换、LoRA 合并、量化、Ollama 导出、缓存数据集导出、Megatron/HF 互转、推送到 Hub。

    关键属性：
        args_class: 参数类类型，固定为 ExportArguments。
        args: 解析后的导出参数实例。
    """

    # 指定该管道使用的参数类类型
    args_class = ExportArguments
    # 类型标注：实例属性 args 为上述参数类（方便静态检查）
    args: args_class

    # 运行导出主流程：按优先级顺序选择并执行对应分支
    def run(self):
        """
        函数功能：
            读取导出参数并依次判断各导出/转换分支，执行匹配的功能模块。

        入参：
            无（使用 self.args 内的参数）。

        返回值：
            None

        示例：
            >>> SwiftExport(["--merge_lora", "--output_dir", "./out"]).run()
        """
        # 便捷引用参数对象，避免多次属性访问
        args = self.args
        # 若需要将 Swift 适配器转换为 PEFT 目录格式，则先执行转换并替换第一个适配器路径
        if args.to_peft_format:
            args.adapters[0] = swift_to_peft_format(args.adapters[0], args.output_dir)
        # 若需要合并 LoRA，则可能需要在合并阶段临时清空 output_dir 以避免冲突
        if args.merge_lora:
            # 先保存原始输出目录，后续恢复
            output_dir = args.output_dir
            # 若还计划执行后续步骤（转PEFT/量化/Ollama/推Hub），则在合并时将输出目录置空
            if args.to_peft_format or args.quant_method or args.to_ollama or args.push_to_hub:
                args.output_dir = None
            # 执行 LoRA 合并流程（可能将合并结果输出到 adapters[0] 或指定目录）
            merge_lora(args)
            # 恢复原始输出目录，确保后续步骤使用用户指定的目录
            args.output_dir = output_dir  # recover
        # 若指定量化方法，则执行量化导出
        if args.quant_method:
            quantize_model(args)
        # 否则若指定导出为 Ollama，则执行 Ollama 导出
        elif args.to_ollama:
            export_to_ollama(args)
        # 否则若指定导出缓存数据集，则调用对应函数
        elif args.to_cached_dataset:
            export_cached_dataset(args)
        # 否则若指定导出为 Megatron-core，则进行 HF -> mcore 转换
        elif args.to_mcore:
            # 延迟导入，避免非必要依赖
            from swift.megatron import convert_hf2mcore
            convert_hf2mcore(args)
        # 否则若指定从 Megatron-core 转回 HF，则进行 mcore -> HF 转换
        elif args.to_hf:
            # 延迟导入，避免非必要依赖
            from swift.megatron import convert_mcore2hf
            convert_mcore2hf(args)
        # 否则若指定推送到 Hub，则执行推送逻辑
        elif args.push_to_hub:
            # 若 adapters 存在，默认推送第一个适配器目录；否则推送 model_dir
            model_dir = args.adapters and args.adapters[0] or args.model_dir
            # 基本校验：必须存在可推送的目录
            assert model_dir, f'model_dir: {model_dir}'
            # 调用 Hub 客户端执行推送，传递模型ID、目录、Token、私有仓库开关与提交信息
            args.hub.push_to_hub(
                args.hub_model_id,
                model_dir,
                token=args.hub_token,
                private=args.hub_private_repo,
                commit_message=args.commit_message)


# 导出主入口：构造并运行导出管道
def export_main(args: Optional[Union[List[str], ExportArguments]] = None):
    """
    函数功能：
        便捷入口。根据输入参数构造 SwiftExport 管道并执行主流程。

    入参：
        args (Optional[Union[List[str], ExportArguments]]): 参数列表或参数对象。

    返回值：
        None

    示例：
        >>> export_main(["--to_ollama", "--output_dir", "./ollama-out"])  
    """
    # 创建导出管道并执行主流程
    return SwiftExport(args).main()
