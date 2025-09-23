"""
模块功能概述：
本模块提供将经过模板与分词器编码的训练/验证数据集缓存到磁盘的导出工具：
- ExportCachedDataset: 继承训练管道（SwiftSft），复用数据与模板准备流程，编码并保存HF数据集。
- export_cached_dataset: 便捷函数入口，构造管道并执行导出主流程。
"""

# 版权声明：阿里巴巴及其附属公司保留所有权利
# Copyright (c) Alibaba, Inc. and its affiliates.

# 导入os模块：用于路径拼接与文件系统相关操作
import os
# 从typing导入类型注解：List/Optional/Union用于描述参数类型
from typing import List, Optional, Union

# 从swift.llm导入导出参数类：承载导出相关的命令行/配置参数
from swift.llm import ExportArguments
# 从swift.llm.train导入基础训练管线：复用其模板/模型/分词器与数据集准备逻辑
from swift.llm.train import SwiftSft
# 从swift.utils导入日志工具：统一日志接口
from swift.utils import get_logger

# 初始化模块级日志器：用于记录导出过程信息
logger = get_logger()


# 定义导出缓存数据集的管道类：复用SwiftSft的数据准备与编码能力
class ExportCachedDataset(SwiftSft):
    """
    类功能：
        利用SwiftSft的模板与模型/分词器初始化流程，构建训练/验证数据集，
        执行模板编码后保存至磁盘，以便后续快速加载与复用。

    关键属性：
        args_class: 参数类，固定为ExportArguments。
        args: 解析后的导出参数实例。
    """

    # 指定该管道使用的参数类类型（导出相关参数）
    args_class = ExportArguments
    # 类型标注：实例属性args为上述参数类（供IDE与静态检查使用）
    args: args_class

    # 构造函数：完成模板、模型/分词器与processor初始化
    def __init__(self, args: Optional[Union[List[str], ExportArguments]] = None) -> None:
        """
        函数功能：
            初始化导出管道，复用父类的初始化能力，并按模板配置准备processor与模型/分词器。

        入参：
            args (Optional[Union[List[str], ExportArguments]]): 参数列表或参数对象；
                为空时使用默认参数。

        返回值：
            None

        示例：
            >>> ExportCachedDataset(["--output_dir", "./cache"]).main()
        """
        # 调用父类（SwiftSft）的构造函数，完成基础参数解析与状态初始化
        super(SwiftSft, self).__init__(args)
        # 训练信息占位（本导出流程不使用训练，保留字段以兼容父类接口）
        self.train_msg = {}  # dummy
        # 处理器占位，稍后由模板进行初始化
        self.processor = None
        # 准备模板：包括chat模板/系统提示等，供后续编码与processor初始化使用
        self._prepare_template()
        # 准备模型与分词器：是否加载模型由模板use_model字段决定
        self._prepare_model_tokenizer(load_model=self.template.use_model)
        # 使用模板初始化processor（可能依赖于分词器与特殊tokens）
        self.template.init_processor(self.processor)

    # 主流程：获取数据集→编码→展示信息→保存到磁盘
    def main(self):
        """
        函数功能：
            执行导出主流程：
            1) 构造原始训练/验证数据集
            2) 通过模板/processor进行编码
            3) 展示数据集基本信息
            4) 将编码后的数据集保存至指定输出目录

        入参：
            无（使用self.args内的参数）。

        返回值：
            None

        示例：
            >>> ExportCachedDataset(["--output_dir", "./cache"]).main()
        """
        # 获取原始训练/验证数据集（依据参数与模板组合生成）
        train_dataset, val_dataset = self._get_dataset()
        # 对训练/验证数据集进行编码（tokenize/pack等），返回编码后的数据集
        train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)
        # 展示编码后数据集的基本信息（如样本数/示例等）
        self._show_dataset(train_dataset, val_dataset)
        # 将训练集缓存保存到磁盘的train子目录
        train_dataset.save_to_disk(os.path.join(self.args.output_dir, 'train'))
        # 若存在验证集，则同样保存到val子目录
        if val_dataset is not None:
            val_dataset.save_to_disk(os.path.join(self.args.output_dir, 'val'))
        # 记录导出完成日志与输出目录位置
        logger.info(f'Dataset saved to `{self.args.output_dir}`')


# 便捷导出入口：构造管道并执行主流程
def export_cached_dataset(args: Optional[Union[List[str], ExportArguments]] = None):
    """
    函数功能：
        便捷导出函数。根据入参构造ExportCachedDataset对象并执行其main流程。

    入参：
        args (Optional[Union[List[str], ExportArguments]]): 参数列表或参数对象；
            为空时使用默认参数。

    返回值：
        None

    示例：
        >>> export_cached_dataset(["--output_dir", "./cache"])  
    """
    # 创建导出管道实例并调用其主流程main，返回执行结果（若有）
    return ExportCachedDataset(args).main()
