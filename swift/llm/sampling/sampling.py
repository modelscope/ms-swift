"""模块功能概述：
该模块实现了 Swift 框架的采样（Sampling）功能，用于从语言模型中生成训练数据。

核心功能：
1. SwiftSampling 类：继承自 SwiftPipeline，封装完整的采样流程管理；
2. 支持多种采样器类型：vanilla（普通采样）、mcts（蒙特卡洛树搜索）、distill（蒸馏采样）；
3. 数据集分片处理：支持将大数据集拆分为多个分片并行处理；
4. 断点续传机制：支持从中断处继续采样，避免重复计算；
5. 批量采样：支持批量处理数据以提高效率；
6. 文件管理：自动管理输出文件、临时文件、恢复文件和检查点状态。

应用场景：
- 为强化学习（RL）训练生成策略模型的采样数据；
- 使用奖励模型（PRM/ORM）对模型输出进行评分和筛选；
- 在蒸馏场景中从教师模型生成训练数据；
- 大规模数据集的分布式采样处理。

典型使用：
    # 基本用法：普通采样
    $ swift sample --model_id_or_path qwen/Qwen-7B --dataset alpaca-zh --sampler_type sample
    
    # 使用 MCTS 采样器 + 奖励模型
    $ swift sample --model_id_or_path qwen/Qwen-7B --dataset math --sampler_type mcts \\
                   --prm_model prm_model_path --orm_model orm_model_path
    
    # 数据集分片 + 断点续传
    $ swift sample --model_id_or_path qwen/Qwen-7B --dataset large_dataset \\
                   --data_range 0,10 --resume True --output_file shard_0.jsonl
    
    # 代码调用
    >>> from swift.llm.sampling import sampling_main, SamplingArguments
    >>> args = SamplingArguments(
    ...     model_id_or_path='qwen/Qwen-7B',
    ...     dataset=['alpaca-zh'],
    ...     sampler_type='sample',
    ...     num_sampling_per_gpu_batch_size=8,
    ...     output_dir='./samples'
    ... )
    >>> sampling_main(args)
"""
# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，标注代码版权所有者
import os  # 引入 os 模块，用于文件路径操作、目录创建等系统操作
import shutil  # 引入 shutil 模块，用于文件复制、移动等高级文件操作
import time  # 引入 time 模块，用于生成时间戳（文件备份命名）
from typing import List, Optional, Union  # 引入类型注解，用于参数和返回值的类型提示

import json  # 引入 json 模块，用于读写检查点状态文件（保存采样进度）

from swift.llm import SamplingArguments, SwiftPipeline, load_dataset  # 引入 Swift 框架的核心类：SamplingArguments（采样参数类）、SwiftPipeline（基础管道类）、load_dataset（数据集加载函数）
from swift.utils import get_logger  # 引入日志工具函数，用于创建模块级日志记录器

logger = get_logger()  # 创建模块级日志记录器，用于输出运行时信息、警告和错误日志


class SwiftSampling(SwiftPipeline):
    """类功能：
    定义 Swift 采样类，继承自 SwiftPipeline，负责从语言模型中生成采样数据。
    
    核心职责：
        1. 参数管理：使用 SamplingArguments 类型管理采样配置；
        2. 采样器选择：根据 sampler_type 参数动态加载对应的采样器（VanillaSampler、MctsSampler、DistillSampler）；
        3. 数据集处理：加载数据集并根据 data_range 进行分片；
        4. 采样流程控制：批量处理数据、调用采样器生成输出；
        5. 断点续传管理：保存和恢复采样进度，支持中断后继续；
        6. 文件输出管理：管理输出文件、临时文件、恢复文件和检查点状态文件。
    
    继承关系：
        - 继承自 SwiftPipeline，复用参数解析、日志记录等基础功能。
    
    属性：
        - args_class: 指定参数类为 SamplingArguments；
        - args: 采样参数实例，包含模型路径、数据集、采样器类型、批量大小等配置；
        - sampler: 采样器实例（VanillaSampler、MctsSampler 或 DistillSampler）；
        - cur_piece: 当前处理的数据分片索引（用于分布式采样）；
        - total_piece: 数据集总分片数量。
    
    实际使用示例：
        示例 1：基本采样流程（普通采样器）
        >>> from swift.llm.sampling import SwiftSampling, SamplingArguments
        >>> args = SamplingArguments(
        ...     model_id_or_path='qwen/Qwen-7B-Chat',
        ...     dataset=['alpaca-zh'],
        ...     sampler_type='sample',  # 使用普通采样器
        ...     num_sampling_per_gpu_batch_size=4,
        ...     num_sampling_per_gpu_batches=10,
        ...     output_dir='./output',
        ...     output_file='samples.jsonl'
        ... )
        >>> sampler = SwiftSampling(args)
        >>> sampler.main()  # 执行采样流程
        # 输出：./output/samples.jsonl（包含 40 条采样数据）
        
        示例 2：使用 MCTS 采样器 + 奖励模型
        >>> args = SamplingArguments(
        ...     model_id_or_path='qwen/Qwen-Math-7B',
        ...     dataset=['math'],
        ...     sampler_type='mcts',  # 使用 MCTS 采样器
        ...     prm_model='prm_model_path',  # 过程奖励模型
        ...     orm_model='orm_model_path',  # 结果奖励模型
        ...     num_sampling_per_gpu_batch_size=2
        ... )
        >>> sampler = SwiftSampling(args)
        >>> sampler.run()  # 执行采样
        
        示例 3：分片采样 + 断点续传
        >>> # 第一个分片（处理数据集的前 1/10）
        >>> args = SamplingArguments(
        ...     model_id_or_path='qwen/Qwen-7B',
        ...     dataset=['large_dataset'],
        ...     data_range=[0, 10],  # 当前分片索引 0，总分片数 10
        ...     resume=True,  # 启用断点续传
        ...     output_file='shard_0.jsonl'
        ... )
        >>> sampler = SwiftSampling(args)
        >>> # 假设采样到第 5 批次时程序中断
        >>> sampler.run()
        >>> # 重新启动后会从第 6 批次继续，无需重复计算前 5 批次
    """
    args_class = SamplingArguments  # 指定该管道使用的参数类为 SamplingArguments，用于参数解析和验证
    args: args_class  # 类型注解：声明 args 属性的类型为 SamplingArguments 实例

    def __init__(self, args: Optional[Union[List[str], SamplingArguments]] = None) -> None:
        """函数功能：
        初始化 SwiftSampling 实例，接受命令行参数列表或 SamplingArguments 实例，配置采样器、数据分片参数，并保存配置文件。
        
        参数：
            args (Optional[Union[List[str], SamplingArguments]]): 
                - 可选参数，支持三种形式：
                  1. None: 从 sys.argv 读取命令行参数；
                  2. List[str]: 命令行参数列表（如 ['--model_id_or_path', 'qwen/Qwen-7B']）；
                  3. SamplingArguments: 已实例化的参数对象。
        
        返回值：
            None
        
        初始化流程：
            1. 调用父类初始化方法，解析参数；
            2. 保存参数配置到输出目录；
            3. 创建输出目录；
            4. 初始化数据分片参数（cur_piece、total_piece）；
            5. 根据 sampler_type 动态加载对应的采样器实例。
        
        实际使用示例：
            示例 1：从命令行参数初始化
            >>> sampler = SwiftSampling()  # 自动读取 sys.argv
            
            示例 2：使用参数列表初始化
            >>> sampler = SwiftSampling([
            ...     '--model_id_or_path', 'qwen/Qwen-7B',
            ...     '--dataset', 'alpaca-zh',
            ...     '--sampler_type', 'sample'
            ... ])
            
            示例 3：使用参数对象初始化
            >>> args = SamplingArguments(
            ...     model_id_or_path='qwen/Qwen-7B',
            ...     sampler_type='mcts',
            ...     data_range=[0, 4]  # 第 1 个分片，共 4 个分片
            ... )
            >>> sampler = SwiftSampling(args)
            >>> print(sampler.cur_piece)  # 输出: 0
            >>> print(sampler.total_piece)  # 输出: 4
        """
        super().__init__(args)  # 调用父类 SwiftPipeline 的初始化方法，完成参数解析和基础配置
        self.args.save_args()  # 将参数配置保存到输出目录下的配置文件（通常为 args.json），便于后续查看和复现
        os.makedirs(self.args.output_dir, exist_ok=True)  # 创建输出目录，若目录已存在则不报错（exist_ok=True）
        self.cur_piece = 0  # 初始化当前数据分片索引为 0（默认处理第一个分片）
        self.total_piece = 1  # 初始化总分片数为 1（默认不分片，处理整个数据集）

        if self.args.data_range:  # 若参数中指定了 data_range（数据分片范围）
            self.cur_piece, self.total_piece = self.args.data_range  # 从 data_range 中解包获取当前分片索引和总分片数（如 [0, 10] 表示第 1 个分片，共 10 个分片）

        if self.args.sampler_type == 'sample':  # 若采样器类型为 'sample'（普通采样器）
            from swift.llm.sampling.vanilla_sampler import VanillaSampler  # 延迟导入 VanillaSampler 类（避免循环导入，减少启动时间）
            self.sampler = VanillaSampler(self.args)  # 实例化普通采样器，传入参数对象
        elif self.args.sampler_type == 'mcts':  # 若采样器类型为 'mcts'（蒙特卡洛树搜索采样器）
            from swift.llm.sampling.mcts import MctsSampler  # 延迟导入 MctsSampler 类
            self.sampler = MctsSampler(self.args)  # 实例化 MCTS 采样器，传入参数对象
        elif self.args.sampler_type == 'distill':  # 若采样器类型为 'distill'（蒸馏采样器）
            from swift.llm.sampling.distill_sampler import DistillSampler  # 延迟导入 DistillSampler 类
            self.sampler = DistillSampler(self.args)  # 实例化蒸馏采样器，传入参数对象
        else:  # 否则（采样器类型不支持）
            raise ValueError(f'Unsupported sampler type: {self.args.sampler_type}')  # 抛出 ValueError 异常，提示不支持的采样器类型

    def _get_dataset(self):
        """函数功能：
        私有方法，加载数据集并根据 data_range 参数进行分片处理，返回当前分片的数据子集。
        
        参数：
            无（使用 self.args 中的配置参数）
        
        返回值：
            Dataset: Hugging Face Dataset 对象，包含当前分片的数据样本
        
        处理流程：
            1. 从 args 中获取数据集加载参数；
            2. 调用 load_dataset 加载完整数据集；
            3. 计算每个分片的长度（总长度 / 总分片数）；
            4. 根据 cur_piece 选择对应的数据子集；
            5. 返回分片后的数据集。
        
        实际使用示例：
            示例 1：不分片（默认）
            >>> sampler = SwiftSampling(['--dataset', 'alpaca-zh'])
            >>> dataset = sampler._get_dataset()
            >>> print(len(dataset))  # 输出: 10000（假设完整数据集有 10000 条）
            
            示例 2：分片处理（4 个分片，取第 1 个）
            >>> sampler = SwiftSampling([
            ...     '--dataset', 'alpaca-zh',
            ...     '--data_range', '0,4'
            ... ])
            >>> dataset = sampler._get_dataset()
            >>> print(len(dataset))  # 输出: 2500（10000 / 4）
            
            示例 3：分片处理（4 个分片，取第 4 个）
            >>> sampler.cur_piece = 3  # 最后一个分片（索引从 0 开始）
            >>> dataset = sampler._get_dataset()
            >>> # 返回索引 7500-9999 的数据（共 2500 条）
        """
        args = self.args  # 获取采样参数对象（简化代码，提高可读性）
        dataset_kwargs = args.get_dataset_kwargs()  # 调用参数对象的方法获取数据集加载参数（如 dataset_test_ratio、dataset_sample 等），返回字典
        sampling_dataset, _ = load_dataset(  # 调用 load_dataset 函数加载数据集，返回元组 (训练集, 验证集)，这里只取训练集
            args.dataset, split_dataset_ratio=0., shuffle=args.dataset_shuffle, **dataset_kwargs)  # 传入数据集名称、不拆分验证集（split_dataset_ratio=0.）、是否打乱（dataset_shuffle）和其他参数
        logger.info(f'Sampling_dataset: {sampling_dataset}')  # 记录日志：输出加载的数据集信息（包含数据集大小、特征列等）
        dataset_len = len(sampling_dataset)  # 获取完整数据集的长度（样本总数）
        piece_len = dataset_len // self.total_piece  # 计算每个分片的长度：数据集总长度整除总分片数（使用整除 // 确保结果为整数）
        sampling_dataset = sampling_dataset.select(range(piece_len * self.cur_piece, piece_len * (self.cur_piece + 1)))  # 使用 select 方法选择当前分片的数据：起始索引为 piece_len * cur_piece，结束索引为 piece_len * (cur_piece + 1)
        return sampling_dataset  # 返回分片后的数据集对象


    def run(self):
        """函数功能：
        主运行方法，执行完整的采样流程，包括数据集加载、批量采样、断点续传、文件管理等。
        
        参数：
            无（使用 self.args 中的配置参数）
        
        返回值：
            None
        
        执行流程：
            1. 创建输出目录和准备文件路径；
            2. 检查是否需要覆盖已存在的输出文件；
            3. 处理断点续传逻辑（加载检查点状态）；
            4. 加载和分片数据集；
            5. 批量迭代处理数据：
               a. 截取当前批次数据；
               b. 调用采样器生成输出；
               c. 写入临时文件并刷新；
               d. 备份到恢复文件；
               e. 保存检查点状态；
            6. 完成后重命名文件（临时文件 -> 最终输出文件）。
        
        实际使用示例：
            示例 1：完整采样流程（无断点续传）
            >>> sampler = SwiftSampling([
            ...     '--model_id_or_path', 'qwen/Qwen-7B',
            ...     '--dataset', 'alpaca-zh',
            ...     '--num_sampling_per_gpu_batch_size', '8',
            ...     '--num_sampling_per_gpu_batches', '100',
            ...     '--output_file', 'samples.jsonl'
            ... ])
            >>> sampler.run()
            # 生成 100 批次 × 8 样本 = 800 条采样数据
            # 输出文件：./output/samples.jsonl
            
            示例 2：断点续传（程序中断后恢复）
            >>> # 第一次运行（假设在第 50 批次时中断）
            >>> sampler.run()
            >>> # 程序中断...检查点状态已保存到 ckpt_state.json
            >>> 
            >>> # 第二次运行（启用断点续传）
            >>> sampler = SwiftSampling([
            ...     '--resume', 'True',  # 启用断点续传
            ...     # ...其他参数相同
            ... ])
            >>> sampler.run()
            # 从第 51 批次继续，跳过前 50 批次
            
            示例 3：覆盖已存在的输出文件
            >>> sampler = SwiftSampling([
            ...     '--override_exist_file', 'True',  # 覆盖已存在文件
            ...     # ...其他参数
            ... ])
            >>> sampler.run()
            # 即使输出文件已存在，也会重新采样并覆盖
        """
        os.makedirs(self.args.output_dir, exist_ok=True)  # 创建输出目录，若目录已存在则不报错（确保输出目录存在）
        iter_file = os.path.join(self.args.output_dir, self.args.output_file)  # 拼接最终输出文件路径（如 ./output/samples.jsonl）
        resume_file = os.path.join(self.args.output_dir, self.args.output_file + '.resume')  # 拼接恢复文件路径（如 ./output/samples.jsonl.resume），用于断点续传
        tmp_file = os.path.join(self.args.output_dir, self.args.output_file + '.tmp')  # 拼接临时文件路径（如 ./output/samples.jsonl.tmp），用于写入过程中的缓存
        ckpt_state_file = os.path.join(self.args.output_dir, 'ckpt_state.json')  # 拼接检查点状态文件路径（保存当前采样的批次索引）
        if os.path.exists(iter_file) and not self.args.override_exist_file:  # 若最终输出文件已存在且未设置覆盖标志
            return  # 直接返回，跳过采样流程（避免重复计算）

        index_resume = -1  # 初始化恢复索引为 -1（表示从头开始采样）
        write_mode = 'w'  # 初始化文件写入模式为 'w'（覆盖写入）
        if self.args.resume:  # 若启用断点续传（resume=True）
            write_mode = 'a'  # 修改文件写入模式为 'a'（追加写入，保留已有内容）
            if os.path.exists(resume_file):  # 若恢复文件已存在（说明之前有未完成的采样）
                shutil.copyfile(resume_file, tmp_file)  # 将恢复文件复制到临时文件（恢复之前的进度）

            if os.path.exists(ckpt_state_file):  # 若检查点状态文件存在（保存了上次采样的批次索引）
                with open(ckpt_state_file, 'r') as ckpt_state:  # 以只读模式打开检查点状态文件
                    data = json.load(ckpt_state)  # 从 JSON 文件中加载检查点数据（字典格式，如 {'index': 49}）
                    index_resume = data.get('index', -1)  # 获取上次采样的批次索引，若不存在则默认为 -1
                    logger.info(f'Loaded index_resume: {index_resume}')  # 记录日志：输出恢复的批次索引
        else:  # 否则（未启用断点续传，从头开始）
            if os.path.exists(tmp_file):  # 若临时文件存在（可能是之前未清理的文件）
                os.remove(tmp_file)  # 删除临时文件，确保从干净状态开始

        dataset = self._get_dataset()  # 调用私有方法加载和分片数据集
        dataset_len = len(dataset)  # 获取当前分片数据集的长度（样本总数）
        total_iters = int(dataset_len // self.args.num_sampling_per_gpu_batch_size)  # 计算总迭代次数：数据集长度整除批量大小（如 800 样本 / 8 批量大小 = 100 次迭代）

        if self.args.num_sampling_per_gpu_batches is None or self.args.num_sampling_per_gpu_batches > total_iters:  # 若未指定采样批次数或指定的批次数超过总迭代次数
            self.args.num_sampling_per_gpu_batches = total_iters  # 将采样批次数设置为总迭代次数（采样整个数据集）

        with open(tmp_file, write_mode) as f:  # 以指定模式（'w' 或 'a'）打开临时文件进行写入
            for _index in range(self.args.num_sampling_per_gpu_batches):  # 遍历所有采样批次（从 0 到 num_sampling_per_gpu_batches-1）
                if _index <= index_resume:  # 若当前批次索引小于等于恢复索引（说明该批次已完成）
                    continue  # 跳过该批次，继续下一个批次（避免重复采样）
                logger.info(f' Sampling index:{_index}')  # 记录日志：输出当前采样的批次索引
                slices = dataset[self.args.num_sampling_per_gpu_batch_size  # 从数据集中切片提取当前批次的数据：起始索引为 batch_size * _index
                                 * _index:self.args.num_sampling_per_gpu_batch_size * (_index + 1)]  # 结束索引为 batch_size * (_index + 1)（如第 0 批次取索引 0-7 的数据）
                slices = self.sampler.truncate_input(slices)  # 调用采样器的 truncate_input 方法截断输入（避免超过模型的最大长度限制）
                generated = self.sampler.do_sample(slices)  # 调用采样器的 do_sample 方法生成采样输出（返回 JSONL 格式的字符串列表）
                f.writelines(generated)  # 将生成的采样结果写入临时文件（每行一个 JSON 对象）
                f.flush()  # 刷新文件缓冲区，立即将数据写入磁盘（防止数据丢失）
                shutil.copy(tmp_file, resume_file)  # 将临时文件复制到恢复文件（备份当前进度，用于断点续传）
                with open(ckpt_state_file, 'w') as ckpt_state:  # 以覆盖写入模式打开检查点状态文件
                    json.dump({'index': _index}, ckpt_state)  # 将当前批次索引保存到检查点状态文件（JSON 格式，如 {'index': 5}）

        if os.path.exists(iter_file):  # 若最终输出文件已存在（可能是之前的采样结果）
            shutil.move(iter_file, iter_file + '.' + str(int(time.time())))  # 将旧的输出文件重命名为带时间戳的备份文件（如 samples.jsonl.1699999999），避免覆盖丢失数据
        shutil.move(resume_file, iter_file)  # 将恢复文件重命名为最终输出文件（采样完成，生成最终结果）
        logger.info(f'Sample file {iter_file} generated.')  # 记录日志：输出最终生成的采样文件路径


def sampling_main(args: Optional[Union[List[str], SamplingArguments]] = None):
    """函数功能：
    采样主入口函数，封装了 SwiftSampling 的实例化和主流程执行。
    
    参数：
        args (Optional[Union[List[str], SamplingArguments]]): 
            - 可选参数，支持三种形式：
              1. None: 从 sys.argv 读取命令行参数；
              2. List[str]: 命令行参数列表（如 ['--model_id_or_path', 'qwen/Qwen-7B']）；
              3. SamplingArguments: 已实例化的参数对象。
    
    返回值：
        None（函数会执行采样流程并保存结果到文件）
    
    实际使用示例：
        示例 1：从命令行参数启动（脚本模式）
        >>> # 在 Python 脚本中
        >>> from swift.llm.sampling import sampling_main
        >>> if __name__ == '__main__':
        ...     sampling_main()  # 自动读取 sys.argv
        # 运行：python script.py --model_id_or_path qwen/Qwen-7B --dataset alpaca-zh
        
        示例 2：使用参数列表启动（编程模式）
        >>> sampling_main([
        ...     '--model_id_or_path', 'qwen/Qwen-7B',
        ...     '--dataset', 'alpaca-zh',
        ...     '--sampler_type', 'sample',
        ...     '--num_sampling_per_gpu_batch_size', '8',
        ...     '--num_sampling_per_gpu_batches', '100',
        ...     '--output_dir', './samples',
        ...     '--output_file', 'alpaca_samples.jsonl'
        ... ])
        # 生成 800 条采样数据到 ./samples/alpaca_samples.jsonl
        
        示例 3：使用参数对象启动（高级模式）
        >>> from swift.llm import SamplingArguments
        >>> args = SamplingArguments(
        ...     model_id_or_path='qwen/Qwen-Math-7B',
        ...     dataset=['math'],
        ...     sampler_type='mcts',
        ...     prm_model='prm_path',
        ...     orm_model='orm_path',
        ...     num_sampling_per_gpu_batch_size=4,
        ...     temperature=0.8,
        ...     top_p=0.95
        ... )
        >>> sampling_main(args)  # 使用预配置的参数对象启动
        
        示例 4：分片采样 + 断点续传
        >>> sampling_main([
        ...     '--model_id_or_path', 'qwen/Qwen-7B',
        ...     '--dataset', 'large_dataset',
        ...     '--data_range', '0,10',  # 第 1 个分片，共 10 个分片
        ...     '--resume', 'True',  # 启用断点续传
        ...     '--output_file', 'shard_0.jsonl'
        ... ])
        # 采样数据集的前 1/10，支持中断后继续
    """
    return SwiftSampling(args).main()  # 实例化 SwiftSampling 类（传入参数对象或参数列表），并调用继承自 SwiftPipeline 的 main 方法（该方法会执行参数解析、日志配置、调用 run 方法等完整流程）
