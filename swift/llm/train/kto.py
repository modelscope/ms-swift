"""
脚本用途:
- 定义 KTO（Kahneman-Tversky Optimization）相关的数据预处理与数据集构造逻辑。
- 通过批内右移（batch-level shift）构造 KL 对比样本，供 KTO 训练中的 KL 项计算使用。

主要功能:
- `KTOPreprocessor`: 行级预处理器，将批内样本右移一位以生成 `rejected_response`。
- `_get_kl_dataset`: 基于总 batch size 对数据集进行洗牌并按批进行右移处理。
- `prepare_kto_dataset`: 计算并校验批大小，构造训练/验证集的 KTO 视图，并给出权重建议范围提示。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings  # 标准库：发出警告信息
from typing import Any, Dict, Optional  # 类型提示：任意类型、字典、可选

from datasets import Dataset as HfDataset  # 第三方：HuggingFace 数据集类型别名

from swift.utils import get_dist_setting, get_logger  # 项目工具：分布式信息与日志
from ..dataset import RowPreprocessor  # 项目内：数据行级预处理器基类

logger = get_logger()  # 初始化模块级日志记录器


class KTOPreprocessor(RowPreprocessor):
    """KTO 专用的行级预处理器。

    作用:
        - 对一个批次内的样本进行右移，将每条样本的“被拒绝回复”设置为批内上一个样本的回复，
          用于构造 KL 对比项。
    """

    def batched_preprocess(self, batched_row: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """对一个批次的样本进行右移处理。

        参数:
            batched_row: 一个批次的样本字典，至少包含键 `messages`（列表，内部为对话轮消息）。
            **kwargs: 额外参数（未使用）。

        返回:
            更新后的批次字典，新增键 `rejected_response`，为右移得到的相邻样本的助手回复内容。
        """
        batched_row = dict(batched_row)  # 拷贝为普通字典，避免对上游数据产生副作用
        messages = batched_row['messages']  # 取出批次的对话消息列表
        batch_size = len(messages)  # 批次大小
        kl_messages = [messages[-1]] + messages[:-1]  # 右移一位：最后一条放到最前，其余顺次后移

        kl_response = []  # 收集右移后对应样本的“助手回复”作为被拒绝回复
        for i in range(batch_size):  # 遍历批内索引
            kl_message = kl_messages[i][-1]  # 取对应样本的最后一条消息（期望为助手回复）
            assert kl_message['role'] == 'assistant'  # 断言最后一条消息为助手角色
            kl_response.append(kl_message['content'])  # 记录该助手回复内容
        # The name rejected_response is just for convenience in processing.
        batched_row['rejected_response'] = kl_response  # 命名为 rejected_response 便于下游处理

        return batched_row  # 返回增强后的批次数据


def _get_kl_dataset(dataset: Optional[HfDataset],
                    total_batch_size: int,
                    num_proc: int,
                    seed: Optional[int] = None) -> Optional[HfDataset]:
    """为 KTO 生成 KL 数据集视图：按批右移一位以构造对比样本。

    参数:
        dataset: 原始数据集（可能为 None）。
        total_batch_size: 全局总 batch size（世界大小×每卡 batch×累积步）。
        num_proc: 预处理并行进程数。
        seed: 随机种子，用于洗牌。

    返回:
        处理后的数据集；若输入为 None 则返回 None。
    """
    # Shift one position to the right in each batch.
    if dataset is None:  # 无数据集时直接返回
        return  # 返回 None
    dataset = dataset.shuffle(seed)  # 先对数据集洗牌，避免固定配对
    return KTOPreprocessor()(dataset, batch_size=total_batch_size, num_proc=num_proc)  # 批处理并右移


def prepare_kto_dataset(args, train_dataset, val_dataset):
    """构造并校验 KTO 训练/验证数据集。

    动作:
        - 计算全局总 batch size；若过小则报错（KL 项退化）。
        - 将训练/验证集转换为 KL 视图（批内右移）。
        - 统计正/负样本数量，若不均衡则计算推荐权重区间并给出提示。

    返回:
        (train_dataset, val_dataset): 处理后的训练与验证数据集。
    """
    world_size = get_dist_setting()[2]  # 读取分布式世界大小（rank 数）
    total_batch_size = (world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps)  # 全局总 batch
    if total_batch_size <= 1:  # 批大小过小
        raise ValueError('Batch size is 1 (too small). KTO will not work properly because the KL term '
                         'will be equivalent to the implied reward.')  # 抛出错误：KL 项将退化
    train_dataset = _get_kl_dataset(train_dataset, total_batch_size, args.dataset_num_proc, args.data_seed)  # 右移训练集
    val_dataset = _get_kl_dataset(val_dataset, total_batch_size, args.dataset_num_proc, args.data_seed)  # 右移验证集

    label = train_dataset['label']  # 读取标签（0/1 二分类）
    num_desirable = max(sum(label), 1)  # 正样本数量（至少为 1）
    num_undesirable = max(len(label) - num_desirable, 1)  # 负样本数量（至少为 1）

    if num_desirable != num_undesirable:  # 样本不均衡时，给出权重建议范围
        # The lower and upper bounds come from Eq. (8) of https://huggingface.co/papers/2402.01306
        des_weight_lower_bound = round((num_undesirable * args.undesirable_weight / num_desirable) * 1, 2)  # 正样本权重下界
        des_weight_upper_bound = round((num_undesirable * args.undesirable_weight / num_desirable) * 1.33, 2)  # 正样本权重上界
        und_weight_lower_bound = round((num_desirable * args.desirable_weight / num_undesirable) / 1.33, 2)  # 负样本权重下界
        und_weight_upper_bound = round((num_desirable * args.desirable_weight / num_undesirable) / 1, 2)  # 负样本权重上界

        des_weight_in_range = des_weight_lower_bound <= args.desirable_weight <= des_weight_upper_bound  # 正权重是否在建议区间
        und_weight_in_range = und_weight_lower_bound <= args.undesirable_weight <= und_weight_upper_bound  # 负权重是否在建议区间

        if not (des_weight_in_range or und_weight_in_range):  # 若两者都不在建议范围
            logger.info(f'desirable_weight: {args.desirable_weight}, undesirable_weight: {args.undesirable_weight}')  # 打印当前权重
            warnings.warn(  # 发出警告，提示建议区间
                f"""
        You have different amounts of desirable/positive and undesirable/negative examples but the
        weights on the desirable and undesirable losses don't seem to be in an ideal range. Based
        on your data, we recommend EITHER desirable_weight in [{des_weight_lower_bound}, '{des_weight_upper_bound}]
        or undesirable_weight in [{und_weight_lower_bound}, {und_weight_upper_bound}] (but NOT BOTH).
        See the documentation on how to optimally set these weights.""", UserWarning)
    return train_dataset, val_dataset  # 返回处理后的数据集
