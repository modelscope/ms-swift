"""
swift.llm.data_loader 模块：提供在分布式/张量并行（TP）场景下的数据加载与分发工具。

功能：
- BatchSamplerShard：依据 rank/world_size 对索引做切分与（可选）打乱，产出各批次索引；
- DataLoaderShard：在 DataLoader 基础上，按需把 batch 移动到指定设备；
- DataLoaderDispatcher：在多进程中协调 rank 0 拉取与分发 batch，支持跳过若干批次（断点续训）。
"""

from typing import Optional  # 可选类型注解

import torch  # PyTorch 核心库
import torch.distributed as dist  # 分布式通信工具
from torch.utils.data import DataLoader  # PyTorch 数据加载器
from tqdm import tqdm  # 进度条显示

from swift.llm import to_device  # 工具函数：将数据迁移到设备


class BatchSamplerShard:
    """
    分片批采样器：在分布式/TP 环境下按 rank 划分样本索引并产出批次。

    - 在 __iter__ 中根据 shuffle 与种子生成/选择当前 rank 的索引切片；
    - 支持 drop_last 控制最后不完整批次的丢弃；
    - 通过 set_epoch(epoch) 更新当前迭代的随机种子（常用于分布式每个 epoch 的打乱）。
    """

    def __init__(self,
                 total_samples: int,
                 batch_size: int,
                 shuffle: bool,
                 drop_last: bool,
                 data_seed: Optional[int],
                 tp_size: int = 1,
        ):

        pass    

    def __init__(self,
                 total_samples: int,
                 batch_size: int,
                 shuffle: bool,
                 drop_last: bool,
                 data_seed: Optional[int],
                 tp_size: int = 1):
        """
        参数:
            total_samples: 数据集总样本数。
            batch_size: 每批大小。
            shuffle: 是否打乱样本顺序。
            drop_last: 是否丢弃最后不满一个 batch 的样本。
            data_seed: 数据打乱的随机种子；None 时使用 0。
            tp_size: 张量并行大小（用于 rank/world_size 的换算）。
        """
        self.tp_size = tp_size  # 张量并行大小
        self.total_samples = total_samples // self.world_size  # 当前 rank 负责的样本数量
        self.batch_size = batch_size  # 批大小
        self.shuffle = shuffle  # 是否打乱
        self.drop_last = drop_last  # 是否丢弃末批
        self.base_seed = data_seed or 0  # 基础随机种子
        self.curr_seed = self.base_seed  # 当前迭代使用的种子

    @property
    def rank(self):
        """
        当前进程的数据并行 rank（按张量并行折算）。
        """
        return (dist.get_rank() // self.tp_size) if dist.is_initialized() else 0  # 未初始化分布式则视为 0

    @property
    def world_size(self):
        """
        数据并行 world_size（按张量并行折算）。
        """
        return (dist.get_world_size() // self.tp_size) if dist.is_initialized() else 1  # 未初始化则视为单机 1

    def __iter__(self):
        """
        生成当前 rank 的批次索引序列。
        """
        start_idx = self.rank * self.total_samples  # 本 rank 起始全局索引
        if self.shuffle:  # 需要打乱
            generator = torch.Generator()  # 独立随机发生器
            generator.manual_seed(self.curr_seed)  # 以当前种子播种
            total_idx = torch.randperm(self.total_samples * self.world_size, generator=generator).tolist()  # 全局打乱
            total_idx = total_idx[start_idx:start_idx + self.total_samples]  # 取本 rank 负责的切片
        else:  # 不打乱则按顺序切片
            total_idx = list(range(start_idx, start_idx + self.total_samples))

        batch = []  # 累积当前批次
        # Last batch if not complete will be dropped.  # 若 drop_last=True 则丢弃不满一批的尾部
        for idx in total_idx:  # 遍历样本索引
            batch.append(idx)  # 追加到当前批
            if len(batch) == self.batch_size:  # 满批则产出
                yield batch  # 产出该批索引
                batch = []  # 重置批缓存
        if not self.drop_last and len(batch) > 0:  # 允许保留尾部不满批次
            yield batch  # 产出尾批
        return  # 结束迭代

    def set_epoch(self, epoch: int):
        """
        设置当前 epoch 的随机种子（用于每个 epoch 的打乱）。
        """
        self.curr_seed = self.base_seed + epoch  # 基础种子加 epoch 偏移

    def __len__(self) -> int:
        """
        返回本 rank 的批次数。
        - drop_last=True：向下取整
        - drop_last=False：向上取整
        """
        if self.drop_last:  # 丢弃尾批
            return self.total_samples // self.batch_size  # 向下取整
        else:  # 保留尾批
            return (self.total_samples + self.batch_size - 1) // self.batch_size  # 向上取整


class DataLoaderShard(DataLoader):
    """
    DataLoader 的轻量封装：
    - 在 __iter__
     中按需把 batch 移动到指定设备（device）。
    - 提供 set_epoch 转发，便于与分布式 sampler 或自定义 batch_sampler 联动。
    """

    def __init__(self, dataset, device=None, **dataloader_params):
        """
        参数:
            dataset: 数据集对象。
            device: 可选，目标设备（如 'cuda:0'）。
            dataloader_params: 透传给 torch.utils.data.DataLoader 的参数（如 sampler/batch_sampler 等）。
        """
        self.device = device  # 记录目标设备
        super().__init__(dataset, **dataloader_params)  # 初始化父类 DataLoader

    def set_epoch(self, epoch: int):
        """
        为底层 batch_sampler 或 sampler 设置 epoch（若其支持）。
        """
        if self.batch_sampler is not None and hasattr(self.batch_sampler, 'set_epoch'):  # 优先批采样器
            self.batch_sampler.set_epoch(epoch)  # 同步 epoch
        elif self.sampler is not None and hasattr(self.sampler, 'set_epoch'):  # 其次普通采样器
            self.sampler.set_epoch(epoch)  # 同步 epoch

    def __iter__(self):
        """
        迭代产出 batch；若设置了 device，则把 batch 移动到该设备上。
        """
        for item in super().__iter__():  # 逐批获取父类产出的数据
            if self.device:  # 需要迁移设备
                item = to_device(item, self.device)  # 把张量递归移动到 device
            yield item  # 产出 batch


class DataLoaderDispatcher:
    """
    数据加载分发器：在分布式环境中由 rank 0 统一拉取 batch 并分发到各 rank。

    - 支持训练启动时跳过若干批次（skip_batches），方便断点续训；
    - 通过 scatter_object_list 在进程组内分发对象（None 表示无数据结束）。
    """

    def __init__(self, base_dataloader, device=None, skip_batches: int = 0):
        """
        参数:
            base_dataloader: 基础 DataLoader（通常是各 rank 同构）。
            device: 可选，目标设备；若设置则把分发后的 batch 迁移到此设备。
            skip_batches: 训练开始时需要跳过的批次数（仅 rank 0 负责实际跳过）。
        """
        self.base_dataloader = base_dataloader  # 保存底层 DataLoader
        self.device = device  # 目标设备
        self.skip_batches = skip_batches  # 启动时跳过的批次数

    @property
    def rank(self):
        """
        当前进程在进程组内的 rank。
        """
        return dist.get_rank(self.group) if dist.is_initialized() else 0  # 未初始化则视为 0

    @property
    def world_size(self):
        """
        当前进程组内的 world_size。
        """
        return dist.get_world_size(self.group) if dist.is_initialized() else 1  # 未初始化则视为 1

    @property
    def group(self):
        """
        进程组对象；未初始化分布式时返回占位值 1。
        """
        return dist.group.WORLD if dist.is_initialized() else 1  # WORLD 组

    def _scatter_object_list(self, inputs):
        """
        将 rank 0 收集的对象列表按 rank 分发到各进程；未初始化分布式时返回首个元素。

        参数:
            inputs: 在 rank 0 为长度 world_size 的列表；其他 rank 传入 None。

        返回:
            当前 rank 对应的对象元素。
        """
        if not dist.is_initialized():  # 单机/未初始化场景
            return inputs[0]  # 直接返回第一个元素
        outputs = [None]  # 预留接收列表（当前 rank 接收一个元素）
        global_src_rank = dist.get_global_rank(self.group, 0)  # 获取全局源 rank（通常为 0）
        dist.scatter_object_list(outputs, inputs, global_src_rank, group=self.group)  # 分发对象
        return outputs[0]  # 返回当前 rank 收到的对象

    def _skip_batches(self, base_iter):
        """
        仅在 rank 0 上跳过指定数量的批次，并消费各 rank 对应数量的数据以保持对齐。
        """
        if self.rank == 0 and self.skip_batches > 0:  # 仅 rank 0 执行
            for _ in tqdm(range(self.skip_batches), dynamic_ncols=True, desc='Skip Batches: '):  # 进度条显示
                [next(base_iter) for _ in range(self.world_size)]  # 按 world_size 次数消费底层迭代器

    def __iter__(self):
        """
        调度与分发数据：由 rank 0 拉取并分发到各 rank，直到迭代结束。
        """
        base_iter = iter(self.base_dataloader)  # 获取底层迭代器
        self._skip_batches(base_iter)  # 可选跳过若干批次
        while True:  # 主循环
            if self.rank == 0:  # 仅 rank 0 从底层拉取数据
                try:
                    data = [next(base_iter) for _ in range(self.world_size)]  # 收集每个 rank 的一个 batch
                except StopIteration:  # 到达末尾
                    data = [None] * self.world_size  # 用 None 填充以广播终止
                data = self._scatter_object_list(data)  # 将数据按 rank 分发
            else:  # 非 rank 0 等待接收
                data = self._scatter_object_list(None)  # 接收 rank 0 分发的数据
            if data is None:  # 收到终止信号
                break  # 结束迭代
            if self.device:  # 需要迁移设备
                data = to_device(data, self.device)  # 移动到目标设备
            yield data  # 产出当前 rank 的 batch
