"""
模块功能概述：
本模块提供基于二进制分片文件与索引文件的高效数据集读写工具，适用于需要对样本进行顺序写入、
并在训练/推理时按索引随机读取的场景。通过mmap减少内存拷贝，提升读取性能。

包含组件：
1) IndexedDatasetBuilder：将对象序列化为pickle字节流写入bin分片，并构建idx索引文件。
2) BinReader：对单个bin分片建立内存映射，支持按偏移读取字节切片。
3) IndexedDataset：PyTorch Dataset实现，按索引定位样本所在分片与偏移，读取并反序列化为对象。

使用示例（简要）：
>>> builder = IndexedDatasetBuilder("my_dataset")
>>> builder.add_items([
...     {"length": 3, "data": [1, 2, 3]},
...     {"length": 2, "data": [4, 5]},
... ])
>>> builder.finalize()
>>> dataset = IndexedDataset("my_dataset")
>>> sample = dataset[0]  # 读取第一个样本
"""
# Copyright (c) Alibaba, Inc. and its affiliates.

# 导入bisect模块：用于在有序列表中查找插入位置，这里用于根据偏移定位对应的分片编号
import bisect
# 导入mmap模块：实现文件的内存映射，支持高效、零拷贝的随机读取
import mmap
# 导入os模块：用于文件路径拼接、环境变量获取与文件操作
import os
# 导入pickle模块：用于对象的序列化与反序列化（写入/读取样本）
import pickle
# 导入threading模块：用于在后台线程中异步写入bin文件，避免主线程阻塞
import threading
# 从queue模块导入Queue：作为生产者-消费者队列，缓存待写入的样本批次
from queue import Queue
# 从typing导入类型注解：Any表示任意类型，List表示列表类型
from typing import Any, List

# 从modelscope工具集中导入get_cache_dir：用于获取模型/数据缓存根目录
from modelscope.hub.utils.utils import get_cache_dir
# 从PyTorch导入Dataset基类：使IndexedDataset可直接作为DataLoader的数据源
from torch.utils.data import Dataset


# 定义IndexedDatasetBuilder类：负责写入分片bin文件与生成索引文件
class IndexedDatasetBuilder:
    """
    类功能：
        构建可索引的数据集的写入器。将传入的样本（任意可pickle对象）序列化为字节，
        按顺序写入二进制分片文件，并维护每条样本的字节长度与累积偏移，最终写出索引文件。

    属性简介：
        CHUNK_SIZE (float): 单个分片阈值（字节数），达到阈值后滚动到下一个分片。
        cache_dir (str): 当前数据集的缓存目录。
        n_shard (int): 已创建的分片数量（初始为1，表示第0号分片）。
        bin_path (str): 当前正在写入的分片bin文件路径。
        idx_path (str): 索引文件路径（记录所有样本的起始偏移与长度等）。
        bin_file (IO): 当前打开的二进制文件对象，用于写入样本字节。
        length_list (List[int]): 记录每个样本的长度（用于校验/统计等场景）。
        idx_list (List[int]): 记录每个样本在全局字节流中的累积偏移（首元素0）。
        shard_offset (List[int]): 每个分片在全局字节流中的起始偏移，用于定位所在分片。
        _thread (threading.Thread|None): 后台写入线程。
        _queue (Queue): 生产者-消费者队列，缓存待写入的样本批次。

    示例：
        >>> builder = IndexedDatasetBuilder("demo_ds")
        >>> builder.add_items([
        ...     {"length": 3, "data": [1, 2, 3]},
        ...     {"length": 4, "data": [4, 5, 6, 7]}
        ... ])
        >>> builder.finalize()
    """

    # 定义单个分片的最大字节阈值（达到该阈值则开启新分片）；1e10约等于10GB
    CHUNK_SIZE = 1e10

    # 初始化写入器：准备缓存目录、分片与索引路径，创建第0号分片文件
    def __init__(self, dataset_name: str):
        """
        函数功能：
            初始化构建器，依据数据集名称准备缓存目录与文件路径，创建第一个分片文件，
            并初始化写入所需的索引结构与队列/线程管理对象。

        入参：
            dataset_name (str): 数据集名称，用于确定缓存目录（隔离不同数据集）。

        返回值：
            None

        示例：
            >>> builder = IndexedDatasetBuilder("pack_cache")
        """
        # 计算并创建缓存目录（如不存在）；与IndexedDataset保持一致的目录结构
        self.cache_dir = IndexedDataset.get_cache_dir(dataset_name)
        # 初始化分片计数，1表示当前存在第0号分片
        self.n_shard = 1
        # 计算第0号分片的bin文件路径（形如 data-00000.bin）
        self.bin_path = os.path.join(self.cache_dir, IndexedDataset.BIN_FNAME.format(0))
        # 计算索引文件路径（固定为 data.idx）
        self.idx_path = os.path.join(self.cache_dir, IndexedDataset.IDX_FNAME)
        # 若之前存在残留的第0号bin文件，先行删除，确保重新构建
        if os.path.exists(self.bin_path):
            # 检测到旧的分片文件，执行删除以清理残留
            os.remove(self.bin_path)
        # 以追加二进制方式创建并打开当前分片文件，准备写入
        self.bin_file = open(self.bin_path, 'ab')
        # 初始化样本长度列表（便于统计/调试）
        self.length_list = []
        # 初始化累积偏移列表，首元素为0，代表第一个样本起点
        self.idx_list = [0]
        # 初始化每个分片的全局起始偏移，第0个分片从0开始
        self.shard_offset = [0]
        # 后台写入线程句柄，初始为None，按需启动
        self._thread = None
        # 生产者-消费者队列，缓存写入批次；最大容量1000以限制内存使用
        self._queue = Queue(maxsize=1000)

    # 后台写入线程的工作函数：持续从队列取出批次并写入到当前bin分片
    def _write_worker(self):
        """
        函数功能：
            在独立线程中消费待写入队列，逐条序列化样本并写入分片bin文件，同时更新
            样本的累积偏移列表与长度列表。当当前分片大小达到阈值时，滚动到下一个分片。

        入参：
            无

        返回值：
            None

        示例：
            （内部使用，无需直接调用）
        """
        # 无限循环，持续处理队列中的写入任务，直到收到终止信号
        while True:
            # 从队列中取出一个批次（可能阻塞等待）
            items = self._queue.get()
            # 若取到None，表示写入结束信号，跳出循环
            if items is None:
                # 收到终止信号，退出写入循环
                break
            # 准备本批次的字节缓冲列表，用于一次性写入提升IO效率
            bin_buffer = []
            # 遍历批次中的每个样本
            for item in items:
                # 将样本对象序列化为pickle字节流
                item_buffer = pickle.dumps(item)
                # 追加到本批次的写入缓冲
                bin_buffer.append(item_buffer)
                # 计算并记录当前样本在全局字节流中的结束位置（作为下一条的起点）
                self.idx_list.append(self.idx_list[-1] + len(item_buffer))
                # 记录样本的逻辑长度字段（由上游提供），便于统计/检验
                self.length_list.append(item['length'])
            # 将本批次所有样本的字节拼接后一次性写入当前分片文件
            self.bin_file.write(b''.join(bin_buffer))
            # 计算当前分片已写入的总字节数（相对该分片起始偏移）
            offset = self.idx_list[-1] - self.shard_offset[-1]
            # 若超过单分片阈值，则关闭当前分片并开启新分片
            if offset >= self.CHUNK_SIZE:
                # 关闭当前分片文件以刷新和释放句柄
                self.bin_file.close()
                # 计算下一个分片的bin文件路径（按分片编号递增）
                self.bin_path = os.path.join(self.cache_dir, IndexedDataset.BIN_FNAME.format(self.n_shard))
                # 记录新分片在全局字节流中的起始偏移（累加当前分片大小）
                self.shard_offset.append(self.shard_offset[-1] + offset)
                # 分片计数+1，表示开启了新的分片编号
                self.n_shard += 1
                # 若新分片文件已存在（异常残留），则先删除以确保干净写入
                if os.path.exists(self.bin_path):
                    os.remove(self.bin_path)
                # 打开新的分片文件，继续后续写入
                self.bin_file = open(self.bin_path, 'ab')

    # 向写入器一次性追加一批样本（异步写入队列）
    def add_items(self, items: List[Any]) -> None:
        """
        函数功能：
            追加一批样本到写入队列。首次调用时会启动后台写入线程。

        入参：
            items (List[Any]): 待写入的一批样本，每个元素需可被pickle序列化，且应包含键"length"。

        返回值：
            None

        示例：
            >>> builder.add_items([
            ...     {"length": 3, "data": [1, 2, 3]},
            ...     {"length": 1, "data": [4]}
            ... ])
        """
        # 若后台写入线程尚未启动，则初始化并启动为守护线程
        if self._thread is None:
            # 创建后台线程，目标函数为写入工作者；daemon=True确保主进程退出时不阻塞
            self._thread = threading.Thread(target=self._write_worker, daemon=True)
            # 启动后台写入线程
            self._thread.start()
        # 将本批样本放入队列，供后台线程消费
        self._queue.put(items)

    # 结束写入：等待后台线程完成并写出索引文件
    def finalize(self):
        """
        函数功能：
            结束写入流程：发送终止信号、等待后台线程退出，关闭当前分片文件，
            将内存中的索引结构写出到磁盘的idx文件中。

        入参：
            无

        返回值：
            None

        示例：
            >>> builder.finalize()
        """
        # 若后台线程存在，则发送None作为终止标记并等待其结束
        if self._thread is not None:
            # 向队列发送终止信号，提示写入线程结束循环
            self._queue.put(None)
            # 阻塞等待写入线程安全退出
            self._thread.join()
        # 关闭当前打开的分片文件句柄
        self.bin_file.close()
        # 组织索引对象，包含累积偏移、样本长度、分片数量与各分片起始偏移
        idx_obj = {
            # 所有样本的累积偏移列表（长度=样本数+1）
            'idx': self.idx_list,
            # 样本逻辑长度列表（由写入端提供）
            'length': self.length_list,
            # 分片总数量
            'n_shard': self.n_shard,
            # 各分片在全局字节流中的起始偏移
            'shard_offset': self.shard_offset,
        }
        # 将索引对象序列化写入idx文件，供读取端快速构建数据集
        with open(self.idx_path, 'wb') as f:
            pickle.dump(idx_obj, f)


# 定义BinReader类：单个bin分片的读取器，封装mmap读操作
class BinReader:
    """
    类功能：
        打开指定的bin分片文件，并通过mmap建立内存映射，以便根据偏移与长度高效读取字节数据。

    属性简介：
        bin_path (str): 分片文件路径。
        file (IO): 打开的文件对象。
        mm (mmap.mmap|None): 内存映射对象；若文件为空等特殊情况，则为None。

    示例：
        >>> reader = BinReader("/path/to/data-00000.bin")
        >>> buf = reader.read_buffer(0, 10)
    """

    # 初始化读取器：打开文件并尝试建立只读内存映射
    def __init__(self, bin_path: str):
        """
        函数功能：
            打开给定路径的bin文件，并建立只读mmap映射；若文件为空等情况，则mm为None。

        入参：
            bin_path (str): bin分片文件的绝对或相对路径。

        返回值：
            None

        示例：
            >>> BinReader("./data-00000.bin")
        """
        # 记录分片文件路径，便于调试与错误信息提示
        self.bin_path = bin_path
        # 以二进制只读方式打开分片文件
        self.file = open(bin_path, 'rb')
        try:
            # 建立对整个文件的只读内存映射（长度0表示映射整个文件）
            self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:
            # 异常分支：例如文件为空时，mmap可能抛出ValueError，降级为None以便后续逻辑处理
            self.mm = None

    # 读取指定偏移与长度的字节切片
    def read_buffer(self, offset: int, size: int) -> bytes:
        """
        函数功能：
            从已映射的mmap对象中，按给定偏移与长度读取对应的字节切片。

        入参：
            offset (int): 读取起始位置（从文件头相对偏移）。
            size (int): 读取字节数。

        返回值：
            bytes: 读取到的字节切片。

        示例：
            >>> reader.read_buffer(0, 16)
        """
        # 基本边界检查：偏移与长度必须非负，且读取范围不得越界
        if offset < 0 or size < 0 or offset + size > len(self.mm):
            # 触发错误以提示调用方参数非法
            raise ValueError('Invalid offset or size')
        # 返回指定范围的字节切片（不产生额外拷贝）
        return self.mm[offset:offset + size]

    # 析构方法：释放mmap与文件句柄
    def __del__(self):
        """
        函数功能：
            在对象销毁前，关闭mmap映射与文件句柄，避免资源泄露。

        入参：
            无

        返回值：
            None

        示例：
            >>> del reader  # 触发资源释放
        """
        # 若mmap对象存在，则先关闭映射
        if self.mm is not None:
            self.mm.close()
        # 关闭底层文件句柄
        self.file.close()


# 定义IndexedDataset类：基于索引文件与分片bin的PyTorch数据集
class IndexedDataset(Dataset):
    """
    类功能：
        读取由IndexedDatasetBuilder构建的数据集。通过索引文件定位每条样本的全局偏移，
        再结合分片起始偏移定位具体分片与分片内偏移，最后从bin读取并反序列化为对象。

    属性简介：
        BIN_FNAME (str): 分片文件命名模板，如 'data-00000.bin'。
        IDX_FNAME (str): 索引文件固定文件名 'data.idx'。
        idx_list (List[int]): 所有样本的累积偏移（长度为样本数+1）。
        length_list (List[int]): 每条样本的长度信息（来自写入端）。
        n_shard (int): 分片数量。
        shard_offset (List[int]): 每个分片在全局字节流中的起始偏移。
        bin_readers (List[BinReader]): 每个分片对应的读取器。

    示例：
        >>> ds = IndexedDataset("my_dataset")
        >>> first = ds[0]
        >>> n = len(ds)
    """

    # 分片文件名称模板：使用5位数字左侧补零
    BIN_FNAME = 'data-{:05d}.bin'
    # 索引文件固定名称
    IDX_FNAME = 'data.idx'

    # 静态方法：根据数据集名返回（并创建）缓存目录
    @staticmethod
    def get_cache_dir(dataset_name: str):
        """
        函数功能：
            依据环境变量与默认规则生成数据集缓存目录：
            优先使用环境变量PACKING_CACHE，否则使用modelscope全局cache下的'tmp'子目录，
            最后拼接数据集名形成最终目录，并确保目录存在。

        入参：
            dataset_name (str): 数据集名称，用于区分不同数据集的缓存隔离。

        返回值：
            str: 可用的缓存目录绝对路径。

        示例：
            >>> IndexedDataset.get_cache_dir("demo")
        """
        # 优先从环境变量PACKING_CACHE获取缓存根目录；否则退回到modelscope默认缓存/tmp下
        cache_dir = os.getenv('PACKING_CACHE') or os.path.join(get_cache_dir(), 'tmp')
        # 将数据集名称追加到缓存根目录，形成数据集专属目录
        cache_dir = os.path.join(cache_dir, dataset_name)
        # 若目录不存在则创建，exist_ok=True确保并发安全
        os.makedirs(cache_dir, exist_ok=True)
        # 基本校验：dataset_name不能为空
        assert dataset_name is not None, f'dataset_name: {dataset_name}'
        # 返回最终可用的缓存目录
        return cache_dir

    # 初始化数据集：读取索引文件并打开各分片的读取器
    def __init__(self, dataset_name: str):
        """
        函数功能：
            根据数据集名称加载索引文件，构建样本偏移与分片信息，并为每个分片创建BinReader，
            从而实现按索引读取样本的能力。

        入参：
            dataset_name (str): 数据集名称（需与构建阶段一致）。

        返回值：
            None

        示例：
            >>> ds = IndexedDataset("my_dataset")
        """
        # 记录数据集名称，便于调试或后续使用
        self.dataset_name = dataset_name
        # 计算该数据集的缓存目录（需与构建器一致）
        cache_dir = self.get_cache_dir(dataset_name)
        # 拼接索引文件路径（固定为data.idx）
        self.idx_path = os.path.join(cache_dir, self.IDX_FNAME)
        # 读取并反序列化索引对象，获取偏移、长度与分片信息
        with open(self.idx_path, 'rb') as f:
            idx_obj = pickle.load(f)
        # 提取所有样本的累积偏移列表
        self.idx_list = idx_obj['idx']
        # 提取所有样本的长度列表
        self.length_list = idx_obj['length']
        # 读取分片数量
        self.n_shard = idx_obj['n_shard']
        # 读取每个分片的全局起始偏移
        self.shard_offset = idx_obj['shard_offset']
        # 初始化分片读取器列表
        self.bin_readers = []
        # 为每个分片构建一个BinReader实例
        for i in range(self.n_shard):
            # 计算当前分片的bin文件路径
            bin_path = os.path.join(cache_dir, self.BIN_FNAME.format(i))
            # 创建并保存分片读取器，便于后续快速读取
            self.bin_readers.append(BinReader(bin_path))

    # 根据整数索引返回对应样本对象（支持负索引）
    def __getitem__(self, index: int):
        """
        函数功能：
            返回指定索引的样本对象。支持负索引（Python风格）。内部基于idx与shard_offset
            定位分片与偏移，再从对应bin文件中读取并反序列化得到样本。

        入参：
            index (int): 样本索引（可为负）。

        返回值：
            Any: 反序列化得到的样本对象。

        示例：
            >>> sample = ds[0]
        """
        # 若为负索引，转换为等价的正向索引
        if index < 0:
            index = index % len(self)
        # 取出当前样本的起始偏移与下一样本的起始偏移，差值即为字节长度
        idx, idx_next = self.idx_list[index], self.idx_list[index + 1]
        # 在分片起始偏移列表中，查找idx应落入的分片位置（右侧二分）
        num_shard = bisect.bisect_right(self.shard_offset, idx)
        # 取出该分片的全局起始偏移，用于计算分片内局部偏移
        offset = self.shard_offset[num_shard - 1]
        # 从对应分片读取器中按局部偏移与长度读取字节切片
        buffer = self.bin_readers[num_shard - 1].read_buffer(idx - offset, idx_next - idx)
        # 反序列化字节为原始对象并返回
        return pickle.loads(buffer)

    # 返回数据集中样本数量
    def __len__(self):
        """
        函数功能：
            返回数据集包含的样本数目。

        入参：
            无

        返回值：
            int: 样本数量。

        示例：
            >>> n = len(ds)
        """
        # idx_list长度比样本数多1，因此减1得到样本总数
        return len(self.idx_list) - 1
