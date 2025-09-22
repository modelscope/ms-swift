"""
模块功能概述：
本模块提供围绕大语言模型（LLM）训练/推理的数据处理实用工具与数据集封装：
- sample_dataset: 对HuggingFace数据集按样本数进行抽样（支持重复采样与随机打乱）。
- LazyLLMDataset: 惰性tokenize/encode的数据集，训练时遇到坏样本可跳过以保证稳定性。
- calculate_matched_group: 使用装箱算法将若干样本按长度上限进行分组匹配。
- PackingDataset: 预计算并缓存“打包后的索引”的Dataset，__getitem__按组取样并packing。
- IterablePackingDataset: 基于多进程+队列的“边取边pack”可迭代数据集，适合大规模流式数据。
- EncodePreprocessor: 行级预处理器，封装模板encode；可选择仅写入length供后续packing。

简要示例：
>>> from datasets import Dataset as HfDataset
>>> ds = HfDataset.from_dict({"text": ["hello", "world"]})
>>> ds2 = sample_dataset(ds, dataset_sample=3, shuffle=True)
"""

# 版权声明：阿里巴巴及其附属公司保留所有权利
# 该行用于标注本文件的版权信息与归属
# Copyright (c) Alibaba, Inc. and its affiliates.

# 导入多进程模块并简写为mp：用于创建子进程和进程间通信队列
import multiprocessing as mp
# 从typing导入类型提示工具：用于静态类型检查与更清晰的接口定义
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

# 导入数值计算库numpy：用于索引采样、打乱与数组拼接等操作
import numpy as np
# 导入分布式通信库torch.distributed：用于多进程/多机环境下的对象广播
import torch.distributed as dist
# 从datasets库导入HuggingFace数据集类型并重命名为HfDataset：统一类型标注
from datasets import Dataset as HfDataset
# 从PyTorch导入Dataset与IterableDataset：定义自定义数据集与可迭代数据集
from torch.utils.data import Dataset, IterableDataset
# 导入tqdm进度条：用于显示打包进度
from tqdm import tqdm

# 从swift.utils导入工具函数：get_logger获取日志器，is_dist/is_master用于分布式状态判断
from swift.utils import get_logger, is_dist, is_master
# 从上级template模块导入MaxLengthError：用于区分可忽略的超长错误
from ..template import MaxLengthError
# 从当前包的preprocessor模块导入RowPreprocessor：定义行级预处理基类
from .preprocessor import RowPreprocessor

# 初始化模块级日志器：供本模块内部统一打印日志使用
logger = get_logger()

# 仅在类型检查阶段导入以避免循环依赖与运行时开销
if TYPE_CHECKING:
    # 仅用于类型提示的Template类引用（运行时不会真正导入）
    from swift.llm import Template


# 定义函数：按给定样本数抽样HF数据集（支持重复采样与随机打乱）
def sample_dataset(
    dataset: HfDataset,
    dataset_sample: Optional[int],
    shuffle: bool = True,
    random_state: Optional[np.random.RandomState] = None,
) -> HfDataset:
    """
    函数功能：
        根据期望样本数对给定HF数据集进行抽样。若期望数大于数据集长度，使用重复采样；
        可选是否先打乱再取余数样本。

    入参：
        dataset (HfDataset): HuggingFace数据集实例（不支持可迭代流式数据集）。
        dataset_sample (Optional[int]): 期望抽样得到的样本总数；为None时返回原数据集。
        shuffle (bool): 当需要补充余数样本时，是否对索引进行随机打乱后再截取。
        random_state (Optional[np.random.RandomState]): 指定随机状态以保证可复现；
            若为None且需要shuffle，将内部创建新的随机状态。

    返回值：
        HfDataset: 抽样后的新数据集视图（通过select索引实现，不会拷贝原始数据）。

    示例：
        >>> sampled = sample_dataset(dataset, dataset_sample=1000, shuffle=True)
    """
    # 若未指定抽样数，直接返回原数据集
    if dataset_sample is None:
        return dataset

    # 计算整倍数重复次数：可整除部分通过重复索引实现
    n_repeat_sample = dataset_sample // len(dataset)
    # 计算剩余样本数：用于补齐不足整倍数的部分
    n_remain_sample = dataset_sample % len(dataset)
    # 若既有整倍数重复又有余数，提示将执行重复采样（日志级别warning）
    if n_repeat_sample >= 1 and n_remain_sample >= 1:
        logger.warning(
            f'dataset_sample:{dataset_sample} is greater than len(dataset):{len(dataset)}, '
            'repeated sampling will be performed.'
        )
    # 生成整倍数重复的基础索引序列：np.tile按重复次数复制索引范围
    idx = np.tile(range(len(dataset)), n_repeat_sample)
    # 若仍有余数样本需要补齐
    if n_remain_sample >= 1:
        # 当需要随机打乱时
        if shuffle:
            # 若未提供随机状态，则创建新的以保证随机性
            if random_state is None:
                random_state = np.random.RandomState()
            # 从打乱后的全索引中截取前n_remain_sample个作为补充索引
            idx_remain = random_state.permutation(len(dataset))[:n_remain_sample]
        else:
            # 不打乱时，直接使用从0开始的顺序索引补齐余数
            idx_remain = np.arange(n_remain_sample)
        # 将整倍数索引与余数索引拼接得到最终索引序列
        idx = np.concatenate([idx, idx_remain])
    # 使用HF的select方法根据索引选择子集，形成抽样后的数据集视图
    dataset = dataset.select(idx)
    # 返回抽样后的数据集
    return dataset


# 定义惰性编码数据集：按需对原始样本进行encode，失败时在训练中跳过坏样本
class LazyLLMDataset(Dataset):
    """
    类功能：
        在__getitem__时才对样本进行encode/tokenize，若编码失败（例如模板异常），
        非strict模式下会尝试多次并跳过坏样本，避免中断训练。

    关键属性：
        dataset (HfDataset): 原始HF数据集。
        encode_func (Callable): 对单条样本进行编码的函数，需支持return_length参数。
        n_try_fetch (int): 最大尝试次数；strict=True时固定为1。
        strict (bool): 严格模式，出错即抛异常；否则跳过坏样本。
        random_state (np.random.RandomState): 用于随机选择备选样本。
        traceback_limit (int): 最多打印几次详细堆栈，避免刷屏。
    """

    # 构造函数：保存配置并初始化内部状态
    def __init__(
        self,
        dataset: HfDataset,
        encode_func: Callable[[Dict[str, Any]], Dict[str, Any]],
        *,
        n_try_fetch: int = 10,
        strict: bool = False,
        random_state: Optional[Union[np.random.RandomState, int]] = None,
        traceback_limit: int = 10,
    ) -> None:
        """
        函数功能：
            初始化惰性数据集，配置编码函数、重试策略与随机选择策略。

        入参：
            dataset (HfDataset): 原始HF数据集。
            encode_func (Callable): 编码函数，签名形如 f(row, return_length=True) -> Dict。
            n_try_fetch (int): 每次getitem最多尝试次数（strict=True时将被置为1）。
            strict (bool): 是否严格模式；严格模式下首次失败即抛出异常。
            random_state (Optional[Union[np.random.RandomState, int]]): 随机状态或种子。
            traceback_limit (int): 最多打印的异常堆栈次数上限。

        返回值：
            None

        示例：
            >>> ds = LazyLLMDataset(dataset, encode_func, n_try_fetch=5, strict=False)
        """
        # 保存原始数据集引用
        self.dataset = dataset
        # 保存用户提供的编码函数
        self.encode_func = encode_func

        # 若为严格模式，仅允许尝试1次；否则限定最大尝试次数不超过数据集长度
        n_try_fetch = 1 if strict else min(n_try_fetch, len(self.dataset))
        # 基本校验：尝试次数至少为1
        assert n_try_fetch >= 1
        # 保存严格模式标志
        self.strict = strict
        # 保存最终的最大尝试次数
        self.n_try_fetch = n_try_fetch

        # 归一化random_state：若不是RandomState实例，则用其构造一个实例（支持传入seed或None）
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        # 保存随机状态
        self.random_state = random_state

        # 保存最多打印异常堆栈次数
        self.traceback_limit = traceback_limit
        # 当前已打印异常堆栈的次数计数器
        self._traceback_counter = 0
        # 轮询备选索引用的游标
        self._idx = 0
        # 将数据集长度范围进行随机排列，转为列表供轮询使用
        self._idx_list = self.random_state.permutation(len(self.dataset)).tolist()

    # 读取单条样本：若失败则在限制内重试其它随机样本
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        函数功能：
            返回索引为idx的编码后样本；若编码失败且非严格模式，则从预生成的随机索引列表
            中按轮询方式尝试其它样本，最多尝试n_try_fetch次。

        入参：
            idx (int): 请求的样本索引；当为str时，直接走HF数据集的键访问。

        返回值：
            Dict[str, Any]: 编码后的样本字典，需包含length等字段。

        示例：
            >>> item = ds[0]
        """
        # 若传入的是字符串索引，则直接委托HF数据集按键访问返回原始数据
        if isinstance(idx, str):
            return self.dataset[idx]
        # 尝试最多n_try_fetch次
        for i in range(self.n_try_fetch):
            # 记录本次尝试次数（用于后续判断是否最后一次）
            n_try = i
            # 第一次使用用户请求的索引
            if i == 0:
                i = idx
            else:
                # 后续尝试使用随机排列列表中的索引，并轮询推进游标
                i = self._idx_list[self._idx]
                self._idx = (self._idx + 1) % len(self.dataset)
            # 从原始数据集中取出该条数据
            data = self.dataset[i]
            try:
                # 调用编码函数进行编码，并强制要求返回长度信息
                return self.encode_func(data, return_length=True)
            except Exception:
                # 若已达最后一次尝试，或处于严格模式，则抛出异常
                if n_try == self.n_try_fetch - 1 or self.strict:
                    if self.strict:
                        logger.warning('To avoid errors, you can pass `strict=False`.')
                    raise
                # 若允许打印堆栈且未超过次数上限，则打印一次并递增计数
                if self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                    import traceback
                    logger.info(traceback.format_exc())
                    logger.warning(
                        '👆👆👆There are errors in the template.encode, '
                        'and another piece of data will be randomly selected.'
                    )
                    self._traceback_counter += 1

        # 多次尝试后依然失败，给出更具操作性的错误提示
        raise ValueError(
            'Failed to retrieve the dataset. You can avoid this issue by increasing `max_length` or '
            'modifying the `truncation_strategy`.'
        )

    # 返回数据集长度：直接转交给底层HF数据集
    def __len__(self) -> int:
        """
        函数功能：
            返回惰性数据集的样本总数（与底层HF数据集一致）。

        入参：
            无

        返回值：
            int: 样本数量。

        示例：
            >>> n = len(ds)
        """
        return len(self.dataset)


# 定义函数：使用装箱算法将样本按长度进行匹配打包
def calculate_matched_group(template, sequences, is_finished: bool = True):
    """
    函数功能：
        使用binpacking算法按模板的最大长度约束，将序列按总长度打包为若干组；
        若未结束（is_finished=False），则保留最后一组作为残留以便下轮继续填充。

    入参：
        template: 模板对象，需包含max_length属性用于约束每组总长度上限。
        sequences: 待打包的序列列表，元素形如 (payload, length)。
        is_finished (bool): 是否所有数据已经喂入；若否，将把最后一组作为残留返回。

    返回值：
        Tuple[List[List], List]: (已完成的若干组, 残留的未满一组)。

    示例：
        >>> seqs, rest = calculate_matched_group(template, [(x, l) for x,l in data], True)
    """
    # 边界情况：若没有任何待打包元素，返回两个空列表
    if len(sequences) == 0:
        return [], []
    # 引用论文背景：https://arxiv.org/pdf/2404.10830
    # 动态导入binpacking库：将元素以length作为重量进行装箱
    import binpacking
    # 使用恒定体积装箱：weight_pos=1指明长度字段位置，限制每组不超过template.max_length
    sequences = binpacking.to_constant_volume(sequences, template.max_length, weight_pos=1)
    # 若存在至少一组且未结束，将最后一组作为残留，其余作为完成组
    if sequences and not is_finished:
        sequences, ret_sequences = sequences[:-1], sequences[-1]
    else:
        # 已结束或没有分组，残留为空
        ret_sequences = []
    # 返回（已完成的分组，残留分组）
    return sequences, ret_sequences


# 定义预计算打包索引的数据集：一次性离线计算所有pack组
class PackingDataset(Dataset):
    """
    类功能：
        基于静态数据集与模板，预计算打包后的索引列表与对应长度，__getitem__按组取出并调用模板进行packing。

    关键属性：
        template: 模板对象，需提供max_length与packing_row等接口。
        dataset: 底层数据集（应包含'length'列以供打包）。
        num_proc (int): 预留参数，当前实现未使用多进程。
        strict (bool): 是否严格处理编码错误（传递给模板阶段使用）。
        load_from_cache_file (bool): 预留参数，控制是否从缓存加载。
        workers (List): 预留的工作进程列表（当前未使用）。
        packed_idx (List[List[int]]): 预计算的分组索引。
        packed_length (List[int]): 每组的总长度（sum of lengths）。
    """

    # 构造函数：保存配置并（在主进程）创建打包索引，然后在分布式环境中广播
    def __init__(
        self,
        template,
        dataset,
        num_proc: int = 1,
        *,
        strict: bool = False,
        load_from_cache_file: bool = True,
        **kwargs,
    ):
        """
        函数功能：
            初始化打包数据集；在主进程上创建pack索引，并在分布式初始化完成时广播给其他进程。

        入参：
            template: 模板对象（需具备packing能力）。
            dataset: 含'length'列的底层数据集。
            num_proc (int): 预留并发处理参数。
            strict (bool): 是否严格模式。
            load_from_cache_file (bool): 是否从缓存加载（当前未用）。
            **kwargs: 预留扩展参数。

        返回值：
            None

        示例：
            >>> ds = PackingDataset(template, dataset)
        """
        # 指示模板进入packing模式（可能影响encode行为）
        template._packing = True
        # 保存模板与数据集引用
        self.template = template
        self.dataset = dataset
        # 保存预留参数：进程数与严格模式
        self.num_proc = num_proc
        self.strict = strict
        # 保存是否从缓存加载标志
        self.load_from_cache_file = load_from_cache_file
        # 预留工作进程列表（当前未使用）
        self.workers = []
        # 仅主进程预计算打包索引与总长度；其他进程等待广播
        self.packed_idx, self.packed_length = self.create_packed_idx() if is_master() else (None, None)
        # 若处于分布式环境且通信已初始化，则广播对象列表到所有进程
        if dist.is_initialized() and is_dist():
            obj_list = [(self.packed_idx, self.packed_length)]
            dist.broadcast_object_list(obj_list)
            self.packed_idx, self.packed_length = obj_list[0]
        
    # 创建打包索引：基于length列进行装箱，生成每组的样本索引与总长度
    def create_packed_idx(self):
        """
        函数功能：
            读取数据集中'length'列，结合模板最大长度，通过装箱生成分组索引与每组总长度。

        入参：
            无

        返回值：
            Tuple[List[List[int]], List[int]]: (每组的样本索引列表, 每组的长度和)。

        示例：
            >>> packed_idx, packed_len = self.create_packed_idx()
        """
        # 获取每条样本的长度数组
        lengths = self.dataset['length']
        # 构造形如 (样本索引, 长度) 的列表，供装箱算法使用
        data = [(i, length) for i, length in enumerate(lengths)]
        # 初始化滑动窗口起点索引
        i = 0
        # 设置每批送入装箱算法的批大小，平衡效果与速度
        PACKING_BATCH_SIZE = 1000
        # 初始化输入缓存、结果索引与结果长度列表
        input_data, packed_idx, packed_length = [], [], []
        # 使用tqdm显示总进度条（total为样本数），动态列宽，描述为'Packing: '
        with tqdm(total=len(data), dynamic_ncols=True, desc='Packing: ') as prog_bar:
            # 持续地按批次推进直至处理完所有数据
            while True:
                # 取当前批次的新数据并追加到输入缓存
                # NOTE: Python 的列表切片（slice）是“安全切片”，即使起始索引超出范围，也不会报 IndexError，而是返回空列表 []
                new_data = data[i:i + PACKING_BATCH_SIZE]
                input_data += new_data
                # 更新进度条，增量为本批次数据量
                prog_bar.update(len(new_data))
                # 若缓存已空（无数据可供装箱），跳出循环
                if not input_data:
                    break
                # 前移批次起点
                i += PACKING_BATCH_SIZE
                # 标记本轮结束状态（处理到末尾）
                is_finished = i >= len(data)
                # 调用装箱函数：返回完成的分组与可能的残留缓存
                sequences, input_data = calculate_matched_group(self.template, input_data, is_finished=is_finished)
                # 将每个分组中的样本索引提取出来并追加到结果列表
                packed_idx += [[x[0] for x in seq] for seq in sequences]
                # 计算每个分组的长度和并追加
                packed_length += [sum(x[1] for x in seq) for seq in sequences]
        # 返回打包索引与每组长度
        return packed_idx, packed_length

    # 读取一组数据：根据预计算索引取出多条样本并调用模板进行packing
    def __getitem__(self, index):
        """
        函数功能：
            根据index索引取出预计算的样本序列，并使用模板进行打包，返回单个打包样本。

        入参：
            index (int): 组索引。

        返回值：
            Any: 模板packing_row返回的已打包样本。

        示例：
            >>> batch = self[0]
        """
        # 取出该组对应的样本索引序列
        sequence = self.packed_idx[index]
        # 根据索引从底层数据集中取出多条样本，形成列表
        row = [self.dataset[i] for i in sequence]
        # 调用模板的packing_row对该组样本进行拼接/裁剪等处理
        return self.template.packing_row(row)

    # 返回打包后的组数
    def __len__(self):
        """
        函数功能：
            返回预计算的打包组合数量。

        入参：
            无

        返回值：
            int: 组的数量。

        示例：
            >>> n = len(self)
        """
        return len(self.packed_idx)


# 定义可迭代的流式打包数据集：使用子进程并通过队列与装箱协作完成packing
class IterablePackingDataset(IterableDataset):
    """
    类功能：
        边迭代边打包的数据集实现。通过多进程对样本进行编码，主进程聚合返回，
        按interval批次做装箱，并将打包后的样本按需yield。

    关键属性：
        template: 模板对象。
        dataset: 可迭代/可索引的数据源。
        num_proc (int): 后台编码进程数。
        packing_interval (int): 每轮送入编码/装箱的样本数上限。
        strict (bool): 编码异常时是否严格抛错（忽略MaxLengthError）。
        cyclic (bool): 是否循环遍历数据源（无限流）。
        _in_queue/_out_queue (mp.Queue): 进/出队列，用于传递原始数据与编码结果。
        workers (List[mp.Process]): 后台工作进程列表。
    """

    # 构造函数：启动子进程，初始化队列与控制参数
    def __init__(
        self,
        template,
        dataset,
        num_proc: int = 1,
        *,
        packing_interval: int = 128,
        strict: bool = False,
        cyclic: bool = False,
        **kwargs,
    ):
        """
        函数功能：
            初始化流式打包数据集：设置并启动编码子进程，准备队列与控制参数。

        入参：
            template: 模板对象。
            dataset: 数据源。
            num_proc (int): 子进程数量。
            packing_interval (int): 每轮处理的样本上限。
            strict (bool): 是否严格处理编码异常。
            cyclic (bool): 是否循环遍历数据。
            **kwargs: 预留参数。

        返回值：
            None

        示例：
            >>> ds = IterablePackingDataset(template, dataset, num_proc=2)
        """
        # 指示模板进入packing模式
        template._packing = True
        # 保存模板与数据引用
        self.template = template
        self.dataset = dataset
        # 保存子进程数量与严格模式参数
        self.num_proc = num_proc
        self.strict = strict

        # 保存packing间隔样本数
        self.packing_interval = packing_interval
        # 创建进/出队列用于与子进程通信
        self._in_queue = mp.Queue()
        self._out_queue = mp.Queue()
        # 初始化工作进程列表
        self.workers = []
        # 保存是否循环取数标志
        self.cyclic = cyclic
        # 按num_proc创建并启动子进程，每个进程运行_processor作为工作循环
        for _ in range(self.num_proc):
            worker = mp.Process(target=self._processor, daemon=True)
            worker.start()
            self.workers.append(worker)

    # 子进程工作循环：从输入队列取数据，编码后放入输出队列
    def _processor(self):
        """
        函数功能：
            子进程中执行的循环：持续从_in_queue取样本，调用模板encode，
            若strict且异常并非MaxLengthError则抛出；否则将结果放入_out_queue。

        示例：
            （内部使用，无需直接调用）
        """
        # 持续处理直到主进程结束（守护进程随主进程退出）
        while True:
            # 从输入队列取出一个编号与数据对
            i, data = self._in_queue.get()
            # 预设编码结果为空字典，用于标识失败情形
            encoded_data = {}
            try:
                # 尝试调用模板进行编码，并要求返回长度信息
                encoded_data = self.template.encode(data, return_length=True)
            except Exception as e:
                # 严格模式下，除最大长度异常外，其他异常需要抛出以便上层处理
                if self.strict and not isinstance(e, MaxLengthError):
                    raise
            # 将结果（可能为空）放入输出队列，保留其在批次中的位置i
            self._out_queue.put((i, encoded_data))

    # 主进程：将若干条样本放入输入队列，返回实际放入的样本数
    def _put_data_in_queue(self, iterator) -> int:
        """
        函数功能：
            从迭代器中最多取packing_interval条数据，放入输入队列，返回本轮放入的样本数。

        入参：
            iterator: 数据迭代器。

        返回值：
            int: 实际放入队列的样本数量。

        示例：
            >>> n = self._put_data_in_queue(iter(self.dataset))
        """
        # 在当前轮内，按顺序为每条数据分配一个位置索引i
        for i in range(self.packing_interval):
            try:
                # 从迭代器取出下一条数据
                data = next(iterator)
            except StopIteration:
                # 迭代器耗尽，返回当前已放入的数量
                return i
            # 将位置索引与数据放入输入队列
            self._in_queue.put((i, data))
        # 若循环完整执行，说明放满了interval条，返回总数
        return i + 1

    # 主进程：从输出队列收集编码结果，按原位置还原顺序并返回累加后的结果列表
    def _fetch_data_out_queue(self, last_res, num_samples):
        """
        函数功能：
            从_out_queue取回num_samples条编码结果，按位置索引放回列表，
            过滤掉编码失败项，并累加到last_res后返回。

        入参：
            last_res (List): 上轮剩余/未处理的结果列表。
            num_samples (int): 本轮期望收集的结果数量。

        返回值：
            List: 追加本轮结果后的总结果列表，元素为 (编码结果, 长度)。

        示例：
            >>> data = self._fetch_data_out_queue([], 32)
        """
        # 初始化固定长度的占位列表，用于按i放置结果
        res = [None] * num_samples
        # 逐项从输出队列取回结果
        for _ in range(num_samples):
            i, data = self._out_queue.get()
            # 若编码结果为空（失败），跳过
            if not data:
                continue
            # 将编码结果与其长度组成二元组，按位置i放回
            res[i] = (data, len(data['input_ids']))
        # 过滤None占位，得到有效结果列表
        res = [data for data in res if data]
        # 将本轮有效结果追加到累计结果中
        last_res += res
        # 返回累计后的结果
        return last_res

    @staticmethod
    def cyclic_iter(iterable):
        """
        函数功能：
            对任意可迭代对象进行无限循环迭代的生成器。

        入参：
            iterable: 任意可迭代对象。

        返回值：
            Iterator: 无限循环地yield元素。

        示例：
            >>> it = IterablePackingDataset.cyclic_iter([1, 2])
        """
        # 外层无限循环，确保源被反复遍历
        while True:
            # 内层遍历一次可迭代对象，将元素逐个产出
            for x in iterable:
                yield x

    # 迭代器：边放入队列、边取回结果、边打包、边yield
    def __iter__(self):
        """
        函数功能：
            实现可迭代接口：不断地向子进程投喂数据并收集编码结果，按最大长度进行装箱，
            将打包后的样本逐个yield，直至数据源耗尽（或循环模式下持续）。

        入参：
            无

        返回值：
            Iterator: 逐个产出的打包样本。

        示例：
            >>> for packed in self: ...
        """
        # 快速检测数据集是否为空：若立刻抛出StopIteration，则直接结束
        try:
            next(iter(self.dataset))
        except StopIteration:
            return

        # 根据是否循环模式，选择不同的迭代器实现
        if self.cyclic:
            iterator = self.cyclic_iter(self.dataset)
        else:
            iterator = iter(self.dataset)
        # 用于累积编码结果（payload, length）对
        data = []
        # 主循环：每轮放入一批数据，取回结果，做装箱并yield
        while True:
            # 放入一批数据，获取本轮实际样本数
            num_samples = self._put_data_in_queue(iterator)
            # 若不足一个packing_interval，则本轮结束后整体结束
            finished = num_samples != self.packing_interval
            # 从输出队列取回本轮结果，并与历史结果累计
            data = self._fetch_data_out_queue(data, num_samples)
            # 按模板最大长度进行装箱，返回完成的分组与残留（覆盖data为残留）
            sequences, data = calculate_matched_group(self.template, data, is_finished=finished)
            # 临时缓冲本轮要产出的已打包样本
            res = []
            # 遍历每个分组，提取payload并调用模板进行packing
            for row in sequences:
                packed = self.template.packing_row([r[0] for r in row])
                res.append(packed)
            # 逐个yield本轮的已打包样本
            yield from res
            # 若数据源已耗尽（非循环模式），跳出主循环
            if finished:
                break


# 定义行级编码预处理器：可选择仅预先计算length供packing使用
class EncodePreprocessor(RowPreprocessor):
    """
    类功能：
        行级预处理器，封装对模板encode的调用；可选择仅写入length到原row，
        以便后续的PackingDataset/IterablePackingDataset使用。

    关键属性：
        template (Template): 模板对象。
        pre_tokenize (bool): 是否仅预先计算长度而不返回完整编码结果。
    """

    # 构造函数：保存模板引用与预标志
    def __init__(self, template: 'Template', pre_tokenize: bool = False):
        """
        函数功能：
            初始化预处理器，保存模板与是否预标注长度的配置。

        入参：
            template (Template): 模板对象。
            pre_tokenize (bool): 为True时，仅返回附加了length字段的原row。

        返回值：
            None

        示例：
            >>> pp = EncodePreprocessor(template, pre_tokenize=True)
        """
        # 调用父类构造函数，完成基类初始化
        super().__init__()
        # 保存模板引用
        self.template = template
        # 保存是否仅进行预标注长度的开关
        self.pre_tokenize = pre_tokenize

    # 行级预处理：编码或仅写入length
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        函数功能：
            对单行row执行模板encode；若pre_tokenize为True，则仅将length写回row并返回row。

        入参：
            row (Dict[str, Any]): 单条原始样本。

        返回值：
            Optional[Dict[str, Any]]: 编码后的样本字典，或仅写入长度后的原row。

        示例：
            >>> out = self.preprocess({"text": "hello"})
        """
        # 调用模板encode以获得完整编码（含length）
        encoded = self.template.encode(row, return_length=True)
        # 若仅需预标注长度，则将length写回到原row，并覆盖encoded为row
        if self.pre_tokenize:
            row['length'] = encoded['length']
            encoded = row
        # 返回编码结果或仅带length的原row
        return encoded
