"""
模块功能
-------
本模块提供统一的数据集加载与后处理入口：
1. 支持从本地路径、ModelScope、HuggingFace 三种来源加载数据集；
2. 通过 `DatasetSyntax` 解析命令行式数据集描述（含子集与采样量语法）；
3. 按注册信息 `DatasetMeta/SubsetDataset` 调用相应预处理器，完成列重命名、样本标准化；
4. 支持数据集拼接/交错混合、采样、划分训练/验证集、随机置乱、流式迭代等；
5. 提供 ModelScope 手工下载小工具方法。

快速示例
------
>>> # 1) 单个数据集（自动解析来源，并应用注册的预处理器）
>>> train_ds, val_ds = load_dataset('swift/alpaca:default#1000', split_dataset_ratio=0.1)
>>> len(train_ds), len(val_ds)

>>> # 2) 多个数据集合并，并按权重交错混合
>>> train_ds, _ = load_dataset(['swift/alpaca', 'swift/sharegpt'], interleave_prob=[0.7, 0.3])

设计要点
------
- 分布式环境中利用 `safe_ddp_context` 控制单/多进程安全下载与缓存复用；
- 统一的列映射/去除无用列能力，保证下游训练输入格式一致；
- 兼容 HF 与 MS 两套 Hub 接口，由 `get_hub` 动态分发。
"""

# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明
import os  # 操作系统路径与环境变量工具
import platform  # 判断操作系统平台，用于路径兼容
import re  # 正则表达式，解析数据集名称
import shutil  # 文件复制等操作工具
from contextlib import nullcontext  # 空上下文管理器，占位用于统一 with 逻辑
from dataclasses import dataclass, field  # 数据类工具，简化配置对象定义
from functools import partial  # 偏函数，用于预绑定上下文参数
from tempfile import TemporaryDirectory  # 创建临时目录，便于安全下载
from typing import Dict, List, Literal, Optional, Tuple, Union  # 类型注解

import numpy as np  # 数值与随机工具
from datasets import Dataset as HfDataset  # HF 数据集类（内存数据集）
from datasets import concatenate_datasets, interleave_datasets  # HF 数据集合并/交错混合
from datasets import load_dataset as hf_load_dataset  # HF 通用加载函数（本地/远程）
from modelscope.hub.api import ModelScopeConfig  # ModelScope 账号配置（cookies）
from modelscope.utils.config_ds import MS_CACHE_HOME  # ModelScope 缓存主目录

from swift.hub import get_hub  # 根据 use_hf 选择 HF/MS Hub 适配器
from swift.utils import download_ms_file, get_logger, get_seed, safe_ddp_context, use_hf_hub  # 常用工具
from .preprocessor import RowPreprocessor  # 预处理基类，提供列重命名/裁剪等能力
from .register import DATASET_MAPPING, DATASET_TYPE, DatasetMeta, SubsetDataset  # 数据集注册信息结构
from .utils import sample_dataset  # 数据集采样工具

logger = get_logger()  # 模块级日志记录器

_dataset_meta_mapping = None  # 懒初始化的“数据源到 DatasetMeta”映射缓存


@dataclass
class DatasetSyntax:
    """
    数据集语法解析器：将命令行风格字符串解析为结构化字段。

    语法
    ----
    - 格式：`[hf|ms::]dataset_id_or_path[:subset1/subset2][#dataset_sample]`
    - 示例：`hf::swift/alpaca:default#1000`、`/path/to/data.jsonl#500`、`swift/sharegpt`

    字段
    ----
    - dataset: 原始数据集标识（本地路径或仓库 ID）。
    - subsets: 需要选择的子集名称列表。
    - dataset_sample: 采样条数（None 表示全量）。
    - use_hf: 指明使用 HF(True)/MS(False)，None 代表未显式指定。
    """
    dataset: str  # 原始数据集标识（路径或仓库 ID）
    subsets: List[str] = field(default_factory=list)  # 子集列表，默认空列表
    dataset_sample: Optional[int] = None  # 采样条数，默认不采样
    use_hf: Optional[bool] = None  # 指定 hub 来源，默认由外部或环境判断

    def __post_init__(self):
        """
        初始化后置钩子：根据 `dataset` 判断数据集类型（本地路径或仓库）。

        - 若是文件路径，则认为是 `'path'` 类型；
        - 否则（目录或 ID），暂定为 `'repo'` 类型，由后续逻辑进一步确定 HF/MS。
        """
        if os.path.isfile(self.dataset):  # 若为存在的本地文件
            self.dataset_type = 'path'  # 标记为本地路径类型
        else:  # dataset_id or dataset_dir  # 否则视为仓库/目录类型
            self.dataset_type = 'repo'  # 初始标记为仓库类型

    def get_raw(self):
        """
        以原始语法拼接并返回数据集描述字符串（用于展示/日志）。

        返回
        ----
        - str: 形如 `dataset:subset1/subset2#sample` 的描述。
        """
        subsets = '/'.join(self.subsets)  # 将子集名称用 '/' 拼接
        dataset_sample = '' if self.dataset_sample is None else f'#{self.dataset_sample}'  # 追加采样量标记
        return f'{self.dataset}{subsets}{dataset_sample}'  # 返回拼接后的原始字符串

    @staticmethod
    def _safe_split(s: str,
                    sep: str,
                    use_0: bool,
                    split_mode: Literal['left', 'right'] = 'left') -> Tuple[Optional[str], Optional[str]]:
        """
        安全的二段分割工具。

        参数
        ----
        - s: 待分割字符串。
        - sep: 分隔符。
        - use_0: 当仅分割出一个片段时，返回值放在 part0（True）还是 part1（False）。
        - split_mode: 使用从左分割 `split` 还是从右分割 `rsplit`。

        返回
        ----
        - Tuple[Optional[str], Optional[str]]: (part0, part1)，分割失败处返回 None。

        示例
        ----
        >>> DatasetSyntax._safe_split('a:b', ':', True)
        ('a', 'b')
        >>> DatasetSyntax._safe_split('a', ':', True)
        ('a', None)
        >>> DatasetSyntax._safe_split('a', ':', False)
        (None, 'a')
        """
        if s is None or len(s) == 0:  # 空字符串直接返回空对
            return None, None  # 无可分割内容
        if split_mode == 'left':  # 选择从左侧分割
            part = s.split(sep, 1)  # 最多分割一次
        else:  # 从右侧分割
            part = s.rsplit(sep, 1)  # 最多分割一次
        if len(part) == 1:  # 分割未发生
            if use_0:  # 结果放在 part0
                part = part[0], None  # (值, None)
            else:  # 结果放在 part1
                part = None, part[0]  # (None, 值)
        else:  # 正常分割得到两个片段
            assert len(part) == 2  # 防御性断言
        return part  # 返回二元组

    @classmethod
    def parse(cls, dataset: str) -> 'DatasetSyntax':
        """
        函数功能：
            解析命令行式数据集字符串为 `DatasetSyntax` 对象。

        参数：
            - dataset: 原始描述字符串，形如 `hf::id:path#sample` 或本地路径。

        返回值：
            - DatasetSyntax: 解析后的结构化对象。

        使用示例：
            >>> DatasetSyntax.parse('hf::swift/alpaca:default#100')
            DatasetSyntax(...)
        """
        # hf/ms::dataset_id or dataset_path:subset1/subset2/subset3#dataset_sample  # 语法提示
        if os.path.exists(dataset):  # 若本地存在该路径（文件或目录）
            use_hf = None  # 不强制指定 hub 来源
        else:  # 远程仓库语法，需要解析前缀
            use_hf, dataset = cls._safe_split(dataset, '::', False)  # 分离 hub 前缀与剩余部分
            if isinstance(use_hf, str):  # 若前缀存在
                use_hf = use_hf.lower()  # 规范化大小写
            use_hf = {'hf': True, 'ms': False}.get(use_hf)  # 转换为布尔或 None
        if os.path.exists(dataset):  # 若剩余部分为本地路径
            other, dataset_sample = dataset, None  # 不包含采样段
        else:  # 解析采样段（从右分割）
            other, dataset_sample = cls._safe_split(dataset, '#', True, 'right')  # 提取 #sample
        if os.path.exists(other):  # 若 other 为本地路径
            dataset, subsets = other, None  # 无子集段
        else:  # 解析子集段（':' 左分割）
            dataset, subsets = cls._safe_split(other, ':', True)  # 提取 :subsets

        if subsets is not None:  # 若存在子集描述
            subsets = [subset.strip() for subset in subsets.split('/')]  # 以 '/' 拆分并去空格
        if dataset_sample is not None:  # 若存在采样量
            dataset_sample = int(dataset_sample)  # 转为整数
        return cls(dataset.strip(), subsets or [], dataset_sample, use_hf)  # 返回解析结果

    def get_dataset_meta(self, use_hf: bool):
        """
        根据解析结果与外部偏好，获取匹配的 `DatasetMeta`（数据集注册信息）。

        参数
        ----
        - use_hf: 外部指定是否使用 HF hub（用于仓库类型）。

        返回
        ----
        - DatasetMeta: 匹配到的注册信息；若找不到，尝试后缀名匹配；仍失败则返回空 `DatasetMeta()`。
        """
        dataset_meta_mapping = self._get_dataset_meta_mapping()  # 获取缓存映射
        dataset_type = self.dataset_type  # 初始数据源类型
        if dataset_type == 'path':  # 本地路径，按 (path, 绝对/小写) 精确匹配
            dataset_meta = dataset_meta_mapping.get((dataset_type, self.dataset.lower()))  # 取已注册的路径映射
        else:  # 仓库类型，需区分本地目录/远程 ID 与 HF/MS
            dataset_type = 'repo' if os.path.isdir(self.dataset) else {True: 'hf', False: 'ms'}[use_hf]  # 进一步细分
            dataset_meta = dataset_meta_mapping.get((dataset_type, self.dataset.lower()))  # 查表
        return dataset_meta or self._get_matched_dataset_meta(dataset_meta_mapping) or DatasetMeta()  # 兜底返回

    @staticmethod
    def _get_dataset_meta_mapping() -> Dict[Tuple[str, str], DatasetMeta]:
        """
        构建“(来源类型, 键) -> DatasetMeta” 的查找表，并做全局缓存。

        返回
        ----
        - Dict[(str, str), DatasetMeta]: 键为 ('path'|'repo'|'ms'|'hf', 小写键)。
        """
        global _dataset_meta_mapping  # 使用模块级缓存
        if _dataset_meta_mapping is not None:  # 已构建则直接返回
            return _dataset_meta_mapping
        _dataset_meta_mapping = {}  # 初始化映射
        for dataset_meta in DATASET_MAPPING.values():  # 遍历已注册数据集
            if dataset_meta.dataset_path is not None:  # 若注册了本地路径
                dataset_type = 'repo' if os.path.isdir(dataset_meta.dataset_path) else 'path'  # 判断路径是目录还是文件
                _dataset_meta_mapping[(dataset_type, dataset_meta.dataset_path.lower())] = dataset_meta  # 记录映射
            if dataset_meta.ms_dataset_id is not None:  # 注册了 MS ID
                _dataset_meta_mapping[('ms', dataset_meta.ms_dataset_id.lower())] = dataset_meta  # 记录映射
            if dataset_meta.hf_dataset_id is not None:  # 注册了 HF ID
                _dataset_meta_mapping[('hf', dataset_meta.hf_dataset_id.lower())] = dataset_meta  # 记录映射
        return _dataset_meta_mapping  # 返回构建好的映射

    @staticmethod
    def get_dataset_name(dataset_id: str) -> str:
        """
        从路径或缓存目录字符串中提取数据集名称（兼容 HF 本地缓存路径）。

        参数
        ----
        - dataset_id: 原始路径或 ID。

        返回
        ----
        - str: 提取到的短名称。
        """
        # compat hf hub  # 兼容 HF 本地缓存路径格式
        dataset_id = dataset_id.rstrip('/')  # 去除末尾斜杠
        match_ = re.search('/datasets--.+?--(.+?)/snapshots/', dataset_id)  # 匹配 HF 缓存目录中的数据集名
        if match_ is not None:  # 命中缓存路径格式
            return match_.group(1)  # 返回捕获组

        dataset_name = dataset_id.rsplit('/', 1)[-1]  # 常规从右分割获取最后一段
        if platform.system().lower() == 'windows':  # Windows 下可能包含反斜杠
            dataset_name = dataset_name.rsplit('\\', 1)[-1]  # 兼容反斜杠
        return dataset_name  # 返回数据集短名

    def _get_matched_dataset_meta(self, dataset_meta_mapping):
        """
        通过“后缀短名”匹配 `DatasetMeta`（不要求精确路径/ID 匹配）。

        参数
        ----
        - dataset_meta_mapping: 已构建的精确映射表。

        返回
        ----
        - Optional[DatasetMeta]: 匹配到的注册信息或 None。
        """
        suffix_dataset_meta_mapping = {}  # 短名到 meta 的映射
        for dataset_name, dataset_meta in dataset_meta_mapping.items():  # 遍历所有注册项
            dataset_name = self.get_dataset_name(dataset_name[1]).lower()  # 提取短名并小写
            suffix_dataset_meta_mapping[dataset_name] = dataset_meta  # 建立短名映射
        dataset_name = self.get_dataset_name(self.dataset).lower()  # 当前输入的短名
        dataset_meta = suffix_dataset_meta_mapping.get(dataset_name)  # 尝试匹配
        return dataset_meta  # 返回结果（可能为 None）


class DatasetLoader:
    """
    数据集加载器：封装多来源加载、自动预处理与后处理的静态方法集合。

    - 核心职责：下载/加载、子集选择、预处理、列映射、拼接/交错、采样、切分、置乱。
    - 适配分布式：通过 `safe_ddp_context` 在多进程环境下安全执行下载/缓存。
    """

    @staticmethod
    def download_ms_dataset(ms_dataset_id: str, files: List[str], force_download: bool = False) -> str:
        """
        从 ModelScope 仓库手动下载指定文件集合到本地缓存。

        参数
        ----
        - ms_dataset_id: ModelScope 的数据集 ID。
        - files: 需要下载的相对文件路径列表。
        - force_download: 是否强制重新下载（覆盖本地已存在文件）。

        返回
        ----
        - str: 本地缓存数据集目录路径（raw 子目录）。

        示例
        ----
        >>> DatasetLoader.download_ms_dataset('AI-ModelScope/LLaVA-Pretrain', ['images.zip'])
        """
        assert isinstance(files, list)  # 参数检查：files 必须为列表
        url = f'http://www.modelscope.cn/api/v1/datasets/{ms_dataset_id}/repo?Revision=master&FilePath={{fpath}}'  # 下载 URL 模板
        cache_dir = os.path.join(MS_CACHE_HOME, 'datasets', ms_dataset_id, 'master')  # 数据集缓存目录
        local_dir = os.path.join(cache_dir, 'raw')  # 原始文件存放目录
        tmp_dir = os.path.join(cache_dir, 'tmp')  # 临时下载目录
        os.makedirs(local_dir, exist_ok=True)  # 创建原始目录
        os.makedirs(tmp_dir, exist_ok=True)  # 创建临时目录
        cookies = ModelScopeConfig.get_cookies()  # 获取登录 cookies（如需要权限）
        with TemporaryDirectory(dir=tmp_dir) as temp_dir:  # 在缓存/tmp 内创建临时目录
            for remote_fpath in files:  # 遍历需要下载的文件列表
                url = url.format(fpath=remote_fpath)  # 格式化 URL（替换文件路径参数）
                temp_fpath = os.path.join(temp_dir, remote_fpath)  # 临时文件路径（保持层级）
                local_fpath = os.path.join(local_dir, remote_fpath)  # 目标落地路径
                if not force_download and os.path.exists(local_fpath):  # 已存在且不强制
                    continue  # 跳过下载
                download_ms_file(url, temp_fpath, cookies)  # 执行下载到临时文件
                shutil.copy2(temp_fpath, local_fpath)  # 拷贝临时文件到 raw 目录

        return local_dir  # 返回原始目录路径

    @staticmethod
    def _concat_datasets(datasets: List[HfDataset]) -> Optional[HfDataset]:
        """
        合并多个 HF 数据集为单个数据集（按顺序拼接）。

        参数
        ----
        - datasets: 数据集列表。

        返回
        ----
        - Optional[HfDataset]: 合并后的数据集，空列表返回 None。
        """
        if len(datasets) == 0:  # 空列表
            return  # 返回 None
        if len(datasets) == 1:  # 单个数据集
            return datasets[0]  # 直接返回
        return concatenate_datasets(datasets)  # 使用 HF 提供的拼接工具

    @staticmethod
    def _interleave_datasets(datasets, *args, **kwargs):
        """
        按概率权重等策略交错混合多个数据集。

        参数
        ----
        - datasets: 数据集列表。
        - *args, **kwargs: 透传给 `datasets.interleave_datasets` 的参数，如 probabilities、seed 等。

        返回
        ----
        - Optional[HfDataset]: 交错后的数据集，空列表返回 None。
        """
        if len(datasets) == 0:  # 空列表
            return  # 返回 None
        if len(datasets) == 1:  # 单个数据集
            return datasets[0]  # 直接返回
        return interleave_datasets(datasets, *args, **kwargs)  # 调用 HF 的交错函数

    @staticmethod
    def _load_dataset_path(
        dataset_path: str,
        dataset_meta: DatasetMeta,
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        strict: bool = False,
        streaming: bool = False,
        columns: Optional[Dict[str, str]] = None,
        remove_unused_columns: bool = True,
    ) -> HfDataset:
        """
        函数功能：
            从本地文件路径加载数据集，并调用注册预处理器进行标准化处理。

        参数：
            - dataset_path: 本地文件路径（csv/json/jsonl/txt 等）。
            - dataset_meta: 数据集注册信息，内部包含 `preprocess_func`。
            - num_proc: 预处理多进程数量。
            - load_from_cache_file: 是否使用缓存文件。
            - strict: 预处理遇到异常是否抛出。
            - streaming: 是否以流式方式加载。
            - columns: 可选的列重命名映射。
            - remove_unused_columns: 是否移除未使用列。

        返回值：
            - HfDataset: 标准化后的 HF 数据集。
        注意：
            - 虽然当前函数支持.txt格式，但是在hf_load_dataset中并不支持.txt，可能是需要文件内容符合某种特定格式（比如json）。
        """
        ext = os.path.splitext(dataset_path)[1].lstrip('.')  # 提取文件扩展名
        file_type = {'jsonl': 'json', 'txt': 'text'}.get(ext) or ext  # 兼容 jsonl/txt 到 HF 类型
        kwargs = {'split': 'train', 'streaming': streaming, 'num_proc': num_proc}  # 加载参数
        if file_type == 'csv':  # CSV 特殊处理避免将空串视作 NaN
            kwargs['na_filter'] = False  # 关闭 NA 过滤
        with safe_ddp_context(None, True):  # 分布式安全上下文（仅主进程实际执行）
            dataset = hf_load_dataset(file_type, data_files=dataset_path, **kwargs)  # 通过 HF 加载本地文件
        if columns:  # 若提供列映射
            dataset = RowPreprocessor.safe_rename_columns(dataset, columns)  # 安全重命名列
        dataset = dataset_meta.preprocess_func(  # 调用注册的预处理器进行标准化
            dataset, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict)
        if remove_unused_columns:  # 如需裁剪无用列
            dataset = RowPreprocessor.remove_useless_columns(dataset)  # 移除非必要字段
        return dataset  # 返回处理结果

    @staticmethod
    def _load_repo_dataset(
        dataset_id: str,
        subset: SubsetDataset,
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        streaming: bool = False,
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        strict: bool = False,
        revision: Optional[str] = None,
        download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
        columns: Optional[Dict[str, str]] = None,
        remove_unused_columns: bool = True,
    ) -> HfDataset:
        """
        从仓库（本地目录/HF/MS）加载指定子集与切分，应用预处理并返回合并后的数据集。

        参数
        ----
        - dataset_id: 数据集 ID 或本地目录路径。
        - subset: 目标子集配置（名称、预处理函数、切分列表等）。
        - num_proc 等: 见 `_load_dataset_path`。

        返回
        ----
        - HfDataset: 合并后的数据集。
        """
        datasets = []  # 存放每个 split 的数据集
        if os.path.isdir(dataset_id):  # 若传入的是本地目录
            retry = 1  # 本地目录无需重试
            load_context = nullcontext  # 空上下文，无需 DDP 屏障
            use_hf = True  # 将其当作 HF 本地目录兼容处理
            dataset_str = f'Use local folder, dataset_dir: {dataset_id}'  # 记录日志
            # The dataset downloaded from modelscope will have an additional dataset_infos.json file.  # MS 下载目录有额外文件
            dataset_infos_path = os.path.join(dataset_id, 'dataset_infos.json')  # 该文件会干扰 HF 解析
            if os.path.isfile(dataset_infos_path):  # 若存在
                os.rename(dataset_infos_path, f'{dataset_infos_path}_bak')  # 暂时改名以规避
        elif dataset_id.startswith('/'):
            # 若是绝对路径但目录不存在，则报错提醒
            raise ValueError(f'The local path does not exist, dataset_id: `{dataset_id}`. '
                             f'os.path.exists(dataset_id): {os.path.exists(dataset_id)}')
        else:  # 远程仓库 ID
            retry = 3  # 网络加载允许重试
            load_context = partial(safe_ddp_context, hash_id=dataset_id, use_barrier=True)  # DDP 同步下载
            dataset_str_f = 'Downloading the dataset from {hub}, dataset_id: {dataset_id}'  # 日志模板
            if use_hf:  # 选择 hub 名称
                dataset_str = dataset_str_f.format(hub='HuggingFace', dataset_id=dataset_id)  # HF 日志
            else:
                dataset_str = dataset_str_f.format(hub='ModelScope', dataset_id=dataset_id)  # MS 日志
        logger.info(dataset_str)  # 记录加载来源信息
        hub = get_hub(use_hf)  # 获取对应 hub 适配器
        for split in subset.split:  # 遍历需要的切分（train/validation/test 等）
            i = 1  # 当前重试次数
            with load_context():  # 在指定的上下文（DDP 或空）中执行
                while True:  # 直到成功或耗尽重试次数
                    try:
                        dataset = hub.load_dataset(  # 调用 hub 适配器加载
                            dataset_id,
                            subset.subset,
                            split,
                            streaming=streaming,
                            revision=revision,
                            download_mode=download_mode,
                            hub_token=hub_token,
                            num_proc=num_proc)
                    except Exception as e:  # 捕获异常
                        if i == retry:  # 达到重试上限
                            raise  # 向上抛出
                        i += 1  # 增加重试次数
                        logger.error(f'Dataset {dataset_id} load failed: subset_name={subset.subset},'
                                     f'split={split} with error: {e}')  # 打印错误日志并重试
                    else:  # 成功
                        break  # 跳出重试循环
            if hasattr(dataset, '_hf_ds'):  # 某些 hub 返回包装对象，取出底层 HF 数据集
                dataset = dataset._hf_ds  # 解包
                if streaming and isinstance(dataset, HfDataset):  # 若流式且为内存数据集
                    dataset = dataset.to_iterable_dataset()  # 转为可迭代数据集
            if columns:  # 如需列重命名
                dataset = RowPreprocessor.safe_rename_columns(dataset, columns)  # 执行重命名
            dataset = subset.preprocess_func(  # 执行子集预处理函数
                dataset, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict)
            if remove_unused_columns:  # 去除无用列
                dataset = RowPreprocessor.remove_useless_columns(dataset)  # 裁剪字段
            datasets.append(dataset)  # 收集该 split 的结果
        return DatasetLoader._concat_datasets(datasets)  # 合并所有 split

    @staticmethod
    def _select_subsets(subsets: List[str], dataset_meta: DatasetMeta) -> List[SubsetDataset]:
        """
        根据用户输入的子集名称列表，选择并规范化 `SubsetDataset` 列表。

        规则
        ----
        - 未提供子集且注册表只有 0/1 个子集：使用该子集（或空）。
        - 未提供子集但存在 `default`：使用 `default`。
        - 提供了 `all` 但注册表未显式提供 `all`：选择所有非弱子集。
        - 名称未注册则构造一个临时 `SubsetDataset`。
        """
        subset_mapping = {subset.name: subset for subset in dataset_meta.subsets}  # 名称到对象的映射
        subset_names = list(subset_mapping.keys())  # 可选子集名称集合
        if not subsets:  # 未指定子集
            if len(subset_names) <= 1:  # 0 或 1 个注册子集
                subsets = subset_names  # 使用现有（可能为空）
            elif 'default' in subset_names:  # 多个子集时优先 default
                subsets = ['default']  # 仅选择 default
            else:  # 无法确定默认
                raise ValueError(f'Please provide subsets. available subsets: {subset_names}')  # 提示用户指定
        elif len(subsets) == 1 and subsets[0] == 'all' and 'all' not in subset_names:  # 用户请求 all，但注册表无 all
            subsets = [subset_name for subset_name in subset_names if not subset_mapping[subset_name].is_weak_subset]  # 选择非弱子集

        subsets = [  # 将名称替换为对象（若未知则新建占位子集）
            subset_mapping[subset_name] if subset_name in subset_mapping else SubsetDataset(subset=subset_name)
            for subset_name in subsets
        ]
        return [subset.set_default(dataset_meta) for subset in subsets]  # 补齐默认配置并返回

    @staticmethod
    def shuffle_dataset(dataset, seed: int, buffer_size: int = 1000):
        """
        统一的随机置乱接口：兼容 HF 内存数据集与可迭代数据集。

        参数
        ----
        - dataset: 目标数据集（HfDataset 或 IterableDataset）。
        - seed: 随机种子。
        - buffer_size: 仅对可迭代数据集有效的缓冲大小。
        """
        if isinstance(dataset, HfDataset):  # HF 内存数据集
            with safe_ddp_context(None, True):  # 分布式下仅主进程置乱
                return dataset.shuffle(seed=seed)  # 使用 HF 自带的 shuffle
        else:  # 可迭代数据集
            return dataset.shuffle(seed=seed, buffer_size=buffer_size)  # 传入缓冲大小进行置乱

    @staticmethod
    def post_process(
        train_dataset: DATASET_TYPE,
        *,
        dataset_sample: Optional[int] = None,
        split_dataset_ratio: float = 0.,
        streaming: bool = False,
        shuffle: bool = True,
        random_state: Optional[np.random.RandomState] = None,
    ) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
        """
        数据集后处理：采样与划分训练/验证集（兼容流式与非流式）。

        参数
        ----
        - train_dataset: 原始训练数据集（或可迭代数据集）。
        - dataset_sample: 采样总量（None 表示不限制）。
        - split_dataset_ratio: 验证集比例（0 表示不切分，1 表示全部作为验证）。
        - streaming: 是否为流式数据集。
        - shuffle: 是否置乱（非流式分割时生效）。
        - random_state: 随机状态或种子。

        返回
        ----
        - (train_dataset, val_dataset): 处理后的训练与验证数据集（可能为 None）。
        """
        assert dataset_sample is None or dataset_sample > 0  # 采样量需为正或 None
        assert 0 <= split_dataset_ratio <= 1  # 比例范围校验
        if streaming:  # 流式场景
            if dataset_sample is None:  # 未指定采样量
                if split_dataset_ratio == 0:  # 不切分
                    val_dataset = None  # 验证集为空
                elif split_dataset_ratio == 1:  # 全部作为验证
                    train_dataset, val_dataset = None, train_dataset  # 交换
                else:  # 流式不支持按比例切分（除 0 或 1）
                    raise ValueError('The IterableDataset does not support splitting the training set '
                                     'and validation set when dataset_sample is None.')
            else:  # 指定采样量
                # not shuffle  # 流式下不置乱，直接顺序截取
                train_dataset = train_dataset.take(dataset_sample)  # 先截取总样本
                val_sample = int(dataset_sample * split_dataset_ratio)  # 计算验证集样本数
                val_dataset = None if val_sample == 0 else train_dataset.take(val_sample)  # 取前 val_sample 作为验证
                if val_sample:  # 若验证集非空
                    train_dataset = train_dataset.skip(val_sample)  # 训练集跳过验证部分
        else:  # 非流式场景
            if dataset_sample is None:  # 未指定采样量
                dataset_sample = len(train_dataset)  # 默认全量
            if split_dataset_ratio == 0:  # 不切分
                train_dataset = sample_dataset(train_dataset, dataset_sample, shuffle, random_state)  # 先采样
                val_dataset = None  # 无验证集
            elif split_dataset_ratio == 1:  # 全部作为验证
                train_dataset, val_dataset = None, train_dataset  # 切换
                val_sample = dataset_sample  # 验证集样本数即总样本数
                # Avoid duplication in the val_dataset.  # 避免越界
                assert val_sample <= len(val_dataset), f'val_sample: {val_sample}, len(val_dataset): {len(val_dataset)}'
                val_dataset = sample_dataset(val_dataset, val_sample, shuffle, random_state)  # 对验证集采样
            else:  # 同时存在训练与验证
                # Avoid duplication in the val_dataset.  # 计算合理的切分点
                train_len = min(len(train_dataset), dataset_sample)  # 可用总量
                val_sample = max(int(train_len * split_dataset_ratio), 1)  # 至少取 1 作为验证
                train_sample = dataset_sample - val_sample  # 训练样本数
                assert train_sample > 0  # 保证训练集非空
                with safe_ddp_context(None, True):  # 使用 HF 的分割工具（仅主进程执行）
                    train_dataset, val_dataset = train_dataset.train_test_split(
                        test_size=val_sample, shuffle=shuffle, seed=get_seed(random_state)).values()
                train_dataset = sample_dataset(train_dataset, train_sample, shuffle, random_state)  # 再对训练集采样
        return train_dataset, val_dataset  # 返回二元组

    @staticmethod
    def load(
        dataset_syntax: Optional[DatasetSyntax] = None,
        dataset_meta: Optional[DatasetMeta] = None,
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        streaming: bool = False,
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        strict: bool = False,
        download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
        columns: Optional[Dict[str, str]] = None,
        remove_unused_columns: bool = True,
    ) -> HfDataset:
        """
        加载单个数据集（路径或仓库），并按子集/切分合并为一个 HF 数据集。

        参数
        ----
        - dataset_syntax: 数据集语法对象。
        - dataset_meta: 注册的元信息（含加载/预处理函数等）。
        - 其他关键字参数：控制并行、缓存、流式、列映射等行为。

        返回
        ----
        - HfDataset: 合并后的 HF 数据集。
        """
        if dataset_syntax.dataset_type == 'path':  # 本地文件路径
            dataset = DatasetLoader._load_dataset_path(  # 直接从路径加载
                dataset_syntax.dataset,
                dataset_meta=dataset_meta,
                num_proc=num_proc,
                load_from_cache_file=load_from_cache_file,
                strict=strict,
                streaming=streaming,
                columns=columns,
                remove_unused_columns=remove_unused_columns,
            )
        else:  # 仓库类型（本地目录/HF/MS）
            subsets: List[SubsetDataset] = DatasetLoader._select_subsets(dataset_syntax.subsets, dataset_meta)  # 选择子集
            revision = dataset_meta.hf_revision if use_hf else dataset_meta.ms_revision  # 选择版本（HF/MS）
            datasets = []  # 收集每个子集/切分的结果
            for subset in subsets:  # 遍历选中的子集
                dataset = DatasetLoader._load_repo_dataset(  # 加载该子集
                    dataset_syntax.dataset,
                    subset,
                    use_hf=use_hf,
                    hub_token=hub_token,
                    num_proc=num_proc,
                    load_from_cache_file=load_from_cache_file,
                    strict=strict,
                    revision=revision,
                    streaming=streaming,
                    download_mode=download_mode,
                    columns=columns,
                    remove_unused_columns=remove_unused_columns,
                )
                datasets.append(dataset)  # 收集
            dataset = DatasetLoader._concat_datasets(datasets)  # 合并所有子集/切分
        return dataset  # 返回合并后的数据集


def init_self_cognition_preprocessor(
    dataset_meta: Optional[DatasetMeta],
    model_name: Optional[Union[Tuple[str, str], List[str]]] = None,
    model_author: Optional[Union[Tuple[str, str], List[str]]] = None,
) -> None:
    """
    初始化自我认知任务的预处理器参数（如模型名称与作者的中英配置）。

    参数
    ----
    - dataset_meta: 目标数据集的元信息（含预处理器实例）。
    - model_name: 模型名称（支持 [zh, en] 或二元组）。
    - model_author: 模型作者（支持 [zh, en] 或二元组）。
    """
    if dataset_meta is None or model_name is None and model_author is None:  # 无需配置则直接返回
        return  # 退出函数
    kwargs = {}  # 收集归一化后的参数
    # zh, en  # 两种语言的字段
    for key in ['name', 'author']:  # 处理 name 与 author 两个键
        val = locals()[f'model_{key}']  # 取入参对应值
        if isinstance(val, str):  # 若是单字符串
            val = [val]  # 统一转为列表形式
        if val is not None and val[0] is not None and (len(val) == 1 or val[1] is None):  # 仅提供一种语言时
            val = (val[0], val[0])  # 复制为 (zh, zh) 或 (en, en)
        kwargs[key] = val  # 写入参数字典
    
    from .dataset.llm import SelfCognitionPreprocessor  # 延迟导入，避免循环依赖
    preprocess_funcs = [dataset_meta.preprocess_func]  # 根预处理函数（可能即为 SelfCognitionPreprocessor）
    preprocess_funcs += [subset.preprocess_func for subset in dataset_meta.subsets if isinstance(subset, SubsetDataset)]  # 追加子集预处理器
    for preprocess_func in preprocess_funcs:  # 遍历所有相关预处理器
        if isinstance(preprocess_func, SelfCognitionPreprocessor):  # 找到目标类
            preprocess_func.set_name_author(**kwargs)  # 注入名称与作者配置
    logger.info_once(f"SelfCognitionPreprocessor has been successfully configured with name: {kwargs['name']}, "
                     f"author: {kwargs['author']}.")  # 仅打印一次的成功日志


def load_dataset(
    datasets: Union[List[str], str],
    *,
    split_dataset_ratio: float = 0.,
    seed: Union[int, np.random.RandomState, None] = 42,
    num_proc: int = 1,
    load_from_cache_file: bool = True,
    shuffle: bool = False,
    streaming: bool = False,
    interleave_prob: Optional[List[float]] = None,
    stopping_strategy: Literal['first_exhausted', 'all_exhausted'] = 'first_exhausted',
    shuffle_buffer_size: int = 1000,
    use_hf: Optional[bool] = None,
    hub_token: Optional[str] = None,
    strict: bool = False,
    download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
    columns: Optional[Dict[str, str]] = None,  # columns_mapping
    remove_unused_columns: bool = True,
    # self-cognition
    model_name: Optional[Union[Tuple[str, str], List[str]]] = None,  # zh, en
    model_author: Optional[Union[Tuple[str, str], List[str]]] = None,
) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
    """
    统一入口：按配置加载一个或多个已注册数据集，返回训练/验证数据集。

    参数
    ----
    - datasets: 单个或多个数据集描述字符串（支持 `DatasetSyntax` 语法）。
    - split_dataset_ratio: 验证集比例（0~1）。
    - seed: 随机种子或 `np.random.RandomState`。
    - num_proc: 预处理并行度。
    - load_from_cache_file: 是否读取缓存。
    - shuffle: 是否最终置乱。
    - streaming: 是否以流式加载。
    - interleave_prob: 多数据集交错权重列表（与 datasets 同长度）。
    - stopping_strategy: 交错停止策略。
    - shuffle_buffer_size: 流式数据集置乱缓冲大小。
    - use_hf: 指定使用 HF(True)/MS(False)，None 自动判断。
    - hub_token: 访问远程 hub 的凭据。
    - strict: 预处理异常是否抛出。
    - download_mode: 下载策略。
    - columns: 列重命名映射。
    - remove_unused_columns: 是否去除未使用列。
    - model_name/model_author: 自我认知任务参数（中英）。

    返回
    ----
    - (train_dataset, val_dataset): 训练与验证数据集（可能为 None）。

    示例
    ----
    >>> load_dataset(['swift/alpaca', 'swift/sharegpt'], split_dataset_ratio=0.1, interleave_prob=[0.5, 0.5])
    """
    init_self_cognition_preprocessor(DATASET_MAPPING.get('self-cognition'), model_name, model_author)  # 配置自我认知预处理器
    if isinstance(datasets, str):  # 若传入单个字符串
        datasets = [datasets]  # 归一化为列表
    if not isinstance(seed, np.random.RandomState):  # 将整数种子转为 RandomState
        seed = np.random.RandomState(seed)  # 统一随机接口
    if streaming:  # 流式模式不支持多进程 map
        num_proc = None  # 置空并行度
    train_datasets = []  # 收集训练数据集（或组合）
    val_datasets = []  # 收集验证数据集（或组合）
    load_kwargs = {  # 通用加载关键字参数
        'num_proc': num_proc,
        'load_from_cache_file': load_from_cache_file,
        'strict': strict,
        'download_mode': download_mode,
        'columns': columns,
        'streaming': streaming,
        'hub_token': hub_token,
        'remove_unused_columns': remove_unused_columns,
    }
    use_hf_default = use_hf  # 基准 hub 选择
    if use_hf_default is None:  # 未显式指定时
        use_hf_default = True if use_hf_hub() else False  # 根据环境判断是否可用 HF hub
    for dataset in datasets:  # 逐个数据集处理
        dataset_syntax = DatasetSyntax.parse(dataset)  # 解析语法
        use_hf = dataset_syntax.use_hf or use_hf_default  # 优先使用语法中指定的 hub，否则用默认值
        # compat dataset_name  # 兼容注册名：若传入的是注册键，转换为真实路径或 ID
        if dataset_syntax.dataset in DATASET_MAPPING:  # 命中注册键
            dataset_meta = DATASET_MAPPING[dataset_syntax.dataset]  # 取注册信息
            if dataset_syntax.use_hf is None and dataset_meta.dataset_path is not None:  # 未指定 hub 且存在本地路径
                dataset_syntax.dataset = dataset_meta.dataset_path  # 使用本地路径
                dataset_syntax.dataset_type = 'path'  # 标记为本地类型
            else:  # 根据 hub 选择 ID
                dataset_syntax.dataset = dataset_meta.hf_dataset_id if use_hf else dataset_meta.ms_dataset_id  # 设置远程 ID
        else:  # 未注册键，按路径/ID 获取对应 meta
            dataset_meta = dataset_syntax.get_dataset_meta(use_hf)  # 查找或构建元信息
        load_function = dataset_meta.load_function  # 加载函数（可被覆写）
        train_dataset = load_function(dataset_syntax, dataset_meta, **load_kwargs, use_hf=use_hf)  # 实际加载数据集
        train_dataset, val_dataset = DatasetLoader.post_process(  # 采样与切分
            train_dataset,
            dataset_sample=dataset_syntax.dataset_sample,
            split_dataset_ratio=split_dataset_ratio,
            streaming=streaming,
            shuffle=shuffle,
            random_state=seed,
        )
        if train_dataset is not None:  # 训练集存在则收集
            train_datasets.append(train_dataset)
        if val_dataset is not None:  # 验证集存在则收集
            val_datasets.append(val_dataset)

    if interleave_prob is None:  # 未指定交错权重
        train_datasets = DatasetLoader._concat_datasets(train_datasets)  # 直接拼接训练集
        val_datasets = DatasetLoader._concat_datasets(val_datasets)  # 直接拼接验证集
    else:  # 按权重交错
        train_datasets = DatasetLoader._interleave_datasets(
            train_datasets, interleave_prob, seed=get_seed(seed), stopping_strategy=stopping_strategy)  # 交错训练集
        val_datasets = DatasetLoader._interleave_datasets(
            val_datasets, interleave_prob, seed=get_seed(seed), stopping_strategy=stopping_strategy)  # 交错验证集

    if shuffle:  # 最终可选置乱
        if train_datasets:  # 训练集非空
            train_datasets = DatasetLoader.shuffle_dataset(
                train_datasets, seed=get_seed(seed), buffer_size=shuffle_buffer_size)  # 置乱训练集
        if val_datasets:  # 验证集非空
            val_datasets = DatasetLoader.shuffle_dataset(
                val_datasets, seed=get_seed(seed), buffer_size=shuffle_buffer_size)  # 置乱验证集
    return train_datasets, val_datasets  # 返回二元组
