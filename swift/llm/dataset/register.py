# Copyright (c) Alibaba, Inc. and its affiliates.
"""swift.llm.dataset.register

该模块负责“数据集注册与检索”的统一入口，提供数据集元信息 `DatasetMeta`、子集信息 `SubsetDataset` 的数据结构，
以及面向外部/配置文件（例如 `dataset_info.json`）的注册函数。核心能力包括：

- 以模型仓库（ModelScope/HuggingFace）ID、本地路径或自定义名称为键，维护全局 `DATASET_MAPPING` 映射表。
- 支持在 JSON 配置中以更简洁的语法定义列映射与预处理器，并自动实例化为 `AutoPreprocessor` 或 `MessagesPreprocessor`。
- 提供便捷函数批量注册/单条注册、查询当前可用数据集列表（会根据是否使用 HF Hub 返回不同标识）。

快速示例：

>>> # 从 JSON 文件路径注册
>>> from swift.llm.dataset.register import register_dataset_info, get_dataset_list
>>> metas = register_dataset_info('/path/to/dataset_info.json')  # 读取并注册  # doctest: +SKIP
>>> all_names = get_dataset_list()  # 查询可用数据集标识列表                # doctest: +SKIP
>>> len(metas), len(all_names)  # 返回注册的元信息对象与名称数量            # doctest: +SKIP

注意：该模块仅维护“元信息与注册流程”，真实数据加载在 `DatasetMeta.load_function` 所指函数中完成（默认指向 `DatasetLoader.load`）。
"""
import os  # 导入 os，用于路径拼接、绝对化与文件存在性判断等文件系统操作
from copy import deepcopy  # 导入 deepcopy，用于安全地复制嵌套对象，避免引用共享导致副作用
from dataclasses import dataclass, field  # 导入 dataclass/field，用于声明数据类与默认工厂
from re import L
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型注解，提升可读性与静态检查

import json  # 导入 json，用于读取/解析数据集信息配置

from swift.utils import get_logger, use_hf_hub  # 导入日志工具与是否使用 HF Hub 的判断函数
from .preprocessor import DATASET_TYPE, AutoPreprocessor, MessagesPreprocessor  # 导入数据统一类型与预处理器

PreprocessFunc = Callable[..., DATASET_TYPE]  # 预处理函数类型别名：入参灵活，返回标准化数据集类型
LoadFunction = Callable[..., DATASET_TYPE]  # 数据加载函数类型别名：用于从元信息加载实际数据集
logger = get_logger()  # 初始化模块级日志记录器，用于输出注册与处理信息


@dataclass  # 使用 dataclass 简化子集数据结构的定义与初始化
class SubsetDataset:
    """数据集子集定义。

    用于描述一个数据集中的“子集（subset）”及其独立的处理/切分配置。

    - `name`: 子集的标识名，用于匹配使用；若不填，默认使用 `subset`。
    - `subset`: 在 Hub（如 MS/HF）上的子集名，例如 `default`/`train` 等。
    - `split`: 该子集需要加载的数据切分列表，如 `["train", "validation"]`。
    - `preprocess_func`: 子集独享的预处理函数，若为空则回退到 `DatasetMeta.preprocess_func`。
    - `is_weak_subset`: 弱子集标记；当数据集指定 "all" 时，弱子集可被跳过。

    示例：
    >>> sd = SubsetDataset(subset='default', split=['train'])  # 定义一个默认子集            # doctest: +SKIP
    >>> sd.name  # name 未提供时，初始化后会被设置为 subset 同名                     # doctest: +SKIP
    'default'
    """
    # `Name` is used for matching subsets of the dataset, and `subset` refers to the subset_name on the hub.
    name: Optional[str] = None  # 子集的显示名称；若为 None，将在初始化中被设为 subset
    # If set to None, then subset is set to subset_name.
    subset: str = 'default'  # Hub 上的 subset 名称，默认使用 'default'

    # Higher priority. If set to None, the attributes of the DatasetMeta will be used.
    split: Optional[List[str]] = None  # 子集的切分列表；为 None 时回退使用 DatasetMeta 的 split
    preprocess_func: Optional[PreprocessFunc] = None  # 子集级预处理器；为 None 时回退使用 DatasetMeta 的预处理器

    # If the dataset specifies "all," weak subsets will be skipped.
    is_weak_subset: bool = False  # 是否为弱子集；当加载 all 时可被跳过以节省资源

    def __post_init__(self):  # dataclass 初始化后置钩子，用于补全/约束字段
        """在对象创建完成后做字段缺省填充。

        将 `name` 为空的情况统一回退为 `subset`，保证子集对象总有易记的标识名。

        示例：
        >>> SubsetDataset(subset='s1').name  # 未显式提供 name，则使用 subset
        's1'
        """
        if self.name is None:  # 若未提供 name
            self.name = self.subset  # 将 name 设置为 subset，保持标识一致

    def set_default(self, dataset_meta: 'DatasetMeta') -> 'SubsetDataset':  # 生成带默认字段回填的新子集对象
        """将当前子集对象与 `DatasetMeta` 合并，回填缺省字段。

        当子集未指定 `split` 或 `preprocess_func` 时，从 `dataset_meta` 中取同名字段填充，
        返回一个新的 `SubsetDataset` 实例，避免修改原对象。

        Args:
            dataset_meta: 数据集元信息对象，提供默认 `split` 与 `preprocess_func`。

        Returns:
            新的 `SubsetDataset`，其缺省字段已根据 `dataset_meta` 补全。

        示例：
        >>> dm = DatasetMeta(split=['train'])
        >>> SubsetDataset(subset='s').set_default(dm).split
        ['train']
        """
        subset_dataset = deepcopy(self)  # 复制当前对象，避免直接修改原始实例
        for k in ['split', 'preprocess_func']:  # 遍历需要从 DatasetMeta 回填的字段键
            v = getattr(subset_dataset, k)  # 读取当前对象对应字段的值
            if v is None:  # 若该字段未设置
                setattr(subset_dataset, k, deepcopy(getattr(dataset_meta, k)))  # 使用 DatasetMeta 的值深拷贝填充
        return subset_dataset  # 返回填充后的新对象


@dataclass  # 使用 dataclass 定义数据集元信息容器
class DatasetMeta:
    """数据集元信息。

    同时支持三类数据源标识：ModelScope ID、HuggingFace ID、本地路径；还可用自定义
    `dataset_name` 作为注册键。并允许为所有子集设置默认的 `split` 与 `preprocess_func`。

    字段说明：
    - `ms_dataset_id`/`hf_dataset_id`/`dataset_path`: 三选一或组合，用于定位数据源。
    - `dataset_name`: 自定义名称；若提供，会作为 `DATASET_MAPPING` 的键以覆盖三元组键。
    - `ms_revision`/`hf_revision`: 对应 Hub 的版本/修订号。
    - `subsets`: 子集列表；元素可为 `SubsetDataset` 或字符串（后者将在初始化时转换）。
    - `split`: 默认切分列表，适用于所有子集（可被子集自身覆盖）。
    - `preprocess_func`: 默认预处理器，优先进行列映射再执行预处理逻辑。
    - `load_function`: 实际的数据加载函数，默认指向 `DatasetLoader.load`。
    - `tags`/`help`: 元信息标签与帮助说明，便于人类可读与工具筛选。
    - `huge_dataset`: 大规模数据集标志，供上游逻辑做资源/策略优化。

    示例：
    >>> meta = DatasetMeta(hf_dataset_id='tatsu-lab/alpaca')
    >>> isinstance(meta.preprocess_func, AutoPreprocessor)
    True
    """
    ms_dataset_id: Optional[str] = None  # ModelScope 数据集 ID（可选）
    hf_dataset_id: Optional[str] = None  # HuggingFace 数据集 ID（可选）
    dataset_path: Optional[str] = None  # 本地数据集路径（可选）
    dataset_name: Optional[str] = None  # 自定义注册名称（优先级高于三元组键）
    ms_revision: Optional[str] = None  # ModelScope 数据集修订版本（可选）
    hf_revision: Optional[str] = None  # HuggingFace 数据集修订版本（可选）

    subsets: List[Union[SubsetDataset, str]] = field(default_factory=lambda: ['default'])  # 子集列表，默认含 'default'
    # Applicable to all subsets.
    split: List[str] = field(default_factory=lambda: ['train'])  # 默认切分，适用于所有子集（可被覆盖）
    # First perform column mapping, then proceed with the preprocess_func.
    preprocess_func: PreprocessFunc = field(default_factory=lambda: AutoPreprocessor())  # 默认预处理器：先映射列再处理
    load_function: Optional[LoadFunction] = None  # 数据加载函数；为空时在初始化中设置为 DatasetLoader.load

    tags: List[str] = field(default_factory=list)  # 可选标签，便于检索/筛选
    help: Optional[str] = None  # 帮助说明字符串
    huge_dataset: bool = False  # 是否为超大规模数据集

    def __post_init__(self):  # dataclass 初始化后置钩子
        """在对象创建后补齐默认加载函数并规范化子集结构。

        - 若 `load_function` 未提供，则设置为 `DatasetLoader.load`。
        - 若 `subsets` 中存在字符串元素，则转换为 `SubsetDataset(subset=字符串)`。

        示例：
        >>> dm = DatasetMeta(subsets=['default', 'valid'])
        >>> [type(s).__name__ for s in dm.subsets]
        ['SubsetDataset', 'SubsetDataset']
        """
        from .loader import DatasetLoader  # 延迟导入，避免循环依赖
        if self.load_function is None:  # 若未设置加载函数
            self.load_function = DatasetLoader.load  # 使用默认的 DatasetLoader.load
        for i, subset in enumerate(self.subsets):  # 遍历子集列表
            if isinstance(subset, str):  # 若子集是字符串（简写）
                self.subsets[i] = SubsetDataset(subset=subset)  # 规范化为 SubsetDataset 对象


DATASET_MAPPING: Dict[Tuple[str, str, str], DatasetMeta] = {}  # 全局数据集注册表：以 (ms_id, hf_id, path) 为键


def get_dataset_list():
    """获取可用数据集的标识列表。

    返回的内容会根据是否使用 HuggingFace Hub 而有所不同：
    - 若 `use_hf_hub()` 为 True，则收集键中的第 2 项（HF 数据集 ID）。
    - 否则收集键中的第 1 项（ModelScope 数据集 ID）。

    Returns:
        List[str]: 可用于展示/选择的数据集标识列表（过滤掉空字符串/None）。

    示例：
    >>> # 已完成若干数据集注册后
    >>> names = get_dataset_list()  # 依据当前环境选择返回 HF 或 MS 的标识
    >>> isinstance(names, list)
    True
    """
    datasets = []  # 存放返回的标识字符串列表
    for key in DATASET_MAPPING:  # 遍历已注册的数据集键
        if use_hf_hub():  # 若当前环境使用 HF Hub
            if key[1]:  # 若 HF ID 存在且非空
                datasets.append(key[1])  # 追加 HF ID 到结果列表
        else:  # 否则（优先使用 ModelScope）
            if key[0]:  # 若 MS ID 存在且非空
                datasets.append(key[0])  # 追加 MS ID 到结果列表
    return datasets  # 返回最终标识列表


def register_dataset(dataset_meta: DatasetMeta, *, exist_ok: bool = False) -> None:
    """注册单个数据集到全局映射表。

    以 `dataset_meta.dataset_name`（若提供）为唯一键；否则以三元组
    `(ms_dataset_id, hf_dataset_id, dataset_path)` 作为键。默认不允许覆盖已存在项。

    Args:
        dataset_meta (DatasetMeta): 待注册的数据集元信息对象。
        exist_ok (bool, optional): 若为 True，允许覆盖已存在键；否则抛出异常。默认 False。

    Raises:
        ValueError: 当 `exist_ok=False` 且键已存在时抛出。

    示例：
    >>> dm = DatasetMeta(hf_dataset_id='tatsu-lab/alpaca')
    >>> register_dataset(dm)  # 成功注册
    """
    if dataset_meta.dataset_name:  # 若提供了自定义名称
        dataset_name = dataset_meta.dataset_name  # 使用自定义名称作为注册键
    else:  # 否则使用三元组键
        dataset_name = (
            dataset_meta.ms_dataset_id,
            dataset_meta.hf_dataset_id,
            dataset_meta.dataset_path,
        )  # 三元组用于唯一定位同一数据源
    if not exist_ok and dataset_name in DATASET_MAPPING:  # 若不允许覆盖且键已存在
        raise ValueError(
            f'The `{dataset_name}` has already been registered in the DATASET_MAPPING.'
        )  # 抛出重复注册错误，提示键已存在

    DATASET_MAPPING[dataset_name] = dataset_meta  # 写入全局映射表，完成注册/覆盖


def _preprocess_d_info(d_info: Dict[str, Any], *, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """预处理从 JSON/字典中读取的数据集信息。

    该函数将用户简写配置转化为 `DatasetMeta` 可直接解包的参数：
    - 处理 `columns` 与 `messages` 字段，自动构造合适的 `preprocess_func`。
    - 规范化 `dataset_path` 为绝对路径（可结合 `base_dir` 处理相对路径）。
    - 将 `subsets` 中的字典元素递归转换为 `SubsetDataset` 对象。

    Args:
        d_info (Dict[str, Any]): 从配置读取的原始数据集信息字典。
        base_dir (Optional[str], optional): 若提供，则相对路径将基于该目录进行拼接。

    Returns:
        Dict[str, Any]: 处理完成、可直接用于实例化 `DatasetMeta` 的字典。

    示例：
    >>> raw = {  # 简写配置示例 
    ...     'dataset_path': './data/demo.jsonl',
    ...     'columns': {'instruction': 'query', 'output': 'response'},
    ...     'subsets': ['default', {'subset': 'valid', 'split': ['validation']}],
    ... }
    >>> info = _preprocess_d_info(raw, base_dir='/root/project')
    >>> 'preprocess_func' in info and 'dataset_path' in info
    True
    """
    d_info = deepcopy(d_info)  # 深拷贝入参，避免直接修改调用方传入的字典

    columns = None  # 列映射的占位变量，默认无列映射
    if 'columns' in d_info:  # 若提供了 columns 字段（列名映射）
        columns = d_info.pop('columns')  # 取出列映射，并从字典中移除以免与后续解包冲突

    if 'messages' in d_info:  # 若提供了消息式数据的列定义（多轮对话/消息）
        d_info['preprocess_func'] = MessagesPreprocessor(  # 构造消息预处理器
            **d_info.pop('messages'), columns=columns  # 将 messages 的配置解包并传入列映射
        )
    else:  # 否则回退为通用的 AutoPreprocessor
        d_info['preprocess_func'] = AutoPreprocessor(columns=columns)  # 传入可能存在的列映射

    if 'dataset_path' in d_info:  # 若提供了数据路径（本地文件或目录）
        dataset_path = d_info.pop('dataset_path')  # 取出原始路径字符串
        if base_dir is not None and not os.path.isabs(dataset_path):  # 若需要基于基目录解析相对路径
            dataset_path = os.path.join(base_dir, dataset_path)  # 与基目录拼接得到完整路径
        dataset_path = os.path.abspath(os.path.expanduser(dataset_path))  # 绝对化并展开 ~ 用户目录

        d_info['dataset_path'] = dataset_path  # 回写规范化后的绝对路径

    if 'subsets' in d_info:  # 若提供了子集列表
        subsets = d_info.pop('subsets')  # 取出子集列表
        for i, subset in enumerate(subsets):  # 遍历每个子集项
            if isinstance(subset, dict):  # 若该子集以字典形式提供（需要进一步处理）
                subsets[i] = SubsetDataset(**_preprocess_d_info(subset))  # 递归预处理并实例化为 SubsetDataset
        d_info['subsets'] = subsets  # 回写处理完成的子集列表
    return d_info  # 返回可用于 DatasetMeta(**d_info) 的规范化字典


def _register_d_info(d_info: Dict[str, Any], *, base_dir: Optional[str] = None) -> DatasetMeta:
    """将单条数据集信息注册到全局映射表。

    先调用 `_preprocess_d_info` 对简写/路径等进行规范化处理，再实例化 `DatasetMeta` 并
    调用 `register_dataset` 完成注册。

    Args:
        d_info (Dict[str, Any]): 单条原始数据集信息。
        base_dir (Optional[str], optional): 相对路径的基目录。默认 None。

    Returns:
        DatasetMeta: 完成注册的元信息对象。

    示例：
    >>> dm = _register_d_info({'hf_dataset_id': 'tatsu-lab/alpaca'})                 # doctest: +SKIP
    >>> isinstance(dm, DatasetMeta)                                                  # doctest: +SKIP
    True
    """
    d_info = _preprocess_d_info(d_info, base_dir=base_dir)  # 先对字典做预处理，得到规范化配置
    dataset_meta = DatasetMeta(**d_info)  # 基于处理后的字典创建 DatasetMeta 实例
    register_dataset(dataset_meta)  # 写入全局映射表
    return dataset_meta  # 返回实例，便于上游继续使用


def register_dataset_info(dataset_info: Union[str, List[str], None] = None) -> List[DatasetMeta]:
    """从配置文件/JSON 字符串批量注册数据集。

    该函数支持三种调用方式：
    1) `dataset_info` 为 None：将使用模块内置的 `data/dataset_info.json` 文件。
    2) `dataset_info` 为字符串：若对应本地文件存在则读取文件，否则当作 JSON 字符串解析。
    3) `dataset_info` 为列表：直接视为已解析的配置对象集合。

    Args:
        dataset_info (Union[str, List[str], None], optional): 配置文件路径、JSON 字符串或已解析列表。

    Returns:
        List[DatasetMeta]: 所有成功注册的数据集元信息对象列表。

    示例：
    >>> metas = register_dataset_info()  # 使用默认的 dataset_info.json                 # doctest: +SKIP
    >>> isinstance(metas, list)                                                        # doctest: +SKIP
    True
    """
    # dataset_info_path: path, json or None
    if dataset_info is None:  # 未提供则使用包内默认配置文件
        dataset_info = os.path.join(os.path.dirname(__file__), 'data', 'dataset_info.json')  # 组装默认文件路径
    assert isinstance(dataset_info, (str, list))  # 仅接受 str 或 list 两种类型
    base_dir = None  # 用于相对路径解析的基目录，初始为空
    log_msg = None  # 用于日志展示的路径/关键字段
    if isinstance(dataset_info, str):  # 若传入为字符串
        dataset_path = os.path.abspath(os.path.expanduser(dataset_info))  # 绝对化路径并展开 ~
        if os.path.isfile(dataset_path):  # 若该字符串对应本地文件
            log_msg = dataset_path  # 记录日志展示用的文件路径
            base_dir = os.path.dirname(dataset_path)  # 用文件所在目录作为相对路径基准
            with open(dataset_path, 'r', encoding='utf-8') as f:  # 打开并读取 JSON 文件
                dataset_info = json.load(f)  # 解析为 Python 对象（通常为 list 或 dict）
        else:  # 否则将其视为 JSON 字符串
            dataset_info = json.loads(dataset_info)  # json 字符串解析为 Python 对象
    if len(dataset_info) == 0:  # 若解析结果为空列表
        return []  # 直接返回空结果
    res = []  # 存放注册完成后的 DatasetMeta 列表
    for d_info in dataset_info:  # 遍历配置中的每条数据集定义
        res.append(_register_d_info(d_info, base_dir=base_dir))  # 预处理并注册，收集返回的 DatasetMeta

    if log_msg is None:  # 若未记录文件路径（即传入的是对象或 JSON 字符串）
        log_msg = dataset_info if len(dataset_info) < 5 else list(dataset_info.keys())  # 简短展示一部分关键信息
    logger.info(f'Successfully registered `{log_msg}`.')  # 记录成功注册日志，便于排查与追踪
    return res  # 返回已注册的元信息对象列表
