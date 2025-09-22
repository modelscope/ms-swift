"""
模块功能
-------
本模块定义了多种数据预处理基类与工具，负责将原始数据集样本列（文本、对话、多模态字段等）
规范化为训练所需的标准结构（如 `messages/query/response/images/videos/audios/tools/objects`）。

核心能力
-------
- `RowPreprocessor`: 抽象基类，提供批处理映射、字段重命名、消息校验、MM 数据转换等能力；
- `ResponsePreprocessor/AlpacaPreprocessor`: 统一将 `query/response/history/system` 转换为 `messages`；
- `MessagesPreprocessor`: 兼容多种消息字段/角色名，修复历史并对齐到标准 `messages`；
- `ClsPreprocessor`: 在响应式预处理基础上补充 `label`；
- `AutoPreprocessor`: 根据数据集特征自动选择合适的预处理器；
- 其余工具函数：修补函数、ArrowWriter 打补丁等，保障流式与分布式环境下映射稳定。

典型用法
-------
>>> proc = ResponsePreprocessor(columns={'input': 'query', 'answer': 'response'})
>>> new_ds = proc(dataset, num_proc=4, load_from_cache_file=True)

说明：代码已按行附加中文注释或在函数/类文档中详述各步骤的作用，便于维护与排障。
"""

# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明
import ast  # 安全解析字符串字面量
import os  # OS 路径与环境变量操作
from collections import Counter  # 计数器，用于统计列重命名冲突
from contextlib import contextmanager  # 上下文管理器装饰器
from typing import Any, Callable, Dict, List, Optional, Union  # 类型注解工具

import numpy as np  # 随机与数组工具
from datasets import Dataset as HfDataset  # HF 常规数据集
from datasets import Image  # HF 图像列类型
from datasets import IterableDataset as HfIterableDataset  # HF 可迭代数据集
from datasets import Sequence, Value  # HF 类型系统：序列与值类型

from swift.llm import history_to_messages  # 将 (query, response) 历史转换为 messages 的工具
from swift.utils import get_logger, is_dist, is_master, safe_ddp_context  # 日志与分布式工具

DATASET_TYPE = Union[HfDataset, HfIterableDataset]  # 统一表示两种 HF 数据集类型

logger = get_logger()  # 模块级日志记录器


class RowPreprocessor:
    """
    行级预处理器抽象基类：提供字段对齐、消息校验、多模态数据规整、批处理映射等通用能力。

    使用说明
    -------
    - 子类需实现 `preprocess(self, row)` 将单条样本转为标准结构；
    - 调用 `__call__` 可对整个数据集执行 map 操作并返回新数据集。
    """
    standard_keys = [
        'messages', 'rejected_response', 'rejected_images', 'label', 'images', 'videos', 'audios', 'tools', 'objects',
        'channel', 'margin'  # 预处理后可能出现的标准字段集合
    ]

    def __init__(self,
                 *,
                 columns: Optional[Dict[str, str]] = None,
                 dataset_sample: Optional[int] = None,
                 random_state: Optional[Union[np.random.RandomState, int]] = 42,
                 traceback_limit: int = 10) -> None:
        self.columns = columns or {}  # 列映射：源列名 -> 目标标准列名
        self.origin_columns = self.columns.copy()  # 最高优先级的原始映射，冲突时优先
        images_keys = ['images', 'image']  # 可能的图像列别名
        audios_keys = ['audios', 'audio']  # 可能的音频列别名
        videos_keys = ['videos', 'video']  # 可能的视频列别名
        for mm_type in ['images', 'audios', 'videos']:  # 统一补充多模态列别名映射
            keys = locals()[f'{mm_type}_keys']  # 取对应别名列表
            for key in keys:  # 将别名映射到标准列名
                self.columns[key] = mm_type

        self.traceback_limit = traceback_limit  # 记录可打印回溯的最大次数
        self._traceback_counter = 0  # 已打印回溯计数
        self.dataset_sample = dataset_sample  # 可选采样条数，用于快速调试
        if not isinstance(random_state, np.random.RandomState):  # 归一化为 RandomState
            random_state = np.random.RandomState(random_state)  # 以 seed 初始化
        self.random_state = random_state  # 保存随机状态

    @staticmethod
    def _check_messages(row: Dict[str, Any]) -> None:
        """校验 `messages` 结构与角色字段的合法性，移除非标准键。"""
        if 'messages' not in row:  # 无消息字段则跳过
            return
        messages = row['messages']  # 取消息列表
        assert len(messages) > 0, f'messages: {messages}'  # 至少一条
        # fix swift/SlimOrca  # 兼容性：只保留 role/content 两键
        for message in messages:
            keys = set(message.keys()) - {'role', 'content'}  # 找到多余键
            for key in keys:  # 逐个移除
                message.pop(key)

        for message in messages:  # 遍历检查每条消息
            role, content = message['role'], message['content']  # 取角色与内容
            # The terms 'tool' and 'tool_response' have the same meaning, ensuring compatibility.  # 兼容工具消息
            assert role in {'system', 'user', 'tool_call', 'tool_response', 'tool', 'assistant'}, f'message: {message}'  # 角色合法
            assert content is not None, f'message: {message}'  # 内容不可为 None

    @staticmethod
    def _cast_mm_data(row: Dict[str, Any]) -> None:
        """将多模态字段统一为标准结构：images/rejected_images -> [{'bytes','path'}]，videos/audios -> list。"""
        for key in ['images', 'rejected_images']:  # 处理图像类字段
            images = row.get(key, None)  # 读取字段
            if images is None:  # 无则跳过
                continue

            if isinstance(images, str) or (isinstance(images, list) and images and isinstance(images[0], str)):  # 字符串或字符串列表
                if isinstance(images, str):  # 单字符串转列表
                    images = [images]
                for i, image in enumerate(images):  # 包装为 dict 结构
                    images[i] = {'bytes': None, 'path': image}
                row[key] = images  # 回写
            elif isinstance(images, dict):  # 单个 dict 转列表
                row[key] = [images]

        for key in ['videos', 'audios']:  # 处理视频/音频字段
            mm_data = row.get(key)  # 读取字段
            if mm_data is None:  # 无则跳过
                continue
            elif isinstance(mm_data, str):  # 单字符串 -> 列表
                row[key] = [mm_data]

    @staticmethod
    def _check_rejected_response(row: Dict[str, Any]) -> None:
        """
        兼容 DPO/ORPO：
        - 若提供 `rejected_messages`，合并与 `messages` 对齐，并提取 `rejected_response`；
        - 若已有 `rejected_response`，需与最后一条 assistant 回复不同，否则报错。
        """
        if 'rejected_messages' in row:  # 同时提供正/负消息
            chosen_messages = row['messages']  # 正样本消息
            rejected_messages = row['rejected_messages']  # 负样本消息
            messages = []  # 合并后的消息
            rejected_response = None  # 存放拒绝回复
            for chosen_user, chosen_assistant, rejected_user, rejected_assistant in zip(
                    chosen_messages[::2], chosen_messages[1::2], rejected_messages[::2], rejected_messages[1::2]):  # 成对遍历
                assert chosen_user == rejected_user  # 用户消息应一致
                messages.append(chosen_user)  # 添加用户
                messages.append(chosen_assistant)  # 添加正样本助手
                if chosen_assistant != rejected_assistant:  # 如正负助手不同
                    rejected_response = rejected_assistant['content']  # 记录负样本内容
            row['messages'] = messages  # 回写合并消息
            row['rejected_response'] = rejected_response  # 回写拒绝回复

        if 'rejected_response' in row:  # 明确给出拒绝回复
            messages = row['messages']  # 当前消息
            rejected_response = row['rejected_response']  # 负样本回复
            if rejected_response is None or rejected_response == messages[-1]['content']:  # 不应为空或等于最后一条正样本
                raise ValueError(f'rejected_response: {rejected_response}')  # 抛错提示

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """抽象方法：子类需实现对单条样本的规范化。"""
        raise NotImplementedError  # 由子类实现

    def prepare_dataset(self, dataset: DATASET_TYPE) -> DATASET_TYPE:
        """可在映射前对数据集进行准备（下载媒体/修复列等），默认直接返回。"""
        return dataset  # 默认不变

    @staticmethod
    def batched_to_rows(batched_row: Dict[str, Any]):
        """将 batched 行转换为行列表，方便逐行处理。"""
        keys = list(batched_row.keys())  # 取所有键
        batch_size = len(batched_row[keys[0]])  # 批大小按第一个键长度决定
        return [{key: batched_row[key][i] for key in keys} for i in range(batch_size)]  # 逐样本重组

    @staticmethod
    def rows_to_batched(rows: List[Dict[str, Any]]):
        """将行列表重新拼成 batched 结构，填补缺失列使长度一致。"""
        batched = {}  # 结果容器
        for i, row in enumerate(rows):  # 枚举行
            for k, v in row.items():  # 遍历键值
                if k not in batched:  # 新键则补齐之前位置
                    batched[k] = [None] * i
                batched[k].append(v)  # 追加当前值
            # Make all the lengths of v the same.  # 对缺失列用 None 补齐
            for k in set(batched.keys()) - set(row.keys()):
                batched[k].append(None)
        return batched  # 返回 batched 字典

    @staticmethod
    def _remove_prefix_keys(row, prefix: str):
        """移除字典键的前缀（兼容流式/GRPO 缓存字段命名）。"""
        for k in list(row.keys()):  # 遍历原始键列表
            if k.startswith(prefix):  # 命中前缀
                new_k = k[len(prefix):]  # 去掉前缀后的新键
                new_v = row.pop(k)  # 弹出旧键的值
                if new_k not in row:  # 避免覆盖
                    row[new_k] = new_v  # 写回新键

    @staticmethod
    def _check_objects(row):
        """规范 objects 字段顺序并检查 bbox 合法性（坐标有序/长度 2 或 4）。"""
        objects = row.get('objects')  # 读取对象字段
        if objects is None:  # 无则跳过
            return
        new_objects = {}  # 规范后的对象
        # Ensure the order  # 保持键顺序一致
        for k in ['ref', 'bbox', 'bbox_type', 'image_id']:
            if k in objects.keys():  # 存在则保留
                new_objects[k] = objects[k]
        row['objects'] = new_objects  # 回写
        bbox = new_objects['bbox']  # 取出 bbox 列表

        # check bbox  # 合法性检查
        for box in bbox:  # 遍历每个框
            assert len(box) in {2, 4}, f'len(box): {len(box)}'  # 支持点或矩形
            if len(box) == 2:  # 点框无需调整
                continue
            if box[0] > box[2]:  # 确保 x1<=x2
                box[0], box[2] = box[2], box[0]
            if box[1] > box[3]:  # 确保 y1<=y2
                box[1], box[3] = box[3], box[1]

    def batched_preprocess(self, batched_row: Dict[str, Any], *, strict: bool,
                           ignore_max_length_error: bool) -> Dict[str, Any]:
        """
        对 batched 行执行安全的逐行预处理：
        - 支持子类 `preprocess` 返回单条或多条样本；
        - 校验/修补 objects/messages/rejected_response/mm 数据；
        - 在非 strict 模式下，对异常样本进行过滤并限量打印回溯。

        参数
        ----
        - batched_row: batched 格式的输入字典
        - strict: True 时遇到错误直接抛出；False 时过滤错误样本
        - ignore_max_length_error: True 时忽略模板长度相关错误

        返回
        ----
        - Dict[str, Any]: batched 结构的标准化结果
        """
        from ...template import MaxLengthError  # 延迟导入，避免循环依赖
        batched_row = dict(batched_row)  # 复制，避免原地修改
        assert len(batched_row) > 0  # 非空断言
        self._remove_prefix_keys(batched_row, '__@')  # compat streaming  # 去除流式前缀
        rows = self.batched_to_rows(batched_row)  # 拆为逐行

        new_rows = []  # 收集合法行
        for row in rows:  # 遍历每行
            try:
                row = self.preprocess(row)  # 由子类实现
                # support [row1, row2, ...]  # 允许返回列表或单条/None
                if row is None:
                    row = []  # 过滤
                if isinstance(row, dict):  # 单条 -> 列表
                    row = [row]
                for r in row:  # 对每个返回样本做校验与规整
                    self._check_objects(r)  # 规范 objects
                    self._check_messages(r)  # 校验 messages
                    self._check_rejected_response(r)  # 处理拒绝回复
                    self._cast_mm_data(r)  # 统一 MM 数据格式
            except Exception as e:  # 捕获预处理异常
                if strict:  # 严格模式抛出
                    logger.warning('To avoid errors, you can pass `strict=False`.')  # 提示可切换
                    raise
                if isinstance(e, MaxLengthError) and ignore_max_length_error:  # 可忽略的长度错误
                    pass
                elif self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:  # 限量打印回溯
                    import traceback  # 延迟导入
                    logger.info(traceback.format_exc())  # 打印堆栈
                    logger.warning('👆👆👆There are errors in the dataset, the data will be deleted')  # 告警过滤
                    self._traceback_counter += 1  # 计数
                row = []  # 该样本丢弃
            new_rows += row  # 累加结果
        res = self.rows_to_batched(new_rows)  # 重新拼为 batched
        self._remove_prefix_keys(res, '__#')  # compat GRPO  # 去除 solution 缓存前缀
        if len(res) == 0:  # 若全部被过滤
            res['messages'] = []  # 至少保留空 messages 列

        return res  # 返回结果

    @staticmethod
    def get_features_dataset(dataset: DATASET_TYPE) -> DATASET_TYPE:
        """确保 dataset 带有 features（对 IterableDataset 需先解析）。"""
        if dataset.features is None:  # 无 features 需解析
            assert isinstance(dataset, HfIterableDataset)  # 仅可迭代数据集有此情形
            dataset = dataset._resolve_features()  # 解析特征
        return dataset  # 返回带 features 的数据集

    @staticmethod
    def safe_rename_columns(dataset, columns):
        """安全地重命名列：仅对存在的列进行重命名，且避免目标名冲突。"""
        dataset = RowPreprocessor.get_features_dataset(dataset)  # 确保有 features
        columns_keys = {k.lower(): k for k in dataset.features.keys()}  # lower -> 原始大小写键
        safe_columns = {columns_keys[k.lower()]: v for k, v in columns.items() if k.lower() in columns_keys}  # 过滤存在的源列

        counter = Counter(safe_columns.values())  # 统计目标名出现次数
        for k, new_k in list(safe_columns.items()):  # 移除会产生目标名冲突的映射
            if counter[new_k] > 1:
                # For example, if "response" and "answer" match, then no processing is done.  # 冲突则跳过
                safe_columns.pop(k)
                continue

        # e.g. Keep {'query': 'query'} to ensure that the query has the highest priority.  # 去掉同名映射（无意义）
        safe_columns = {k: v for k, v in safe_columns.items() if k != v}
        if safe_columns:  # 存在需重命名的列
            dataset = dataset.rename_columns(safe_columns)  # 执行重命名

        return dataset  # 返回数据集

    def _rename_columns(self, dataset: DATASET_TYPE) -> DATASET_TYPE:
        """两阶段重命名：先按 origin_columns，再按 columns；流式数据集追加前缀以兼容写入。"""
        dataset = self.safe_rename_columns(dataset, self.origin_columns)  # 先应用原始高优先级映射
        dataset = self.safe_rename_columns(dataset, self.columns)  # 再应用补充映射
        if isinstance(dataset, HfIterableDataset):  # 流式数据集写入兼容
            # fix: https://github.com/huggingface/datasets/issues/6408  # 加前缀绕过写入限制
            columns = {k: f'__@{k}' for k in RowPreprocessor.standard_keys if k in dataset.features}
            if columns:
                dataset = dataset.rename_columns(columns)  # 重命名加前缀
        return dataset  # 返回数据集

    @staticmethod
    def remove_useless_columns(dataset: DATASET_TYPE) -> DATASET_TYPE:
        """仅保留标准键列，去除无用列以节省存储与传输。"""
        dataset = RowPreprocessor.get_features_dataset(dataset)  # 确保 features 可用
        features = dataset.features  # 取特征描述
        k_list = [k for k in RowPreprocessor.standard_keys if k in features]  # 仅标准列
        if len(k_list) != len(features):  # 存在冗余列
            dataset = dataset.select_columns(k_list)  # 选择子集
        return dataset  # 返回裁剪后数据集

    @staticmethod
    @contextmanager
    def _patch_arrow_writer():
        """为 ArrowWriter 打补丁，确保写入时标准列 features 正确声明（尤其 messages/images/objects）。"""
        # fix AI-ModelScope/ms_agent_for_agentfabric:all  # 针对部分数据集的兼容修复
        from datasets.arrow_writer import ArrowWriter  # 导入 ArrowWriter 类

        def _new_init(self, schema=None, features=None, *args, **kwargs):  # 替换构造函数

            if features is not None:  # 若 features 存在则补充标准列 schema
                features['messages'] = [{'role': Value(dtype='string'), 'content': Value(dtype='string')}]
                features['images'] = [{'bytes': Value(dtype='binary'), 'path': Value(dtype='string')}]
                features['objects'] = {
                    'ref': Sequence(feature=Value(dtype='string'), length=-1),
                    'bbox': Sequence(feature=Sequence(feature=Value(dtype='float64'), length=-1), length=-1),
                    'bbox_type': Value(dtype='string'),
                    'image_id': Sequence(feature=Value(dtype='int64'), length=-1),
                }
            ArrowWriter.__origin_init__(self, schema, features, *args, **kwargs)  # 调用原始构造

        ArrowWriter.__origin_init__ = ArrowWriter.__init__  # 备份原始 __init__
        ArrowWriter.__init__ = _new_init  # 注入新构造
        try:
            yield  # 进入补丁作用范围
        finally:
            ArrowWriter.__init__ = ArrowWriter.__origin_init__  # 恢复原构造
            del ArrowWriter.__origin_init__  # 清理备份引用

    def _cast_pil_image(self, dataset):
        """将可解码的 Image 列切换为非解码模式，避免 map 时隐式解码带来的开销。"""
        features = dataset.features  # 当前特征定义
        for col in ['images', 'rejected_images']:  # 两个图像相关列
            if (col in features and isinstance(features[col], Image) and getattr(features[col], 'decode', False)):
                dataset = dataset.cast_column(col, Image(decode=False))  # 关闭 decode 标志
        return dataset  # 返回数据集

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        strict: bool = False,
        batch_size: Optional[int] = None,
    ) -> DATASET_TYPE:
        """
        对 HF 数据集执行标准化预处理（支持并行/缓存/流式）：
        - 可选采样；
        - 列重命名与准备；
        - 批处理映射并捕获异常样本；
        - 兼容 `solution` 字段保留（GRPO）。

        参数
        ----
        - dataset: HF 数据集或可迭代数据集
        - num_proc: 并行进程数（仅 HfDataset 生效）
        - load_from_cache_file: 是否使用缓存文件
        - strict: 严格模式（异常即抛出）
        - batch_size: 映射批大小，默认 1000（HfDataset）或 16（迭代式）

        返回
        ----
        - 预处理后的数据集（与输入类型一致）
        """
        from ..utils import sample_dataset  # 数据子采样工具
        if batch_size is None:  # 设置默认批大小
            batch_size = 1000 if isinstance(dataset, HfDataset) else 16
        if self.dataset_sample is not None:  # 若要求采样
            dataset = sample_dataset(dataset, self.dataset_sample, True, self.random_state)  # 采样后返回

        map_kwargs = {'batched': True, 'batch_size': batch_size}  # map 公共参数
        if isinstance(dataset, HfDataset):  # 常规数据集支持多进程与缓存
            if not load_from_cache_file and is_dist() and not is_master():  # 分布式下非主进程强制使用缓存
                load_from_cache_file = True
            map_kwargs.update({
                'num_proc': num_proc,
                'load_from_cache_file': load_from_cache_file,
            })
        # compat GRPO: The solution field will be retained.  # 兼容保留 solution 字段
        dataset = RowPreprocessor.get_features_dataset(dataset)  # 确保 features 可用
        if 'solution' in dataset.features:  # 若包含 solution 列
            with safe_ddp_context(None, True):  # DDP 安全上下文
                dataset = dataset.map(lambda x: {'__#solution': x['solution']}, **map_kwargs)  # 临时缓存 solution
        dataset = self._rename_columns(dataset)  # 应用列重命名逻辑
        dataset = self.prepare_dataset(dataset)  # 子类准备（下载/修复）
        dataset = self._cast_pil_image(dataset)  # 调整图像 decode 行为

        ignore_max_length_error = True if isinstance(dataset, HfDataset) and num_proc > 1 else False  # 多进程忽略长度错
        with self._patch_arrow_writer(), safe_ddp_context(None, True):  # 写入补丁与 DDP 安全环境
            try:
                dataset_mapped = dataset.map(
                    self.batched_preprocess,  # 批处理预处理函数
                    fn_kwargs={
                        'strict': strict,
                        'ignore_max_length_error': ignore_max_length_error
                    },
                    remove_columns=list(dataset.features.keys()),  # 移除原列，仅保留新列
                    **map_kwargs)
            except NotImplementedError:  # 子类未实现 preprocess 时跳过
                pass
        if isinstance(dataset_mapped, HfDataset) and len(dataset) != len(dataset_mapped):  # 过滤统计
            logger.info(
                f'Dataset filtered, origin length: {len(dataset)}, filtered dataset length: {len(dataset_mapped)}')

        return dataset_mapped  # 返回映射后的数据集


class ResponsePreprocessor(RowPreprocessor):
    """
    响应式预处理器：兼容早期 ms-swift 数据格式，将 `system/query/response/history`
    统一转换为标准 `messages` 序列。
    """

    def __init__(self, *, columns: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """扩展列映射：常见的 system/query/response 别名归一化到标准键。"""
        super().__init__(columns=columns, **kwargs)  # 初始化基类
        system_keys = ['system', 'system_prompt']  # system 别名
        query_keys = ['query', 'prompt', 'input', 'instruction', 'question', 'problem']  # query 别名
        response_keys = ['response', 'answer', 'output', 'targets', 'target', 'answer_key', 'answers', 'solution'
                         ] + ['text', 'completion', 'content']  # response 别名
        for key in system_keys:  # 归一化 system
            self.columns[key] = 'system'
        for key in query_keys:  # 归一化 query
            self.columns[key] = 'query'
        for key in response_keys:  # 归一化 response
            self.columns[key] = 'response'

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将 `query/response/history/system` 组装为标准 `messages`：
        - response 支持 list/tuple，并可配置随机选取；
        - history 支持字符串字面量表示。
        """
        response = row.pop('response', None)  # 取出响应并从行中移除
        if response is not None:
            if isinstance(response, (list, tuple)):  # 响应可能是多个候选
                from transformers.utils import strtobool  # 字符串转布尔
                # sometimes response is a list, pick one randomly  # 可随机选择回应
                if strtobool(os.environ.get('RANDOM_DATASET_RESPONSE', 'True')):  # 受环境变量控制
                    response = self.random_state.choice(response)  # 随机挑选
                else:
                    response = response[0]  # 取第一个
        history = row.pop('history', None) or []  # 取历史对话，无则空列表
        query = row.pop('query', None)  # 取 query 并移除
        system = row.pop('system', None)  # 取 system 并移除
        if isinstance(history, str):  # e.g. "[['query1', 'response1']]"  # 字符串形式历史
            history = ast.literal_eval(history)  # 安全解析
        history.append([query, response])  # 追加当前轮

        row.update({'messages': history_to_messages(history, system)})  # 转为标准 messages
        return row  # 返回


class AlpacaPreprocessor(ResponsePreprocessor):
    """
    兼容 Alpaca 风格数据：`instruction/input/output` -> `query/response` 并生成 messages。
    """

    @classmethod
    def concat_inst_input(cls, instruction, input_):
        """拼接 `instruction` 与 `input` 生成 `query`，若一方为空则取另一方。"""
        if instruction and input_:  # 两者皆有
            query = f'{instruction}\n{input_}'  # 以换行拼接
        else:
            query = instruction or input_  # 取存在的一方
        assert isinstance(query, str), f'query: {query}'  # 断言为字符串
        return query  # 返回查询

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从 `instruction/input/output` 构造 `query/response`，并调用父类生成 messages。"""
        instruction = row.pop('instruction', None)  # 取并移除 instruction
        input_ = row.pop('input', None)  # 取并移除 input
        output = row.pop('output', None)  # 取并移除 output
        if output is not None:  # 存在输出则作为响应
            row['response'] = output
        row['query'] = self.concat_inst_input(instruction, input_)  # 生成 query
        return super().preprocess(row)  # 交由父类生成 messages


def default_repair_messages(s: Union[str, Any]) -> Any:
    """默认修补函数：若输入为字符串，则用 `ast.literal_eval` 解析为 Python 对象。"""
    if isinstance(s, str):  # 字符串形式
        return ast.literal_eval(s)  # 安全解析
    return s  # 非字符串直接返回


class MessagesPreprocessor(RowPreprocessor):
    """
    消息式预处理器：兼容各种消息键名与角色名，修复并对齐为标准 `messages` 序列。

    关键参数
    -------
    - role_key/content_key: 消息中角色与内容键名（默认自动匹配）；
    - user_role/assistant_role/system_role: 角色命名别名（'human'/'gpt' 等）；
    - columns: 输入列到标准列的映射；
    - repair_messages: 修补函数，支持字符串表达的消息历史；
    - inner_key: 当 messages 是嵌套结构时取其子键。
    """

    def __init__(
            self,
            *,
            # If set to None, automatic matching will be performed.
            role_key: Optional[str] = None,  # 'role', 'from'
            content_key: Optional[str] = None,  # 'content', 'value'
            user_role: Optional[str] = None,  # 'user', 'human'
            assistant_role: Optional[str] = None,  # 'assistant', 'gpt', 'bot'
            system_role: str = 'system',
            # 'conversation', 'conversations' -> 'messages'
            columns: Optional[Dict[str, str]] = None,
            repair_messages: Callable[[Union[str, List[Dict[str, str]]]],
                                      Optional[List[Dict[str, str]]]] = default_repair_messages,
            inner_key: Optional[str] = None,
            **kwargs):
        super().__init__(columns=columns, **kwargs)  # 初始化父类
        self.role_keys = ['role', 'from'] if role_key is None else [role_key]  # 角色键候选
        self.content_keys = ['content', 'value'] if content_key is None else [content_key]  # 内容键候选
        self.user_roles = ['user', 'human'] if user_role is None else [user_role]  # 用户角色别名
        self.assistant_roles = ['assistant', 'gpt', 'bot'] if assistant_role is None else [assistant_role]  # 助手别名
        self.tool_call_roles = ['function_call']  # 工具调用角色别名
        self.tool_response_roles = ['function_response', 'observation', 'observations']  # 工具响应别名

        self.system_role = system_role  # 系统角色名
        self.repair_messages = repair_messages  # 消息修补函数
        self.inner_key = inner_key  # 嵌套消息键

        message_keys = ['messages', 'conversation', 'conversations']  # 常见消息键
        for key in message_keys:  # 归一化为 messages
            self.columns[key] = 'messages'
        # sharegptq  # 系统提示键归一化
        system_keys = ['system', 'system_prompt']
        if system_role not in system_keys:  # 补充自定义系统键
            system_keys.append(system_role)
        for key in system_keys:  # 归一化为 system
            self.columns[key] = 'system'

    @staticmethod
    def _is_sharegpt_format(message: Dict[str, str]) -> bool:
        """判断消息是否为 ShareGPT 键风格（无 role/content）。"""
        if 'role' in message or 'content' in message:  # 含标准键则不是 ShareGPT 风格
            return False
        return True  # 否则是

    def sharegpt_to_messages(self, messages: List[Dict[str, str]], system: Optional[str]) -> List[Dict[str, str]]:
        """将 ShareGPT 风格消息转为标准 `messages` 列表。"""
        self._to_std_key(messages, 'user', self.user_roles)  # 统一用户键
        self._to_std_key(messages, 'assistant', self.assistant_roles)  # 统一助手键
        new_messages = []  # 输出列表
        if system is not None:  # 有系统提示则置于首位
            new_messages.append({'role': 'system', 'content': system})
        for message in messages:  # 交替加入 user/assistant
            user_message = {'role': 'user', 'content': message['user']}
            assistant_message = {'role': 'assistant', 'content': message['assistant']}
            new_messages.append(user_message)
            new_messages.append(assistant_message)
        return new_messages  # 返回

    def to_std_messages(self, messages: List[Dict[str, str]], system: Optional[str]) -> None:
        """就地将混合角色名对齐为标准角色名，并在必要时插入 system 消息。"""
        if messages[0]['role'] == self.system_role:  # 首条系统消息角色名对齐
            messages[0]['role'] = 'system'
        elif system is not None:  # 否则若提供 system 文本，则插入首条
            messages.insert(0, {'role': 'system', 'content': system})
        for message in messages:  # 遍历对齐角色
            role = message['role']
            if role in self.user_roles:
                message['role'] = 'user'
            elif role in self.assistant_roles:
                message['role'] = 'assistant'
            elif role.replace('-', '_') in self.tool_call_roles:  # function-call 别名
                message['role'] = 'tool_call'
            elif role.replace('-', '_') in self.tool_response_roles:  # function-response 别名
                message['role'] = 'tool_response'

    @staticmethod
    def _to_std_key(messages: List[Dict[str, str]], std_key: str, optional_keys: List[str]) -> None:
        """将消息的可选键之一映射为标准键 `std_key`（如 user/assistant）。"""
        for message in messages:  # 遍历每条消息
            for key in optional_keys:  # 尝试每个候选键
                if key in message:  # 命中则替换
                    message[std_key] = message.pop(key)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """修复与标准化 `messages`：支持 rejected_messages、嵌套键与字符串历史。"""
        if 'rejected_messages' in row:  # 提前规范负样本消息
            row['rejected_messages'] = MessagesPreprocessor.preprocess(
                self, {'messages': row['rejected_messages']})['messages']
        messages = row['messages']  # 取消息
        if self.inner_key is not None:  # 内嵌键场景
            messages = messages[self.inner_key]
        messages: Optional[List[Dict[str, str]]] = self.repair_messages(messages)  # 修补消息（字符串 -> 列表）
        if not messages or isinstance(messages, str):  # 修补失败则跳过
            return
        self._to_std_key(messages, 'role', self.role_keys)  # 对齐 role 键
        self._to_std_key(messages, 'content', self.content_keys)  # 对齐 content 键
        system = row.pop('system', None)  # 取出 system 文本
        if self._is_sharegpt_format(messages[0]):  # ShareGPT 风格
            messages = self.sharegpt_to_messages(messages, system)
        else:
            self.to_std_messages(messages, system)  # inplace 标准化
        row['messages'] = messages  # 写回消息
        return row  # 返回


class ClsPreprocessor(ResponsePreprocessor):
    """
    分类预处理器：在响应式预处理基础上，将 `label` 转为整型。
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """调用父类生成 messages，随后将 `label` 转为整型。"""
        res = super().preprocess(row)  # 先生成 messages
        res['label'] = int(res['label'])  # label -> int
        return res  # 返回


class AutoPreprocessor:
    """
    预处理器自动选择器：根据数据集特征自动选择 `Messages/Alpaca/Response` 预处理器。

    用法
    ---
    >>> auto = AutoPreprocessor(columns={'instruction': 'instruction'})
    >>> ds2 = auto(ds)
    """

    def __init__(self, *, columns: Optional[Dict[str, str]] = None, **kwargs) -> None:
        self.columns = columns or {}  # 需要提前重命名的列映射
        self.kwargs = kwargs  # 传递给具体预处理器的其他参数

    def _get_preprocessor(self, dataset: DATASET_TYPE) -> RowPreprocessor:
        """根据 features 选择最适合的预处理器类型。"""
        features = dataset.features  # 特征字典
        for key in ['conversation', 'conversations', 'messages']:  # 若有消息类字段
            if key in features:
                return MessagesPreprocessor(**self.kwargs)  # 使用消息预处理
        if 'instruction' in features and 'input' in features:  # Alpaca 风格
            return AlpacaPreprocessor(**self.kwargs)
        return ResponsePreprocessor(**self.kwargs)  # 默认响应式

    def __call__(
        self,
        dataset: DATASET_TYPE,
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        strict: bool = False,
    ) -> DATASET_TYPE:
        """先安全重命名列，再选择并调用具体预处理器完成数据标准化。"""
        dataset = RowPreprocessor.safe_rename_columns(dataset, self.columns)  # 先做列对齐
        preprocessor = self._get_preprocessor(dataset)  # 自动选择具体预处理器
        return preprocessor(dataset, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict)  # 执行
