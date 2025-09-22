"""模块说明：
该模块定义了用于模型评测（evaluation）的参数数据类与初始化逻辑。
- EvalArguments：继承自 DeployArguments，扩展评测数据集、评测后端、生成配置、输出目录等能力。
模块通过集中化的配置解析与合法性校验，确保在评测之前完成 URL 兼容、数据集规范化、路径规范化等准备工作。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt  # 导入 datetime 模块并命名为 dt，用于时间戳生成
import os  # 导入 os，用于路径拼接与目录创建
from dataclasses import dataclass, field  # 导入 dataclass 与 field，用于数据类与默认工厂
from typing import Dict, List, Literal, Optional, Union  # 导入类型注解工具

from swift.utils import get_logger, json_parse_to_dict  # 导入日志器获取与 JSON 字符串解析函数
from .base_args import to_abspath  # 导入路径绝对化工具函数
from .deploy_args import DeployArguments  # 导入部署参数基类，评测参数在其基础上扩展

logger = get_logger()  # 初始化模块级日志器，用于统一打印


@dataclass  # 数据类装饰器，自动生成 __init__/__repr__ 等方法
class EvalArguments(DeployArguments):  # 评测参数类，继承部署参数
    """
    类说明：评测参数数据类，继承自 `DeployArguments`，用于定义模型评测所需的配置项。

    主要职责：
    - 指定评测数据集、评测后端与生成配置。
    - 处理评测 URL 兼容、解析 JSON 形参、规范输出目录。

    属性：
        eval_dataset: 评测数据集名称列表，大小写不敏感。
        eval_limit: 每个数据集的样本上限；None 表示不限。
        dataset_args: 数据集相关的额外参数（dict 或 JSON 字符串）。
        eval_generation_config: 生成推理配置（dict 或 JSON 字符串）。
        eval_output_dir: 评测输出目录。
        eval_backend: 评测后端，支持 'Native'/'OpenCompass'/'VLMEvalKit'。
        local_dataset: 是否从 OpenCompass 等来源下载本地数据集。
        temperature: 生成温度；0 表示更确定性的输出。
        verbose: 是否输出更详细的日志。
        eval_num_proc: 评测时使用的进程数。
        extra_eval_args: 额外评测参数（dict 或 JSON 字符串）。
        eval_url: 若提供，将直接使用该 URL 进行评测而不进行部署。
    """
    eval_dataset: List[str] = field(default_factory=list)  # 评测数据集名称列表，默认空
    eval_limit: Optional[int] = None  # 每个数据集的样本上限，None 表示不限制
    dataset_args: Optional[Union[Dict, str]] = None  # 数据集额外参数（dict/JSON 字符串）
    eval_generation_config: Optional[Union[Dict, str]] = field(default_factory=dict)  # 生成配置（默认空字典）
    eval_output_dir: str = 'eval_output'  # 评测输出目录
    eval_backend: Literal['Native', 'OpenCompass', 'VLMEvalKit'] = 'Native'  # 评测后端选择
    local_dataset: bool = False  # 是否下载额外数据集到本地

    temperature: Optional[float] = 0.  # 生成温度，默认 0（更保守）；None 表示不覆盖
    verbose: bool = False  # 是否输出详细信息
    eval_num_proc: int = 16  # 评测使用的并行进程数
    extra_eval_args: Optional[Union[Dict, str]] = field(default_factory=dict)  # 额外评测参数
    # If eval_url is set, ms-swift will not perform deployment operations and
    # will directly use the URL for evaluation.
    eval_url: Optional[str] = None  # 若设置，将直接使用该 URL 评测并跳过部署

    def _init_eval_url(self):  # 规范化 eval_url（兼容 OpenAI 风格路径）
        """
        函数说明：对 eval_url 进行兼容性处理，去除 OpenAI 风格路径后缀 '/chat/completions'。

        参数：
            self: 当前实例。

        返回：
            None

        示例：
            >>> args = EvalArguments(eval_url='http://host:port/v1/chat/completions')
            >>> args._init_eval_url()
            >>> args.eval_url
            'http://host:port/v1'
        """
        # [compat]
        if self.eval_url and 'chat/completions' in self.eval_url:  # 若为 OpenAI 风格路径
            self.eval_url = self.eval_url.split('/chat/completions', 1)[0]  # 去除路径后缀，保留基地址

    def __post_init__(self):  # 数据类初始化后的钩子
        """
        函数说明：在数据类初始化后，完成评测相关的初始化：URL 兼容、数据集校验、参数解析与路径规范。

        参数：
            self: 当前实例。

        返回：
            None

        示例：
            >>> args = EvalArguments()
            >>> # 初始化后 eval_output_dir 将被转为绝对路径
        """
        super().__post_init__()  # 调用父类初始化，继承部署侧的通用逻辑
        self._init_eval_url()  # 规范化评测 URL
        self._init_eval_dataset()  # 规范化并校验评测数据集
        self.dataset_args = json_parse_to_dict(self.dataset_args)  # 将可能的 JSON 字符串解析为字典
        self.eval_generation_config = json_parse_to_dict(self.eval_generation_config)  # 解析生成配置
        self.extra_eval_args = json_parse_to_dict(self.extra_eval_args)  # 解析额外评测参数
        self.eval_output_dir = to_abspath(self.eval_output_dir)  # 评测输出目录转为绝对路径
        logger.info(f'eval_output_dir: {self.eval_output_dir}')  # 打印评测输出目录

    @staticmethod  # 静态方法，与实例无关
    def list_eval_dataset(eval_backend=None):  # 列出各后端支持的数据集
        """
        函数说明：返回一个映射，描述不同评测后端各自支持的数据集列表。

        参数：
            eval_backend: 可选，仅在缺少依赖而用户显式请求某后端时用于抛错判定。

        返回：
            Dict：键为评测后端常量，值为该后端支持的数据集名称列表。

        示例：
            >>> datasets = EvalArguments.list_eval_dataset()
            >>> isinstance(datasets, dict)
            True
        """
        from evalscope.constants import EvalBackend  # 导入评测后端常量
        from evalscope.benchmarks.benchmark import BENCHMARK_MAPPINGS  # 导入基准数据集映射
        from evalscope.backend.opencompass import OpenCompassBackendManager  # 导入 OpenCompass 管理器
        res = {  # 构建各后端到数据集列表的映射
            EvalBackend.NATIVE: list(BENCHMARK_MAPPINGS.keys()),  # 原生后端支持的数据集列表
            EvalBackend.OPEN_COMPASS: OpenCompassBackendManager.list_datasets(),  # OpenCompass 支持的数据集
        }
        try:  # 尝试引入 VLM 评测后端（可能缺少依赖）
            from evalscope.backend.vlm_eval_kit import VLMEvalKitBackendManager  # 导入 VLM 后端管理器
            vlm_datasets = VLMEvalKitBackendManager.list_supported_datasets()  # 获取 VLM 支持的数据集
            res[EvalBackend.VLM_EVAL_KIT] = vlm_datasets  # 添加到结果映射
        except ImportError:  # 缺失依赖时
            # fix cv2 import error
            if eval_backend == 'VLMEvalKit':  # 若用户显式请求 VLM 后端则需要抛错
                raise  # 继续向上抛出异常
        return res  # 返回后端-数据集映射

    def _init_eval_dataset(self):  # 规范化与校验评测数据集
        """
        函数说明：将 `eval_dataset` 标准化为大小写不敏感的合法数据集名称列表。

        参数：
            self: 当前实例。

        返回：
            None；若包含不受支持的数据集名，将抛出 ValueError。

        示例：
            >>> args = EvalArguments(eval_dataset=['MMLU', 'C-Eval'])
            >>> args._init_eval_dataset()
        """
        if isinstance(self.eval_dataset, str):  # 兼容单字符串输入
            self.eval_dataset = [self.eval_dataset]  # 统一转为列表

        all_eval_dataset = self.list_eval_dataset(self.eval_backend)  # 获取该后端支持的数据集
        dataset_mapping = {dataset.lower(): dataset for dataset in all_eval_dataset[self.eval_backend]}  # 构建大小写不敏感映射
        valid_dataset = []  # 存放规范化后的合法数据集名
        for dataset in self.eval_dataset:  # 遍历用户提供的数据集
            if dataset.lower() not in dataset_mapping:  # 若数据集不受支持
                raise ValueError(  # 抛出详细错误信息
                    f'eval_dataset: {dataset} is not supported.\n'  # 错误信息第一行
                    f'eval_backend: {self.eval_backend} supported datasets: {all_eval_dataset[self.eval_backend]}')  # 错误信息第二行
            valid_dataset.append(dataset_mapping[dataset.lower()])  # 追加规范化后的数据集名
        self.eval_dataset = valid_dataset  # 写回校验与规范化后的列表

        logger.info(f'eval_backend: {self.eval_backend}')  # 打印评测后端
        logger.info(f'eval_dataset: {self.eval_dataset}')  # 打印最终的数据集列表

    def _init_result_path(self, folder_name: str) -> None:  # 初始化评测结果路径
        """
        函数说明：初始化评测结果输出目录与结果文件路径，并在未指定外部 URL 时创建评测结果目录结构。

        参数：
            folder_name: 预期的结果目录名称（此处固定使用 'eval_result'）。

        返回：
            None

        示例：
            >>> args = EvalArguments()
            >>> args._init_result_path('eval_result')
        """
        self.time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')  # 记录时间戳，精确到微秒
        result_dir = self.ckpt_dir or f'result/{self.model_suffix}'  # 优先使用 ckpt_dir，否则按模型后缀构造目录
        os.makedirs(result_dir, exist_ok=True)  # 创建目录（若已存在则忽略）
        self.result_jsonl = to_abspath(os.path.join(result_dir, 'eval_result.jsonl'))  # 结果 JSONL 文件绝对路径
        if not self.eval_url:  # 若未指定外部评测 URL
            super()._init_result_path('eval_result')  # 调用父类构建 'eval_result' 目录结构

    def _init_torch_dtype(self) -> None:  # 初始化 PyTorch dtype（若需要）
        """
        函数说明：当使用外部 eval_url 进行评测时，跳过模型 dtype 初始化；否则沿用父类逻辑。

        参数：
            self: 当前实例。

        返回：
            None

        示例：
            >>> args = EvalArguments(eval_url='http://host:port/v1')
            >>> args._init_torch_dtype()
        """
        if self.eval_url:  # 外链评测不需要加载本地模型
            self.model_dir = self.eval_output_dir  # 复用评测输出目录作为模型目录
            return  # 提前返回，跳过父类 dtype 初始化
        super()._init_torch_dtype()  # 调用父类方法进行 dtype 初始化
