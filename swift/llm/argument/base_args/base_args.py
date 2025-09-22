"""模块文档注释：
该脚本定义了用于大语言模型（LLM）训练与推理流程的基础参数类 `BaseArguments` 以及
兼容处理类 `CompatArguments`，并封装了以下核心能力：
- 参数聚合：组合生成（Generation）、量化（Quantize）、数据（Data）、模板（Template）、模型（Model）等多组参数；
- 兼容迁移：从 checkpoint 目录加载历史参数并进行回填；
- 适配器管理：下载/校验 LoRA/REFT 等适配器目录；
- 外部扩展：动态导入外部插件与自定义数据注册脚本；
- 初始化流程：Hub 登录、分布式设备设置、延迟分词策略、模板与模型构建；
- I/O 能力：保存/加载参数至输出目录或 checkpoint 目录。
"""

# 版权声明：本文件版权归 Alibaba 及其关联方所有
# Copyright (c) Alibaba, Inc. and its affiliates.

# 导入标准库：os，用于路径和环境变量操作
import os
# 导入 dataclasses 工具，用于声明数据类与字段；fields 用于反射字段列表
from dataclasses import dataclass, field, fields
# 导入类型注解工具，提升类型可读性：Any/Dict/List/Literal/Optional/Union
from typing import Any, Dict, List, Literal, Optional, Union

# 导入标准库：json，用于序列化/反序列化参数到文件
import json

# 导入 Hub 相关函数，用于实例化 hub 客户端（支持 ModelScope/HF）
from jinja2 import pass_context
from swift.hub import get_hub
# 导入 LLM 相关构建工具：处理器、模板、模型-分词器构建、模板获取、unsloth 加载与安全下载
from swift.llm import Processor, Template, get_model_tokenizer, get_template, load_by_unsloth, safe_snapshot_download
# 导入工具函数：根据模型/适配器推断 checkpoint 目录
from swift.llm.utils import get_ckpt_dir
# 导入插件系统中额外可用的 tuner 注册表
from swift.plugin import extra_tuners
# 导入通用工具：JSON 校验、分布式环境信息、日志、外部模块导入、分布式判断、主进程判断、
#               JSON 字符串解析为 dict、设备设置、是否使用 HF Hub 的环境判断
from swift.utils import (check_json_format, get_dist_setting, get_logger, import_external_file, is_dist, is_master,
                         json_parse_to_dict, set_device, use_hf_hub)

# 导入局部相对模块：数据/生成/模型/量化/模板参数定义
from .data_args import DataArguments
from .generation_args import GenerationArguments
from .model_args import ModelArguments
from .quant_args import QuantizeArguments
from .template_args import TemplateArguments

# 创建模块级日志记录器，用于输出信息/调试日志
logger = get_logger()


# 定义工具函数：返回支持的 tuner 名称集合
def get_supported_tuners():
    """返回受支持的调优器（tuner）集合，包含内置与外部插件注册的 tuner。

    返回：
        set[str]: 例如 {'lora','full','longlora',...} 与外部 `extra_tuners` 的并集。
    """
    # 返回内置与外部注册 tuner 的并集
    return {'lora', 'full', 'longlora', 'adalora', 'llamapro', 'adapter', 'vera', 'boft', 'fourierft', 'reft', 'bone'
            } | set(extra_tuners.keys())


# 使用 dataclass 声明兼容性参数类
@dataclass
class CompatArguments:
    """兼容性参数容器。

    该类用于对历史/旧版参数进行兼容转换：
    - 将 `ckpt_dir` 识别为模型或适配器目录并迁移到对应字段；
    - 将旧版 `lora_modules` 合并到当前的 `adapters` 列表中。
    """
    # 兼容字段：checkpoint 目录；可能指向基础模型或适配器
    ckpt_dir: Optional[str] = None
    # 兼容字段：旧版 LoRA 模块路径列表，将被合并进 `adapters`
    lora_modules: List[str] = field(default_factory=list)

    # 内部方法：处理 ckpt_dir，将其识别为模型或适配器并写入相应字段
    def _handle_ckpt_dir(self: 'BaseArguments'):
        """将 `ckpt_dir` 判定为适配器或模型目录并迁移到当前参数结构。

        异常：
            AssertionError: 当 `ckpt_dir` 不是有效目录时抛出错误。
        """
        # 断言 ckpt_dir 必须为有效目录
        assert os.path.isdir(self.ckpt_dir), f'self.ckpt_dir: {self.ckpt_dir}'
        # 若目录包含适配器配置文件或 reft 子目录，则视为适配器目录
        if (os.path.exists(os.path.join(self.ckpt_dir, 'adapter_config.json'))
                or os.path.exists(os.path.join(self.ckpt_dir, 'default', 'adapter_config.json'))
                or os.path.exists(os.path.join(self.ckpt_dir, 'reft'))):
            # 若已在 adapters 中，则无需重复插入
            if self.ckpt_dir in self.adapters:
                return
            # 插入到 adapters 首位，确保最高优先级
            self.adapters.insert(0, self.ckpt_dir)
        else:
            # 否则视为基础模型目录，写入到 model 字段
            self.model = self.ckpt_dir
        # 使用完后清空 ckpt_dir，避免重复处理
        self.ckpt_dir = None

    # dataclass 初始化后置钩子：在实例化后执行兼容迁移逻辑
    def __post_init__(self: 'BaseArguments'):
        """兼容处理钩子：把 `ckpt_dir` 与 `lora_modules` 迁移到统一字段。

        - 若提供 `ckpt_dir`：判定其为模型或适配器并迁移；
        - 若存在 `lora_modules`：将其合并到 `adapters`。
        """
        # 若提供 ckpt_dir，则进行判定与迁移
        if self.ckpt_dir is not None:
            self._handle_ckpt_dir()

        # 若存在历史 lora_modules，则合并入 adapters
        if len(self.lora_modules) > 0:
            self.adapters += self.lora_modules


# 使用 dataclass 声明基础参数聚合类，组合多个参数类
@dataclass
class BaseArguments(CompatArguments, GenerationArguments, QuantizeArguments, DataArguments, TemplateArguments,
                    ModelArguments):
    """基础参数聚合类。

    该数据类聚合 `GenerationArguments`、`QuantizeArguments`、`DataArguments`、
    `TemplateArguments`、`ModelArguments` 等多种参数配置，并在初始化中完成：
    - 适配器路径标准化与安全下载；
    - checkpoint 目录推断与历史参数加载；
    - 外部插件与自定义数据注册脚本导入；
    - 模型关键字参数解析与环境变量设置；
    - 分布式信息收集、延迟分词策略初始化、Hub 登录；
    - 模板与模型构建辅助方法提供。

    参数：
        tuner_backend (Literal['peft','unsloth']): 调优后端类型；
        train_type (str): 训练/微调类型（支持多类 tuner 与 'full'）；
        seed (int): 随机种子；
        model_kwargs (Optional[Union[dict,str]]): 模型构建的额外关键字参数；
        load_args (bool): 是否从 checkpoint 目录加载历史参数；
        load_data_args (bool): 是否加载数据集相关配置；
        packing (bool): 数据集是否启用样本打包；
        lazy_tokenize (Optional[bool]): 是否启用延迟分词；
        use_hf (bool): 是否启用 Hugging Face Hub；
        hub_token (Optional[str]): Hub 访问令牌；
        custom_register_path (List[str]): 数据集注册脚本路径；
        ignore_args_error (bool): 是否忽略参数错误（Notebook 兼容）；
        use_swift_lora (bool): 是否使用 swift lora（高级选项）。
    """
    # 指定调优后端类型，默认为 peft
    tuner_backend: Literal['peft', 'unsloth'] = 'peft'
    # 训练/微调类型，默认 'lora'，并在 metadata 中提示可选项列表
    train_type: str = field(default='lora', metadata={'help': f'train_type choices: {list(get_supported_tuners())}'})
    # 适配器路径列表（可多项），用于加载/合并多个适配器
    adapters: List[str] = field(default_factory=list)
    # 外部插件（Python 脚本）路径列表
    external_plugins: List[str] = field(default_factory=list)

    # 随机种子，默认 42
    seed: int = 42
    # 传入模型构建的额外关键字参数（dict 或 JSON 字符串）
    model_kwargs: Optional[Union[dict, str]] = None
    # 是否从 checkpoint 目录加载历史参数
    load_args: bool = True
    # 是否加载数据相关配置
    load_data_args: bool = False
    # dataset
    # 是否启用样本打包
    packing: bool = False
    # 是否启用延迟分词（None 表示自动根据条件判定）
    lazy_tokenize: Optional[bool] = None
    # 已缓存的数据集路径/名称列表
    cached_dataset: List[str] = field(default_factory=list)
    # 自定义数据注册脚本路径列表（.py 文件）
    custom_register_path: List[str] = field(default_factory=list)  # .py
    # hub
    # 是否使用 Hugging Face Hub
    use_hf: bool = False
    # None: use env var `MODELSCOPE_API_TOKEN`
    # Hub 访问令牌；未设置时可从环境变量读取
    hub_token: Optional[str] = field(
        default=None, metadata={'help': 'SDK token can be found in https://modelscope.cn/my/myaccesstoken'})
    # dist
    # 分布式训练超时时间（毫秒）
    ddp_timeout: int = 18000000
    # 分布式后端类型（可选）
    ddp_backend: Optional[str] = None

    # extra
    # 是否忽略参数错误（True 便于在 Notebook 环境下运行）
    ignore_args_error: bool = False  # True: notebook compatibility
    # 是否使用 swift 的 lora（高级选项，不建议随意开启）
    use_swift_lora: bool = False  # True for using tuner_backend == swift, don't specify this unless you know what you are doing # noqa

    # 预留方法：由具体训练器覆盖，以准备训练参数
    def _prepare_training_args(self, training_args: Dict[str, Any]) -> None:
        """准备训练参数占位方法，供子类或具体训练器覆盖实现。"""
        # 当前无默认实现
        pass

    # 初始化延迟分词策略
    def _init_lazy_tokenize(self):
        """根据模型/流式/打包配置自动判定并校验 `lazy_tokenize`。"""
        # 若未显式指定，则根据多模态/流式/打包策略自动判定
        if self.lazy_tokenize is None:
            if self.model_meta.is_multimodal and not self.streaming and not self.packing:
                self.lazy_tokenize = True
            else:
                self.lazy_tokenize = False
            # 输出最终判定结果
            logger.info(f'Setting args.lazy_tokenize: {self.lazy_tokenize}')
        # 若启用了延迟分词，则不得与 packing/streaming 同时启用
        if self.lazy_tokenize:
            if self.packing:
                raise ValueError('Packing and lazy_tokenize are incompatible.')
            if self.streaming:
                raise ValueError('Streaming and lazy_tokenize are incompatible.')

    # 初始化自定义数据注册脚本
    def _init_custom_register(self) -> None:
        """导入自定义 .py 注册脚本并注册到数据管线。"""
        # 若为字符串则包装为列表，统一处理
        if isinstance(self.custom_register_path, str):
            self.custom_register_path = [self.custom_register_path]
        # 未提供则直接返回
        if not self.custom_register_path:
            return
        # 逐个导入外部注册脚本
        for path in self.custom_register_path:
            import_external_file(path)
        # 记录成功日志
        logger.info(f'Successfully registered {self.custom_register_path}.')

    # 导入外部插件（Python 文件）
    def _import_external_plugins(self):
        """导入外部插件脚本以扩展功能（如数据处理或自定义逻辑）。"""
        # 若为字符串则包装为列表
        if isinstance(self.external_plugins, str):
            self.external_plugins = [self.external_plugins]
        # 未配置则返回
        if not self.external_plugins:
            return
        # 逐个导入插件文件
        for external_plugin in self.external_plugins:
            import_external_file(external_plugin)
        # 记录成功日志
        logger.info(f'Successfully imported external_plugins: {self.external_plugins}.')

    @staticmethod
    def _check_is_adapter(adapter_dir: str) -> bool:
        """判断给定目录是否为适配器目录。

        参数：
            adapter_dir (str): 目录路径。
        返回：
            bool: 若包含适配器配置/标记文件则为 True，否则 False。
        """
        # 检查是否存在适配器配置或 reft 子目录
        if (os.path.exists(os.path.join(adapter_dir, 'adapter_config.json'))
                or os.path.exists(os.path.join(adapter_dir, 'default', 'adapter_config.json'))
                or os.path.exists(os.path.join(adapter_dir, 'reft'))):
            return True
        return False

    # 初始化适配器：标准化为列表并执行安全下载
    def _init_adapters(self):
        """标准化 `adapters` 字段并通过 Hub 安全下载到本地缓存路径。"""
        # 兼容：若传入为字符串，则包装成列表
        if isinstance(self.adapters, str):
            self.adapters = [self.adapters]
        # 对每个适配器路径执行安全下载，返回本地路径列表
        self.adapters = [
            safe_snapshot_download(adapter, use_hf=self.use_hf, hub_token=self.hub_token) for adapter in self.adapters
        ]

    # dataclass 初始化后置钩子：完成完整初始化流程
    def __post_init__(self):
        """完成 BaseArguments 初始化：Hub 设置、兼容迁移、适配器与 checkpoint、插件加载、
        模型参数与数据流初始化、分布式信息、延迟分词、Hub 登录等。
        """
        # 若显式或环境要求使用 HF Hub，则设置标记与环境变量
        if self.use_hf or use_hf_hub():
            self.use_hf = True
            os.environ['USE_HF'] = '1'
        # 父类兼容处理：迁移 ckpt_dir/lora_modules 到统一字段
        CompatArguments.__post_init__(self)
        # 初始化并下载适配器
        self._init_adapters()
        # 推断 checkpoint 目录与按需加载历史参数
        self._init_ckpt_dir()
        # 导入自定义数据注册脚本
        self._init_custom_register()
        # 导入外部插件
        self._import_external_plugins()
        # 解析并设置模型相关 kwargs 到环境变量
        self._init_model_kwargs()
        # 初始化数据流配置（如 streaming）
        self._init_stream()
        # 说明：Transformers 的 Seq2SeqTrainingArguments 的 world_size 属性不可直接赋值
        # 获取分布式相关信息：全局/本地 rank 与 world_size
        self.rank, self.local_rank, self.global_world_size, self.local_world_size = get_dist_setting()
        # 记录分布式信息到日志
        logger.info(f'rank: {self.rank}, local_rank: {self.local_rank}, '
                    f'world_size: {self.global_world_size}, local_world_size: {self.local_world_size}')
        # 若 train_type 不属于外部自定义 tuner，则检查 adapters 合法性
        if self.train_type not in extra_tuners:
            for adapter in self.adapters:
                assert self._check_is_adapter(adapter), (
                    f'`{adapter}` is not an adapter, please try using `--model` to pass it.')
        # 依次调用各父类的 __post_init__ 完成分支参数初始化
        ModelArguments.__post_init__(self)
        QuantizeArguments.__post_init__(self)
        TemplateArguments.__post_init__(self)
        DataArguments.__post_init__(self)
        # 兼容：若 cached_dataset 为字符串则包装为列表
        if isinstance(self.cached_dataset, str):
            self.cached_dataset = [self.cached_dataset]
        # 计算与校验延迟分词策略
        self._init_lazy_tokenize()
        # 初始化 Hub 实例
        self.hub = get_hub(self.use_hf)
        # 若提供 token 则尝试登录
        if self.hub.try_login(self.hub_token):
            logger.info('hub login successful!')

    # 解析模型关键字参数并设置到环境变量
    def _init_model_kwargs(self):
        """解析 `model_kwargs` 并将其写入进程环境变量。

        将传入的字典或 JSON 字符串转换为字典，并将键名转为大写后设置到 `os.environ`。
        """
        # 将可能的 JSON 字符串解析为字典
        self.model_kwargs: Dict[str, Any] = json_parse_to_dict(self.model_kwargs)
        # 遍历项并写入环境变量（键名大写，值转字符串）
        for k, v in self.model_kwargs.items():
            k = k.upper()
            os.environ[k] = str(v)

    @property
    def is_adapter(self) -> bool:
        """是否处于适配器训练模式（非 'full' 即视为适配器模式）。"""
        # 返回当前训练类型是否不是 'full'
        return self.train_type not in {'full'}

    @property
    def supported_tuners(self):
        """返回受支持的 tuner 名称集合。"""
        # 调用工具函数返回集合
        return get_supported_tuners()

    @property
    def adapters_can_be_merged(self):
        """返回可执行权重合并（merge）的适配器类型集合。"""
        # 仅这些类型支持直接合并
        return {'lora', 'longlora', 'llamapro', 'adalora'}

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str):
        """基于指定 checkpoint 目录构造参数对象并加载历史参数。

        参数：
            checkpoint_dir (str): checkpoint 根目录。
        返回：
            BaseArguments: 已加载历史参数并补齐字段的实例。
        """
        # 通过 __new__ 构造未初始化实例（避免默认 __init__）
        self = super().__new__(cls)
        # 标记允许加载数据相关参数
        self.load_data_args = True
        # 指定 checkpoint 目录
        self.ckpt_dir = checkpoint_dir
        # 从 checkpoint 读取历史参数
        self.load_args_from_ckpt()
        # 收集 BaseArguments 的所有字段名
        all_keys = list(f.name for f in fields(BaseArguments))
        # 对缺失字段赋值 None，保证实例字段完整
        for key in all_keys:
            if not hasattr(self, key):
                setattr(self, key, None)
        # 返回实例
        return self

    # 推断并初始化 checkpoint 目录（兼容 megatron 字段）
    def _init_ckpt_dir(self, adapters=None):
        """根据 model/adapters 推断 checkpoint 目录，并按需加载参数。"""
        # compat megatron：兼容 mcore_model/load 等字段
        model = self.model or getattr(self, 'mcore_model', None) or getattr(self, 'load', None)
        # 兼容：从 adapters/mcore_adapters 中取得适配器列表
        adapters = adapters or self.adapters or getattr(self, 'mcore_adapters', None)
        # 计算 checkpoint 目录
        self.ckpt_dir = get_ckpt_dir(model, adapters)
        # 若需要，从 checkpoint 目录加载历史参数
        if self.ckpt_dir and self.load_args:
            self.load_args_from_ckpt()

    # 从 checkpoint 目录读取并回填历史参数
    def load_args_from_ckpt(self) -> None:
        """从 `self.ckpt_dir/args.json` 读取历史参数，并按规则回填到当前实例。"""
        # 组装 args.json 路径
        args_path = os.path.join(self.ckpt_dir, 'args.json')
        # 校验 args.json 必须存在
        assert os.path.exists(args_path), f'args_path: {args_path}'
        # 读取旧参数到字典
        with open(args_path, 'r', encoding='utf-8') as f:
            old_args = json.load(f)
        # 强制覆盖的键（无条件从旧参数覆盖）
        force_load_keys = [
            # base_args
            'train_type',
            # model_args
            'task_type',
            # quant_args
            'bnb_4bit_quant_type',
            'bnb_4bit_use_double_quant',
        ]
        # 条件加载键：若当前值为 None 或空列表，则从旧参数加载
        # If the current value is None or an empty list and it is among the following keys
        load_keys = [
            'custom_register_path',
            'external_plugins',
            # model_args
            'model',
            'model_type',
            'model_revision',
            'torch_dtype',
            'attn_impl',
            'new_special_tokens',
            'num_labels',
            'problem_type',
            'rope_scaling',
            'max_model_len',
            # quant_args
            'quant_method',
            'quant_bits',
            'hqq_axis',
            'bnb_4bit_compute_dtype',
            # template_args
            'template',
            'system',
            'truncation_strategy',
            'agent_template',
            'norm_bbox',
            'use_chat_template',
            'response_prefix',
        ]
        # 若为 Megatron 派生类，则不强制覆盖，且不加载 use_chat_template
        if 'megatron' in self.__class__.__name__.lower():
            force_load_keys = []
            load_keys.remove('use_chat_template')
        # 获取 DataArguments 的所有字段名，用于有条件加载数据相关参数
        data_keys = list(f.name for f in fields(DataArguments))
        # 遍历旧参数，按策略更新当前实例
        for key, old_value in old_args.items():
            # 跳过旧值为 None 的项
            if old_value is None:
                continue
            # 无条件覆盖或在允许加载的数据键集合中且标记允许加载数据
            if key in force_load_keys or self.load_data_args and key in data_keys:
                setattr(self, key, old_value)
            # 若当前值缺失或为空（列表/元组），且键在 load_keys 中，则回填旧值
            value = getattr(self, key, None)
            if key in load_keys and (value is None or isinstance(value, (list, tuple)) and len(value) == 0):
                setattr(self, key, old_value)
        # 输出成功日志
        logger.info(f'Successfully loaded {args_path}.')

    # 保存当前参数到输出目录
    def save_args(self, output_dir=None) -> None:
        """将当前参数序列化保存到 `output_dir/args.json`（仅主节点执行）。"""
        # 仅主进程执行保存，避免分布式并发写
        if is_master():
            # 若未显式传入，则使用实例的 output_dir
            output_dir = output_dir or self.output_dir
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            # 目标文件路径
            fpath = os.path.join(output_dir, 'args.json')
            # 记录保存路径
            logger.info(f'The {self.__class__.__name__} will be saved in: {fpath}')
            # 写入 JSON 文件，ensure_ascii=False 以保留中文
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(check_json_format(self.__dict__), f, ensure_ascii=False, indent=2)

    # 初始化设备（分布式场景）
    def _init_device(self):
        """在分布式场景下设置当前进程的可见设备（例如 GPU 设备 ID）。"""
        # 分布式训练时设置当前进程设备
        if is_dist():
            set_device()

    # 获取模板实例
    def get_template(self, processor: Optional['Processor'], template_type: Optional[str] = None) -> 'Template':
        """根据当前配置获取对话模板实例。

        参数：
            processor (Optional[Processor]): 预处理/后处理器。
            template_type (Optional[str]): 模板类型，默认使用 `self.template`。
        返回：
            Template: 模板实例。
        """
        # 收集模板所需关键字参数
        template_kwargs = self.get_template_kwargs()
        # 若未显式指定模板类型，则使用默认模板
        template_type = template_type or self.template
        # 构造模板实例
        template = get_template(template_type, processor, **template_kwargs)
        # 返回模板
        return template

    # 构建模型与处理器/分词器
    def get_model_processor(self,
                            *,
                            model=None,
                            model_type=None,
                            model_revision=None,
                            task_type=None,
                            num_labels=None,
                            **kwargs):
        """构建并返回模型与分词器（或处理器）。

        参数：
            model (Optional[str]): 模型路径或标识。
            model_type (Optional[str]): 模型类型标识。
            model_revision (Optional[str]): 模型版本。
            task_type (Optional[str]): 任务类型。
            num_labels (Optional[int]): 标签数量（分类任务）。
            **kwargs: 额外关键字参数，将与实例的 `model_kwargs` 合并。
        返回：
            任意：`get_model_tokenizer` 或 `load_by_unsloth` 返回的对象。
        """
        # 若采用 unsloth 后端，则交由其加载逻辑处理
        if self.tuner_backend == 'unsloth':
            return load_by_unsloth(self)
        # 合并实例级模型关键字参数
        kwargs.update(self.get_model_kwargs())
        # compat rlhf：统一关键参数命名
        kwargs['model_id_or_path'] = model or self.model
        kwargs['model_type'] = model_type or self.model_type
        kwargs['model_revision'] = model_revision or self.model_revision
        kwargs['task_type'] = task_type or self.task_type
        kwargs['num_labels'] = num_labels or self.num_labels

        # 返回由 swift.llm 提供的构建结果
        return get_model_tokenizer(**kwargs)
