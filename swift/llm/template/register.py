"""
模块功能：
    该模块实现 ms-swift 框架的模板注册系统，提供全局模板映射表和注册/获取模板的核心功能。
    通过统一的注册机制管理所有模型的对话模板（如 Qwen、LLaMA、ChatGLM 等），支持动态注册和检索。

核心功能：
    1. 全局模板映射表（TEMPLATE_MAPPING）：存储所有已注册的模板元信息
    2. 模板注册（register_template）：将新的模板元信息注册到全局映射表
    3. 模板实例化（get_template）：根据模板类型创建对应的 Template 实例
    4. 模板元信息获取（get_template_meta）：获取指定模板的元信息（不实例化）

应用场景：
    - 模型模板注册：为新支持的模型注册对话模板（如添加新模型时）
    - 模板查询：根据模型类型查找对应的模板配置
    - 模板初始化：在训练或推理前实例化对应的模板对象
    - 模板扩展：支持自定义模板的动态注册（插件系统）

使用示例：
    >>> # 示例1：注册自定义模板
    >>> from swift.llm.template import TemplateMeta, register_template, Template
    >>> 
    >>> # 创建模板元信息
    >>> custom_meta = TemplateMeta(
    ...     template_type='my_custom_model',
    ...     template_cls=Template,
    ...     prefix=['<s>'],
    ...     prompt=['User: {{QUERY}}\nAssistant: '],
    ...     chat_sep=['</s>']
    ... )
    >>> 
    >>> # 注册到全局映射表
    >>> register_template(custom_meta)
    >>> 
    >>> # 示例2：获取并实例化模板
    >>> from swift.llm.template import get_template
    >>> from transformers import AutoTokenizer
    >>> 
    >>> tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
    >>> template = get_template(
    ...     template_type='qwen',
    ...     processor=tokenizer,
    ...     max_length=2048,
    ...     truncation_strategy='left'
    ... )
    >>> template.mode = 'train'
    >>> 
    >>> # 示例3：查询模板元信息（不实例化）
    >>> from swift.llm.template import get_template_meta
    >>> 
    >>> meta = get_template_meta('llama3')
    >>> print(meta.template_type)  # 输出: 'llama3'
    >>> print(meta.prefix)  # 输出: ['<|begin_of_text|>']
"""
# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict, Literal, Optional  # 类型注解：Dict字典，Literal字面量类型，Optional可选类型

from ..utils import Processor  # 导入处理器类：统一管理tokenizer和image_processor
from .base import Template  # 导入核心模板类：所有具体模板的基类
from .template_meta import TemplateMeta  # 导入模板元信息类：存储模板配置（prefix、prompt、chat_sep等）

# 全局模板映射表：键为模板类型字符串（如'qwen'），值为对应的TemplateMeta元信息对象
# 所有通过register_template注册的模板都会存储在这里
TEMPLATE_MAPPING: Dict[str, TemplateMeta] = {}


def register_template(template_meta: TemplateMeta, *, exist_ok: bool = False) -> None:
    """
    功能：
        将模板元信息注册到全局映射表 TEMPLATE_MAPPING 中，使其可被 get_template 检索和实例化。
        注册时会检查模板类型是否已存在，默认情况下不允许重复注册同一模板类型。

    参数：
        template_meta (TemplateMeta): 模板元信息对象，包含模板类型、前缀、提示词等配置。
            - 必须包含 template_type 字段作为唯一标识符
            - 示例：TemplateMeta(template_type='qwen', prefix=['<|im_start|>'], ...)
        exist_ok (bool): 是否允许覆盖已存在的模板，默认 False（不允许）。
            - False: 若模板类型已存在则抛出 ValueError 异常
            - True: 若模板类型已存在则静默覆盖（用于更新模板配置）

    返回：
        None: 无返回值，通过修改全局 TEMPLATE_MAPPING 完成注册。

    示例：
        >>> # 示例1：注册新模板
        >>> meta = TemplateMeta(template_type='my_model', prefix=['<s>'], prompt=['User: {{QUERY}}\nBot: '])
        >>> register_template(meta)
        >>> 
        >>> # 示例2：更新已存在的模板（需设置 exist_ok=True）
        >>> updated_meta = TemplateMeta(template_type='my_model', prefix=['<start>'], prompt=['Q: {{QUERY}}\nA: '])
        >>> register_template(updated_meta, exist_ok=True)  # 覆盖之前的注册
    """
    template_type = template_meta.template_type  # 提取模板类型字符串（唯一标识符）
    # 重复注册检查：若模板已存在且不允许覆盖，抛出异常
    if not exist_ok and template_type in TEMPLATE_MAPPING:
        raise ValueError(f'The `{template_type}` has already been registered in the TEMPLATE_MAPPING.')
    # 注册模板：将元信息对象存入全局映射表
    TEMPLATE_MAPPING[template_type] = template_meta


def get_template(
    template_type: str,
    processor: Processor,
    default_system: Optional[str] = None,
    max_length: Optional[int] = None,
    *,
    truncation_strategy: Literal['raise', 'left', 'right'] = 'raise',
    max_pixels: Optional[int] = None,  # h * w
    agent_template: Optional[str] = None,
    norm_bbox: Literal['norm1000', 'none', None] = None,
    use_chat_template: bool = True,
    remove_unused_columns: bool = True,
    # train
    padding_free: bool = False,
    padding_side: Literal['left', 'right'] = 'right',
    loss_scale: str = 'default',
    sequence_parallel_size: int = 1,
    # infer/deploy
    response_prefix: Optional[str] = None,
    template_backend: Literal['swift', 'jinja'] = 'swift',
) -> 'Template':
    """
    功能：
        根据模板类型从全局映射表中获取模板元信息，并实例化为 Template 对象。
        该函数是获取模板的主要入口，支持训练和推理场景的各种配置参数。

    参数：
        template_type (str): 模板类型标识符，必须已通过 register_template 注册。
            - 示例：'qwen', 'llama3', 'chatglm2' 等
        processor (Processor): 处理器对象，包含 tokenizer 和 image_processor。
            - 用于文本编码和多模态数据预处理
        default_system (Optional[str]): 默认系统提示词，覆盖模板元信息中的默认值。
            - 示例：'You are a helpful assistant.'
        max_length (Optional[int]): 最大序列长度（包含输入+输出），超出时根据 truncation_strategy 处理。
            - 示例：2048, 4096
        truncation_strategy (Literal['raise', 'left', 'right']): 超长序列截断策略。
            - 'raise': 抛出 MaxLengthError 异常
            - 'left': 左截断（保留最新的对话轮次）
            - 'right': 右截断（截断输出部分）
        max_pixels (Optional[int]): 图像最大像素数（高×宽），超出时会缩放图像。
            - 示例：1024*1024 = 1048576
        agent_template (Optional[str]): Agent 模板类型，用于 function calling 场景。
            - 示例：'react', 'tool_call'
        norm_bbox (Literal['norm1000', 'none', None]): Grounding 任务的边界框归一化模式。
            - 'norm1000': 坐标范围 0-1000
            - 'none': 像素坐标
        use_chat_template (bool): 是否使用 chat template（Jinja后端需要）。
        remove_unused_columns (bool): 是否移除未使用的数据列（数据集预处理）。
        
        # 训练相关参数
        padding_free (bool): 是否启用 padding-free 模式（packing训练）。
        padding_side (Literal['left', 'right']): Padding 方向。
            - 'right': 右padding（训练常用）
            - 'left': 左padding（推理常用）
        loss_scale (str): Loss 权重策略。
            - 'default': 默认策略（assistant部分计算loss）
            - 'all': 所有token计算loss
            - 'last_round': 仅最后一轮assistant计算loss
        sequence_parallel_size (int): 序列并行大小（Megatron-LM）。
        
        # 推理相关参数
        response_prefix (Optional[str]): 生成响应的前缀字符串。
        template_backend (Literal['swift', 'jinja']): 模板后端类型。
            - 'swift': 自定义模板系统（功能更强大）
            - 'jinja': HuggingFace标准模板（兼容性好）

    返回：
        Template: 实例化的模板对象，可用于 encode/decode 等操作。

    示例：
        >>> # 示例1：训练场景 - 基础配置
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
        >>> template = get_template(
        ...     template_type='qwen',
        ...     processor=tokenizer,
        ...     max_length=2048,
        ...     truncation_strategy='left',
        ...     loss_scale='default'
        ... )
        >>> 
        >>> # 示例2：多模态推理场景
        >>> from transformers import AutoProcessor
        >>> processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')
        >>> template = get_template(
        ...     template_type='qwen2_vl',
        ...     processor=processor,
        ...     max_pixels=1024*1024,
        ...     padding_side='left',
        ...     template_backend='swift'
        ... )
        >>> 
        >>> # 示例3：Agent 场景
        >>> template = get_template(
        ...     template_type='qwen',
        ...     processor=tokenizer,
        ...     agent_template='react',
        ...     use_chat_template=False
        ... )
    """
    # 1> 从全局映射表中获取模板元信息
    template_meta = TEMPLATE_MAPPING[template_type]
    
    # 2> 提取模板类（可能是 Template 或其子类，如 QwenTemplate、LlamaTemplate等）
    template_cls = template_meta.template_cls
    
    # 3> 实例化模板对象：将所有参数传递给模板类的构造函数
    return template_cls(
        processor,  # 处理器对象（tokenizer + image_processor）
        template_meta,  # 模板元信息（prefix、prompt、chat_sep等配置）
        default_system,  # 自定义系统提示词
        max_length,  # 最大序列长度
        truncation_strategy=truncation_strategy,  # 截断策略
        max_pixels=max_pixels,  # 图像最大像素数
        agent_template=agent_template,  # Agent模板类型
        norm_bbox=norm_bbox,  # 边界框归一化模式
        use_chat_template=use_chat_template,  # 是否使用chat template
        remove_unused_columns=remove_unused_columns,  # 是否移除未使用列
        # 训练参数
        padding_free=padding_free,  # 是否padding-free
        padding_side=padding_side,  # padding方向
        loss_scale=loss_scale,  # loss权重策略
        sequence_parallel_size=sequence_parallel_size,  # 序列并行大小
        # 推理参数
        response_prefix=response_prefix,  # 响应前缀
        template_backend=template_backend,  # 模板后端类型
    )


def get_template_meta(template_type: str) -> TemplateMeta:
    """
    功能：
        根据模板类型获取模板元信息对象（不实例化 Template）。
        用于查询模板配置信息，比实例化 Template 更轻量。

    参数：
        template_type (str): 模板类型标识符。
            - 示例：'qwen', 'llama3', 'chatglm2'

    返回：
        TemplateMeta: 模板元信息对象，包含 template_type, prefix, prompt, chat_sep 等字段。

    示例：
        >>> # 查询模板配置
        >>> meta = get_template_meta('qwen')
        >>> print(f"Prefix: {meta.prefix}")
        >>> print(f"Prompt: {meta.prompt}")
        >>> print(f"Default system: {meta.default_system}")
    """
    # 直接从全局映射表返回元信息对象（无需实例化）
    return TEMPLATE_MAPPING[template_type]
