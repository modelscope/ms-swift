"""模块功能概述：
该模块实现 ms-swift 框架的核心模板系统 `Template` 类，负责将各种格式的输入数据转换为
大语言模型可接受的标准化输入格式。该模板系统是整个框架中连接用户数据与模型推理的核心桥梁。

核心功能：
1. **多任务支持**：
   - Causal LM（因果语言模型）：标准的自回归语言模型训练和推理
   - Sequence Classification（序列分类）：文本分类、情感分析等任务
   - PRM（Process Reward Model）：过程奖励模型，用于强化学习
   - Embedding：文本向量化，支持anchor-positive-negative三元组
   - Reranker/Generative Reranker：文档排序任务

2. **多模态处理**：
   - 图像输入：支持PIL.Image、路径、bytes等多种格式，自动加载和预处理
   - 视频输入：支持视频文件加载和帧提取
   - 音频输入：支持音频文件加载和预处理
   - 特殊标签替换：<image>、<video>、<audio>等标签的智能处理
   - Grounding任务：支持bbox和ref-object标签的归一化和替换

3. **多种训练模式**：
   - 标准训练（train/pt）：普通的监督学习
   - RLHF（Reinforcement Learning from Human Feedback）：基于人类反馈的强化学习
   - KTO（Kahneman-Tversky Optimization）：KT优化算法
   - GKD（Generalized Knowledge Distillation）：广义知识蒸馏

4. **编码与解码**：
   - encode方法：将用户输入转换为input_ids、labels、loss_scale等模型输入
   - decode方法：将模型生成的token序列解码为可读文本
   - 支持流式和非流式两种模式
   - 特殊token和占位符的安全处理

5. **模板后端支持**：
   - Swift后端：自定义的模板系统，支持复杂的对话格式和多轮对话
   - Jinja后端：使用tokenizer.apply_chat_template，兼容HuggingFace标准

6. **高级特性**：
   - Loss Scale：支持对不同token段设置不同的loss权重
   - Truncation：多种截断策略（左截断、右截断、抛异常）
   - Padding：支持左右padding，适配不同的训练和推理场景
   - Packing：将多个短序列拼接为长序列，提高GPU利用率
   - Sequence Parallel：序列并行支持，适配大规模训练
   - Agent Template：支持function calling和tool使用

7. **推理框架集成**：
   - vLLM：高性能推理框架集成
   - LMDeploy：支持PyTorch和TurboMind两种后端
   - SGLang：高效的语言模型推理框架
   - 标准transformers推理

8. **Data Collator**：
   - 批量数据整理和padding
   - 多模态数据（pixel_values、image_sizes等）的拼接
   - 支持各种任务的专用collator（RLHF、KTO、Embedding、Reranker等）
   - Attention mask生成（支持Megatron的4D causal mask）

主要类：
- Template：核心模板类，实现所有编码、解码和数据处理逻辑
- MaxLengthError：序列长度超限异常

适用场景：
- LLM模型的微调训练（SFT、RLHF等）
- 多模态模型的训练和推理
- 各类下游任务的适配（分类、排序、embedding等）
- 高性能推理部署（vLLM、LMDeploy等）

使用示例：
>>> # 初始化模板
>>> template = Template(['<|system|>'], ['<|user|>'], ['<|assistant|>'], ['<|end|>'])
>>> template.init_processor(processor)
>>> 
>>> # 编码输入
>>> inputs = StdTemplateInputs(messages=[
...     {"role": "user", "content": "Hello"},
...     {"role": "assistant", "content": "Hi there!"}
... ])
>>> encoded = template.encode(inputs)
>>> # encoded包含：input_ids, labels, loss_scale等
>>> 
>>> # 解码输出
>>> response = template.decode(generate_ids)
"""
# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，标注代码版权所有者

# ===== 标准库导入 =====
import hashlib  # 引入hashlib：用于计算文件哈希值（如保存PIL图像时生成唯一文件名）
import inspect  # 引入inspect：用于获取函数签名信息，检查参数和返回类型
import math  # 引入math：用于数学计算（如序列长度的对数计算、向上取整等）
import os  # 引入os：用于文件路径操作、环境变量读取等
import re  # 引入re：用于正则表达式匹配（如特殊标签替换、工具调用解析等）
from contextlib import contextmanager, nullcontext  # 引入上下文管理器：contextmanager用于创建临时patch（如flash_attention_forward），nullcontext用于条件性上下文
from copy import deepcopy  # 引入deepcopy：用于深拷贝复杂数据结构（如inputs字典），避免修改原始数据
from dataclasses import asdict  # 引入asdict：将数据类实例转换为字典，便于序列化和传递
from functools import partial, wraps  # 引入functools工具：partial用于函数参数固定，wraps用于装饰器保持原函数元信息
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Tuple, Union  # 引入类型注解：TYPE_CHECKING用于避免循环导入，其他用于明确参数和返回值类型

# ===== PyTorch核心库 =====
import torch  # 引入torch：PyTorch核心库，用于张量操作和深度学习
import torch.nn as nn  # 引入nn：神经网络模块，用于模型定义和层操作
import torch.nn.functional as F  # 引入F：函数式API，用于padding、激活函数等无状态操作
from modelscope.hub.utils.utils import get_cache_dir  # 从ModelScope获取缓存目录路径，用于下载和存储模型文件
from peft import PeftModel  # 引入PEFT模型类：用于识别和提取PEFT（参数高效微调）模型的base_model
from PIL import Image  # 引入PIL图像库：用于图像加载、格式转换和保存操作
from torch.nn.utils.rnn import pad_sequence  # 引入pad_sequence：用于批量序列的右padding操作（变长序列对齐）
from transformers import StoppingCriteriaList  # 引入停止条件列表：用于生成时的自定义停止逻辑（如遇到特定stop words）
from transformers.integrations import is_deepspeed_zero3_enabled  # 引入DeepSpeed Zero-3检测函数：判断是否启用了Zero-3并注册hook
from transformers.utils import strtobool  # 引入字符串转布尔函数：将"true"/"false"等字符串转换为布尔值

# ===== swift框架内部导入 =====
from swift.utils import get_env_args, get_logger  # 引入swift工具：get_env_args读取环境变量配置，get_logger创建日志记录器
from ..utils import Processor, ProcessorMixin  # 引入处理器类：Processor用于多模态数据预处理，ProcessorMixin提供处理器混入能力
from .template_inputs import InferRequest, StdTemplateInputs, TemplateInputs  # 引入模板输入数据类：InferRequest用于推理请求，StdTemplateInputs/TemplateInputs定义标准输入格式
from .utils import Context, ContextType, StopWordsCriteria, fetch_one, findall, split_str_parts_by  # 引入模板工具函数：Context上下文对象，StopWordsCriteria停止词判断，fetch_one获取单个元素，findall查找所有匹配，split_str_parts_by字符串分割
from .vision_utils import load_audio, load_batch, load_image, rescale_image  # 引入视觉/音频工具函数：load_image加载图像，load_audio加载音频，load_batch批量加载，rescale_image缩放图像

# ===== 日志和类型检查 =====
logger = get_logger()  # 创建模块级日志记录器，用于输出调试信息、警告和错误
if TYPE_CHECKING:  # 类型检查模式（仅在静态类型检查时执行，运行时跳过，避免循环导入）
    from .template_meta import TemplateMeta  # 导入TemplateMeta类型：用于类型注解，不在运行时导入以避免循环依赖


class MaxLengthError(ValueError):
    """类功能：
    `MaxLengthError` 是一个自定义异常类，用于在输入序列长度超过模板设定的最大长度限制时抛出异常。
    当模板的截断策略（`truncation_strategy`）设置为 'raise' 时，编码过程会主动检查序列长度，
    若超过 `max_length` 则抛出此异常，提醒调用者当前数据行的长度超限，需要调整配置或数据。

    继承关系说明：
    - 继承自 `ValueError`：因为序列长度超限本质上是一个输入值错误问题
    - 作为 `ValueError` 的子类，可以被标准的异常处理机制捕获
    - 通过自定义异常类型，调用者可以精确识别并处理"长度超限"这一特定错误场景

    应用场景：
    1. **数据验证**：在训练或推理前验证输入序列是否符合长度要求
    2. **严格模式**：当 `truncation_strategy='raise'` 时，拒绝处理超长序列，避免自动截断导致信息丢失
    3. **调试辅助**：帮助开发者快速定位数据集中存在的超长样本问题
    4. **配置检查**：提示用户需要调整 `max_length` 参数或修改截断策略（如改为 'left' 或 'right'）

    使用示例：
    >>> # 场景1：捕获并处理长度超限异常
    >>> template = Template(processor, template_meta, max_length=512, truncation_strategy='raise')
    >>> try:
    ...     inputs = StdTemplateInputs(messages=[{"role": "user", "content": "很长的文本..." * 1000}])
    ...     encoded = template.encode(inputs)
    ... except MaxLengthError as e:
    ...     print(f"序列长度超限: {e}")
    ...     # 可以选择：1) 跳过该样本；2) 调整max_length；3) 改用'left'/'right'截断策略
    >>> 
    >>> # 场景2：使用自动截断避免异常
    >>> template = Template(processor, template_meta, max_length=512, truncation_strategy='left')
    >>> encoded = template.encode(inputs)  # 自动左截断，不抛出异常
    >>> 
    >>> # 场景3：批量处理时过滤超长样本
    >>> valid_samples = []
    >>> for sample in dataset:
    ...     try:
    ...         encoded = template.encode(sample)
    ...         valid_samples.append(encoded)
    ...     except MaxLengthError:
    ...         print(f"跳过超长样本: {sample['id']}")
    ...         continue
    """
    pass  # 异常类仅作为标记，不需要额外实现


class Template(ProcessorMixin):
    """类功能：
    Template 是 ms-swift 框架的核心模板类，负责将用户输入的对话数据、多模态数据转换为
    大语言模型可接受的标准化输入格式（input_ids, labels等），并提供解码、数据整理等功能。
    该类是连接原始数据与模型训练/推理的关键桥梁。

    核心职责：
    1. 输入编码（encode）：将对话消息、图像、视频、音频等多模态输入转换为token序列
    2. 输出解码（decode）：将模型生成的token ID序列转换为可读文本
    3. 数据整理（data_collator）：批量数据的padding、拼接和attention mask生成
    4. 多模态处理：加载、预处理图像/视频/音频，替换特殊标签（<image>等）
    5. 模板应用：支持Swift和Jinja两种模板后端，处理系统提示、用户/助手轮次
    6. 截断策略：支持左截断、右截断或抛出异常三种方式处理超长序列
    7. Loss权重：为不同token段设置不同的loss_scale权重
    8. 多任务支持：适配因果语言模型、分类、embedding、reranker等多种任务类型
    9. 多训练模式：支持标准训练、RLHF、KTO、GKD等训练模式
    10. 推理框架集成：适配vLLM、LMDeploy、SGLang等高性能推理框架

    继承关系说明：
    - 继承自 `ProcessorMixin`：提供processor（tokenizer + image_processor等）的混入能力
    - 通过ProcessorMixin获得tokenizer、image_processor等属性的访问能力
    - 作为混入类的子类，可以灵活地与不同的processor实现配合

    应用场景：
    1. 模型微调训练：
    - SFT（Supervised Fine-Tuning）：监督微调
    - RLHF：基于人类反馈的强化学习
    - KTO/GKD：高级训练算法
    2. 多模态模型训练：处理包含图像、视频、音频的多模态输入
    3. 推理部署：
    - vLLM高性能推理
    - LMDeploy（PyTorch/TurboMind后端）
    - SGLang推理框架
    - 标准transformers推理
    4. 下游任务适配：
    - 文本分类（Sequence Classification）
    - 文本向量化（Embedding）
    - 文档排序（Reranker）
    - 过程奖励模型（PRM）
    5. Agent/工具调用：支持function calling和tool使用的格式化

    使用示例：
    >>> # 示例1：基础训练场景 - 初始化模板并编码对话数据
    >>> from swift.llm.template import get_template
    >>> from transformers import AutoTokenizer
    >>> 
    >>> # 加载tokenizer和模板
    >>> tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
    >>> template = get_template('qwen', tokenizer)
    >>> template.mode = 'train'  # 设置为训练模式
    >>> 
    >>> # 编码对话数据
    >>> inputs = StdTemplateInputs(messages=[
    ...     {"role": "user", "content": "你好，请介绍一下自己"},
    ...     {"role": "assistant", "content": "我是一个AI助手，很高兴为您服务"}
    ... ])
    >>> encoded = template.encode(inputs)
    >>> # encoded包含: input_ids, labels, loss_scale等训练所需字段
    >>> 
    >>> # 示例2：多模态场景 - 处理包含图像的输入
    >>> inputs = StdTemplateInputs(
    ...     messages=[
    ...         {"role": "user", "content": "<image>这张图片里有什么？"},
    ...         {"role": "assistant", "content": "图片中是一只可爱的猫咪"}
    ...     ],
    ...     images=["path/to/image.jpg"]  # 支持路径、PIL.Image、bytes等多种格式
    ... )
    >>> encoded = template.encode(inputs)
    >>> 
    >>> # 示例3：推理场景 - 解码生成的token序列
    >>> template.mode = 'pt'  # 设置为推理模式
    >>> generate_ids = model.generate(input_ids, max_new_tokens=100)
    >>> response = template.decode(generate_ids[0], input_ids[0])
    >>> print(response)  # 输出：模型生成的回复文本
    >>> 
    >>> # 示例4：RLHF训练 - 编码chosen/rejected对
    >>> template.mode = 'rlhf'
    >>> inputs = StdTemplateInputs(
    ...     messages=[{"role": "user", "content": "写一首诗"}],
    ...     chosen={"role": "assistant", "content": "春眠不觉晓，处处闻啼鸟"},
    ...     rejected={"role": "assistant", "content": "这是一首糟糕的诗"}
    ... )
    >>> encoded = template.encode(inputs)
    >>> # encoded包含: chosen_input_ids, rejected_input_ids等RLHF所需字段
    >>> 
    >>> # 示例5：批量数据整理（Data Collator）
    >>> batch_encoded = [template.encode(inp) for inp in inputs_list]
    >>> batch = template.data_collator(batch_encoded)
    >>> # batch包含: input_ids (padded), attention_mask, labels等，可直接送入模型
    """
    # ===== 类属性：特殊标签和占位符配置 =====
    special_tokens = ['<image>', '<video>', '<audio>', '<bbox>', '<ref-object>', '<cot-process>', '<start-image>']  # 特殊标签列表：用于标识多模态内容和特殊功能区域
    special_keys = ['images', 'videos', 'audios', 'objects']  # 特殊输入键：对应多模态数据的键名

    image_placeholder = ['<image>']  # 图像占位符：在文本中表示图像位置的标签
    video_placeholder = ['<video>']  # 视频占位符：在文本中表示视频位置的标签
    audio_placeholder = ['<audio>']  # 音频占位符：在文本中表示音频位置的标签
    cot_process_placeholder = ['ки']  # CoT过程占位符：用于思维链（Chain-of-Thought）过程标记
    placeholder_tokens = []  # 占位符token列表：用于更清晰地打印，存储特殊占位符的token ID（For clearer printing）
    load_images = True  # 是否加载图像：控制是否实际加载和处理图像数据
    skip_prompt = True  # 是否跳过prompt：在计算loss时是否跳过用户输入部分（仅对assistant回复计算loss）
    use_model = False  # 是否使用模型：某些模板需要模型参与编码过程（如视觉编码器）
    norm_bbox = 'norm1000'  # bbox归一化方式：'norm1000'表示归一化到0-1000范围，'none'表示不归一化

    is_encoder_decoder = False  # 是否为编码器-解码器架构：如T5、BART等模型为True，GPT等为False

    def __init__(
        self,
        processor: Optional[Processor],  # 处理器对象：包含tokenizer、image_processor等，用于数据预处理
        template_meta: 'TemplateMeta',  # 模板元数据：定义对话格式、系统提示、特殊token等模板配置
        default_system: Optional[str] = None,  # 默认系统提示：覆盖template_meta中的默认系统提示
        max_length: Optional[int] = None,  # 最大序列长度：超过此长度将根据truncation_strategy处理
        *,  # 以下参数必须使用关键字传递
        truncation_strategy: Literal['raise', 'left', 'right'] = 'raise',  # 截断策略：'raise'抛异常/'left'左截断/'right'右截断
        max_pixels: Optional[int] = None,  # 图像最大像素数：用于缩放图像以减少显存占用，None表示不限制（如512*512）
        agent_template: Optional[str] = None,  # Agent模板名称：用于function calling和tool使用的格式化模板
        norm_bbox: Literal['norm1000', 'none', None] = None,  # bbox归一化方式：'norm1000'归一化到0-1000/'none'不归一化/None使用类默认值
        use_chat_template: bool = True,  # 是否使用对话模板：False时转换为生成模板（不使用角色标签）
        remove_unused_columns: bool = True,  # 是否移除未使用的列：数据预处理时移除不需要的字段
        # 以下参数仅用于训练（only for train）
        padding_free: bool = False,  # 是否启用padding-free模式：通过packing减少padding，提高训练效率
        padding_side: Literal['left', 'right'] = 'right',  # padding方向：训练batch_size>=2时的padding方向，'right'右padding/'left'左padding
        loss_scale: str = 'default',  # loss权重函数：为不同token段设置不同的loss权重，'default'为默认策略
        sequence_parallel_size: int = 1,  # 序列并行大小：用于长序列训练的并行策略，1表示不使用序列并行
        # 以下参数用于推理/部署（infer/deploy）
        response_prefix: Optional[str] = None,  # 响应前缀：覆盖template_meta中的响应前缀，用于生成时的引导
        template_backend: Literal['swift', 'jinja'] = 'swift',  # 模板后端：'swift'使用自定义模板系统/'jinja'使用tokenizer.apply_chat_template
    ) -> None:
        """函数功能：
        初始化Template实例，配置模板的各项参数，包括处理器、模板元数据、截断策略、
        多模态配置、训练参数、推理参数等。这是Template类的核心初始化方法。

        参数：
        - processor (Optional[Processor]): 处理器对象，包含tokenizer和image_processor等，
            用于文本和多模态数据的预处理。可以为None，稍后通过init_processor设置。
        - template_meta (TemplateMeta): 模板元数据对象，定义了对话格式、系统提示、
            特殊token、前缀后缀等模板的核心配置信息。
        - default_system (Optional[str]): 默认系统提示文本，如果提供则覆盖template_meta
            中定义的默认系统提示。用于自定义模型的system角色内容。
        - max_length (Optional[int]): 序列的最大长度限制。超过此长度的序列将根据
            truncation_strategy进行处理。None表示不限制长度。
        - truncation_strategy (Literal['raise', 'left', 'right']): 截断策略。
            'raise'：抛出MaxLengthError异常；
            'left'：从左侧（序列开头）截断；
            'right'：从右侧（序列末尾）截断。
        - max_pixels (Optional[int]): 图像的最大像素数限制（高×宽），用于自动缩放图像
            以减少显存占用。例如512*512=262144。None表示不限制图像尺寸。
        - agent_template (Optional[str]): Agent模板的名称，用于function calling和
            tool使用场景的消息格式化。None则使用template_meta中的agent_template。
        - norm_bbox (Literal['norm1000', 'none', None]): bbox坐标的归一化方式。
            'norm1000'：归一化到0-1000整数范围；
            'none'：不进行归一化；
            None：使用类属性的默认值。
        - use_chat_template (bool): 是否使用对话模板。True时使用完整的角色标签
            （user/assistant等），False时转换为生成模板（仅保留内容）。
        - remove_unused_columns (bool): 是否在数据预处理时移除未使用的列，减少内存占用。
        - padding_free (bool): 是否启用padding-free模式。启用时会通过packing技术
            将多个短序列拼接为长序列，减少padding浪费，提高训练效率。
        - padding_side (Literal['left', 'right']): padding的方向。训练时batch_size>=2
            需要padding对齐，'right'表示在序列右侧padding，'left'表示在左侧padding。
        - loss_scale (str): loss权重函数的名称。用于为不同的token段设置不同的loss权重，
            如可以降低prompt部分的权重，增加response部分的权重。
        - sequence_parallel_size (int): 序列并行的大小。用于超长序列训练，将序列切分
            到多个设备上并行计算。1表示不使用序列并行。
        - response_prefix (Optional[str]): 响应的前缀文本。如果提供则覆盖template_meta
            中的response_prefix，用于在生成时引导模型的输出格式。
        - template_backend (Literal['swift', 'jinja']): 模板后端的选择。
            'swift'：使用ms-swift自定义的模板系统，功能更丰富；
            'jinja'：使用tokenizer.apply_chat_template，兼容HuggingFace标准。

        返回值：
        - None
        使用示例：
        >>> # 示例1：基础初始化
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
        >>> processor = Processor(tokenizer=tokenizer)
        >>> template_meta = TemplateMeta(...)
        >>> template = Template(processor, template_meta, max_length=2048)
        >>> 
        >>> # 示例2：训练场景配置
        >>> template = Template(
        ...     processor, template_meta,
        ...     max_length=2048,
        ...     truncation_strategy='left',  # 左截断保留最新对话
        ...     padding_side='right',  # 右padding
        ...     loss_scale='default',  # 默认loss权重
        ...     padding_free=True  # 启用packing提高效率
        ... )
        >>> 
        >>> # 示例3：多模态配置
        >>> template = Template(
        ...     processor, template_meta,
        ...     max_pixels=512*512,  # 限制图像为512x512
        ...     norm_bbox='norm1000'  # bbox归一化到0-1000
        ... )
        >>> 
        >>> # 示例4：Agent/工具调用配置
        >>> template = Template(
        ...     processor, template_meta,
        ...     agent_template='qwen',  # 使用Qwen的工具调用格式
        ...     default_system='你是一个具有工具调用能力的AI助手'
        ... )
        """
        # ===== 导入依赖模块 =====
        from .template_meta import TemplateMeta  # 导入TemplateMeta类：用于类型检查和实例化
        from swift.plugin import agent_templates, loss_scale_map  # 导入agent模板映射和loss权重函数映射
        
        # ===== 初始化内部状态标志 =====
        self._processor_inited = False  # Processor初始化标志：标记processor是否已完成初始化
        self._version = 'v2'  # 模板版本号：用于避免load_from_cache_file缓存导致的兼容性问题（Avoid compatibility issues caused by load_from_cache_file caching.）
        self.max_length = max_length  # 保存最大长度配置
        self.model = None  # 模型引用：某些模板需要模型参与编码（如视觉编码器），初始化为None

        # ===== 处理模板元数据 =====
        if not use_chat_template:  # 若不使用对话模板
            template_meta = template_meta.to_generate_template_meta()  # 转换为生成模板：移除角色标签，仅保留内容
        else:  # 若使用对话模板
            template_meta = deepcopy(template_meta)  # 深拷贝模板元数据：避免修改原始对象
        
        # ===== 配置系统提示和响应前缀 =====
        template_meta.check_system(default_system)  # 检查系统提示的有效性：验证default_system是否合法（if default_system is None. not change self.default_system）
        if default_system is not None:  # 若提供了自定义系统提示
            template_meta.default_system = default_system  # 覆盖模板元数据中的默认系统提示
        if response_prefix is not None:  # 若提供了自定义响应前缀
            template_meta.response_prefix = response_prefix  # 覆盖模板元数据中的响应前缀

        # ===== 保存核心配置属性 =====
        self.template_meta: TemplateMeta = template_meta  # 保存模板元数据对象
        self.use_chat_template = use_chat_template  # 保存是否使用对话模板的标志
        self.remove_unused_columns = remove_unused_columns  # 保存是否移除未使用列的标志
        self.template_backend = template_backend  # 保存模板后端选择（swift或jinja）
        self.max_length = max_length  # 再次保存最大长度（确保覆盖）
        self.truncation_strategy = truncation_strategy  # 保存截断策略
        self.loss_scale = loss_scale_map[loss_scale]()  # 从映射中获取loss权重函数实例并保存
        self.max_pixels = max_pixels  # 保存图像最大像素数限制
        self.padding_side = padding_side  # 保存padding方向
        self.sequence_parallel_size = sequence_parallel_size  # 保存序列并行大小
        self.padding_free = padding_free  # 保存是否启用padding-free模式
        
        # ===== 配置Agent模板 =====
        agent_template = agent_template or template_meta.agent_template  # 使用传入的agent_template，若无则使用template_meta中的默认值
        self._agent_template = agent_template  # 保存agent模板名称（用于内部引用）
        self.agent_template = agent_templates[agent_template]()  # 从映射中获取agent模板实例并保存

        # ===== 配置bbox归一化方式 =====
        self.norm_bbox = norm_bbox or self.norm_bbox  # 使用传入的norm_bbox，若无则使用类属性的默认值
        
        # ===== 特殊处理：编码器-解码器架构 =====
        if self.is_encoder_decoder:  # 若为编码器-解码器架构（如T5、BART）
            self.skip_prompt = False  # 不跳过prompt：编码器-解码器模型需要完整的输入序列
        
        # ===== 初始化运行模式和任务类型 =====
        self.mode: Literal['pt', 'vllm', 'lmdeploy', 'sglang',  # 推理模式（infer）
                           'train', 'rlhf', 'kto', 'gkd'] = 'pt'  # 训练模式（train），默认为'pt'（PyTorch推理）
        self.task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'prm', 'reranker',
                                'generative_reranker'] = 'causal_lm'  # 任务类型：默认为因果语言模型
        
        # ===== 初始化其他内部状态 =====
        self._packing = self.padding_free  # Packing标志：与padding_free保持一致
        self.use_megatron = False  # 是否使用Megatron格式：用于生成4D causal attention mask
        self._handles = []  # Hook句柄列表：用于存储注册的forward hook，便于后续移除
        self._deepspeed_initialize = None  # DeepSpeed初始化函数：用于DeepSpeed Zero-3场景

        # ===== 初始化Processor（如果提供） =====
        if processor is not None:  # 若在初始化时提供了processor
            self.init_processor(processor)  # 立即初始化processor，完成tokenizer等的设置

    def init_processor(self, processor: Processor) -> None:
        """函数功能：
        初始化Processor对象，将tokenizer、image_processor、model_info等信息绑定到
        Template实例。这个方法会从processor中提取模型配置、任务类型等关键信息，
        并完成占位符token的ID转换和模板元数据的初始化。

        参数：
        - processor (Processor): 处理器对象，包含tokenizer、image_processor、
            model_info、model_meta等信息

        返回值：
        - None

        使用示例：
        >>> # 示例1：在__init__后初始化processor
        >>> template = Template(None, template_meta)  # processor暂时为None
        >>> processor = Processor(tokenizer=tokenizer, model_info=model_info)
        >>> template.init_processor(processor)  # 后续初始化processor
        >>> 
        >>> # 示例2：重复调用会被忽略
        >>> template.init_processor(processor)  # 第一次初始化
        >>> template.init_processor(processor)  # 第二次调用会直接返回，不重复初始化
        """
        if processor is None or self._processor_inited:  # 若processor为None或已经初始化过
            return  # 直接返回，避免重复初始化
        self._processor_inited = True  # 标记processor已初始化，防止后续重复调用
        
        # ===== 保存processor及相关信息 =====
        self.processor = processor  # 保存processor对象，提供tokenizer等访问入口
        self.model_info = processor.model_info  # 保存模型信息对象：包含max_model_len、模型架构等
        self.config = self.model_info.config  # 保存模型配置对象：Transformers的PretrainedConfig
        self.task_type = self.model_info.task_type  # 保存任务类型：从model_info提取任务类型（可能覆盖__init__中的默认值）

        # ===== 处理模型元数据和最大长度 =====
        self.model_meta = processor.model_meta  # 保存模型元数据：包含is_multimodal等模型特性信息
        if self.max_length is None:  # 若在__init__时未指定max_length
            self.max_length = self.model_info.max_model_len  # 使用模型的默认最大长度（从config或model_info获取）
        
        # ===== 打印配置信息日志 =====
        # NOTE: repr(obj) 是 Python 的内置函数，用于获取对象的"官方字符串表示形式"，生成一个尽可能准确、可用于调试的字符串
        logger.info(f'default_system: {repr(self.template_meta.default_system)}')  # 记录默认系统提示
        logger.info(f'max_length: {self.max_length}')  # 记录最大序列长度
        logger.info(f'response_prefix: {repr(self.template_meta.response_prefix)}')  # 记录响应前缀
        logger.info(f'agent_template: {self._agent_template}')  # 记录agent模板名称
        if self.model_meta.is_multimodal:  # 若为多模态模型
            logger.info(f'norm_bbox: {self.norm_bbox}')  # 记录bbox归一化方式

        # ===== 获取tokenizer并转换占位符token =====
        tokenizer = self.tokenizer  # 获取tokenizer对象（通过ProcessorMixin的属性访问）

        # ===== 转换占位符token为ID =====
        for i, token in enumerate(self.placeholder_tokens):  # 遍历所有占位符token
            if isinstance(token, str):  # 若token是字符串形式（尚未转换为ID）
                self.placeholder_tokens[i] = tokenizer.convert_tokens_to_ids(token)  # 将字符串token转换为对应的token ID
        
        # ===== 初始化模板元数据 =====
        self.template_meta.init(tokenizer)  # 调用template_meta的init方法，完成特殊token ID的设置等初始化工作
    @staticmethod  # 静态方法：不依赖实例状态
    def _load_image(image, load_images: bool):
        """函数功能：
        加载图像数据，支持多种输入格式（路径、bytes、PIL.Image等）。
        根据load_images参数决定是否立即加载图像为PIL.Image对象，或仅保留路径。

        参数：
        - image: 图像输入，支持多种格式：
            * str: 文件路径或URL
            * dict: {'path': str, 'bytes': bytes} 字典
            * PIL.Image: 已加载的图像对象
            * bytes: 图像二进制数据
        - load_images (bool): 是否立即加载图像为PIL.Image对象。
            True: 将所有格式转换为PIL.Image；
            False: 尽可能保留路径字符串（用于vLLM等框架的延迟加载）

        返回值：
        - 加载后的图像（PIL.Image或str路径）
        """
        if load_images:  # 若需要立即加载图像
            if isinstance(image, dict) and 'bytes' in image:  # 若为字典格式且包含bytes字段
                image = image['bytes'] or image['path']  # 优先使用bytes，若无则使用path
            image = load_image(image)  # 调用load_image工具函数：将各种格式转换为PIL.Image对象
        else:  # 若不需要立即加载（延迟加载模式，用于vLLM等）
            if isinstance(image, dict):  # 若为字典格式
                path = image['path']  # 提取path字段
                if path and (path.startswith('http') or os.path.exists(path)):  # 若path有效（URL或本地文件存在）
                    image = path  # 保留路径字符串，不加载图像（延迟加载）
                else:  # 若path无效或不存在
                    image = load_image(image['bytes'])  # 从bytes加载图像
            elif not isinstance(image, str):  # 若不是字符串（可能是PIL.Image或bytes）
                image = load_image(image)  # 加载为PIL.Image对象
        return image  # 返回处理后的图像

    @staticmethod
    def _get_height_width(inputs: StdTemplateInputs) -> None:
        """函数功能：
        从inputs.images中提取所有图像的宽度和高度，并保存到inputs.objects字典中。
        这些尺寸信息用于后续的bbox归一化等操作。

        参数：
        - inputs (StdTemplateInputs): 标准输入对象，必须包含已加载的images列表

        返回值：
        - None（原地修改inputs.objects，添加'width'和'height'字段）
        """
        width = []  # 初始化宽度列表
        height = []  # 初始化高度列表
        for image in inputs.images:  # 遍历所有图像（PIL.Image对象）
            width.append(image.width)  # 提取图像宽度并添加到列表
            height.append(image.height)  # 提取图像高度并添加到列表
        inputs.objects['width'] = width  # 将宽度列表保存到objects字典
        inputs.objects['height'] = height  # 将高度列表保存到objects字典


    def normalize_bbox(self, inputs: StdTemplateInputs) -> None:
        """函数功能：
        归一化bbox坐标到指定的格式。支持将真实像素坐标或norm1坐标转换为：
        - 'norm1000': 归一化到[0, 1000]整数范围
        - 'none': 保持原始图像像素坐标

        参数：
        - inputs (StdTemplateInputs): 标准输入对象，必须包含：
            * objects['bbox']: bbox列表，每个bbox为[x1, y1, x2, y2, ...]格式
            * objects['width']: 图像宽度列表
            * objects['height']: 图像高度列表
            * objects['bbox_type']: 可选，bbox类型（'real'真实像素/'norm1'归一化到0-1）
            * objects['image_id']: 可选，每个bbox对应的图像索引

        返回值：
        - None（原地修改inputs.objects['bbox']中的坐标值）
        """
        objects = inputs.objects  # 获取objects字典引用
        bbox_list = objects['bbox']  # 获取bbox列表
        width_list = objects['width']  # 获取图像宽度列表
        height_list = objects['height']  # 获取图像高度列表
        bbox_type = objects.pop('bbox_type', None) or 'real'  # 获取并移除bbox_type，默认为'real'（真实像素坐标）
        image_id_list = objects.pop('image_id', None) or []  # 获取并移除image_id列表，默认为空列表
        image_id_list += [0] * (len(bbox_list) - len(image_id_list))  # 若image_id不足，用0填充（默认关联第一张图像）

        for bbox, image_id in zip(bbox_list, image_id_list):  # 遍历每个bbox及其对应的图像ID
            if bbox_type == 'norm1':  # 若bbox已归一化到[0, 1]范围
                width, height = 1, 1  # 原始尺寸设为1（归一化坐标）
            else:  # 若bbox为真实像素坐标
                width, height = width_list[image_id], height_list[image_id]  # 获取对应图像的真实宽高
            
            for i, (x, y) in enumerate(zip(bbox[::2], bbox[1::2])):  # 遍历bbox中的每个坐标点（x, y对）
                # bbox格式为[x1, y1, x2, y2, ...]，bbox[::2]提取所有x坐标，bbox[1::2]提取所有y坐标
                if self.norm_bbox == 'norm1000':  # 若目标归一化方式为'norm1000'
                    norm_width, norm_height = 1000, 1000  # 目标归一化范围为[0, 1000]
                elif self.norm_bbox == 'none':  # 若目标归一化方式为'none'（保持像素坐标）
                    image = inputs.images[image_id]  # 获取对应的图像对象
                    norm_width, norm_height = image.width, image.height  # 目标范围为图像的真实宽高
                bbox[2 * i] = int(round(x / width * norm_width))  # 归一化x坐标：从原始范围转换到目标范围
                bbox[2 * i + 1] = int(round(y / height * norm_height))  # 归一化y坐标：从原始范围转换到目标范围

    def _preprocess_function_call(self, inputs: StdTemplateInputs) -> None:
        """函数功能：
        预处理工具调用（function calling）相关的输入数据。主要完成两个任务：
        1. 解析和格式化tools定义（将JSON字符串转换为字典，并用agent模板包装）
        2. 将连续的tool_call消息合并为单个assistant消息（格式化工具调用内容）

        参数：
        - inputs (StdTemplateInputs): 标准输入对象，可能包含tools和tool_call消息

        返回值：
        - None（原地修改inputs.tools和inputs.messages）
        """
        agent_template = self.agent_template  # 获取agent模板实例
        agent_template.template_meta = self.template_meta  # 设置agent模板的template_meta（for hermes等特定模型）
        
        # ===== 步骤1：处理tools定义 =====
        if inputs.tools:  # 若提供了tools定义
            if isinstance(inputs.tools, str):  # 若tools为JSON字符串
                inputs.tools = agent_template._parse_json(inputs.tools)  # 解析JSON字符串为Python对象
                if not isinstance(inputs.tools, (list, tuple)):  # 若解析结果不是列表或元组（单个工具）
                    inputs.tools = [inputs.tools]  # 转换为列表格式
            elif isinstance(inputs.tools, (list, tuple)):  # 若tools已经是列表或元组
                inputs.tools = [agent_template._parse_json(tool) for tool in inputs.tools]  # 解析列表中的每个tool（可能是JSON字符串）
            else:  # 若tools格式不支持
                raise ValueError(f'inputs.tools: {inputs.tools}')  # 抛出异常
            for i, tool in enumerate(inputs.tools):  # 遍历所有工具定义
                inputs.tools[i] = agent_template.wrap_tool(tool)  # 使用agent模板包装工具定义（添加特定格式）

        # ===== 步骤2：合并连续的tool_call消息 =====
        i = 0  # 初始化消息索引
        messages = inputs.messages  # 获取消息列表引用
        while i < len(messages):  # 遍历所有消息
            if messages[i]['role'] == 'tool_call':  # 若当前消息为tool_call角色
                i_start = i  # 记录tool_call序列的起始位置
                while i + 1 < len(messages) and messages[i + 1]['role'] == 'tool_call':  # 查找连续的tool_call消息
                    i += 1  # 移动到下一个tool_call
                tool_content = self.agent_template._format_tool_calls(messages[i_start:i + 1])  # 格式化所有连续的tool_call为单个内容字符串
                messages[i_start:i + 1] = [{'role': 'assistant', 'content': tool_content}]  # 用单个assistant消息替换所有tool_call消息
                i = i_start + 1  # 移动到合并后消息的下一个位置
            else:  # 若当前消息不是tool_call
                i += 1  # 移动到下一个消息

    def _preprocess_inputs(
        self,
        inputs: StdTemplateInputs,
    ) -> None:
        """函数功能：
        预处理输入数据，这是encode之前的核心准备步骤。主要完成：
        1. 处理工具调用（function calling）
        2. 替换多模态标签（<image>、<start-image>等）
        3. 加载和预处理图像数据（支持images和rejected_images）
        4. 提取图像尺寸信息（用于bbox归一化）
        5. 缩放图像（根据max_pixels限制）

        参数：
        - inputs (StdTemplateInputs): 标准输入对象，包含messages、images、tools等字段

        返回值：
        - None（原地修改inputs对象）
        """
        # ===== 步骤1：预处理工具调用 =====
        self._preprocess_function_call(inputs)  # 处理tools定义和tool_call消息
        
        # ===== 步骤2：替换多模态标签（仅限多模态模型） =====
        if self.model_meta.is_multimodal:  # 若为多模态模型
            self._replace_image_tags(inputs)  # 替换<image>标签为模型特定的图像占位符
            self._replace_start_image_tags(inputs)  # 替换<start-image>标签（某些模型需要）
        # ===== 步骤3：加载和预处理图像数据 =====
        for img_field in ['images', 'rejected_images']:  # 遍历图像字段（包括RLHF的rejected_images）
            images = getattr(inputs, img_field, None)  # 获取图像列表
            if not images:  # 若该字段为空
                continue  # 跳过处理
            
            # 确定是否需要立即加载图像
            load_images = self.load_images or self.mode in {'vllm', 'lmdeploy'}  # 若配置要求加载或使用vLLM/LMDeploy（需要路径）
            load_images_origin = load_images  # 保存原始的load_images标志
            if self.max_pixels is not None or inputs.objects:  # 若需要缩放图像或处理bbox
                load_images = True  # 必须加载图像（需要访问PIL.Image属性）
            
            # 加载所有图像
            if images:  # 若图像列表非空
                for i, image in enumerate(images):  # 遍历每个图像
                    images[i] = self._load_image(image, load_images)  # 加载图像（PIL.Image或保留路径）
            
            # 提取图像尺寸信息
            if inputs.objects:  # 若需要处理objects（bbox等）
                self._get_height_width(inputs)  # 提取所有图像的宽高并保存到inputs.objects
            
            # 缩放图像（减少显存占用）
            if self.max_pixels is not None and images:  # 若设置了max_pixels限制且有图像
                images = [rescale_image(img, self.max_pixels) for img in images]  # 缩放所有图像到max_pixels限制
            # 处理特定模型的图像格式要求（针对 PyTorch 和 Qwen-VL 模型）
            if images and not load_images_origin:  # 如果有图像且原始配置不需要加载（但因max_pixels/objects而被强制加载）
                # 遍历所有图像，将 PIL.Image 对象转换回文件路径格式
                for i, image in enumerate(images):
                    if isinstance(image, Image.Image):  # 如果当前图像是 PIL.Image 对象
                        # 将 PIL.Image 保存为临时文件并返回路径
                        # 这是为了修复某些模型（如 PyTorch 原生模型、Qwen-VL）对图像格式的特殊要求
                        images[i] = self._save_pil_image(image)
            
            # 将处理后的图像列表更新回 inputs 对象的对应字段
            setattr(inputs, img_field, images)

        # ===== 步骤4：加载和预处理音频数据（仅 vLLM 模式） =====
        if self.mode == 'vllm' and inputs.audios:  # 如果使用 vLLM 推理引擎且包含音频数据
            # 从环境变量获取音频采样率配置
            sampling_rate = get_env_args('sampling_rate', int, None)
            # 批量加载音频文件，返回音频数组和采样率
            inputs.audios = load_batch(
                inputs.audios, load_func=partial(load_audio, sampling_rate=sampling_rate, return_sr=True))

    @staticmethod
    def _replace_image_tags(inputs: StdTemplateInputs):
        """
        功能：
        该方法用于向后兼容旧版本的图像标签格式。
        具体地，解析消息中的 <img>路径</img> 标签，提取其中的图像路径并添加到 inputs.images 列表，同时将标签替换为标准的 <image> 占位符。
        
        参数：
            inputs (StdTemplateInputs): 标准模板输入对象，包含 messages 和 images 字段
        
        返回值：
            无（原地修改 inputs 对象的 messages 和 images 字段）
        
        使用示例：
            # 示例1：解析 <img> 标签
            inputs = StdTemplateInputs(
                messages=[{
                    "role": "user",
                    "content": "Look at <img>/path/to/image.jpg</img> and tell me."
                }],
                images=[]
            )
            Template._replace_image_tags(inputs)
            # 结果：
            # inputs.messages[0]['content'] == "Look at <image> and tell me."
            # inputs.images == ['/path/to/image.jpg']
            
            # 示例2：多个图像标签
            inputs = StdTemplateInputs(
                messages=[{
                    "role": "user",
                    "content": "<img>cat.jpg</img> vs <img>dog.jpg</img>"
                }],
                images=[]
            )
            Template._replace_image_tags(inputs)
            # 结果：
            # inputs.messages[0]['content'] == "<image> vs <image>"
            # inputs.images == ['cat.jpg', 'dog.jpg']
            
            # 示例3：images 已存在，跳过处理
            inputs = StdTemplateInputs(
                messages=[{"role": "user", "content": "<img>test.jpg</img>"}],
                images=["existing.jpg"]
            )
            Template._replace_image_tags(inputs)
            # 结果：不做任何修改，直接返回
            
            # 示例4：无效的图像路径
            inputs = StdTemplateInputs(
                messages=[{
                    "role": "user",
                    "content": "<img>nonexistent.jpg</img>"
                }],
                images=[]
            )
            Template._replace_image_tags(inputs)
            # 结果：警告日志输出，但仍替换标签
            # inputs.messages[0]['content'] == "<image>"
            # inputs.images == []  # 无效路径不添加
        """
        # 兼容性检查：如果 images 字段已有内容，说明图像已通过标准方式提供，无需处理
        if inputs.images:
            return
        
        # 初始化图像路径列表
        images = []
        
        # 定义正则表达式模式，匹配 <img>路径</img> 格式
        # (.+?) 表示非贪婪匹配任意字符（图像路径）
        pattern = r'<img>(.+?)</img>'
        
        # 遍历所有消息
        for message in inputs.messages:
            # 获取消息内容
            content = message['content']
            
            # 跳过非字符串内容（如已编码的 token ID 列表）
            if not isinstance(content, str):
                continue
            
            # 使用正则表达式查找所有 <img> 标签中的路径
            for image in re.findall(pattern, content):
                # 验证路径：仅支持本地文件路径
                if os.path.isfile(image):  # 检查文件是否存在
                    # 将有效的图像路径添加到列表
                    images.append(image)
                else:
                    # 记录警告：图像路径无效或文件不存在
                    # warning_once 确保相同的警告只输出一次
                    logger.warning_once(f'Failed to parse image path: `{content}`.', hash_id='<img></img>')
            
            # 将消息中的所有 <img>...</img> 标签替换为标准的 <image> 占位符
            message['content'] = re.sub(pattern, '<image>', content)
        
        # 将提取的图像路径列表更新到 inputs 对象
        inputs.images = images

    @staticmethod
    def _replace_start_image_tags(inputs: StdTemplateInputs):
        """
        功能：
            该方法用于支持图像生成任务的兼容性处理，检测并移除消息末尾的 <start-image> 标签，设置图像生成模式。
            具体地，检查最后一条用户消息是否以 <start-image> 标签结尾。如果是，
            则移除该标签并将 inputs.generate_mode 设置为 True，表示这是一个图像生成请求。
        
        参数：
            inputs (StdTemplateInputs): 标准模板输入对象，包含 messages 字段
        
        返回值：
            无（原地修改 inputs 对象，设置 generate_mode 属性并可能修改最后一条消息的内容）
        
        使用示例：
            # 示例1：图像生成模式
            inputs = StdTemplateInputs(
                messages=[{
                    "role": "user",
                    "content": "Generate a beautiful sunset<start-image>"
                }]
            )
            Template._replace_start_image_tags(inputs)
            # 结果：
            # inputs.messages[-1]['content'] == "Generate a beautiful sunset"
            # inputs.generate_mode == True
            
            # 示例2：普通对话（非生成模式）
            inputs = StdTemplateInputs(
                messages=[{
                    "role": "user",
                    "content": "What is AI?"
                }]
            )
            Template._replace_start_image_tags(inputs)
            # 结果：
            # inputs.messages[-1]['content'] == "What is AI?"  # 保持不变
            # inputs.generate_mode == False
            
            # 示例3：最后一条消息是 assistant（不处理）
            inputs = StdTemplateInputs(
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!<start-image>"}
                ]
            )
            Template._replace_start_image_tags(inputs)
            # 结果：
            # inputs.messages[-1]['content'] == "Hi there!<start-image>"  # 保持不变
            # inputs.generate_mode == False  # 不启用生成模式
            
            # 示例4：标签在中间位置（不处理）
            inputs = StdTemplateInputs(
                messages=[{
                    "role": "user",
                    "content": "<start-image>Generate image"
                }]
            )
            Template._replace_start_image_tags(inputs)
            # 结果：
            # inputs.messages[-1]['content'] == "<start-image>Generate image"  # 保持不变
            # inputs.generate_mode == False  # 标签必须在末尾
        """
        # 初始化生成模式标志为 False（默认为非生成模式）
        generate_mode = False
        
        # 获取消息列表中的最后一条消息
        message = inputs.messages[-1]
        
        # 获取最后一条消息的内容
        content = message['content']
        
        # 检查条件：1) 最后一条消息的角色是 user，2) 内容以 <start-image> 标签结尾
        if message['role'] == 'user' and content.endswith('<start-image>'):
            # 设置为图像生成模式
            generate_mode = True
            
            # 从消息内容中移除 <start-image> 标签
            # 使用切片操作去除末尾的标签文本
            message['content'] = message['content'][:-len('<start-image>')]
        
        # 将生成模式标志保存到 inputs 对象
        # 后续的模板处理逻辑可以根据此标志调整行为
        inputs.generate_mode = generate_mode
    @staticmethod
    def _extend_tokens(
            input_ids: List[int], labels: Optional[List[int]], loss_scale: Optional[List[float]],
            replace_idx_list: List[int],
            get_new_tokens: Callable[[int], List[int]]) -> Tuple[List[int], Optional[List[int]], Optional[List[float]]]:
        """
        功能：
            在指定位置将单个 token 扩展为多个 token 序列。具体地，
            根据 replace_idx_list 中的索引位置，将 input_ids 中对应位置的单个 token 替换为
            由 get_new_tokens 函数生成的多个 token。同时同步更新 labels 和 loss_scale，
            保持它们与 input_ids 的长度一致。此方法常用于多模态模型中将占位符 token
            （如 <image>）扩展为实际的图像特征 token 序列。
        
        参数：
            input_ids (List[int]): 输入的 token ID 序列
                例如：[1, 2, 3, 4, 5] 表示原始的 token 序列
            labels (Optional[List[int]]): 标签序列，用于训练时的损失计算
                - 如果为 None，不处理
                - 扩展位置的标签会被设置为 -100（表示不计算损失）
            loss_scale (Optional[List[float]]): 损失缩放因子序列
                - 如果为 None，不处理
                - 扩展位置继承原位置的缩放因子值
            replace_idx_list (List[int]): 需要替换的索引位置列表
                例如：[1, 3] 表示在位置 1 和位置 3 进行扩展
            get_new_tokens (Callable[[int], List[int]]): 生成新 token 序列的函数
                - 接受索引 i（在 replace_idx_list 中的位置）
                - 返回用于替换的 token 序列
                - 例如：lambda i: [100, 101, 102] 返回 3 个 token
        
        返回值：
            Tuple[List[int], Optional[List[int]], Optional[List[float]]]: 返回三元组
                - 第一个元素：扩展后的 input_ids
                - 第二个元素：扩展后的 labels（如果输入为 None 则返回 None）
                - 第三个元素：扩展后的 loss_scale（如果输入为 None 则返回 None）
        
        使用示例：
            # 示例1：基本用法 - 将单个占位符扩展为多个 token
            input_ids = [1, 999, 3, 999, 5]  # 999 是占位符
            labels = [1, 2, 3, 4, 5]
            loss_scale = [1.0, 1.0, 1.0, 1.0, 1.0]
            replace_idx_list = [1, 3]  # 替换位置 1 和 3 的 token
            
            # 生成新 token：每次返回 3 个 token
            def get_new_tokens(i):
                return [100 + i*10, 101 + i*10, 102 + i*10]
            
            new_ids, new_labels, new_scale = Template._extend_tokens(
                input_ids, labels, loss_scale, replace_idx_list, get_new_tokens
            )
            # 结果：
            # new_ids == [1, 100, 101, 102, 3, 110, 111, 112, 5]
            # new_labels == [1, -100, -100, -100, 3, -100, -100, -100, 5]
            # new_scale == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            
            # 示例2：不同长度的扩展
            input_ids = [10, 20, 30]
            replace_idx_list = [1]
            get_new_tokens = lambda i: [201, 202, 203, 204]  # 扩展为 4 个 token
            
            new_ids, _, _ = Template._extend_tokens(
                input_ids, None, None, replace_idx_list, get_new_tokens
            )
            # new_ids == [10, 201, 202, 203, 204, 30]
            
            # 示例3：多模态场景 - 图像占位符扩展
            # 原始序列：[BOS, <image>, text1, text2, EOS]
            input_ids = [1, 32000, 100, 200, 2]  # 32000 是 <image> token
            labels = [1, -100, 100, 200, 2]
            replace_idx_list = [1]  # 扩展 <image> token
            
            # 图像特征 token 序列（假设图像编码为 256 个 token）
            image_tokens = list(range(50000, 50256))
            get_new_tokens = lambda i: image_tokens
            
            new_ids, new_labels, _ = Template._extend_tokens(
                input_ids, labels, None, replace_idx_list, get_new_tokens
            )
            # new_ids 长度从 5 变为 260
            # new_labels 中图像 token 位置全部为 -100（不计算损失）
            
            # 示例4：多个占位符，不同长度
            input_ids = [1, 888, 3, 999, 5]
            replace_idx_list = [1, 3]
            
            def get_new_tokens(i):
                if i == 0:
                    return [10, 11]  # 第一个位置扩展为 2 个 token
                else:
                    return [20, 21, 22]  # 第二个位置扩展为 3 个 token
            
            new_ids, _, _ = Template._extend_tokens(
                input_ids, None, None, replace_idx_list, get_new_tokens
            )
            # new_ids == [1, 10, 11, 3, 20, 21, 22, 5]
        """
        # 初始化已添加 token 的累计长度（用于调整后续索引位置）
        added_tokens_len = 0
        # 遍历所有需要替换的索引位置
        for i, idx in enumerate(replace_idx_list):
            # 调用 get_new_tokens 函数获取用于替换的新 token 序列
            new_tokens = get_new_tokens(i)
            
            # 获取新 token 序列的长度
            token_len = len(new_tokens)
            
            # 在 input_ids 中进行替换：
            # 1. 取原序列的前半部分：[:idx + added_tokens_len]
            # 2. 插入新 token 序列：new_tokens
            # 3. 连接原序列的后半部分：[added_tokens_len + idx + 1:]
            # 注意：added_tokens_len 用于补偿之前替换操作导致的序列长度变化
            input_ids = input_ids[:idx + added_tokens_len] + new_tokens + input_ids[added_tokens_len + idx + 1:]
            
            # 如果提供了 labels，同步更新
            if labels:
                # 新 token 位置的标签设置为 -100（表示不计算损失）
                # 这在多模态模型中很常见，图像 token 不参与语言建模损失
                labels = labels[:idx + added_tokens_len] + [-100] * token_len + labels[added_tokens_len + idx + 1:]
            
            # 如果提供了 loss_scale，同步更新
            if loss_scale:
                # 获取原位置的缩放因子值
                scale_idx = loss_scale[idx + added_tokens_len]
                # 新 token 位置继承原位置的缩放因子
                loss_scale = loss_scale[:idx + added_tokens_len] + [scale_idx] * token_len + loss_scale[added_tokens_len
                                                                                                        + idx + 1:]
            
            # 更新累计长度：新增的 token 数量为 (token_len - 1)
            # 减 1 是因为替换掉了原来的 1 个 token
            added_tokens_len += token_len - 1
        
        # 返回扩展后的三个序列
        return input_ids, labels, loss_scale

    def forward_context(self, model, inputs):
        return nullcontext()
    
    @staticmethod
    def get_base_model(model):
        if isinstance(model, PeftModel):
            return model.model
        else:
            return model

    def _rlhf_encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """
        功能：
            编码 RLHF（Reinforcement Learning from Human Feedback）训练数据。具体地，
            将输入数据编码为 chosen（被选择的）和 rejected（被拒绝的）两个版本，
            用于基于人类反馈的强化学习训练，如 DPO（Direct Preference Optimization）等算法。
            此方法会分别编码正向样本（chosen）和负向样本（rejected），并将它们的编码结果
            合并到一个字典中，便于后续的对比学习。
        
        参数：
            inputs (StdTemplateInputs): 标准模板输入对象，必须包含以下至少一项：
                - rejected_response: 拒绝的响应文本（用于替换最后一条消息的内容）
                - rejected_images: 拒绝的图像列表（用于多模态 RLHF）
                - margin (可选): 奖励边际值，表示 chosen 和 rejected 之间的奖励差异
        
        返回值：
            Dict[str, Any]: 包含 chosen 和 rejected 编码结果的字典，格式如下：
                {
                    'chosen_input_ids': List[int],        # chosen 版本的 token IDs
                    'chosen_labels': List[int],           # chosen 版本的标签
                    'chosen_attention_mask': List[int],   # chosen 版本的注意力掩码
                    ... (其他 chosen 相关字段)
                    'rejected_input_ids': List[int],      # rejected 版本的 token IDs
                    'rejected_labels': List[int],         # rejected 版本的标签
                    'rejected_attention_mask': List[int], # rejected 版本的注意力掩码
                    ... (其他 rejected 相关字段)
                    'margin': float (可选)                # 奖励边际值
                }
        
        使用示例：
            # 示例1：文本 RLHF 数据编码
            inputs = StdTemplateInputs(
                messages=[
                    {"role": "user", "content": "写一首关于春天的诗"},
                    {"role": "assistant", "content": "春天来了，花儿开了，鸟儿叫了。"}  # chosen
                ],
                rejected_response="春天是个季节。"  # rejected
            )
            template = Template(...)
            encoded = template._rlhf_encode(inputs)
            # encoded 包含：
            # - chosen_input_ids: 编码的 chosen 响应
            # - rejected_input_ids: 编码的 rejected 响应
            # 两者的 prompt 部分相同，只有响应部分不同
            
            # 示例2：带边际值的 RLHF 数据
            inputs = StdTemplateInputs(
                messages=[
                    {"role": "user", "content": "解释量子力学"},
                    {"role": "assistant", "content": "量子力学是研究微观粒子..."}
                ],
                rejected_response="量子力学很复杂。",
                margin=0.5  # 指定奖励边际
            )
            encoded = template._rlhf_encode(inputs)
            # encoded['margin'] == 0.5
            
            # 示例3：多模态 RLHF（图像）
            inputs = StdTemplateInputs(
                messages=[
                    {"role": "user", "content": "<image>描述这张图片"},
                    {"role": "assistant", "content": "这是一只可爱的猫咪"}
                ],
                images=["cat_good.jpg"],           # chosen 图像
                rejected_response="这是一只动物。",  # rejected 响应
                rejected_images=["cat_bad.jpg"]    # rejected 图像
            )
            encoded = template._rlhf_encode(inputs)
            # chosen 使用 cat_good.jpg，rejected 使用 cat_bad.jpg
            
            # 示例4：DPO 训练数据准备
            # 用于 Direct Preference Optimization
            inputs = StdTemplateInputs(
                messages=[
                    {"role": "user", "content": "请给我一些建议"},
                    {"role": "assistant", "content": "建议1：保持积极态度..."}
                ],
                rejected_response="建议：努力工作。"
            )
            encoded = template._rlhf_encode(inputs)
            # DPO 算法会使用 chosen 和 rejected 的对比来优化模型
        """
        # 提取边际值（奖励差异）
        margin = inputs.margin
        
        # 创建 chosen 和 rejected 两个版本的输入
        # chosen_inputs 直接使用原始输入（包含 chosen 响应）
        # rejected_inputs 是深拷贝，用于创建 rejected 版本
        chosen_inputs, rejected_inputs = inputs, deepcopy(inputs)

        # 验证必须提供至少一种拒绝数据（rejected_response 或 rejected_images）
        assert chosen_inputs.rejected_response or chosen_inputs.rejected_images, f'inputs: {inputs}'
        
        # 如果提供了拒绝的响应文本，替换 rejected_inputs 最后一条消息的内容
        if chosen_inputs.rejected_response:
            rejected_inputs.messages[-1]['content'] = chosen_inputs.rejected_response
        
        # 如果提供了拒绝的图像，替换 rejected_inputs 的图像列表
        if chosen_inputs.rejected_images:
            rejected_inputs.images = chosen_inputs.rejected_images
        
        # 分别编码 chosen 和 rejected 版本
        chosen_encoded = self._encode_truncated(chosen_inputs)
        rejected_encoded = self._encode_truncated(rejected_inputs)

        # 初始化结果字典
        encoded = {}
        
        # 遍历 'chosen' 和 'rejected' 两个前缀
        for prefix in ['chosen', 'rejected']:
            # 使用 locals() 动态获取对应的编码结果
            # 例如：prefix='chosen' 时，获取 chosen_encoded
            data = locals()[f'{prefix}_encoded']
            
            # 将编码结果的所有字段添加到结果字典，并加上前缀
            # 例如：'input_ids' -> 'chosen_input_ids' 或 'rejected_input_ids'
            for k, v in data.items():
                encoded[f'{prefix}_{k}'] = v
        
        # 如果提供了边际值，添加到结果字典
        if margin:
            encoded['margin'] = float(margin)
        
        # 返回包含 chosen 和 rejected 编码结果的字典
        return encoded

    def _kto_encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """
        功能：
            编码 KTO（Kahneman-Tversky Optimization）训练数据。具体地，
            KTO 是一种基于二元反馈（好/坏）的优化算法，不需要成对的偏好数据。
            此方法在 RLHF 编码的基础上，添加了二元标签（True/False）来表示响应的质量，
            从而支持 KTO 算法的训练。与传统的 RLHF 需要 chosen/rejected 对比不同，
            KTO 只需要单个样本及其质量标签。
        
        参数：
            inputs (StdTemplateInputs): 标准模板输入对象，必须包含：
                - label: 二元标签，表示响应质量
                  * True/1: 表示这是一个好的响应（desirable）
                  * False/0: 表示这是一个坏的响应（undesirable）
                - rejected_response 或 rejected_images: 拒绝的响应（继承自 RLHF）
                - margin (可选): 奖励边际值
        
        返回值：
            Dict[str, Any]: 包含 chosen、rejected 编码结果和标签的字典，格式如下：
                {
                    'chosen_input_ids': List[int],        # chosen 版本的 token IDs
                    'chosen_labels': List[int],           # chosen 版本的标签
                    'rejected_input_ids': List[int],      # rejected 版本的 token IDs
                    'rejected_labels': List[int],         # rejected 版本的标签
                    'label': bool,                        # 二元标签 (True=好, False=坏)
                    'margin': float (可选)                # 奖励边际值
                }
        
        使用示例：
            # 示例1：好的响应（正样本）
            inputs = StdTemplateInputs(
                messages=[
                    {"role": "user", "content": "解释人工智能"},
                    {"role": "assistant", "content": "人工智能是计算机科学的一个分支..."}
                ],
                rejected_response="AI就是电脑程序。",  # 较差的响应
                label=1  # 标记为好的响应
            )
            template = Template(...)
            encoded = template._kto_encode(inputs)
            # encoded['label'] == True  (表示 chosen 是好的响应)
            
            # 示例2：坏的响应（负样本）
            inputs = StdTemplateInputs(
                messages=[
                    {"role": "user", "content": "如何学习编程？"},
                    {"role": "assistant", "content": "多看书。"}  # 简单的响应
                ],
                rejected_response="学习编程需要系统地学习...",  # 更好的响应
                label=0  # 标记为坏的响应
            )
            encoded = template._kto_encode(inputs)
            # encoded['label'] == False  (表示 chosen 是坏的响应)
            
            # 示例3：KTO 与传统 RLHF 的区别
            # 传统 RLHF：需要明确的 chosen/rejected 对
            # KTO：只需要一个响应 + 质量标签
            
            # KTO 训练数据：
            inputs_good = StdTemplateInputs(
                messages=[{"role": "user", "content": "问题"}, 
                         {"role": "assistant", "content": "好答案"}],
                rejected_response="坏答案",
                label=True  # 标记好答案
            )
            
            inputs_bad = StdTemplateInputs(
                messages=[{"role": "user", "content": "问题"},
                         {"role": "assistant", "content": "坏答案"}],
                rejected_response="好答案",
                label=False  # 标记坏答案
            )
            
            # 示例4：多模态 KTO
            inputs = StdTemplateInputs(
                messages=[
                    {"role": "user", "content": "<image>描述这张图"},
                    {"role": "assistant", "content": "这是一只猫"}
                ],
                images=["cat.jpg"],
                rejected_response="这是动物。",
                rejected_images=["cat_blur.jpg"],
                label=True  # 标记为好的描述
            )
            encoded = template._kto_encode(inputs)
        """
        # 提取并保存原始标签值，然后清空 inputs.label
        # 这样做是为了避免在调用 _rlhf_encode 时 label 字段干扰处理
        label, inputs.label = inputs.label, None
        
        # 调用 RLHF 编码方法，获取 chosen 和 rejected 的编码结果
        # KTO 复用了 RLHF 的编码逻辑，因为都需要对比两个响应
        encoded = self._rlhf_encode(inputs)
        
        # 添加二元标签到编码结果
        # 将标签转换为布尔值：True 表示好的响应，False 表示坏的响应
        encoded['label'] = bool(label)
        
        # 返回包含 chosen、rejected 编码和二元标签的字典
        return encoded
    def _gkd_encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """函数功能：
        编码 GKD（Generalized Knowledge Distillation，广义知识蒸馏）训练数据。
        
        GKD 是一种知识蒸馏技术，通过让学生模型学习教师模型的输出分布来提升学生模型性能。
        关键特点：
        1. 分离 prompt 和 answer：将完整序列分为两部分
           - prompts: 用户输入部分（不含答案）
           - input_ids: 完整序列（prompt + answer）
        2. 训练流程：
           - 教师模型（通常是更大/更好的模型）和学生模型都处理相同的 input_ids
           - 计算学生模型和教师模型在 answer 部分的 logits 分布
           - 使用 KL 散度或 JSD（Jensen-Shannon Divergence）损失让学生模仿教师
           - prompts 字段用于标识哪部分是 prompt，从而只在 answer 部分计算蒸馏损失
        3. 与其他训练模式的区别：
           - 标准训练：只有 input_ids 和 labels，计算交叉熵损失
           - RLHF/KTO：需要 chosen 和 rejected 两个响应，计算偏好损失
           - GKD：需要 prompts 分隔符和完整 input_ids，计算分布匹配损失
        
        本方法的核心任务：
        - 调用 _encode_truncated 获取分段编码结果（包含 prompt_input_ids 和 answer_input_ids）
        - 提取 prompt 部分的 token IDs 并保存为 'prompts' 字段
        - 清理中间字段（prompt_*、*_answer_），只保留必要的 input_ids 和 prompts
        
        参数：
        - inputs (StdTemplateInputs): 标准模板输入对象，包含：
            * messages: 对话消息列表（必须包含用户输入和助手回复）
            * images/videos/audios: 多模态数据（可选）
            * system: 系统提示词（可选）
        
        返回值：
        - Dict[str, Any]: GKD 训练所需的编码字典，包含：
            * input_ids (List[int]): 完整的 token ID 序列（prompt + answer）
            * labels (List[int]): 标签序列（prompt 部分为 -100，answer 部分为实际 token）
            * loss_scale (Optional[List[float]]): loss 权重序列（如果配置了）
            * prompts (List[int]): 仅包含 prompt 部分的 token ID 序列
              - 用于标识 prompt 和 answer 的边界
              - GKD Trainer 使用此字段来只在 answer 部分计算蒸馏损失
            * pixel_values/image_sizes 等: 多模态数据（如果有）
            
            注意：不包含以下中间字段（已被清理）：
            - prompt_input_ids, prompt_labels, prompt_loss_scale
            - answer_input_ids, answer_labels, answer_loss_scale
        
        使用示例：
        >>> # 示例1：基础 GKD 训练数据编码
        >>> template.mode = 'gkd'
        >>> inputs = StdTemplateInputs(messages=[
        ...     {"role": "user", "content": "什么是广义知识蒸馏？"},
        ...     {"role": "assistant", "content": "GKD 是一种让小模型学习大模型输出分布的技术..."}
        ... ])
        >>> encoded = template._gkd_encode(inputs)
        >>> # encoded 结构：
        >>> # {
        >>> #     'input_ids': [101, 1234, ..., 5678, 102],  # 完整序列
        >>> #     'prompts': [101, 1234, ..., 5678],         # 只有 prompt 部分
        >>> #     'labels': [-100, -100, ..., 5678, 102],    # prompt 部分为 -100
        >>> #     'loss_scale': [1.0, 1.0, ..., 1.0]
        >>> # }
        >>> # len(encoded['prompts']) + len(answer) == len(encoded['input_ids'])
        >>> 
        >>> # 示例2：多模态 GKD（图像描述蒸馏）
        >>> inputs = StdTemplateInputs(
        ...     messages=[
        ...         {"role": "user", "content": "<image>描述这张图片"},
        ...         {"role": "assistant", "content": "这是一只可爱的小猫，坐在..."}
        ...     ],
        ...     images=["cat.jpg"]
        ... )
        >>> encoded = template._gkd_encode(inputs)
        >>> # encoded 还会包含: pixel_values, image_sizes 等
        >>> # prompts 包含图像 token 和用户问题
        >>> 
        >>> # 示例3：在 GKD Trainer 中的实际使用
        >>> # 教师模型（大模型）
        >>> teacher_outputs = teacher_model(input_ids=encoded['input_ids'])
        >>> # 学生模型（小模型）
        >>> student_outputs = student_model(input_ids=encoded['input_ids'])
        >>> 
        >>> # 计算 answer 部分的起始位置
        >>> prompt_len = len(encoded['prompts'])
        >>> 
        >>> # 只在 answer 部分计算 KL 散度损失
        >>> teacher_logits = teacher_outputs.logits[:, prompt_len:, :]
        >>> student_logits = student_outputs.logits[:, prompt_len:, :]
        >>> kl_loss = F.kl_div(
        ...     F.log_softmax(student_logits / temperature, dim=-1),
        ...     F.softmax(teacher_logits / temperature, dim=-1),
        ...     reduction='batchmean'
        ... )
        >>> 
        >>> # 示例4：与标准训练的对比
        >>> # 标准训练编码（只有 input_ids 和 labels）
        >>> template.mode = 'train'
        >>> encoded_std = template._encode_truncated(inputs)
        >>> # {'input_ids': [...], 'labels': [...], 'loss_scale': [...]}
        >>> 
        >>> # GKD 编码（额外有 prompts 字段用于分隔）
        >>> template.mode = 'gkd'
        >>> encoded_gkd = template._gkd_encode(inputs)
        >>> # {
        >>> #     'input_ids': [...],    # 与 encoded_std 相同
        >>> #     'labels': [...],       # 与 encoded_std 相同
        >>> #     'prompts': [...],      # 额外字段：标识 prompt 边界
        >>> #     'loss_scale': [...]
        >>> # }
        """
        # 调用 _encode_truncated 方法进行编码
        # 返回的 encoded 包含：input_ids（完整序列）、answer_input_ids（答案部分）、
        # prompt_input_ids（提示部分）、labels、loss_scale 等字段
        encoded = self._encode_truncated(inputs)
        
        # 提取 prompt 部分的 token IDs 并保存为 'prompts' 字段
        # 逻辑：input_ids = prompt_input_ids + answer_input_ids
        # 因此：prompts = input_ids[:-len(answer_input_ids)]
        # 使用 pop 同时移除 answer_input_ids 字段（因为后续不再需要）
        encoded['prompts'] = encoded['input_ids'][:-len(encoded.pop('answer_input_ids'))]
        
        # 清理所有中间字段，只保留必要的字段
        # 需要清理的字段包括：
        # - prompt_input_ids, prompt_labels, prompt_loss_scale（已经合并到 input_ids 等中）
        # - answer_labels, answer_loss_scale（已经合并，answer_input_ids 已在上面 pop 掉）
        for k in list(encoded.keys()):  # 使用 list() 避免迭代时修改字典大小
            # 移除所有以 'prompt_' 开头或以 '_answer_' 结尾的键
            if k.startswith('prompt_') or k.endswith('answer_'):
                encoded.pop(k, None)  # 使用 pop(k, None) 安全删除（即使键不存在也不报错）
        
        # 返回清理后的编码字典
        # 最终包含：input_ids（完整序列）、prompts（仅 prompt 部分）、labels、loss_scale 等
        return encoded

    def _embedding_encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """函数功能：
        编码 Embedding 任务的训练或推理数据，支持文本向量化（Text Embedding）任务。
        
        Embedding 任务是将文本（或多模态数据）映射到高维向量空间，使得语义相似的内容在向量空间中距离更近。
        
        核心概念 - 三元组学习（Triplet Learning）：
        1. Anchor（锚点样本）：查询文本（query），是需要获取向量表示的基准样本
        2. Positive（正样本）：与 anchor 语义相关或相似的文本（response）
        3. Negative（负样本）：与 anchor 语义不相关或不相似的文本（rejected_response）
        
        训练目标：
        - 最小化 anchor 和 positive 之间的距离
        - 最大化 anchor 和 negative 之间的距离
        - 通过对比学习（Contrastive Learning）训练模型生成有区分度的向量表示
        
        支持的损失函数：
        1. cosine_similarity: 余弦相似度损失，使用 MSE 拟合相似度标签
        2. contrastive: 对比学习损失（带 margin），标签为 0/1
        3. online_contrastive: 考虑 hard negative/positive 的对比损失
        4. infonce: InfoNCE 损失，batch 内样本两两对比，不需要显式标签
        
        数据格式示例：
        - 训练：{"query": "锚点文本", "response": "正样本文本", "rejected_response": ["负样本1", "负样本2"], "label": 0.8}
        - 推理：{"query": "查询文本"}
        
        本方法的核心任务：
        1. 训练模式（inputs.messages >= 2）：
           - 构造 anchor：使用 query 部分（messages[-2]）编码
           - 构造 positive：使用 response 部分（messages[-1]）编码
           - 构造 negative：使用 rejected_response 列表逐个编码（可选）
           - 生成标签：positive 使用 inputs.label（默认 1.0），negative 使用 0.0
        2. 推理模式（inputs.messages == 1）：
           - 只编码 anchor（query），用于获取查询文本的向量表示
           - 不生成 labels
        3. 多模态支持：
           - 通过 split_multi_medias 函数分配 images/videos/audios 到各个样本
           - 支持 <image>、<video>、<audio> 标签
        
        参数：
        - inputs (StdTemplateInputs): 标准模板输入对象，包含：
            messages (List[Dict]): 对话消息列表
              - 训练模式：至少 2 条消息，messages[-2] 为 query（user），messages[-1] 为 response（assistant）
              - 推理模式：仅 1 条消息，messages[0] 为 query
            rejected_response (Optional[List[str]]): 负样本列表（hard negatives），仅训练模式使用
            label (Optional[float]): positive 样本的标签值，范围 [0.0, 1.0]，默认 1.0
              - cosine_similarity loss: 表示相似度（0.0-1.0）
              - contrastive loss: 0 表示不相关，1 表示相关
            images/videos/audios (Optional[List]): 多模态数据（可选）
        
        返回值：
        - Dict[str, Any]: 根据模式返回不同结构的字典
        
        训练模式返回值：
        {
            # Anchor 相关字段（query 的编码）
            'anchor_input_ids': List[int],           # anchor 的 token IDs
            'anchor_labels': List[int],              # anchor 的标签（通常全为 -100）
            'anchor_loss_scale': List[float],        # anchor 的 loss 权重
            'anchor_pixel_values': ...,              # anchor 的图像数据（如果有）
            
            # Positive 相关字段（response 的编码）
            'positive_input_ids': List[int],         # positive 的 token IDs
            'positive_labels': List[int],            # positive 的标签（通常全为 -100）
            'positive_loss_scale': List[float],      # positive 的 loss 权重
            'positive_pixel_values': ...,            # positive 的图像数据（如果有）
            
            # Negative 相关字段（rejected_response 的编码，列表形式）
            'negative_input_ids': List[List[int]],   # 所有 negative 的 token IDs 列表
            'negative_labels': List[List[int]],      # 所有 negative 的标签列表
            'negative_loss_scale': List[List[float]], # 所有 negative 的 loss 权重列表
            'negative_pixel_values': List[...],      # 所有 negative 的图像数据列表（如果有）
            
            # 标签
            'labels': List[float]                    # [positive_label, negative_label_0, negative_label_1, ...]
                                                     # positive_label 为 inputs.label（默认 1.0）
                                                     # 所有 negative_label 为 0.0
        }
        
        推理模式返回值：
        {
            'input_ids': List[int],                  # query 的 token IDs
            'loss_scale': List[float],               # loss 权重（虽然推理不需要）
            'pixel_values': ...,                     # 图像数据（如果有）
            # 注意：没有 'labels' 字段
        }
        
        使用示例：
        >>> # 示例1：纯文本 Embedding 训练（带 hard negatives）
        >>> template.task_type = 'embedding'
        >>> template.mode = 'train'
        >>> inputs = StdTemplateInputs(
        ...     messages=[
        ...         {"role": "user", "content": "什么是机器学习？"},      # query (anchor)
        ...         {"role": "assistant", "content": "机器学习是人工智能的一个分支..."}  # response (positive)
        ...     ],
        ...     rejected_response=[
        ...         "今天天气真好",           # hard negative 1
        ...         "我喜欢吃披萨"            # hard negative 2
        ...     ],
        ...     label=0.9  # positive 样本的相似度标签
        ... )
        >>> encoded = template._embedding_encode(inputs)
        >>> # 返回结构：
        >>> # {
        >>> #     'anchor_input_ids': [101, 1234, ...],      # "什么是机器学习？"
        >>> #     'positive_input_ids': [101, 5678, ...],    # "机器学习是..."
        >>> #     'negative_input_ids': [                    # 2 个 hard negatives
        >>> #         [101, 9999, ...],                      # "今天天气真好"
        >>> #         [101, 8888, ...]                       # "我喜欢吃披萨"
        >>> #     ],
        >>> #     'labels': [0.9, 0.0, 0.0]  # [positive_label, neg1_label, neg2_label]
        >>> # }
        >>> 
        >>> # 示例2：多模态 Embedding 训练（图像+文本）
        >>> inputs = StdTemplateInputs(
        ...     messages=[
        ...         {"role": "user", "content": "<image>这张图片展示了什么？"},  # anchor with image
        ...         {"role": "assistant", "content": "<image>一只可爱的猫"}      # positive with image
        ...     ],
        ...     images=["query_image.jpg", "response_image.jpg"],
        ...     rejected_response=["<image>一辆汽车"],
        ...     label=1.0
        ... )
        >>> # 需要再添加 negative 的图像
        >>> inputs.images.append("negative_image.jpg")
        >>> encoded = template._embedding_encode(inputs)
        >>> # 返回结构包含：
        >>> # anchor_pixel_values, positive_pixel_values, negative_pixel_values (列表)
        >>> 
        >>> # 示例3：推理模式（只获取 query 的向量）
        >>> template.mode = 'pt'
        >>> inputs = StdTemplateInputs(
        ...     messages=[{"role": "user", "content": "什么是深度学习？"}]
        ... )
        >>> encoded = template._embedding_encode(inputs)
        >>> # 返回结构（简化版）：
        >>> # {
        >>> #     'input_ids': [101, 1234, ...],  # 只有 query 的编码
        >>> #     # 没有 labels 字段
        >>> # }
        >>> 
        >>> # 示例4：不带 hard negatives 的训练（使用 batch 内负样本）
        >>> inputs = StdTemplateInputs(
        ...     messages=[
        ...         {"role": "user", "content": "Python 是什么？"},
        ...         {"role": "assistant", "content": "Python 是一种编程语言"}
        ...     ],
        ...     rejected_response=[],  # 空列表或 None
        ...     label=1.0
        ... )
        >>> encoded = template._embedding_encode(inputs)
        >>> # 返回结构：
        >>> # {
        >>> #     'anchor_input_ids': [...],
        >>> #     'positive_input_ids': [...],
        >>> #     'negative_input_ids': [],      # 空列表
        >>> #     'labels': [1.0]                # 只有 positive 的标签
        >>> # }
        >>> # 训练时会使用 batch 内其他样本的 positive 作为 negative（InfoNCE）
        """
        # 初始化编码字典和标签列表
        _encoded = {}  # 存储所有编码结果（anchor、positive、negative）
        labels = []    # 存储标签列表：[positive_label, negative_label_1, negative_label_2, ...]
        
        # 判断是推理模式还是训练模式
        # 推理模式：只有 1 条消息（query），训练模式：至少 2 条消息（query + response）
        inference = len(inputs.messages) == 1
        
        # 如果是推理模式，添加一个空的 assistant 消息占位
        # 这样后续处理可以统一使用 messages[-2] (user) 和 messages[-1] (assistant) 的格式
        if inference:
            inputs.messages.append({'role': 'assistant', 'content': ''})

        def split_multi_medias(_inputs):
            """
            功能：
                内部辅助函数：分配多模态数据到各个样本。
                根据 _inputs.messages[-2] 中的多模态标签数量，从全局 inputs 中切分并分配对应的多模态数据到 _inputs
            
            原理：
            - 多模态数据（images/videos/audios）是按顺序存储的
            - 需要根据每个样本中的 <image>/<video>/<audio> 标签数量来切分
            - 例如：inputs.images = [img1, img2, img3, img4]
              * anchor 有 1 个 <image>，分配 img1
              * positive 有 2 个 <image>，分配 img2, img3
              * negative 有 1 个 <image>，分配 img4
            """
            # 从倒数第二条消息（通常是 user 角色）中获取内容
            _content = _inputs.messages[-2]['content']
            
            # 统计该内容中的多模态标签数量
            image_size = len(re.findall('<image>', _content))  # <image> 标签数量
            video_size = len(re.findall('<video>', _content))  # <video> 标签数量
            audio_size = len(re.findall('<audio>', _content))  # <audio> 标签数量
            
            # 从全局 inputs.images 中切分前 image_size 个图像给当前 _inputs
            _inputs.images = inputs.images[:image_size]
            assert len(_inputs.images) == image_size  # 确保数量匹配
            inputs.images = inputs.images[image_size:]  # 从全局列表中移除已分配的图像
            
            # 从全局 inputs.videos 中切分前 video_size 个视频给当前 _inputs
            _inputs.videos = inputs.videos[:video_size]
            assert len(_inputs.videos) == video_size  # 确保数量匹配
            inputs.videos = inputs.videos[video_size:]  # 从全局列表中移除已分配的视频
            
            # 从全局 inputs.audios 中切分前 audio_size 个音频给当前 _inputs
            _inputs.audios = inputs.audios[:audio_size]
            assert len(_inputs.audios) == audio_size  # 确保数量匹配
            inputs.audios = inputs.audios[audio_size:]  # 从全局列表中移除已分配的音频

        # 根据模式分支处理
        if not inference:  # 训练模式
            # ===== 1. 编码 Anchor 样本（查询/query） =====
            # 深拷贝 inputs 避免修改原始数据
            anchor = deepcopy(inputs)
            # 清空 assistant 消息内容（只保留 user 部分作为 anchor）
            anchor.messages[-1]['content'] = ''
            # 清空 rejected_response（anchor 不需要负样本信息）
            anchor.rejected_response = []
            # 分配多模态数据给 anchor
            split_multi_medias(anchor)
            # 编码 anchor
            anchor_encoded = self._encode_truncated(anchor)
            # 将 anchor 编码结果添加到 _encoded，所有键加上 'anchor_' 前缀
            for key in anchor_encoded:
                _encoded[f'anchor_{key}'] = anchor_encoded[key]
            
            # ===== 2. 编码 Positive 样本（正样本/response） =====
            # 深拷贝 inputs
            positive = deepcopy(inputs)
            # 将 response 内容（messages[-1]）移动到 user 位置（messages[-2]）
            # 这样编码时会把 response 当作输入来编码（获取其向量表示）
            positive.messages[-2]['content'] = positive.messages[-1]['content']
            # 清空 assistant 消息内容
            positive.messages[-1]['content'] = ''
            # 清空 rejected_response
            positive.rejected_response = []
            # 分配多模态数据给 positive
            split_multi_medias(positive)
            # 编码 positive
            positive_encoded = self._encode_truncated(positive)
            # 将 positive 编码结果添加到 _encoded，所有键加上 'positive_' 前缀
            for key in positive_encoded:
                _encoded[f'positive_{key}'] = positive_encoded[key]
                # 同时为每个 key 初始化对应的 'negative_{key}' 为空列表
                # 因为 negative 可能有多个，需要用列表存储
                _encoded[f'negative_{key}'] = []
            
            # 添加 positive 样本的标签
            # 如果 inputs.label 存在则使用其值，否则默认为 1.0（表示完全相关）
            labels.append(float(inputs.label) if inputs.label is not None else 1.0)

            # ===== 3. 编码 Negative 样本（负样本/hard negatives） =====
            # 获取 rejected_response 的数量（hard negatives 的数量）
            rejected_len = len(inputs.rejected_response) if inputs.rejected_response else 0
            # 遍历每个 negative 样本
            for i in range(rejected_len):
                # 深拷贝 inputs
                negative = deepcopy(inputs)
                # 将第 i 个 rejected_response 移动到 user 位置
                # 这样编码时会把这个 negative 样本当作输入来编码
                negative.messages[-2]['content'] = negative.rejected_response[i]
                # 清空 assistant 消息内容
                negative.messages[-1]['content'] = ''
                # 清空 rejected_response
                negative.rejected_response = []
                # 分配多模态数据给这个 negative 样本
                split_multi_medias(negative)
                # 编码 negative
                negative_encoded = self._encode_truncated(negative)
                # 将 negative 编码结果添加到对应的列表中
                # 注意：negative 是列表形式，因为可能有多个 hard negatives
                for key in negative_encoded:
                    _encoded[f'negative_{key}'].append(negative_encoded[key])
                # 为这个 negative 样本添加标签 0.0（表示不相关）
                labels.append(0.0)

            # 将标签列表添加到编码结果中
            # labels 格式：[positive_label, negative_label_0, negative_label_1, ...]
            _encoded['labels'] = labels
        else:  # 推理模式
            # 推理模式只需要编码 query（anchor），不需要 positive 和 negative
            # ===== 编码 Anchor（查询） =====
            # 深拷贝 inputs
            anchor = deepcopy(inputs)
            # 清空 assistant 消息内容（只编码 user 输入）
            anchor.messages[-1]['content'] = ''
            # 清空 rejected_response
            anchor.rejected_response = []
            # 分配多模态数据
            split_multi_medias(anchor)
            # 编码 anchor
            _encoded = self._encode_truncated(anchor)
            # 移除 labels 字段（推理模式不需要标签）
            _encoded.pop('labels', None)
        
        # 返回编码结果
        return _encoded
    def _reranker_encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """函数功能：
        编码 Reranker（重排序）任务的训练数据，用于文档相关性排序任务。
        
        Reranker 是信息检索系统中的关键组件，用于对初步检索的文档进行精细化重排序，
        将最相关的文档排在前面。
        
        核心概念：
        1. Query（查询）：用户的搜索查询或问题
        2. Positive Document（正样本文档）：与查询高度相关的文档（response）
        3. Negative Documents（负样本文档）：与查询不相关或相关性较低的文档（rejected_response）
        
        训练目标：
        - 学习判断文档与查询的相关性
        - Positive 样本标签为 1（相关）
        - Negative 样本标签为 0（不相关）
        - 通过对比学习，使模型能够准确区分相关和不相关的文档
        
        两种 Reranker 架构：
        1. 分类式 Reranker（task_type='reranker'）：
           - 基于序列分类架构（如 modernbert-reranker）
           - 输出单个相关性分数
           - 损失函数：pointwise 或 listwise
        2. 生成式 Reranker（task_type='generative_reranker'）：
           - 基于因果语言模型（如 qwen3-reranker）
           - 输出 "yes"/"no" token 的概率
           - 损失函数：生成式分类损失
        
        数据格式示例：
        {"query": "什么是机器学习？", "response": "机器学习是AI的分支...", 
         "rejected_response": ["今天天气不错", "Python是编程语言"]}
        
        支持两种输入模式：
        1. 占位符模式：query 中包含 `{doc}` 占位符
           - 示例：messages = [{"role": "user", "content": "这段文本 {doc} 与机器学习相关吗？"}]
           - 编码时会将 {doc} 替换为实际文档内容（response 或 rejected_response[i]）
           - 适合需要明确指定文档位置的场景
        2. 标准模式：query 不包含 `{doc}` 占位符
           - 示例：messages = [
               {"role": "user", "content": "查询：什么是机器学习？"},
               {"role": "assistant", "content": "文档：机器学习是..."}
             ]
           - 文档内容直接放在 assistant 消息中
           - 适合标准对话格式
        
        本方法的核心任务：
        1. 编码 positive 样本（query + 相关文档）
        2. 编码所有 negative 样本（query + 不相关文档）
        3. 为每个样本生成标签（positive=1, negative=0）
        4. 返回 positive 和 negative 的编码结果及标签
        
        参数：
        - inputs (StdTemplateInputs): 标准模板输入对象，包含：
            messages (List[Dict]): 对话消息列表
              - 占位符模式：1-2 条消息，messages[-2] 包含 {doc} 占位符，messages[-1] 为 positive 文档
              - 标准模式：2 条消息，messages[-2] 为 query，messages[-1] 为 positive 文档
            rejected_response (List[str]): 负样本文档列表（hard negatives）
              - 每个元素是一个不相关或低相关的文档
        
        返回值：
        - Dict[str, Any]: Reranker 训练所需的编码字典
        
        {
            # Positive 相关字段（query + 相关文档的编码）
            'positive_input_ids': List[int],         # positive 的 token IDs
            'positive_labels': List[int],            # positive 的标签（用于计算损失）
            'positive_loss_scale': List[float],      # positive 的 loss 权重
            
            # Negative 相关字段（query + 不相关文档的编码，列表形式）
            'negative_input_ids': List[List[int]],   # 所有 negative 的 token IDs 列表
            'negative_labels': List[List[int]],      # 所有 negative 的标签列表
            'negative_loss_scale': List[List[float]], # 所有 negative 的 loss 权重列表
            
            # 标签
            'labels': List[int]                      # [1, 0, 0, 0, ...]
                                                     # 第一个 1 表示 positive，后续的 0 表示各个 negative
        }
        
        使用示例：
        >>> # 示例1：占位符模式（推荐，格式更灵活）
        >>> template.task_type = 'reranker'
        >>> template.mode = 'train'
        >>> inputs = StdTemplateInputs(
        ...     messages=[
        ...         {"role": "user", "content": "文档 {doc} 是否与机器学习相关？"}
        ...         # 注意：这里只有 1 条消息，包含 {doc} 占位符
        ...     ],
        ...     # 实际的文档内容通过 response 和 rejected_response 提供
        ...     response="机器学习是人工智能的一个重要分支...",  # 会被添加到 messages[-1]
        ...     rejected_response=[
        ...         "今天天气真好，阳光明媚",       # hard negative 1
        ...         "Python 是一种编程语言",        # hard negative 2
        ...         "我喜欢吃披萨和汉堡"            # hard negative 3
        ...     ]
        ... )
        >>> encoded = template._reranker_encode(inputs)
        >>> # 返回结构：
        >>> # {
        >>> #     'positive_input_ids': [101, ...],  # "文档 [机器学习是...] 是否与机器学习相关？"
        >>> #     'negative_input_ids': [
        >>> #         [101, ...],  # "文档 [今天天气...] 是否与机器学习相关？"
        >>> #         [101, ...],  # "文档 [Python是...] 是否与机器学习相关？"
        >>> #         [101, ...]   # "文档 [我喜欢...] 是否与机器学习相关？"
        >>> #     ],
        >>> #     'labels': [1, 0, 0, 0]  # positive=1, 3个negative=0
        >>> # }
        >>> 
        >>> # 示例2：标准模式（不使用占位符）
        >>> inputs = StdTemplateInputs(
        ...     messages=[
        ...         {"role": "user", "content": "查询：什么是深度学习？"},
        ...         {"role": "assistant", "content": "深度学习是机器学习的子领域..."}  # positive
        ...     ],
        ...     rejected_response=[
        ...         "深圳是中国的一座城市",
        ...         "学习编程需要耐心"
        ...     ]
        ... )
        >>> encoded = template._reranker_encode(inputs)
        >>> # 返回结构：
        >>> # {
        >>> #     'positive_input_ids': [...],  # query + positive文档
        >>> #     'negative_input_ids': [
        >>> #         [...],  # query + "深圳是中国的一座城市"
        >>> #         [...]   # query + "学习编程需要耐心"
        >>> #     ],
        >>> #     'labels': [1, 0, 0]
        >>> # }
        >>> 
        >>> # 示例3：Listwise 损失训练（组内排序）
        >>> # 数据格式：每个 query 对应 1 个 positive + n 个 negative
        >>> inputs = StdTemplateInputs(
        ...     messages=[{"role": "user", "content": "相关性：{doc} 与'机器学习'"}],
        ...     response="监督学习是机器学习的主要范式",  # 高相关
        ...     rejected_response=[
        ...         "无监督学习也是机器学习的一种",      # 中等相关
        ...         "Python 有很多机器学习库",          # 低相关
        ...         "今天是星期一"                      # 不相关
        ...     ]
        ... )
        >>> # Listwise 损失会在组内计算相对排序，而不是独立的二分类
        >>> 
        >>> # 示例4：Generative Reranker（生成式）
        >>> template.task_type = 'generative_reranker'
        >>> inputs = StdTemplateInputs(
        ...     messages=[{"role": "user", "content": "文档：{doc}\n问题：这是否与AI相关？"}],
        ...     response="人工智能正在改变世界",
        ...     rejected_response=["今天天气不错"]
        ... )
        >>> # 生成式 reranker 会学习生成 "yes"（相关）或 "no"（不相关）
        """
        # 初始化编码字典和标签列表
        _encoded = {}  # 存储所有编码结果（positive、negative）
        labels = []    # 存储标签列表：[positive_label=1, negative_label_0, negative_label_0, ...]

        # ===== 1. 编码 Positive 样本（query + 相关文档） =====
        # 深拷贝 inputs 避免修改原始数据
        positive = deepcopy(inputs)
        # 清空 rejected_response（positive 不需要负样本信息）
        positive.rejected_response = []
        
        # 判断是否使用占位符模式
        if '{doc}' in positive.messages[-2]['content']:
            # 占位符模式：将 {doc} 替换为 positive 文档内容
            # messages[-1]['content'] 包含实际的 positive 文档
            positive.messages[-2]['content'] = positive.messages[-2]['content'].replace(
                '{doc}', inputs.messages[-1]['content'])
            # 替换后，移除 messages[-1]（因为其内容已经被嵌入到 messages[-2] 中）
            positive.messages.pop(-1)
        # 如果不包含 {doc}，说明是标准模式，messages 已经是正确格式，无需修改
        
        # 编码 positive 样本
        positive_encoded = self._encode_truncated(positive)
        # 将 positive 编码结果添加到 _encoded，所有键加上 'positive_' 前缀
        for key in positive_encoded:
            _encoded[f'positive_{key}'] = positive_encoded[key]
            # 同时为每个 key 初始化对应的 'negative_{key}' 为空列表
            # 因为 negative 可能有多个，需要用列表存储
            _encoded[f'negative_{key}'] = []
        
        # 为 positive 样本添加标签 1（表示相关）
        labels.append(1)

        # ===== 2. 编码 Negative 样本（query + 不相关文档） =====
        # 获取 rejected_response 的数量（hard negatives 的数量）
        rejected_len = len(inputs.rejected_response) if inputs.rejected_response else 0
        # 遍历每个 negative 样本
        for i in range(rejected_len):
            # 深拷贝 inputs
            negative = deepcopy(inputs)
            
            # 判断是否使用占位符模式
            if '{doc}' in negative.messages[-2]['content']:
                # 占位符模式：将 {doc} 替换为第 i 个 negative 文档内容
                negative.messages[-2]['content'] = negative.messages[-2]['content'].replace(
                    '{doc}', negative.rejected_response[i])
                # 替换后，移除 messages[-1]
                negative.messages.pop(-1)
            else:
                # 标准模式：直接将第 i 个 negative 文档放入 messages[-1]['content']
                # 这样就形成了 query + negative 文档的对
                negative.messages[-1]['content'] = negative.rejected_response[i]
            
            # 清空 rejected_response
            negative.rejected_response = []
            # 编码这个 negative 样本
            negative_encoded = self._encode_truncated(negative)
            # 将 negative 编码结果添加到对应的列表中
            # 注意：negative 是列表形式，因为可能有多个 hard negatives
            for key in negative_encoded:
                _encoded[f'negative_{key}'].append(negative_encoded[key])
            
            # 为这个 negative 样本添加标签 0（表示不相关）
            labels.append(0)

        # 将标签列表添加到编码结果中
        # labels 格式：[positive_label=1, negative_label_0, negative_label_1, ...]
        _encoded['labels'] = labels

        # 返回编码结果
        return _encoded

    def _seq_cls_encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """函数功能：
        编码序列分类（Sequence Classification）任务的训练或推理数据。
        
        序列分类是 NLP 中的基础任务，将整个输入序列映射到一个或多个类别标签。
        常见应用包括：
        - 文本分类：新闻分类、主题分类
        - 情感分析：正面/负面/中性情感判断
        - 意图识别：用户意图分类
        - 垃圾邮件检测：正常邮件 vs 垃圾邮件
        
        支持三种问题类型（problem_type）：
        1. single_label_classification（单标签分类）：
           - 每个样本只属于一个类别（互斥）
           - 标签格式：整数索引（如 0, 1, 2）
           - 损失函数：交叉熵损失（CrossEntropyLoss）
           - 示例：情感分类（正面/负面/中性），新闻分类（体育/财经/科技）
        
        2. multi_label_classification（多标签分类）：
           - 每个样本可以属于多个类别（非互斥）
           - 标签格式：浮点数列表（如 [1.0, 0.0, 1.0] 表示属于第 0 和第 2 类）
           - 损失函数：二元交叉熵损失（BCEWithLogitsLoss）
           - 示例：文章标签（可同时包含"技术"、"教育"、"AI"）
        
        3. regression（回归）：
           - 预测连续值而非离散类别
           - 标签格式：浮点数或浮点数列表
           - 损失函数：均方误差损失（MSELoss）
           - 示例：文本相似度评分（0.0-1.0），情感强度预测（-1.0 到 1.0）
        
        本方法的核心任务：
        1. 调用 _encode_truncated 对输入序列进行编码
        2. 移除编码结果中的 labels 字段（因为需要重新设置）
        3. 根据 inputs.label 和问题类型设置正确格式的 labels
        4. 对单标签分类任务，将标签转换为整数类型
        
        参数：
        - inputs (StdTemplateInputs): 标准模板输入对象，包含：
            messages (List[Dict]): 对话消息列表
              - 通常只有 1 条消息（待分类的文本）
              - 或 1-2 条消息（如果包含 system prompt）
            label (Union[int, float, List[float], None]): 类别标签
              - 单标签分类：整数（如 0, 1, 2）或可转换为整数的类型
              - 多标签分类：浮点数列表（如 [1.0, 0.0, 1.0]）
              - 回归：浮点数（如 0.85）或浮点数列表
              - 推理模式：None（不需要标签）
        
        返回值：
        - Dict[str, Any]: 序列分类任务所需的编码字典
        
        {
            'input_ids': List[int],              # 输入序列的 token IDs
            'labels': Union[int, float, List[float]],  # 类别标签（根据 problem_type 类型不同）
                                                       # - single_label: int (如 2)
                                                       # - multi_label: List[float] (如 [1.0, 0.0, 1.0])
                                                       # - regression: float 或 List[float] (如 0.85)
                                                       # - 推理模式：不包含此字段
            'loss_scale': List[float],           # loss 权重序列（通常不用于分类任务）
            'pixel_values': ...,                 # 多模态数据（如果有）
            # 注意：不包含 attention_mask（会在 data_collator 中生成）
        }
        
        使用示例：
        >>> # 示例1：单标签分类（情感分析）
        >>> template.task_type = 'seq_cls'
        >>> template.mode = 'train'
        >>> # 配置模型：3 个类别（负面、中性、正面）
        >>> template.config.num_labels = 3
        >>> template.config.problem_type = 'single_label_classification'
        >>> 
        >>> inputs = StdTemplateInputs(
        ...     messages=[{"role": "user", "content": "这部电影真是太棒了！"}],
        ...     label=2  # 类别 2：正面情感
        ... )
        >>> encoded = template._seq_cls_encode(inputs)
        >>> # 返回结构：
        >>> # {
        >>> #     'input_ids': [101, 1234, 5678, ..., 102],
        >>> #     'labels': 2  # 整数类型（单标签）
        >>> # }
        >>> 
        >>> # 示例2：多标签分类（文章标签）
        >>> template.config.num_labels = 5  # 5 个可能的标签
        >>> template.config.problem_type = 'multi_label_classification'
        >>> 
        >>> inputs = StdTemplateInputs(
        ...     messages=[{"role": "user", "content": "介绍深度学习在计算机视觉中的应用"}],
        ...     label=[1.0, 0.0, 1.0, 1.0, 0.0]  # 属于第 0、2、3 类
        ...     # 可能的标签：[AI, 体育, 技术, 教育, 娱乐]
        ...     # 该文章同时属于：AI、技术、教育
        ... )
        >>> encoded = template._seq_cls_encode(inputs)
        >>> # 返回结构：
        >>> # {
        >>> #     'input_ids': [101, ...],
        >>> #     'labels': [1.0, 0.0, 1.0, 1.0, 0.0]  # 浮点数列表（多标签）
        >>> # }
        >>> 
        >>> # 示例3：回归任务（文本相似度评分）
        >>> template.config.num_labels = 1  # 输出 1 个连续值
        >>> template.config.problem_type = 'regression'
        >>> 
        >>> inputs = StdTemplateInputs(
        ...     messages=[
        ...         {"role": "user", "content": "文本1：机器学习\n文本2：深度学习"}
        ...     ],
        ...     label=0.85  # 相似度分数（0.0-1.0）
        ... )
        >>> encoded = template._seq_cls_encode(inputs)
        >>> # 返回结构：
        >>> # {
        >>> #     'input_ids': [101, ...],
        >>> #     'labels': 0.85  # 浮点数（回归）
        >>> # }
        >>> 
        >>> # 示例4：推理模式（不提供标签）
        >>> template.mode = 'pt'
        >>> inputs = StdTemplateInputs(
        ...     messages=[{"role": "user", "content": "这个产品质量很好"}],
        ...     label=None  # 推理时不需要标签
        ... )
        >>> encoded = template._seq_cls_encode(inputs)
        >>> # 返回结构：
        >>> # {
        >>> #     'input_ids': [101, ...],
        >>> #     # 没有 'labels' 字段
        >>> # }
        >>> 
        >>> # 示例5：自动推断 problem_type（不显式设置）
        >>> template.config.problem_type = None  # 自动推断
        >>> template.config.num_labels = 4
        >>> 
        >>> # 整数标签 -> 自动识别为单标签分类
        >>> inputs1 = StdTemplateInputs(
        ...     messages=[{"role": "user", "content": "新闻文本"}],
        ...     label=1  # 整数
        ... )
        >>> encoded1 = template._seq_cls_encode(inputs1)
        >>> # problem_type 被自动设置为 'single_label_classification'
        >>> # labels 被转换为 int: 1
        >>> 
        >>> # 浮点数列表标签 -> 自动识别为多标签分类
        >>> inputs2 = StdTemplateInputs(
        ...     messages=[{"role": "user", "content": "文章内容"}],
        ...     label=[1.0, 0.0, 1.0, 0.0]  # 浮点数列表
        ... )
        >>> encoded2 = template._seq_cls_encode(inputs2)
        >>> # problem_type 被自动设置为 'multi_label_classification'
        >>> # labels 保持为 List[float]
        """
        # 调用 _encode_truncated 方法对输入进行编码
        # 返回包含 input_ids、labels（causal_lm 格式）、loss_scale 等字段的字典
        encoded = self._encode_truncated(inputs)
        
        # 移除 causal_lm 格式的 labels（因为序列分类不需要逐 token 的标签）
        # 序列分类任务的 labels 是整个序列对应一个标签，而不是每个 token 一个标签
        # pop(key, None) 表示如果 key 不存在也不会报错
        encoded.pop('labels', None)
        
        # 如果提供了标签（训练模式），则处理并添加到编码结果中
        if inputs.label is not None:
            # 获取输入的标签
            labels = inputs.label
            
            # 根据标签类型和模型配置自动推断问题类型
            # _get_problem_type 会根据以下规则判断：
            # 1. 如果 config.problem_type 已设置，直接返回
            # 2. 如果 labels 是列表且首元素是浮点数 -> 'regression'
            # 3. 如果 labels 是列表（非浮点数） -> 'multi_label_classification'
            # 4. 如果 labels 是标量（单个值） -> 'single_label_classification'
            problem_type = self._get_problem_type(self.config, labels=labels)
            
            # 如果是单标签分类，将标签转换为整数类型
            # 这是因为 CrossEntropyLoss 要求标签是整数类型（类别索引）
            if problem_type == 'single_label_classification':
                labels = int(labels)  # 确保标签是整数（如 2 而不是 2.0）
            
            # 将处理后的标签添加到编码结果中
            # 注意：
            # - single_label: labels 是 int (如 2)
            # - multi_label: labels 是 List[float] (如 [1.0, 0.0, 1.0])
            # - regression: labels 是 float 或 List[float] (如 0.85)
            encoded['labels'] = labels
        
        # 返回编码结果
        # 推理模式（inputs.label is None）：不包含 'labels' 字段
        # 训练模式：包含 'labels' 字段，格式取决于 problem_type
        return encoded
    @torch.inference_mode()
    def encode(self,
               inputs: Union[TemplateInputs, Dict[str, Any], InferRequest],  # 输入数据：支持多种格式
               return_template_inputs: bool = False,  # 是否在返回结果中包含处理后的template_inputs
               return_length: bool = False) -> Dict[str, Any]:  # 是否在返回结果中包含序列长度
        """函数功能：
        Template类的核心入口方法！将用户输入的对话数据、多模态数据编码为模型可接受的格式。
        这是整个Template类最重要的方法，根据不同的task_type和mode调用相应的编码子方法。

        主要流程：
        1. 验证processor已初始化
        2. 标准化输入格式为StdTemplateInputs
        3. 预处理输入（加载图像、替换特殊标签等）
        4. 根据task_type和mode选择编码策略：
           - causal_lm: 标准训练/_encode_truncated、RLHF、KTO、GKD
           - seq_cls: 序列分类编码
           - prm: 过程奖励模型编码
           - embedding: 文本向量化编码
           - reranker: 文档排序编码
        5. 清理None值，处理length字段
        6. 可选返回template_inputs和extra_kwargs

        参数：
        - inputs (Union[TemplateInputs, Dict[str, Any], InferRequest]): 输入数据，支持：
            * TemplateInputs/InferRequest数据类实例
            * 字典格式（包含messages、images、tools等字段）
            * StdTemplateInputs标准输入对象
        - return_template_inputs (bool): 是否在返回字典中包含处理后的template_inputs对象，
            默认False。用于调试或需要访问标准化后的输入数据。
        - return_length (bool): 是否在返回字典中包含序列长度信息，默认False。
            True时会计算所有*length字段的最大值并保存。

        返回值：
        - Dict[str, Any]: 编码后的字典，通常包含：
            * input_ids (List[int]): token ID序列
            * labels (Optional[List[int]]): 标签序列（训练时使用，推理时为None）
            * loss_scale (Optional[List[float]]): loss权重序列
            * pixel_values/image_sizes等: 多模态数据（如有）
            * length (Optional[int]): 序列长度（若return_length=True）
            * template_inputs (StdTemplateInputs): 标准化输入（若return_template_inputs=True）
            * 根据不同task_type和mode可能包含额外字段（如RLHF的chosen/rejected）

        使用示例：
        >>> # 示例1：基础训练场景
        >>> template.mode = 'train'
        >>> template.task_type = 'causal_lm'
        >>> inputs = {"messages": [
        ...     {"role": "user", "content": "你好"},
        ...     {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}
        ... ]}
        >>> encoded = template.encode(inputs)
        >>> # encoded: {'input_ids': [...], 'labels': [...], 'loss_scale': [...]}
        >>> 
        >>> # 示例2：多模态输入
        >>> inputs = {"messages": [{"role": "user", "content": "<image>描述这张图"}],
        ...           "images": ["path/to/image.jpg"]}
        >>> encoded = template.encode(inputs)
        >>> # encoded还会包含: pixel_values, image_sizes等
        >>> 
        >>> # 示例3：RLHF模式
        >>> template.mode = 'rlhf'
        >>> inputs = {
        ...     "messages": [{"role": "user", "content": "写一首诗"}],
        ...     "chosen": {"role": "assistant", "content": "好的诗"},
        ...     "rejected": {"role": "assistant", "content": "不好的诗"}
        ... }
        >>> encoded = template.encode(inputs)
        >>> # encoded: {'chosen_input_ids': [...], 'rejected_input_ids': [...], ...}
        >>> 
        >>> # 示例4：推理场景（不生成labels）
        >>> template.mode = 'pt/vllm/sglang/lmdeploy'
        >>> inputs = {"messages": [{"role": "user", "content": "你好"}]}
        >>> encoded = template.encode(inputs, return_length=True)
        >>> # encoded: {'input_ids': [...], 'length': 10} (没有labels)
        """
        # ===== 步骤1：验证processor已初始化 =====
        assert self._processor_inited, ('Please initialize the processor before calling the template.encode method: '
                                        'template.init_processor(processor).')  # 确保processor已初始化，否则无法使用tokenizer
        
        # ===== 步骤2：标准化输入格式 =====
        if isinstance(inputs, (InferRequest, TemplateInputs)):  # 若输入为数据类实例
            inputs = asdict(inputs)  # 转换为字典格式，便于后续处理

        extra_kwargs = {}  # 初始化额外参数字典：用于存储不在标准字段中的自定义参数
        if isinstance(inputs, dict):  # 若输入为字典格式
            inputs = deepcopy(inputs)  # 深拷贝输入，避免修改原始数据
            if self.task_type == 'causal_lm' and not self.is_training:  # 若为因果语言模型且非训练模式（推理）
                InferRequest.remove_response(inputs['messages'])  # 移除messages中的assistant响应，仅保留用户输入
            inputs, extra_kwargs = StdTemplateInputs.from_dict(inputs)  # 将字典转换为StdTemplateInputs对象，分离标准字段和额外参数
        elif isinstance(inputs, StdTemplateInputs):  # 若输入已经是StdTemplateInputs对象
            inputs = deepcopy(inputs)  # 深拷贝输入，避免修改原始对象
        assert isinstance(inputs, StdTemplateInputs)  # 确保此时inputs已经是StdTemplateInputs类型
        
        # ===== 步骤3：预处理输入（加载多模态数据、替换特殊标签、处理工具调用等） =====
        self._preprocess_inputs(inputs)  # 执行预处理：加载图像/视频/音频、替换<image>等标签、处理tools和bbox等

        # ===== 步骤4：根据task_type和mode选择编码策略 =====
        if self.task_type == 'causal_lm':  # 若任务类型为因果语言模型（标准的自回归LM）
            if self.mode in {'train', 'pt', 'vllm', 'lmdeploy', 'sglang'}:  # 标准训练或推理模式
                encoded = self._encode_truncated(inputs)  # 调用标准编码方法：处理消息、tokenize、生成labels、应用截断策略
            elif self.mode == 'rlhf':  # RLHF模式（基于人类反馈的强化学习）
                encoded = self._rlhf_encode(inputs)  # 调用RLHF编码方法：分别编码chosen和rejected响应
            elif self.mode == 'kto':  # KTO模式（Kahneman-Tversky优化）
                encoded = self._kto_encode(inputs)  # 调用KTO编码方法：处理KTO训练所需格式
            elif self.mode == 'gkd':  # GKD模式（广义知识蒸馏）
                encoded = self._gkd_encode(inputs)  # 调用GKD编码方法：分离prompt和answer用于知识蒸馏
        elif self.task_type == 'seq_cls':  # 若任务类型为序列分类
            if self.mode == 'rlhf':  # 序列分类的RLHF模式（用于奖励模型训练）
                encoded = self._rlhf_encode(inputs)  # 先使用RLHF编码
                for prefix in ['chosen', 'rejected']:  # 遍历chosen和rejected前缀
                    encoded.pop(f'{prefix}_labels', None)  # 移除labels字段（分类任务不需要token级别的labels）
                    encoded.pop(f'{prefix}_loss_scale', None)  # 移除loss_scale字段
            else:  # 标准序列分类模式
                encoded = self._seq_cls_encode(inputs)  # 调用序列分类编码方法：编码输入并添加分类标签
        elif self.task_type == 'prm':  # 若任务类型为过程奖励模型（Process Reward Model）
            encoded = self._encode_truncated(inputs)  # 使用标准编码方法（PRM与causal_lm编码类似）
        elif self.task_type == 'embedding':  # 若任务类型为文本向量化
            encoded = self._embedding_encode(inputs)  # 调用embedding编码方法：处理anchor、positive、negative三元组
        elif self.task_type in {'reranker', 'generative_reranker'}:  # 若任务类型为文档排序
            encoded = self._reranker_encode(inputs)  # 调用reranker编码方法：编码positive和negative文档对

        # ===== 步骤5：添加channel信息（如有） =====
        if inputs.channel is not None:  # 若输入包含channel信息（用于多任务或多数据源场景）
            encoded['channel'] = inputs.channel  # 将channel信息添加到编码结果中
        
        # ===== 步骤6：清理None值并处理length字段 =====
        lengths = [0]  # 初始化长度列表，用于收集所有*length字段的值
        for key in list(encoded.keys()):  # 遍历编码结果的所有键（使用list()避免迭代时修改字典）
            if encoded[key] is None:  # 若值为None
                encoded.pop(key)  # 移除该键值对，减少不必要的数据
            elif key.endswith('length'):  # 若键名以'length'结尾（如'length', 'chosen_length'等）
                value = encoded[key]  # 获取length值
                if isinstance(value, int):  # 若为单个整数
                    lengths.append(value)  # 添加到lengths列表
                elif isinstance(value, (tuple, list)):  # 若为列表或元组（多个长度值）
                    lengths += value  # 扩展lengths列表

        # ===== 步骤7：根据return_length决定是否保留length字段 =====
        if return_length:  # 若需要返回length
            encoded['length'] = max(lengths)  # 保存所有长度的最大值
        else:  # 若不需要返回length
            encoded.pop('length', None)  # 移除length字段（如果存在）
        
        # ===== 步骤8：根据参数决定是否添加额外信息 =====
        if return_template_inputs:  # 若需要返回处理后的template_inputs
            encoded['template_inputs'] = inputs  # 将StdTemplateInputs对象添加到结果中
        if not self.remove_unused_columns:  # 若不移除未使用的列
            encoded['_extra_kwargs'] = extra_kwargs  # 保存额外参数（非标准字段）
        return encoded

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        函数功能：
            将多个数据样本打包（packing）成一个批次样本。该方法主要用于 padding_free 模式下，
            将多个短序列拼接成一个长序列，以提高训练效率。通过合并 input_ids、labels 等字段，
            并重新生成 position_ids，实现序列级别的数据打包。
        
        参数：
            row (List[Dict[str, Any]]): 待打包的数据样本列表，每个样本是一个字典，包含以下可能的键：
                - input_ids (List[int]): 输入token序列
                - labels (List[int]): 标签token序列
                - loss_scale (List[float]): 损失缩放因子序列
                - length (int): 序列长度
                - channel (Any): 通道信息（如多模态数据的通道）
                - pixel_values (torch.Tensor, optional): 图像像素值（多模态数据）
                - pixel_values_videos (torch.Tensor, optional): 视频像素值（多模态数据）
                - image_sizes (torch.Tensor, optional): 图像尺寸信息（多模态数据）
        
        返回值：
            Dict[str, Any]: 打包后的批次数据字典，包含以下可能的键：
                - input_ids (List[int]): 拼接后的输入token序列
                - labels (List[int]): 拼接后的标签token序列
                - loss_scale (List[float]): 拼接后的损失缩放因子序列
                - length (int): 所有样本长度的总和
                - channel (List[Any]): 所有样本的通道信息列表
                - position_ids (List[int]): 重新生成的位置编码序列
                - pixel_values (torch.Tensor, optional): 合并后的图像像素值
                - pixel_values_videos (torch.Tensor, optional): 合并后的视频像素值
                - image_sizes (torch.Tensor, optional): 合并后的图像尺寸信息
        
        使用示例：
            >>> template = Template(...)
            >>> row1 = {'input_ids': [1, 2, 3], 'labels': [1, 2, 3], 'length': 3}
            >>> row2 = {'input_ids': [4, 5], 'labels': [4, 5], 'length': 2}
            >>> packed = template.packing_row([row1, row2])
            >>> print(packed['input_ids'])  # [1, 2, 3, 4, 5]
            >>> print(packed['position_ids'])  # [0, 1, 2, 0, 1]
            >>> print(packed['length'])  # 5
        """
        # ===== 步骤1：初始化打包结果字典和辅助变量 =====
        packed = {}  # 存储打包后的结果
        keys = set()  # 收集所有样本中出现的字段名
        length = []  # 收集所有样本的长度信息
        
        # ===== 步骤2：收集所有字段名和长度信息 =====
        for r in row:  # 遍历每个待打包的样本
            keys.update(r.keys())  # 将当前样本的所有字段名添加到集合中
            length.append(r['length'])  # 提取并保存样本的长度
        
        # ===== 步骤3：根据字段类型进行不同的打包操作 =====
        for key in keys:  # 遍历所有收集到的字段名
            if key in {'input_ids', 'labels', 'loss_scale'}:  # 对于序列类字段
                # 使用 sum 函数将所有样本的序列首尾拼接成一个长序列
                # 例如：[[1,2,3], [4,5]] -> [1,2,3,4,5]
                packed[key] = sum((x[key] for x in row), start=[])
            elif key == 'length':  # 对于长度字段
                # 计算所有样本长度的总和，得到打包后的总长度
                packed[key] = sum((x[key] for x in row))
            elif key == 'channel':  # 对于通道字段
                # 将所有样本的通道信息收集成列表（不拼接）
                packed[key] = [x[key] for x in row]
        
        # ===== 步骤4：生成打包后的位置编码 =====
        if 'position_ids' not in packed:  # 如果打包结果中没有 position_ids
            # 为每个样本生成从0开始的位置编码，然后拼接
            # 例如：length=[3,2] -> [[0,1,2], [0,1]] -> [0,1,2,0,1]
            # 这样每个样本的位置编码都是独立的，从0开始计数
            packed['position_ids'] = sum((list(range(x)) for x in length), start=[])

        # ===== 步骤5：处理多模态数据（图像、视频等） =====
        # 调用 _data_collator_mm_data 方法处理 pixel_values、pixel_values_videos、image_sizes 等多模态字段
        packed.update(self._data_collator_mm_data(row))
        
        # ===== 步骤6：返回打包后的批次数据 =====
        return packed

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
    @staticmethod
    def _get_seq_cls_logprobs(pred: int, logprobs: torch.Tensor, top_logprobs: int):
        """
        函数功能：
            为序列分类任务生成预测结果的对数概率信息。该方法用于提取和组织分类模型的预测结果，
            包括预测类别的对数概率以及概率最高的前N个候选类别的详细信息。主要用于序列分类
            （sequence classification）和多标签分类（multi-label classification）任务。
        
        参数：
            pred (int 或 List[int]): 预测的类别索引
                - 对于单标签分类：整数，表示预测的类别索引（如类别3）
                - 对于多标签分类：整数列表，表示预测的多个类别索引（如[0, 2, 5]）
            logprobs (torch.Tensor): 所有类别的对数概率张量
                - 形状：(num_classes,) 一维张量
                - 包含每个类别的对数概率值（log probability）
                - 对于单标签分类：通常来自 log_softmax
                - 对于多标签分类：通常来自 logsigmoid
            top_logprobs (int): 返回概率最高的前N个候选类别数量
                - 用于查看除了预测类别外，其他高概率候选的分布情况
        
        返回值：
            Dict[str, List[Dict[str, Any]]]: 格式化的对数概率信息字典，结构如下：
                {
                    'content': [
                        {
                            'index': <预测类别索引>,
                            'logprobs': <预测类别的对数概率值或值列表>,
                            'top_logprobs': [
                                {'index': <类别索引>, 'logprob': <对数概率值>},
                                ...  # 按概率降序排列的前N个候选
                            ]
                        }
                    ]
                }
        
        使用示例：
            >>> # 单标签分类示例
            >>> logprobs = torch.tensor([-0.1, -2.3, -0.5, -3.2])  # 4个类别的对数概率
            >>> pred = 0  # 预测类别0（概率最高）
            >>> result = Template._get_seq_cls_logprobs(pred, logprobs, top_logprobs=3)
            >>> print(result)
            {
                'content': [{
                    'index': 0,
                    'logprobs': -0.1,
                    'top_logprobs': [
                        {'index': 0, 'logprob': -0.1},
                        {'index': 2, 'logprob': -0.5},
                        {'index': 1, 'logprob': -2.3}
                    ]
                }]
            }
            
            >>> # 多标签分类示例
            >>> logprobs = torch.tensor([-0.2, -0.3, -3.0, -0.1])
            >>> pred = [0, 1, 3]  # 预测多个类别
            >>> result = Template._get_seq_cls_logprobs(pred, logprobs, top_logprobs=2)
            >>> print(result)
            {
                'content': [{
                    'index': [0, 1, 3],
                    'logprobs': [-0.2, -0.3, -0.1],
                    'top_logprobs': [
                        {'index': 3, 'logprob': -0.1},
                        {'index': 0, 'logprob': -0.2}
                    ]
                }]
            }
        """
        # ===== 步骤1：获取概率最高的前N个类别索引 =====
        # 对 logprobs 按降序排序（descending=True），取前 top_logprobs 个索引
        # argsort 返回排序后的索引，[:top_logprobs] 截取前N个，tolist() 转为Python列表
        idxs = logprobs.argsort(descending=True, dim=-1)[:top_logprobs].tolist()
        
        # ===== 步骤2：将 logprobs 张量转换为 Python 列表 =====
        # 转换为列表便于后续索引访问和JSON序列化
        logprobs = logprobs.tolist()
        
        # ===== 步骤3：构建并返回格式化的结果字典 =====
        return {
            'content': [{  # content 字段包含一个列表，通常只有一个元素
                'index': pred,  # 预测的类别索引（单个整数或整数列表）
                # 根据 pred 类型提取对应的对数概率值：
                # - 如果是列表/元组（多标签）：提取每个预测类别的对数概率值列表
                # - 如果是单个整数（单标签）：提取该类别的对数概率值
                'logprobs': [logprobs[p] for p in pred] if isinstance(pred, (list, tuple)) else logprobs[pred],
                # top_logprobs 字段：包含概率最高的前N个候选类别的详细信息
                'top_logprobs': [{
                    'index': idx,  # 候选类别的索引
                    'logprob': logprobs[idx]  # 候选类别的对数概率值
                } for idx in idxs]  # 遍历排序后的前N个索引，构建字典列表
            }]
        }

    @staticmethod
    def _get_problem_type(config, labels=None, logits=None) -> str:
        """
        函数功能：
            自动推断或获取序列分类任务的问题类型（problem type）。该方法用于确定模型处理的任务类型，
            以便选择合适的损失函数和评估指标。支持三种问题类型：回归（regression）、单标签分类
            （single_label_classification）和多标签分类（multi_label_classification）。推断逻辑
            基于配置、标签格式或模型输出的形状，并将结果缓存到config中以避免重复计算。
        
        参数：
            config: 模型配置对象，包含以下属性：
                - problem_type (Optional[str]): 已设置的问题类型，若非None则直接返回
                - num_labels (int): 类别数量，用于单标签分类的验证
            labels (Optional[Union[int, List]]): 标签数据，用于推断问题类型
                - None: 不使用标签推断
                - int: 单个整数标签 -> 推断为单标签分类
                    例如：labels=2（表示第2类）
                - List[float]: 浮点数列表 -> 推断为回归任务
                    例如：labels=[1.5, 2.3, 0.8]（多个回归目标值）
                - List[int]: 整数列表 -> 推断为多标签分类
                    例如：labels=[0, 2, 5]（同时属于类别0、2、5）
            logits (Optional[torch.Tensor]): 模型输出的logits张量，用于推断问题类型
                - None: 不使用logits推断
                - torch.Tensor: 形状为 (..., num_classes) 或 (..., 1)
                    形状 (..., 1): 推断为回归任务（单值输出）
                    形状 (..., N) (N>1): 推断为单标签分类（N个类别）
        
        返回值：
            str: 问题类型，可能的值为：
                - 'regression': 回归任务
                    用于预测连续值，如房价预测、评分预测等
                    损失函数：MSELoss
                - 'single_label_classification': 单标签分类任务
                    每个样本只属于一个类别，如文本分类、情感分析等
                    损失函数：CrossEntropyLoss
                - 'multi_label_classification': 多标签分类任务
                    每个样本可以同时属于多个类别，如标签预测、属性识别等
                    损失函数：BCEWithLogitsLoss
        
        使用示例：
            >>> # 示例1：已设置problem_type（直接返回）
            >>> from transformers import AutoConfig
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')
            >>> config.problem_type = 'single_label_classification'
            >>> problem_type = Template._get_problem_type(config)
            >>> print(problem_type)  # 'single_label_classification'
            
            >>> # 示例2：通过标签推断（单标签分类）
            >>> config = AutoConfig.from_pretrained('bert-base-uncased')
            >>> config.num_labels = 5
            >>> config.problem_type = None
            >>> labels = 2  # 单个整数标签（第2类）
            >>> problem_type = Template._get_problem_type(config, labels=labels)
            >>> print(problem_type)  # 'single_label_classification'
            >>> print(config.problem_type)  # 'single_label_classification'（已缓存）
            
            >>> # 示例3：通过标签推断（回归任务）
            >>> config.problem_type = None
            >>> labels = [1.5, 2.3, 0.8]  # 浮点数列表
            >>> problem_type = Template._get_problem_type(config, labels=labels)
            >>> print(problem_type)  # 'regression'
            
            >>> # 示例4：通过标签推断（多标签分类）
            >>> config.problem_type = None
            >>> labels = [0, 2, 5]  # 整数列表（多个类别）
            >>> problem_type = Template._get_problem_type(config, labels=labels)
            >>> print(problem_type)  # 'multi_label_classification'
            
            >>> # 示例5：通过logits推断（回归任务）
            >>> import torch
            >>> config.problem_type = None
            >>> logits = torch.randn(8, 1)  # 形状 (batch_size, 1)
            >>> problem_type = Template._get_problem_type(config, logits=logits)
            >>> print(problem_type)  # 'regression'
            
            >>> # 示例6：通过logits推断（单标签分类）
            >>> logits = torch.randn(8, 10)  # 形状 (batch_size, 10)，10个类别
            >>> problem_type = Template._get_problem_type(config, logits=logits)
            >>> print(problem_type)  # 'single_label_classification'
        """
        # ===== 步骤1：检查config中是否已设置problem_type =====
        problem_type = config.problem_type  # 获取配置中的问题类型
        if problem_type is not None:  # 若已设置（优先级最高）
            return problem_type  # 直接返回，无需推断
        
        # ===== 步骤2：通过labels推断problem_type =====
        if labels is not None:  # 若提供了labels参数
            if isinstance(labels, (list, tuple)):  # 若labels是列表或元组
                # 检查列表中的元素类型来判断任务类型
                if labels and isinstance(labels[0], float):  # 若列表非空且首元素是浮点数
                    # 浮点数列表 -> 回归任务（预测连续值）
                    # 例如：labels=[1.5, 2.3, 0.8] 表示多个目标值的回归
                    problem_type = 'regression'
                else:  # 若列表元素不是浮点数（通常是整数）
                    # 整数列表 -> 多标签分类（一个样本可以属于多个类别）
                    # 例如：labels=[0, 2, 5] 表示该样本同时属于类别0、2、5
                    problem_type = 'multi_label_classification'
            else:  # 若labels是单个值（标量）
                # 单个整数 -> 单标签分类（每个样本只属于一个类别）
                # 例如：labels=2 表示该样本属于类别2
                problem_type = 'single_label_classification'
                # 断言：类别数量必须大于等于 labels+1（因为类别索引从0开始）
                # 例如：labels=2 要求至少有3个类别（0, 1, 2）
                assert config.num_labels >= labels + 1
        
        # ===== 步骤3：通过logits推断problem_type =====
        if logits is not None:  # 若提供了logits参数（模型输出）
            # 检查logits的最后一个维度（输出维度）
            # logits.shape: (..., num_classes) 或 (..., 1)
            if logits.shape[-1] == 1:  # 若输出维度为1
                # 单值输出 -> 回归任务
                # 例如：logits.shape = (batch_size, 1) -> 每个样本输出一个连续值
                problem_type = 'regression'
            else:  # 若输出维度大于1
                # 多值输出 -> 单标签分类（兼容旧版本）
                # 例如：logits.shape = (batch_size, 10) -> 10个类别的logits
                # 注释：compatible with older versions（兼容旧版本的默认行为）
                problem_type = 'single_label_classification'
        
        assert problem_type is not None  # 确保已成功推断出问题类型
        config.problem_type = problem_type  # 将推断结果保存到config，避免下次重复推断
        return problem_type
    def decode_seq_cls(self, logits: torch.Tensor, top_logprobs: int):
        """
        函数功能：
            解码序列分类（Sequence Classification）任务的模型输出，将logits转换为预测结果和对数概率。
            该方法根据问题类型（回归、单标签分类、多标签分类）采用不同的解码策略，生成可读的预测结果
            和详细的概率信息。对于分类任务，还会提供top-k候选类别的概率分布，便于分析模型的预测置信度。
        
        参数：
            logits (torch.Tensor): 模型输出的logits张量
                - 回归任务：形状 (batch_size, 1)，每个样本一个连续值
                    例如：tensor([[2.3], [1.8], [3.5]]) - 3个样本的回归值
                - 单标签分类：形状 (batch_size, num_classes)，每个样本的类别logits
                    例如：tensor([[0.2, 1.5, 0.1], [2.0, 0.5, 0.8]]) - 2个样本，3个类别
                - 多标签分类：形状 (batch_size, num_labels)，每个样本的标签logits
                    例如：tensor([[0.8, -0.2, 1.5], [0.3, 0.9, -0.5]]) - 2个样本，3个标签
            top_logprobs (int): 返回概率最高的前N个候选类别（仅对分类任务有效）
                用于查看模型的预测分布，便于理解模型的置信度
                例如：top_logprobs=5 表示返回概率最高的前5个类别
        
        返回值：
            Tuple[List, List]: 包含两个列表的元组
                - preds (List): 预测结果列表
                    回归任务：List[float]，每个元素是一个浮点数
                        例如：[2.3, 1.8, 3.5]
                    单标签分类：List[int]，每个元素是预测的类别索引
                        例如：[1, 0, 2] - 预测类别1、0、2
                    多标签分类：List[List[int]]，每个元素是预测的多个类别索引列表
                        例如：[[0, 2], [1], [0, 1, 2]] - 不同样本预测的类别集合
                - logprobs (List): 对数概率信息列表
                    回归任务：List[None]，回归任务没有概率信息
                        例如：[None, None, None]
                    分类任务：List[Dict]，每个元素是格式化的概率信息字典
                        例如：[{'content': [{'index': 1, 'logprobs': -0.5, 'top_logprobs': [...]}]}]
        
        使用示例：
            >>> # 示例1：回归任务
            >>> import torch
            >>> template = Template(...)
            >>> template.config.problem_type = 'regression'
            >>> logits = torch.tensor([[2.3], [1.8], [3.5]])  # 形状: (3, 1)
            >>> preds, logprobs = template.decode_seq_cls(logits, top_logprobs=5)
            >>> print(preds)  # [2.3, 1.8, 3.5]
            >>> print(logprobs)  # [None, None, None]
            
            >>> # 示例2：单标签分类任务
            >>> template.config.problem_type = None  # 重置，让方法自动推断
            >>> logits = torch.tensor([[0.2, 1.5, 0.1],   # 样本1，类别1最高
            ...                        [2.0, 0.5, 0.8]])  # 样本2，类别0最高
            >>> # logits形状: (2, 3) - 2个样本，3个类别
            >>> preds, logprobs = template.decode_seq_cls(logits, top_logprobs=3)
            >>> print(preds)  # [1, 0] - 预测类别索引
            >>> print(len(logprobs))  # 2
            >>> print(logprobs[0]['content'][0]['index'])  # 1（第一个样本预测类别1）
            >>> # logprobs[0]包含top-3类别的概率分布
            
            >>> # 示例3：多标签分类任务
            >>> logits = torch.tensor([[0.8, -0.2, 1.5],   # 样本1
            ...                        [0.3, 0.9, -0.5]])  # 样本2
            >>> # sigmoid后：[[0.69, 0.45, 0.82], [0.57, 0.71, 0.38]]
            >>> # logits形状: (2, 3) - 2个样本，3个标签
            >>> preds, logprobs = template.decode_seq_cls(logits, top_logprobs=3)
            >>> print(preds)  # [[0, 2], [0, 1]] - 样本1预测标签0和2，样本2预测标签0和1
            >>> # 阈值0.5：sigmoid值>=0.5的标签被预测为正类
            
            >>> # 示例4：批量处理
            >>> logits = torch.randn(16, 10)  # 形状: (16, 10) - 16个样本，10个类别
            >>> preds, logprobs = template.decode_seq_cls(logits, top_logprobs=5)
            >>> print(len(preds))  # 16
            >>> print(len(logprobs))  # 16
            >>> print(type(preds[0]))  # <class 'int'>（单标签分类）
        """
        # ===== 步骤1：验证输入类型 =====
        assert isinstance(logits, torch.Tensor)  # 确保logits是torch.Tensor类型
        
        # ===== 步骤2：推断问题类型 =====
        # 根据logits的形状自动推断任务类型（回归/单标签分类/多标签分类）
        problem_type = self._get_problem_type(self.config, logits=logits)

        # ===== 步骤3：根据问题类型解码logits =====
        if problem_type == 'regression':  # 回归任务
            # 处理回归任务的输出
            # logits形状: (batch_size, 1)
            # .squeeze(dim=-1): 移除最后一个维度，形状变为 (batch_size,)
            # 例如：tensor([[2.3], [1.8], [3.5]]) -> tensor([2.3, 1.8, 3.5])
            # .tolist(): 转换为Python列表
            preds = logits.squeeze(dim=-1).tolist()
            
            # 回归任务没有概率分布，创建None列表
            logprobs = [None] * len(preds)
        else:  # 分类任务（单标签或多标签）
            if problem_type == 'single_label_classification':  # 单标签分类
                # 单标签分类：每个样本选择概率最高的一个类别
                # logits形状: (batch_size, num_classes)
                # torch.argmax(logits, dim=-1): 沿最后一个维度（类别维度）取最大值的索引
                # 例如：tensor([[0.2, 1.5, 0.1], [2.0, 0.5, 0.8]])
                #       -> tensor([1, 0])（每个样本的最大值索引）
                # .tolist(): 转换为Python列表
                preds = torch.argmax(logits, dim=-1).tolist()
                
                # 计算对数概率分布
                # torch.log_softmax(logits, -1): 沿最后一个维度计算log-softmax
                # 形状保持不变: (batch_size, num_classes)
                # log_softmax(x) = log(softmax(x)) = log(exp(x) / sum(exp(x)))
                # 避免直接计算softmax再取log导致的数值不稳定
                logprobs = torch.log_softmax(logits, -1)
            else:  # 多标签分类 (problem_type == 'multi_label_classification')
                # 多标签分类：每个样本可以选择多个类别
                # logits形状: (batch_size, num_labels)
                
                # 对每个样本（每一行）单独处理
                preds = []
                for logprob in torch.sigmoid(logits):  # 遍历batch中的每个样本
                    # logprob形状: (num_labels,) - 单个样本的sigmoid概率
                    # torch.sigmoid: 将logits转换为概率 (0, 1)
                    # (logprob >= 0.5): 应用阈值0.5，形状 (num_labels,)，布尔tensor
                    # .nonzero(as_tuple=True)[0]: 获取True位置的索引
                    # .tolist(): 转换为Python列表
                    # 例如：logprob = tensor([0.69, 0.45, 0.82])
                    #       -> (logprob >= 0.5) = tensor([True, False, True])
                    #       -> 索引 [0, 2]（标签0和2被预测为正类）
                    pred = (logprob >= 0.5).nonzero(as_tuple=True)[0].tolist()
                    preds.append(pred)
                
                # 计算对数sigmoid概率
                # F.logsigmoid(logits): 计算log(sigmoid(x))
                # 形状: (batch_size, num_labels)
                # logsigmoid(x) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
                logprobs = F.logsigmoid(logits)

            # ===== 步骤4：格式化概率信息（仅分类任务） =====
            # 为每个样本生成详细的概率信息，包括top-k候选
            # logprobs当前是tensor，形状: (batch_size, num_classes/num_labels)
            # 遍历每个样本，调用_get_seq_cls_logprobs生成格式化的概率信息
            logprobs = [self._get_seq_cls_logprobs(pred, logprobs[i], top_logprobs) 
                       for i, pred in enumerate(preds)]
            # 结果：List[Dict]，每个元素是包含预测和top-k概率的字典
        
        # ===== 步骤5：返回预测结果和概率信息 =====
        return preds, logprobs

    def decode(self,
               generate_ids: List[int],  # 生成的token ID序列：模型生成的完整token列表
               *,  # 关键字分隔符，以下参数必须使用关键字传递
               is_finished: bool = True,  # 生成是否已完成：True表示已结束，会跳过stop tokens
               tokenizer_kwargs=None,  # tokenizer.decode的额外参数：如skip_special_tokens等
               first_token=True,  # 是否为第一个token：True时会添加response_prefix
               **kwargs) -> Any:  # 其他额外参数（预留扩展）
        """函数功能：
        将模型生成的token ID序列解码为可读的文本字符串。这是推理时的核心方法，
        负责将generate_ids转换回自然语言文本，并处理stop tokens和response prefix。

        主要流程：
        1. 准备tokenizer参数（默认不在特殊token间添加空格）
        2. 跳过stop tokens（如eos_token、suffix等）
        3. 使用tokenizer解码token ID序列
        4. 可选添加response_prefix（对于首token）
        5. 返回解码后的文本

        参数：
        - generate_ids (List[int]): 模型生成的token ID列表，通常来自model.generate()的输出
        - is_finished (bool): 生成是否已完成，默认True。True时会跳过末尾的stop tokens
            （如eos_token），False时保留所有token（流式生成的中间状态）
        - tokenizer_kwargs (Optional[Dict]): 传递给tokenizer.decode的额外参数字典，
            如{'skip_special_tokens': True}。默认None会设置{'spaces_between_special_tokens': False}
        - first_token (bool): 是否为生成序列的首个token，默认True。True且存在response_prefix
            时会在解码结果前添加prefix（如某些模型要求的特定前缀）
        - **kwargs: 其他额外参数，预留用于未来扩展

        返回值：
        - Any: 通常为str类型的解码文本，某些特殊情况下可能为其他类型

        使用示例：
        >>> # 示例1：标准解码（生成已完成）
        >>> generate_ids = [128000, 15339, 1917, 264, 11364, 128001]
        >>> response = template.decode(generate_ids, is_finished=True)
        >>> # response: "Hello, this is a test."
        >>> 
        >>> # 示例2：流式解码（生成进行中）
        >>> partial_ids = [128000, 15339, 1917]
        >>> partial_response = template.decode(partial_ids, is_finished=False)
        >>> # is_finished=False时不跳过stop tokens，保留所有内容
        >>> 
        >>> # 示例3：跳过特殊token
        >>> response = template.decode(generate_ids, 
        ...                           tokenizer_kwargs={'skip_special_tokens': True})
        >>> 
        >>> # 示例4：禁用response_prefix
        >>> response = template.decode(generate_ids, first_token=False)
        >>> # 不添加response_prefix，直接返回解码结果
        """
        tokenizer_kwargs = tokenizer_kwargs or {}  # 若未提供tokenizer_kwargs则初始化为空字典
        if 'spaces_between_special_tokens' not in tokenizer_kwargs:  # 若未指定特殊token间的空格处理
            tokenizer_kwargs['spaces_between_special_tokens'] = False  # 默认不在特殊token之间添加空格（保持紧凑输出）
        generate_ids = self.skip_stop_tokens(generate_ids, is_finished)  # 跳过stop tokens：移除末尾的eos_token和模板后缀（仅当is_finished=True时）
        response = self.tokenizer.decode(generate_ids, **tokenizer_kwargs)  # 使用tokenizer将token ID列表解码为文本字符串
        if first_token and self.template_meta.response_prefix:  # 若为首token且模板定义了response_prefix
            response = self.template_meta.response_prefix + response  # 在解码结果前添加response_prefix（某些模型需要特定的响应前缀）
        return response  # 返回最终的解码文本

    def decode_prm(self, input_ids: torch.Tensor, logits: torch.Tensor) -> Any:
        raise NotImplementedError
    
    @contextmanager
    def generate_context(self):
        """
        函数功能：
            生成模式的上下文管理器，用于在生成（推理）阶段临时切换模板的工作模式和多模态钩子状态。
            该方法确保在执行生成操作期间，模板处于正确的推理模式（pt），并在多模态模型中临时移除
            post_encode钩子以避免干扰推理过程。执行完成后，自动恢复原始模式和钩子状态。使用Python的
            上下文管理器协议（with语句），保证资源的正确管理和状态恢复。
        
        返回值：
            Generator: 上下文管理器生成器，使用yield暂停执行
                - yield之前：保存原始状态，切换到生成模式，移除多模态钩子
                - yield之后：恢复原始状态和钩子（在finally块中执行）
        
        使用示例：
            >>> # 示例1：在训练模式下临时切换到生成模式
            >>> template = Template(...)
            >>> template.mode = 'train'  # 当前处于训练模式
            >>> print(template.mode)  # 'train'
            >>> 
            >>> with template.generate_context():
            ...     # 在此上下文中，template.mode自动切换为'pt'
            ...     print(template.mode)  # 'pt'
            ...     # 执行生成操作
            ...     output = model.generate(input_ids, max_new_tokens=100)
            >>> 
            >>> # 退出上下文后，自动恢复原始模式
            >>> print(template.mode)  # 'train'
            
            >>> # 示例2：多模态模型的生成
            >>> template.model_meta.is_multimodal = True
            >>> template.mode = 'rlhf'
            >>> 
            >>> # 在生成前，临时移除post_encode钩子
            >>> with template.generate_context():
            ...     # 此时post_encode钩子已被移除，不会干扰生成
            ...     output = model.generate(input_ids, max_new_tokens=100)
            >>> 
            >>> # 退出后，钩子和模式都已恢复
            >>> print(template.mode)  # 'rlhf'
            
            >>> # 示例3：推理模式下使用（mode='pt'，不需要切换）
            >>> template.mode = 'pt'
            >>> 
            >>> with template.generate_context():
            ...     # mode已经是'pt'，不会切换
            ...     print(template.mode)  # 'pt'
            ...     output = model.generate(input_ids, max_new_tokens=100)
            >>> 
            >>> print(template.mode)  # 'pt'（保持不变）
            
            >>> # 示例4：异常处理（即使发生异常也会恢复状态）
            >>> template.mode = 'kto'
            >>> 
            >>> try:
            ...     with template.generate_context():
            ...         print(template.mode)  # 'pt'
            ...         raise ValueError("模拟错误")
            ... except ValueError:
            ...     pass
            >>> 
            >>> # 即使发生异常，模式也已恢复
            >>> print(template.mode)  # 'kto'
        """
        # ===== 步骤1：保存原始模式 =====
        origin_mode = self.mode  # 保存当前的工作模式（train/rlhf/kto/gkd/pt等）
        
        # ===== 步骤2：切换到推理模式（如果需要） =====
        if self.mode in {'train', 'rlhf', 'kto', 'gkd'}:  # 若当前处于训练相关模式
            # 切换到推理模式（pt = pre-training/prediction/推理模式）
            # 生成操作需要在推理模式下进行，训练模式会影响编码行为
            self.set_mode('pt')
        
        # ===== 步骤3：处理多模态模型的钩子 =====
        is_multimodal = self.model_meta.is_multimodal  # 检查是否为多模态模型
        if is_multimodal:  # 若是多模态模型
            # 移除post_encode钩子（在生成阶段不需要这些钩子）
            # post_encode钩子用于训练时的特殊处理，推理时会干扰正常生成
            # 返回被移除钩子的模型列表，用于后续恢复
            models = self.remove_post_encode_hook()
        
        # ===== 步骤4：执行生成操作（yield暂停执行） =====
        try:
            # yield: 暂停执行，返回控制权给with语句块
            # with语句块中的代码在此处执行
            yield
        # ===== 步骤5：清理和恢复（在finally块中确保执行） =====
        finally:
            # finally块保证无论是否发生异常，都会执行恢复操作
            if is_multimodal:  # 若是多模态模型
                # 重新注册post_encode钩子，恢复训练时的状态
                # 使用之前保存的models列表
                self.register_post_encode_hook(models)
            # 恢复原始的工作模式
            # 例如：从'pt'恢复回'train'或'rlhf'等
            self.set_mode(origin_mode)

    def generate(self, model, *args, **kwargs):
        """
        函数功能：
            模型生成的包装方法，用于统一调用不同类型模型的generate方法。该方法自动处理特定模型
            的generate参数兼容性问题，特别是针对支持use_model_defaults参数的模型（如某些transformers
            版本或自定义模型），自动设置该参数为False以避免使用模型默认配置覆盖用户提供的生成参数。
            该方法作为template和model.generate之间的桥梁，确保生成调用的一致性和兼容性。
        
        参数：
            model: 模型对象，可能是原始模型或被包装的模型（如LoRA、PeftModel等）
                包含generate方法用于文本生成
                例如：AutoModelForCausalLM加载的模型实例
            *args: 位置参数，直接传递给model.generate
                通常包括：
                - input_ids (torch.Tensor): 输入token序列，形状 (batch_size, seq_len)
                - attention_mask (torch.Tensor): 注意力掩码，形状 (batch_size, seq_len)
            **kwargs: 关键字参数，传递给model.generate
                常见参数包括：
                - max_new_tokens (int): 最大生成token数量
                - temperature (float): 采样温度
                - top_p (float): nucleus采样参数
                - top_k (int): top-k采样参数
                - do_sample (bool): 是否使用采样
                - num_beams (int): beam search的beam数量
                - generation_config: 生成配置对象
                等其他transformers.GenerationMixin.generate支持的参数
        
        返回值：
            torch.Tensor 或其他类型: 模型生成的结果，具体类型取决于model.generate的返回值
                - 标准情况：torch.Tensor，形状 (batch_size, total_seq_len)
                    包含输入token和新生成的token
                    例如：tensor([[151644, 8948, 872, ...]])
                - 特殊情况：可能返回GenerateOutput对象（当return_dict_in_generate=True时）
                    包含sequences、scores等详细信息
        
        使用示例：
            >>> # 示例1：基础生成
            >>> from transformers import AutoModelForCausalLM, AutoTokenizer
            >>> template = Template(...)
            >>> model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
            >>> tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
            >>> 
            >>> # 准备输入
            >>> input_text = "你好，请介绍一下自己"
            >>> input_ids = tokenizer.encode(input_text, return_tensors='pt')  # 形状: (1, seq_len)
            >>> 
            >>> # 使用template.generate调用
            >>> output_ids = template.generate(model, input_ids, max_new_tokens=100, 
            ...                                temperature=0.7, do_sample=True)
            >>> # output_ids形状: (1, seq_len + new_tokens_len)
            >>> 
            >>> # 解码输出
            >>> output_text = tokenizer.decode(output_ids[0])
            >>> print(output_text)
            
            >>> # 示例2：批量生成
            >>> input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 形状: (2, 3)
            >>> attention_mask = torch.ones_like(input_ids)  # 形状: (2, 3)
            >>> 
            >>> output_ids = template.generate(model, input_ids, attention_mask=attention_mask,
            ...                                max_new_tokens=50, num_beams=5)
            >>> # output_ids形状: (2, 3 + new_tokens_len)
            
            >>> # 示例3：使用generation_config
            >>> from transformers import GenerationConfig
            >>> generation_config = GenerationConfig(
            ...     max_new_tokens=100,
            ...     temperature=0.7,
            ...     top_p=0.9,
            ...     do_sample=True
            ... )
            >>> output_ids = template.generate(model, input_ids, 
            ...                                generation_config=generation_config)
            
            >>> # 示例4：返回详细生成信息
            >>> output = template.generate(model, input_ids, max_new_tokens=50,
            ...                            return_dict_in_generate=True, output_scores=True)
            >>> print(type(output))  # <class 'transformers.generation.GenerateOutput'>
            >>> print(output.sequences.shape)  # (batch_size, total_seq_len)
            >>> print(len(output.scores))  # 每个生成步骤的logits
            
            >>> # 示例5：LoRA模型的生成
            >>> from peft import get_peft_model
            >>> peft_model = get_peft_model(model, peft_config)  # 包装的模型
            >>> # template.generate会自动处理包装模型，获取base_model
            >>> output_ids = template.generate(peft_model, input_ids, max_new_tokens=100)
        """
        # ===== 步骤1：获取基础模型 =====
        # 从可能被包装的模型（如LoRA、PeftModel）中提取原始的基础模型
        # 某些包装模型的generate方法签名可能与基础模型不同
        base_model = self.get_base_model(model)
        
        # ===== 步骤2：检查基础模型的generate方法签名 =====
        # 使用inspect获取基础模型generate方法的签名信息
        # signature包含方法的所有参数信息（名称、类型、默认值等）
        signature = inspect.signature(base_model.generate)
        
        # ===== 步骤3：处理use_model_defaults参数（兼容性处理） =====
        # 检查两个条件：
        # 1. 基础模型的generate方法是否支持use_model_defaults参数
        # 2. 用户是否已经在kwargs中提供了use_model_defaults参数
        if 'use_model_defaults' in signature.parameters and 'use_model_defaults' not in kwargs:
            # 若模型支持该参数且用户未提供，则设置为False
            # use_model_defaults=False: 不使用模型的默认配置，避免覆盖用户传入的参数
            # 这确保用户通过kwargs传递的参数（如temperature、top_p等）能够生效
            kwargs['use_model_defaults'] = False
        
        # ===== 步骤4：调用模型的generate方法并返回结果 =====
        # 注意：这里使用原始的model（可能是包装模型），而非base_model
        # 因为包装模型（如PeftModel）可能对generate方法有特殊处理
        # *args: 展开位置参数（如input_ids, attention_mask）
        # **kwargs: 展开关键字参数（如max_new_tokens, temperature等）
        return model.generate(*args, **kwargs)
    def skip_stop_tokens(self, generate_ids: List[int], is_finished: bool = True) -> List[int]:
        """
        函数功能：
            跳过（移除）生成序列末尾的停止token，包括EOS token和模板后缀token。该方法用于清理
            模型生成的输出，移除不应该显示给用户的特殊token。根据生成是否完成（is_finished），
            采用不同的处理策略：完成时移除完整的后缀token，未完成时移除部分匹配的后缀token（用于
            流式生成场景）。注意：其他stop_words会被保留并打印，只移除suffix和eos_token。
        
        参数：
            generate_ids (List[int]): 模型生成的token ID序列
                包含完整的生成结果，可能包含EOS token和模板后缀
                例如：[151644, 872, 151643, 151644, 2670, 151643]
                      其中151643可能是<|im_end|>，需要被移除
            is_finished (bool): 生成是否已完成，默认True
                - True: 生成已完成，移除完整的后缀token序列
                    例如：生成结束，移除完整的<|im_end|>
                - False: 生成未完成（流式生成中），移除部分匹配的后缀
                    例如：流式输出时，可能只生成了后缀的一部分，需要移除这部分
        
        返回值：
            List[int]: 清理后的token ID序列，移除了EOS token和模板后缀
                例如：输入 [151644, 872, 151643]，后缀是[151643]
                      输出 [151644, 872]（移除了后缀token）
        
        使用示例：
            >>> # 示例1：生成完成，移除完整后缀
            >>> template = Template(...)
            >>> template.template_meta.suffix = [['<|im_end|>']]
            >>> # 假设<|im_end|>的token ID是151643，eos_token_id是151645
            >>> generate_ids = [151644, 8948, 872, 151643]  # 生成的序列
            >>> result = template.skip_stop_tokens(generate_ids, is_finished=True)
            >>> print(result)  # [151644, 8948, 872]（移除了后缀151643）
            
            >>> # 示例2：生成完成，同时有EOS token和后缀
            >>> generate_ids = [151644, 8948, 872, 151643, 151645]  # 末尾有eos_token
            >>> result = template.skip_stop_tokens(generate_ids, is_finished=True)
            >>> print(result)  # [151644, 8948, 872]（移除了eos和后缀）
            
            >>> # 示例3：流式生成未完成，移除部分匹配的后缀
            >>> # 假设后缀是[100, 101]（两个token）
            >>> template.template_meta.suffix = [[100, 101]]
            >>> generate_ids = [1, 2, 3, 100]  # 只生成了后缀的第一个token
            >>> result = template.skip_stop_tokens(generate_ids, is_finished=False)
            >>> print(result)  # [1, 2, 3]（移除了部分匹配的100）
            
            >>> # 示例4：流式生成未完成，没有后缀匹配
            >>> generate_ids = [1, 2, 3, 4]  # 末尾不是后缀的任何部分
            >>> result = template.skip_stop_tokens(generate_ids, is_finished=False)
            >>> print(result)  # [1, 2, 3, 4]（不修改）
            
            >>> # 示例5：空序列
            >>> generate_ids = []
            >>> result = template.skip_stop_tokens(generate_ids, is_finished=True)
            >>> print(result)  # []（返回空列表）
            
            >>> # 示例6：多token后缀（如"\n { "）
            >>> # 假设后缀是[198, 2]（换行符+ڕ ）
            >>> template.template_meta.suffix = [[198, 2]]
            >>> generate_ids = [1, 2, 3, 198, 2]  # 完整后缀
            >>> result = template.skip_stop_tokens(generate_ids, is_finished=True)
            >>> print(result)  # [1, 2, 3]（移除了两个token的后缀）
        """
        # 注释说明：
        # - 不打印（移除）template_meta.suffix[-1]和eos_token
        # - 但是，其他stop_words会被保留并打印
        
        # ===== 步骤1：获取tokenizer引用 =====
        tokenizer = self.tokenizer  # 获取tokenizer对象，用于访问eos_token_id

        # ===== 步骤2：移除末尾的EOS token（如果存在） =====
        # 检查序列是否非空且最后一个token是eos_token
        if len(generate_ids) > 0 and generate_ids[-1] == tokenizer.eos_token_id:
            # 移除最后一个token（EOS token）
            # 例如：[1, 2, 3, eos_id] -> [1, 2, 3]
            generate_ids = generate_ids[:-1]
        
        # ===== 步骤3：获取并处理模板后缀 =====
        # 注释：跳过suffix和eos_token
        # 获取模板后缀的最后一个元素（通常是结束标记，如<|im_end|>）
        template_suffix = self.template_meta.suffix[-1]
        
        if isinstance(template_suffix, str):  # 若后缀是字符串（尚未转换为token ID）
            # 将后缀字符串编码为token ID列表
            # add_special_tokens=False: 不添加特殊token（如BOS/EOS）
            # [-1:]: 只取最后一个token（修复某些模型的兼容性问题）
            # 注释：[-1:]: fix OpenGVLab/Mini-InternVL-Chat-4B-V1-5
            # 某些模型的后缀可能被编码为多个token，但只需要最后一个
            template_suffix = tokenizer.encode(template_suffix, add_special_tokens=False)[-1:]
        
        # ===== 步骤4：计算后缀长度 =====
        len_tokens = len(template_suffix)  # 后缀包含的token数量

        # ===== 步骤5：根据生成状态移除后缀token =====
        if is_finished and generate_ids[-len_tokens:] == template_suffix:
            # 情况1：生成已完成，且末尾完全匹配后缀
            # generate_ids[-len_tokens:]: 获取末尾len_tokens个token
            # 例如：generate_ids=[1,2,3,100,101], suffix=[100,101], len_tokens=2
            #       -> generate_ids[-2:] = [100,101]，匹配成功
            # 移除末尾的后缀token
            generate_ids = generate_ids[:-len_tokens]
            # 结果：[1, 2, 3]
        elif not is_finished:
            # 情况2：生成未完成（流式生成），可能只生成了后缀的一部分
            # 需要检查并移除部分匹配的后缀
            
            # 从后缀的完整长度开始，逐渐减少，检查是否有部分匹配
            # range(len_tokens, 0, -1): 从len_tokens到1，递减
            # 例如：len_tokens=3 -> 遍历3, 2, 1
            # range(start, stop, step) [start, stop)
            for i in range(len_tokens, 0, -1):
                # 检查序列末尾的i个token是否匹配后缀的前i个token
                # generate_ids[-i:]: 末尾i个token
                # template_suffix[:i]: 后缀的前i个token
                # 例如：i=2, generate_ids末尾=[100,101], suffix前2个=[100,101]
                if generate_ids[-i:] == template_suffix[:i]:
                    # 找到部分匹配，移除末尾的i个token
                    generate_ids = generate_ids[:-i]
                    break  # 找到匹配后立即退出循环
                # 如果没有找到匹配，继续尝试更短的长度
        
        # ===== 步骤6：返回清理后的序列 =====
        return generate_ids

    def prepare_generate_kwargs(self, generate_kwargs: Dict[str, Any], *, model=None) -> Dict[str, Any]:
        """
        函数功能：
            准备和配置模型生成所需的参数字典，主要负责设置停止词（stop words）的停止条件。该方法
            从generation_config或template_meta中获取停止词列表，创建StopWordsCriteria停止判断器，
            并将其添加到stopping_criteria列表中。这确保生成过程在遇到指定的停止词时能够正确终止，
            避免生成不必要的内容。该方法是生成前的必要准备步骤，统一管理停止条件的配置。
        
        参数：
            generate_kwargs (Dict[str, Any]): 生成参数字典，包含传递给model.generate的所有参数
                必须包含的键：
                - 'generation_config': GenerationConfig对象，包含生成配置
                    可能包含stop_words属性（自定义停止词列表）
                可能包含的其他键：
                - 'max_new_tokens': 最大生成token数
                - 'temperature': 采样温度
                - 'top_p', 'top_k': 采样参数
                - 'input_ids': 输入token序列
                等其他生成参数
            model (optional): 模型对象，当前版本未使用，预留参数
                用于未来可能需要根据模型类型调整生成配置的场景
        
        返回值：
            Dict[str, Any]: 更新后的生成参数字典，在原字典基础上添加了停止条件
                新增或更新的键：
                - 'stopping_criteria': StoppingCriteriaList对象
                    包含StopWordsCriteria停止判断器
                    在生成过程中检查是否遇到停止词
                保留原有的所有其他键值对
        
        使用示例：
            >>> # 示例1：基础使用，从template_meta获取停止词
            >>> from transformers import GenerationConfig
            >>> template = Template(...)
            >>> template.template_meta.stop_words = ['<|im_end|>', '<|endoftext|>']
            >>> 
            >>> # 准备生成参数
            >>> generation_config = GenerationConfig(max_new_tokens=100, temperature=0.7)
            >>> generate_kwargs = {
            ...     'generation_config': generation_config,
            ...     'input_ids': torch.tensor([[1, 2, 3]]),
            ...     'do_sample': True
            ... }
            >>> 
            >>> # 准备生成参数，添加停止条件
            >>> generate_kwargs = template.prepare_generate_kwargs(generate_kwargs)
            >>> print('stopping_criteria' in generate_kwargs)  # True
            >>> print(type(generate_kwargs['stopping_criteria']))  
            # <class 'transformers.StoppingCriteriaList'>
            
            >>> # 示例2：generation_config包含自定义停止词（优先级更高）
            >>> generation_config = GenerationConfig(max_new_tokens=100)
            >>> generation_config.stop_words = ['STOP', 'END']  # 自定义停止词
            >>> generate_kwargs = {
            ...     'generation_config': generation_config,
            ...     'input_ids': torch.tensor([[1, 2, 3]])
            ... }
            >>> 
            >>> # 使用generation_config中的停止词（而非template_meta的）
            >>> generate_kwargs = template.prepare_generate_kwargs(generate_kwargs)
            >>> # 内部的StopWordsCriteria使用['STOP', 'END']作为停止词
            
            >>> # 示例3：与model.generate配合使用
            >>> from transformers import AutoModelForCausalLM
            >>> model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
            >>> 
            >>> # 准备输入和配置
            >>> input_ids = tokenizer.encode("你好", return_tensors='pt')
            >>> generation_config = GenerationConfig(max_new_tokens=50)
            >>> generate_kwargs = {
            ...     'generation_config': generation_config,
            ...     'input_ids': input_ids
            ... }
            >>> 
            >>> # 准备参数（添加停止条件）
            >>> generate_kwargs = template.prepare_generate_kwargs(generate_kwargs)
            >>> 
            >>> # 生成（会在遇到停止词时自动停止）
            >>> output_ids = model.generate(**generate_kwargs)
            
            >>> # 示例4：检查停止条件的内容
            >>> generate_kwargs = template.prepare_generate_kwargs(generate_kwargs)
            >>> stopping_criteria = generate_kwargs['stopping_criteria']
            >>> print(len(stopping_criteria))  # 1（包含一个StopWordsCriteria）
            >>> stop_words_criteria = stopping_criteria[0]
            >>> print(type(stop_words_criteria))  
            # <class 'swift.llm.template.utils.StopWordsCriteria'>
        """
        # ===== 步骤1：从generate_kwargs中提取generation_config =====
        # generation_config包含所有生成相关的配置参数
        # 例如：max_new_tokens, temperature, top_p等
        generation_config = generate_kwargs['generation_config']
        
        # ===== 步骤2：获取停止词列表（优先级：generation_config > template_meta） =====
        # 使用getattr安全获取generation_config的stop_words属性
        # - 若generation_config有stop_words属性且不为None，使用该值（优先级高）
        # - 否则使用template_meta.stop_words作为默认值
        # 停止词列表示例：['<|im_end|>', '<|endoftext|>', 'ڕ ']
        stop_words = getattr(generation_config, 'stop_words', None) or self.template_meta.stop_words
        
        # ===== 步骤3：创建停止条件列表并添加到generate_kwargs =====
        # StoppingCriteriaList: transformers库的停止条件列表容器
        # StopWordsCriteria: 自定义的停止词判断器，检查生成的token是否匹配停止词
        # - 参数1：self.tokenizer - 用于将停止词编码为token ID
        # - 参数2：stop_words - 停止词列表（字符串形式）
        # StopWordsCriteria会在生成每个token后检查序列末尾是否出现停止词
        # 若匹配到停止词，则终止生成过程
        generate_kwargs['stopping_criteria'] = StoppingCriteriaList([StopWordsCriteria(self.tokenizer, stop_words)])
        
        # ===== 步骤4：返回更新后的generate_kwargs =====
        # 返回的字典包含原有的所有参数，加上新添加的stopping_criteria
        return generate_kwargs

    @staticmethod
    def _save_pil_image(image: Image.Image) -> str:
        """
        函数功能：
            将PIL.Image对象保存为临时PNG文件并返回文件路径。该方法使用图像内容的SHA256哈希值作为
            文件名，确保相同内容的图像只保存一次（去重），避免重复存储。临时文件保存在ModelScope的
            缓存目录下的'tmp/images'子目录中。该方法主要用于将内存中的PIL.Image对象持久化为文件，
            以满足某些模型（如PyTorch原生模型、Qwen-VL）对图像输入格式为文件路径的要求。
        
        参数：
            image (Image.Image): PIL.Image对象，需要保存的图像
                - 可以是任意PIL支持的图像格式（RGB、RGBA、L等）
                - 图像数据将被转换为字节流用于计算哈希值
                - 最终以PNG格式保存（无损压缩）
        
        返回值：
            str: 保存的图像文件的绝对路径
                格式：<cache_dir>/tmp/images/<sha256_hash>.png
                例如：'/home/user/.cache/modelscope/tmp/images/a1b2c3d4...png'
                - 若文件已存在（相同内容的图像之前已保存），直接返回现有路径
                - 若文件不存在，保存图像后返回新文件路径
        """
        # ===== 步骤1：将PIL.Image对象转换为字节流 =====
        # tobytes(): 将图像的像素数据转换为原始字节序列
        # 字节流包含所有像素的RGB/RGBA值，用于后续计算哈希值
        img_bytes = image.tobytes()
        
        # ===== 步骤2：计算图像内容的SHA256哈希值 =====
        # hashlib.sha256(): 创建SHA256哈希对象
        # hexdigest(): 返回16进制格式的哈希字符串（64个字符）
        # 哈希值作为文件名，相同内容的图像生成相同哈希，实现自动去重
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        
        # ===== 步骤3：构建临时图像存储目录路径 =====
        # get_cache_dir(): 获取ModelScope缓存根目录（通常为~/.cache/modelscope）
        # 临时图像目录结构：<cache_dir>/tmp/images/
        tmp_dir = os.path.join(get_cache_dir(), 'tmp', 'images')
        
        # ===== 步骤4：记录目录创建信息（仅首次记录） =====
        # logger.info_once(): 确保相同的日志信息只输出一次，避免重复日志
        logger.info_once(f'create tmp_dir: {tmp_dir}')
        
        # ===== 步骤5：创建临时目录（如果不存在） =====
        # exist_ok=True: 如果目录已存在不抛出异常，确保目录可用
        os.makedirs(tmp_dir, exist_ok=True)
        
        # ===== 步骤6：构建完整的图像文件路径 =====
        # 文件名格式：<sha256_hash>.png
        # 使用PNG格式保存，因为PNG是无损压缩格式，适合中间数据存储
        img_path = os.path.join(tmp_dir, f'{img_hash}.png')
        
        # ===== 步骤7：检查文件是否已存在，避免重复保存 =====
        if not os.path.exists(img_path):  # 如果文件不存在（新图像或缓存已清理）
            # 将PIL.Image对象保存为PNG文件
            # image.save(): PIL的保存方法，根据文件扩展名自动选择格式
            image.save(img_path)
        # 如果文件已存在，跳过保存步骤（利用哈希去重机制）
        
        # ===== 步骤8：返回图像文件的绝对路径 =====
        return img_path

    @staticmethod
    def _concat_context_list(
            context_list: List[Context],
            res_context_list: List[Context],  # inplace
            res_context_type: List[ContextType],  # inplace
            system: Optional[str] = None,
            query: Optional[str] = None,
            response: Optional[str] = None,
            round0: Optional[int] = None) -> None:
        """
        函数功能：
            拼接上下文列表并替换模板占位符。该方法遍历输入的上下文列表（context_list），将其中的
            模板占位符（如{{SYSTEM}}、{{QUERY}}、{{RESPONSE}}等）替换为实际的内容（系统提示、
            用户问题、助手回复等），并将处理后的结果追加到结果列表（res_context_list和
            res_context_type）中。这是模板系统构建完整对话上下文的核心方法，通过占位符机制实现
            灵活的模板定义和动态内容填充。该方法采用原地修改（inplace）方式，直接修改传入的结果
            列表，不返回新列表。
        参数：
            context_list (List[Context]): 输入的上下文列表，包含模板字符串或token列表
                - Context类型为Union[str, List[int]]，即字符串或token ID列表
                - 可能包含占位符：
                  * '{{RESPONSE}}': 助手回复的占位符（完全匹配）
                  * '{{SYSTEM}}': 系统提示的占位符（字符串内匹配）
                  * '{{QUERY}}': 用户问题的占位符（字符串内匹配）
                  * '{{ROUND0}}': 当前轮次索引（从0开始）的占位符
                  * '{{ROUND1}}': 当前轮次编号（从1开始）的占位符
                - 示例：['<|system|>', '{{SYSTEM}}', '<|user|>', '{{QUERY}}', '<|assistant|>']
            
            res_context_list (List[Context]): 结果上下文列表（原地修改）
                - 函数会将处理后的上下文追加到此列表
                - 占位符会被替换为实际内容
                - 空内容（长度为0）会被跳过
                - 调用前通常为空列表，函数执行后包含完整的上下文序列
            
            res_context_type (List[ContextType]): 结果上下文类型列表（原地修改）
                - 与res_context_list一一对应，标记每个上下文的类型
                - 可能的类型：
                  * ContextType.RESPONSE: 助手回复内容（用于计算loss）
                  * ContextType.OTHER: 其他内容（系统提示、用户问题、模板标签等）
                - 用于后续的loss_scale计算和labels生成
            
            system (Optional[str]): 系统提示文本，用于替换{{SYSTEM}}占位符
                - 默认为None（不替换）
                - 示例：'You are a helpful assistant.'
            
            query (Optional[str]): 用户问题文本，用于替换{{QUERY}}占位符
                - 默认为None（不替换）
                - 示例：'请介绍一下Python语言'
            
            response (Optional[str]): 助手回复文本，用于替换{{RESPONSE}}占位符
                - 默认为None（不替换）
                - 若context_list包含'{{RESPONSE}}'但response为None，会触发断言错误
                - 示例：'Python是一种高级编程语言...'
            
            round0 (Optional[int]): 当前对话轮次索引（从0开始）
                - 用于替换{{ROUND0}}和{{ROUND1}}占位符
                - round0为原始索引，round1 = round0 + 1为人类可读的轮次编号
                - 默认为None（不替换轮次占位符）
                - 示例：round0=0表示第一轮对话，会替换为'0'和'1'
        
        返回值：
            None: 该方法无返回值，通过原地修改res_context_list和res_context_type实现结果传递
        
        使用示例：
            >>> # 示例1：基础使用 - 替换系统提示和用户问题
            >>> context_list = ['<|system|>', '{{SYSTEM}}', '<|user|>', '{{QUERY}}', '<|assistant|>']
            >>> res_context_list = []
            >>> res_context_type = []
            >>> 
            >>> Template._concat_context_list(
            ...     context_list=context_list,
            ...     res_context_list=res_context_list,
            ...     res_context_type=res_context_type,
            ...     system='You are a helpful assistant.',
            ...     query='Hello, how are you?'
            ... )
            >>> 
            >>> print(res_context_list)
            # ['<|system|>', 'You are a helpful assistant.', '<|user|>', 'Hello, how are you?', '<|assistant|>']
            >>> print(res_context_type)
            # [ContextType.OTHER, ContextType.OTHER, ContextType.OTHER, ContextType.OTHER, ContextType.OTHER]
            
            >>> # 示例2：替换助手回复（训练场景）
            >>> context_list = ['<|user|>', '{{QUERY}}', '<|assistant|>', '{{RESPONSE}}', '<|end|>']
            >>> res_context_list = []
            >>> res_context_type = []
            >>> 
            >>> Template._concat_context_list(
            ...     context_list=context_list,
            ...     res_context_list=res_context_list,
            ...     res_context_type=res_context_type,
            ...     query='What is AI?',
            ...     response='AI stands for Artificial Intelligence.'
            ... )
            >>> 
            >>> print(res_context_list)
            # ['<|user|>', 'What is AI?', '<|assistant|>', 'AI stands for Artificial Intelligence.', '<|end|>']
            >>> print(res_context_type)
            # [ContextType.OTHER, ContextType.OTHER, ContextType.OTHER, ContextType.RESPONSE, ContextType.OTHER]
            # 注意：response的类型为ContextType.RESPONSE，用于标记需要计算loss的部分
        """
        # ===== 步骤1：计算轮次编号（round1 = round0 + 1） =====
        # round0: 轮次索引（从0开始），用于程序内部
        # round1: 轮次编号（从1开始），用于用户可读的显示
        round1 = None
        if round0 is not None:  # 如果提供了轮次索引
            # 将整数转换为字符串，便于后续字符串替换
            round1 = str(round0 + 1)  # 第一轮对话：round0=0, round1='1'
            round0 = str(round0)       # round0='0'
        
        # ===== 步骤2：遍历上下文列表，替换占位符并追加到结果列表 =====
        for context in context_list:
            # ===== 步骤2.1：处理字符串类型的上下文（需要替换占位符） =====
            if isinstance(context, str):  # 如果是字符串（而非token ID列表）
                # ===== 步骤2.1.1：特殊处理{{RESPONSE}}占位符 =====
                # {{RESPONSE}}必须完全匹配，且单独作为一个上下文元素
                if '{{RESPONSE}}' == context:  # 完全相等判断（不是字符串包含）
                    # 断言response参数不为None，否则无法替换
                    assert response is not None
                    # 将助手回复追加到结果列表
                    res_context_list.append(response)
                    # 标记类型为RESPONSE，用于后续loss计算（仅对回复部分计算loss）
                    res_context_type.append(ContextType.RESPONSE)
                    # 跳过后续处理，进入下一个context
                    continue
                
                # ===== 步骤2.1.2：替换其他占位符（支持字符串内包含） =====
                # 定义需要替换的占位符列表和对应的实际值列表
                old_str_list = ['{{SYSTEM}}', '{{QUERY}}', '{{ROUND0}}', '{{ROUND1}}']
                new_str_list = [system, query, round0, round1]
                
                # 遍历所有占位符，执行字符串替换
                for (old_str, new_str) in zip(old_str_list, new_str_list):
                    # 只有当新值不为None且占位符存在于字符串中时才替换
                    if new_str is not None and old_str in context:
                        # 断言new_str必须是字符串类型（因为round0/round1已转为字符串）
                        assert isinstance(new_str, str), f'new_str: {new_str}'
                        # 执行字符串替换
                        # 示例：'User {{ROUND0}}: {{QUERY}}' -> 'User 0: Hello'
                        context = context.replace(old_str, new_str)
            
            # ===== 步骤2.2：跳过空内容 =====
            # 如果替换后的内容为空字符串或空列表，跳过不添加
            if len(context) == 0:
                continue
            
            # ===== 步骤2.3：将处理后的上下文追加到结果列表 =====
            res_context_list.append(context)
            # 标记类型为OTHER（非RESPONSE的所有内容）
            # 注意：{{RESPONSE}}的类型在步骤2.1.1中已设置为RESPONSE，不会执行到这里
            res_context_type.append(ContextType.OTHER)
    def _simplify_context_list(self, context_list: List[Context], loss_scale_list: List[float],
                               inputs: StdTemplateInputs) -> Tuple[List[Context], List[float]]:
        """
        函数功能：
            简化和优化上下文列表，通过预处理和合并操作提高编码效率。该方法执行三个核心步骤：
            (1) 分割特殊标签（如<image>、<video>、<audio>），将包含特殊标签的字符串拆分为多个独立
            的上下文元素，便于后续的标签替换操作；
            (2) 预分词处理（pre_tokenize），将特殊标签替换为模型所需的实际内容或token ID（如将
            <image>替换为图像token序列）；
            (3) 合并相邻的字符串上下文，将具有相同loss_scale的连续字符串合并为单个字符串，减少
            上下文元素数量，降低后续tokenization的调用次数，提高处理效率。
            
            该方法是编码流程中的重要优化步骤，在保持语义和loss_scale不变的前提下，最小化上下文
            列表的长度，为后续的_encode_context_list方法提供更高效的输入格式。
        
        参数：
            context_list (List[Context]): 输入的上下文列表
                - Context类型为Union[str, List[int]]，即字符串或token ID列表
                - 可能包含特殊标签（如<image>、<video>、<audio>）
                - 可能包含多个相邻的字符串片段
                - 示例：['<|system|>', 'You are helpful', '<image>', 'Describe this', [1, 2, 3]]
            
            loss_scale_list (List[float]): loss权重列表
                - 与context_list一一对应，每个上下文元素有一个loss_scale值
                - loss_scale=0.0表示不参与loss计算（如系统提示、用户问题）
                - loss_scale=1.0表示正常参与loss计算（如助手回复）
                - 可以有其他值表示不同的权重（如0.5表示半权重）
                - 示例：[0.0, 0.0, 0.0, 1.0, 1.0]（前3个不计算loss，后2个计算loss）
            
            inputs (StdTemplateInputs): 标准模板输入对象
                - 包含messages（对话消息列表）
                - 包含多模态数据（images、videos、audios等）
                - 用于_pre_tokenize方法进行标签替换和内容填充
                - 提供上下文信息用于特殊标签的处理
        
        返回值：
            Tuple[List[Context], List[float]]: 简化后的上下文列表和对应的loss_scale列表
                - 第一个元素：简化后的上下文列表
                  * 特殊标签已被分割和替换
                  * 相邻的同loss_scale字符串已被合并
                  * 列表长度通常小于输入的context_list
                  * 示例：['<|system|>You are helpful', [image_tokens], 'Describe this', [1, 2, 3]]
                - 第二个元素：对应的loss_scale列表
                  * 与简化后的上下文列表一一对应
                  * 长度与简化后的上下文列表相同
                  * 示例：[0.0, 0.0, 1.0, 1.0]（合并后元素减少）
        
        使用示例：
            >>> # 示例1：合并相邻的相同loss_scale字符串
            >>> template = Template(...)
            >>> context_list = ['Hello', ' ', 'world', '!', '<image>', 'Describe']
            >>> loss_scale_list = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            >>> inputs = StdTemplateInputs(messages=[...], images=['image.jpg'])
            >>> 
            >>> simplified_context, simplified_loss = template._simplify_context_list(
            ...     context_list, loss_scale_list, inputs
            ... )
            >>> print(simplified_context)
            # ['Hello world!', [image_token_ids], 'Describe']
            # 注意：前4个相同loss_scale的字符串被合并为1个
            >>> print(simplified_loss)
            # [0.0, 0.0, 1.0]
            
            >>> # 示例2：处理包含token ID列表的上下文
            >>> context_list = ['System: ', 'Be helpful', [151643], 'User: ', 'Hello']
            >>> loss_scale_list = [0.0, 0.0, 0.0, 0.0, 1.0]
            >>> inputs = StdTemplateInputs(messages=[...])
            >>> 
            >>> simplified_context, simplified_loss = template._simplify_context_list(
            ...     context_list, loss_scale_list, inputs
            ... )
            >>> print(simplified_context)
            # ['System: Be helpful', [151643], 'User: ', 'Hello']
            # 注意：token ID列表([151643])不能与字符串合并，所以前2个字符串合并，
            #       后面的'User: '和'Hello'因loss_scale不同而未合并
            >>> print(simplified_loss)
            # [0.0, 0.0, 0.0, 1.0]
        """
        # ===== 步骤1：分割特殊标签（如<image>、<video>、<audio>） =====
        # _split_special_tokens方法将包含特殊标签的字符串拆分为多个元素
        # 例如：'Hello<image>world' -> ['Hello', '<image>', 'world']
        # 这样可以单独处理特殊标签，便于后续的replace_tag操作
        context_list, loss_scale_list = self._split_special_tokens(context_list, loss_scale_list)

        # ===== 步骤2：预分词处理（替换特殊标签为实际内容/token） =====
        # _pre_tokenize方法将特殊标签替换为模型所需的内容
        # 例如：'<image>' -> [image_token_ids]（模型的图像token序列）
        #       '<video>' -> [video_token_ids]
        # 同时处理grounding任务的bbox归一化、object/box标签替换等
        context_list, loss_scale_list = self._pre_tokenize(context_list, loss_scale_list, inputs)

        # ===== 步骤3：合并相邻的同loss_scale字符串 =====
        # 初始化结果列表
        res: List[Context] = []  # 简化后的上下文列表
        res_loss_scale: List[float] = []  # 简化后的loss_scale列表
        
        # 初始化临时缓冲区（用于累积相邻的字符串）
        temp: List[str] = []  # 临时存储待合并的字符串片段
        temp_loss_scale = 0.  # 当前临时缓冲区的loss_scale值
        
        # ===== 步骤3.1：遍历上下文列表，合并相邻的字符串 =====
        for i, (context, loss_scale) in enumerate(zip(context_list, loss_scale_list)):
            # ===== 条件1：可以合并到临时缓冲区 =====
            # 当前context是字符串 且 loss_scale与缓冲区的loss_scale相同
            if isinstance(context, str) and (loss_scale == temp_loss_scale):
                # 将字符串添加到临时缓冲区，等待后续合并
                temp.append(context)
            
            # ===== 条件2：不能合并（需要刷新缓冲区） =====
            else:
                # ===== 步骤3.1.1：刷新临时缓冲区（如果非空） =====
                if len(temp) > 0:
                    # 将缓冲区中的所有字符串合并为一个字符串
                    # 例如：['Hello', ' ', 'world'] -> 'Hello world'
                    res.append(''.join(temp))
                    # 添加对应的loss_scale
                    res_loss_scale.append(temp_loss_scale)
                    # 清空缓冲区，准备下一轮累积
                    temp.clear()
                
                # ===== 步骤3.1.2：处理当前context =====
                if isinstance(context, str):  # 当前是字符串，但loss_scale不同
                    # 将当前字符串加入临时缓冲区（开始新的累积）
                    temp.append(context)
                else:  # 当前是token ID列表（不能合并）
                    # 直接添加到结果列表
                    res.append(context)
                    res_loss_scale.append(loss_scale)
                
                # 更新临时缓冲区的loss_scale为当前值
                temp_loss_scale = loss_scale
        
        # ===== 步骤3.2：处理剩余的临时缓冲区 =====
        # 遍历结束后，如果临时缓冲区还有内容，需要刷新
        if len(temp) > 0:
            # 合并并添加到结果列表
            res.append(''.join(temp))
            res_loss_scale.append(temp_loss_scale)
        return res, res_loss_scale

    @staticmethod
    def _split_special_tokens(context_list: List[Context],
                              loss_scale_list: List[float]) -> Tuple[List[Context], List[float]]:
        """
        函数功能：
            分割字符串中的特殊标签，将包含特殊标签的字符串拆分为多个独立的上下文元素。该方法识别
            并提取字符串中的特殊标签（如<image>、<video>、<audio>等），将每个特殊标签和其周围的
            文本内容分离为独立的上下文元素，便于后续的replace_tag操作进行标签替换。例如，将
            'Hello<image>world'拆分为['Hello', '<image>', 'world']三个独立元素。这种拆分使得
            特殊标签可以被单独识别和处理，为多模态数据的标签替换提供基础。
            
            该方法仅处理字符串类型的上下文，对于token ID列表类型的上下文（已经是token形式）则
            保持不变。拆分后的所有元素继承原上下文的loss_scale值，确保loss权重在拆分过程中不会丢失。
        
        参数：
            context_list (List[Context]): 输入的上下文列表
                - Context类型为Union[str, List[int]]，即字符串或token ID列表
                - 字符串类型的上下文可能包含特殊标签（需要拆分）
                - token ID列表类型的上下文不需要拆分（已经是token形式）
                - 示例：['Hello<image>world', [1, 2, 3], 'Text<video>']
            
            loss_scale_list (List[float]): loss权重列表
                - 与context_list一一对应，每个上下文元素有一个loss_scale值
                - 拆分后，所有拆分出的元素继承原元素的loss_scale值
                - 示例：[0.0, 1.0, 0.5]
        
        返回值：
            Tuple[List[Context], List[float]]: 拆分后的上下文列表和对应的loss_scale列表
                - 第一个元素：拆分后的上下文列表
                  * 字符串中的特殊标签已被分离为独立元素
                  * 列表长度通常大于输入的context_list（因为拆分）
                  * 空字符串和空标签会被过滤掉
                  * 示例：['Hello', '<image>', 'world', [1, 2, 3], 'Text', '<video>']
                - 第二个元素：对应的loss_scale列表
                  * 与拆分后的上下文列表一一对应
                  * 拆分出的元素继承原元素的loss_scale值
                  * 示例：[0.0, 0.0, 0.0, 1.0, 0.5, 0.5]（第1个元素拆分为3个，都是0.0）
        
        使用示例：
            >>> # 示例1：拆分包含单个特殊标签的字符串
            >>> context_list = ['Hello<image>world', 'Some text']
            >>> loss_scale_list = [0.0, 1.0]
            >>> 
            >>> res_context, res_loss = Template._split_special_tokens(context_list, loss_scale_list)
            >>> print(res_context)
            # ['Hello', '<image>', 'world', 'Some text']
            # 第1个元素被拆分为3个独立元素：文本-标签-文本
            >>> print(res_loss)
            # [0.0, 0.0, 0.0, 1.0]
            # 拆分出的3个元素都继承原来的loss_scale=0.0
            
            >>> # 示例2：拆分包含多个特殊标签的字符串，并混合token列表
            >>> context_list = ['Image:<image>Video:<video>End', [151643, 151644], 'Audio:<audio>']
            >>> loss_scale_list = [0.0, 1.0, 0.5]
            >>> 
            >>> res_context, res_loss = Template._split_special_tokens(context_list, loss_scale_list)
            >>> print(res_context)
            # ['Image:', '<image>', 'Video:', '<video>', 'End', [151643, 151644], 'Audio:', '<audio>']
            # 第1个元素拆分为5个，第2个是token列表不拆分，第3个拆分为2个
            >>> print(res_loss)
            # [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5]
            # 注意：token列表([151643, 151644])保持不变，loss_scale也保持为1.0
        """
        # ===== 初始化结果列表 =====
        res: List[Context] = []  # 拆分后的上下文列表
        loss_scale_res: List[float] = []  # 拆分后的loss_scale列表
        # ===== 遍历上下文列表，对每个元素进行拆分处理 =====
        for context, loss_scale in zip(context_list, loss_scale_list):
            # 初始化临时列表，存储当前context的拆分结果
            contexts = []
            
            # ===== 判断当前context是否为字符串类型 =====
            # fetch_one: 从嵌套结构中提取第一个元素（支持str、list、tuple等）
            # 如果context是字符串或包含字符串的结构，返回该字符串
            # 如果context是token ID列表（如[1, 2, 3]），返回整数，isinstance检查为False
            if isinstance(fetch_one(context), str):
                # ===== 字符串类型：需要拆分特殊标签 =====
                # split_str_parts_by: 按照特殊标签列表拆分字符串
                # Template.special_tokens: ['<image>', '<video>', '<audio>', '<bbox>', '<ref-object>', ...]
                # 返回格式：[{'key': '<image>', 'content': 'text1'}, {'key': '', 'content': 'text2'}, ...]
                #   - 'key': 匹配到的特殊标签（如'<image>'），若无匹配则为空字符串
                #   - 'content': 标签后的文本内容或标签本身
                for d in split_str_parts_by(context, Template.special_tokens):
                    # 将'key'（特殊标签）和'content'（文本内容）按顺序添加到列表
                    # 例如：'Hello<image>world' -> [{'key': '', 'content': 'Hello'}, 
                    #                                 {'key': '<image>', 'content': '<image>'},
                    #                                 {'key': '', 'content': 'world'}]
                    # extend后：['', 'Hello', '<image>', '<image>', '', 'world']
                    contexts.extend([d['key'], d['content']])
                
                # ===== 过滤空字符串和空标签 =====
                # 列表推导式：仅保留非空元素（c为真值）
                # 过滤掉空字符串''，只保留有内容的文本和特殊标签
                # 例如：['', 'Hello', '<image>', '<image>', '', 'world'] 
                #       -> ['Hello', '<image>', '<image>', 'world']
                contexts = [c for c in contexts if c]
                
                # ===== 将拆分结果添加到结果列表 =====
                res.extend(contexts)  # 扩展上下文列表
                # 为拆分出的每个元素分配相同的loss_scale值（继承原元素的loss_scale）
                # [loss_scale] * len(contexts): 创建长度为contexts的loss_scale副本列表
                # 例如：如果loss_scale=0.5，contexts有3个元素，则添加[0.5, 0.5, 0.5]
                loss_scale_res.extend([loss_scale] * len(contexts))
            else:
                # ===== 非字符串类型（token ID列表）：不需要拆分 =====
                # 直接添加到结果列表，保持原样
                res.append(context)
                loss_scale_res.append(loss_scale)
        
        return res, loss_scale_res

    def _tokenize(self, context, **tokenizer_kwargs):
        return self.tokenizer(
            context, return_attention_mask=False, add_special_tokens=False, **tokenizer_kwargs)['input_ids']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """Override this function to do your own replace operation.

        This method is used to replace standard tags like `<image>` to some tokens that the model needs.

        Args:
            media_type: The modal.
            index: The index of the medias, for index 0 represents the first elements in `images`
            inputs: The inputs

        Returns:
            The content or input_ids after replacement.
        """
        if media_type == 'image':
            if self.mode == 'lmdeploy':
                return [[-100]]
            return self.image_placeholder
        elif media_type == 'video':
            return self.video_placeholder
        elif media_type == 'audio':
            return self.audio_placeholder
    
    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        """Replace objects referenced by the bbox to contents or input_ids. This is useful in the grounding task.
        Override this function to do your own replace operation.

        Args:
            ref: Description of the bbox
            index: The index in the `objects` key
            inputs: The inputs

        Returns:
            The contents or input_ids replaced
        """
        return [ref]

    def replace_cot_process(self, inputs: StdTemplateInputs) -> List[Context]:
        """Replace the cot process label for PRM training or inference.
        Override this function to do your own replace operation.

        Args:
            inputs: The inputs

        Returns:
            The contents or input_ids replaced
        """
        return [self.cot_process_placeholder]

    @staticmethod
    def _get_bbox_str(bbox: List[int]) -> str:
        point = []
        for x, y in zip(bbox[::2], bbox[1::2]):
            point.append(f'({x},{y})')
        return ','.join(point)

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        """Replace bbox pointing to the objects to contents or input_ids. This is useful in the grounding task.
        Override this function to do your own replace operation.

        Args:
            bbox: [x, y] or [x1, y1, x2, y2]
            index: The index in the `objects` key
            inputs: The inputs

        Returns:
            The contents or input_ids replaced
        """
        return [f'[{self._get_bbox_str(bbox)}]']

    def _pre_tokenize_images(self, context_list: List[Context], loss_scale_list: List[float],
                             inputs: StdTemplateInputs) -> Tuple[List[Context], List[float]]:
        """
        函数功能：
            预分词阶段替换图像标签为模型所需的实际内容或token。遍历上下文列表，将'<image>'标签替换为
            图像占位符token（如特殊token ID或占位符字符串），同时根据模板后端类型调整图像token的loss_scale。
            该方法解决了Qwen2.5-VL grounding任务中bbox位置偏移问题（见issue #3407），确保图像标签在
            分词前被正确替换，避免后续token位置计算错误。
        
        参数：
            context_list (List[Context]): 上下文列表，可能包含'<image>'标签
            loss_scale_list (List[float]): 与context_list对应的loss权重列表
            inputs (StdTemplateInputs): 模板输入对象，包含images等多模态数据
        
        返回值：
            Tuple[List[Context], List[float]]: (替换后的上下文列表, 对应的loss_scale列表)
        
        使用示例：
            >>> # 示例1：替换单个图像标签
            >>> template = Template(...)
            >>> context_list = ['Describe this', '<image>', 'in detail']
            >>> loss_scale_list = [0.0, 0.0, 1.0]
            >>> inputs = StdTemplateInputs(images=['img.jpg'], ...)
            >>> res_context, res_loss = template._pre_tokenize_images(context_list, loss_scale_list, inputs)
            >>> # res_context: ['Describe this', '<image_placeholder>', 'in detail']
            >>> # res_loss: [0.0, 0.0, 1.0]（swift后端下图像token的loss_scale为0.0）
            
            >>> # 示例2：多个图像标签
            >>> context_list = ['Image1:', '<image>', 'Image2:', '<image>']
            >>> loss_scale_list = [0.0, 0.0, 0.0, 0.0]
            >>> inputs = StdTemplateInputs(images=['img1.jpg', 'img2.jpg'], ...)
            >>> res_context, res_loss = template._pre_tokenize_images(context_list, loss_scale_list, inputs)
            >>> # 第1个<image>替换为img1的占位符，第2个<image>替换为img2的占位符
        """
        # 参考：https://github.com/modelscope/ms-swift/issues/3407
        # 修复Qwen2.5-VL grounding任务中的bbox位置偏移问题
        res: List[Context] = []  # 替换后的上下文列表
        res_loss_scale: List[float] = []  # 替换后的loss_scale列表
        inputs.image_idx = 0  # 初始化图像索引，用于追踪当前处理到第几张图像

        for context, loss_scale in zip(context_list, loss_scale_list):
            # 判断是否需要替换图像标签：context是'<image>' 且 是多模态输入 且 还有未处理的图像
            if context == '<image>' and inputs.is_multimodal and inputs.image_idx < len(inputs.images):
                c_list = self.replace_tag('image', inputs.image_idx, inputs)  # 替换为图像占位符token
                inputs.image_idx += 1  # 图像索引递增，指向下一张图像
                # 根据模板后端调整loss_scale：swift后端图像token不计算loss(0.)，其他后端正常计算(1.)
                loss_scale = 0. if self.template_backend == 'swift' else 1.
            else:
                c_list = [context]  # 非图像标签或无图像数据，保持原context
            res += c_list  # 添加到结果列表
            res_loss_scale += [loss_scale] * len(c_list)  # 为替换后的所有token分配相同的loss_scale
        return res, res_loss_scale

    def _pre_tokenize(self, context_list: List[Context], loss_scale_list: List[float],
                      inputs: StdTemplateInputs) -> Tuple[List[Context], List[float]]:
        """
        函数功能：
            预分词阶段将标准标签替换为模型所需的实际内容或token ID。该方法是多模态和特殊任务的核心
            预处理步骤，负责替换所有特殊标签：<image>、<video>、<audio>（多模态标签）、<ref-object>、
            <bbox>（grounding任务标签）、<cot-process>（PRM任务标签）。替换后的内容可能是占位符
            字符串、token ID列表或格式化的坐标字符串。该方法确保在tokenization之前，所有特殊标签都
            已转换为模型可理解的格式，避免标签被错误地当作普通文本分词。
        
        参数：
            context_list (List[Context]): 上下文列表，可能包含各种特殊标签（<image>、<video>等）
            loss_scale_list (List[float]): 与context_list对应的loss权重列表
            inputs (StdTemplateInputs): 模板输入对象，包含多模态数据(images、videos、audios)和
                                       grounding数据(objects中的ref、bbox)
        
        返回值：
            Tuple[List[Context], List[float]]: (替换后的上下文列表, 对应的loss_scale列表)
                - 替换后的上下文列表中，特殊标签已被替换为占位符、token ID或格式化字符串
                - loss_scale列表已根据替换结果调整（多模态token通常为0.0，不参与loss计算）
        
        使用示例：
            >>> # 示例1：多模态标签替换
            >>> template = Template(...)
            >>> context_list = ['Describe', '<image>', 'and', '<video>']
            >>> loss_scale_list = [0.0, 0.0, 1.0, 1.0]
            >>> inputs = StdTemplateInputs(images=['img.jpg'], videos=['video.mp4'], ...)
            >>> res_context, res_loss = template._pre_tokenize(context_list, loss_scale_list, inputs)
            >>> # res_context: ['Describe', '<image_placeholder>', 'and', '<video_placeholder>']
            >>> # res_loss: [0.0, 0.0, 1.0, 0.0]（多模态token的loss_scale被设为0.0）
            
            >>> # 示例2：grounding任务的标签替换
            >>> context_list = ['Object at', '<bbox>', 'is referred as', '<ref-object>']
            >>> loss_scale_list = [1.0, 1.0, 1.0, 1.0]
            >>> inputs = StdTemplateInputs(
            ...     images=['img.jpg'],
            ...     objects={'bbox': [[100, 200, 300, 400]], 'ref': ['person']},
            ...     ...
            ... )
            >>> res_context, res_loss = template._pre_tokenize(context_list, loss_scale_list, inputs)
            >>> # res_context: ['Object at', '[100,200,300,400]', 'is referred as', 'person']
            >>> # <bbox>被替换为归一化后的坐标字符串，<ref-object>被替换为引用对象名称
        """
        # 首先处理图像标签替换
        context_list, loss_scale_list = self._pre_tokenize_images(context_list, loss_scale_list, inputs)
        
        # 如果同时存在图像和objects数据，归一化bbox坐标（用于grounding任务）
        if inputs.images and inputs.objects:
            self.normalize_bbox(inputs)
        
        # 初始化结果列表
        res: List[Context] = []  # 替换后的上下文列表
        res_loss_scale: List[float] = []  # 替换后的loss_scale列表

        # 重置各类多模态和特殊标签的索引计数器（video_idx、audio_idx、object_idx、box_idx）
        for k in ['video', 'audio', 'object', 'box']:
            setattr(inputs, f'{k}_idx', 0)

        # 遍历上下文列表，逐个处理每个context
        for context, loss_scale in zip(context_list, loss_scale_list):
            # 尝试替换视频和音频标签（<video>和<audio>）
            for k in ['video', 'audio']:
                # 检查三个条件：1)context是对应标签 2)是多模态输入 3)还有未处理的数据
                if context == f'<{k}>' and inputs.is_multimodal and getattr(inputs, f'{k}_idx') < len(
                        getattr(inputs, f'{k}s')):
                    # 调用replace_tag方法替换为占位符或token ID
                    c_list = self.replace_tag(k, getattr(inputs, f'{k}_idx'), inputs)
                    # 索引递增，指向下一个video/audio数据
                    setattr(inputs, f'{k}_idx', getattr(inputs, f'{k}_idx') + 1)
                    # 多模态token的loss_scale设为0.0（不参与loss计算）
                    loss_scale = 0.
                    break  # 匹配成功，跳出内层循环
            else:
                # for循环未break（未匹配到video/audio标签），尝试替换grounding和PRM标签
                # 获取ref和bbox列表（grounding任务的引用对象和边界框）
                ref = inputs.objects.get('ref') or []
                bbox = inputs.objects.get('bbox') or []
                
                if context == '<ref-object>' and inputs.ref_idx < len(ref):
                    # 替换<ref-object>标签为实际的引用对象名称
                    idx = inputs.ref_idx
                    c_list = self.replace_ref(ref[idx], idx, inputs)
                    inputs.ref_idx += 1
                elif context == '<bbox>' and inputs.bbox_idx < len(bbox):
                    # 替换<bbox>标签为格式化的边界框坐标字符串
                    idx = inputs.bbox_idx
                    c_list = self.replace_bbox(bbox[idx], idx, inputs)
                    inputs.bbox_idx += 1
                elif context == '<cot-process>' and self.task_type == 'prm':
                    # 替换<cot-process>标签为PRM任务的思维链过程标签
                    c_list = self.replace_cot_process(inputs)
                else:
                    # 不是任何特殊标签，保持原context不变
                    c_list = [context]
            
            # 将替换结果添加到结果列表
            res += c_list
            # 为替换后的所有元素分配相同的loss_scale（通常替换后只有1个元素）
            res_loss_scale += [loss_scale] * len(c_list)
        
        return res, res_loss_scale
    @staticmethod
    def _add_default_tags(inputs: StdTemplateInputs):
        """
        函数功能：
            为多模态输入自动添加缺失的默认标签（如 <image>、<audio>、<video>）。该方法用于
            处理用户提供了多模态数据但未在文本中显式标注相应标签的情况，通过比较实际多模态
            数据数量和文本中标签数量，自动在第一条用户消息前补充缺失的标签，确保模板处理时
            能正确识别和定位多模态数据。
        
        参数：
            inputs (StdTemplateInputs): 标准化的模板输入对象，包含以下字段：
                - messages (List[Dict[str, Any]]): 对话消息列表
                    每个消息包含 'role' 和 'content' 字段
                    role: 'user', 'assistant', 'system', 'tool' 等
                    content: 字符串内容或结构化多模态内容（list/dict）
                - images (Optional[List]): 图像数据列表（路径、URL、PIL.Image等）
                - audios (Optional[List]): 音频文件列表
                - videos (Optional[List]): 视频文件列表
                - system (Optional[str]): 系统提示词
                - rejected_response (Optional[Union[str, List[str]]]): 拒绝的响应（RLHF训练用）
        
        返回值：
            None: 该方法直接修改传入的 inputs 对象（原地修改），不返回任何值。
                  修改后的 inputs.messages[0]['content'] 会在开头添加缺失的多模态标签。
        
        使用示例：
            >>> # 示例1：用户提供了2张图像，但只标注了1个<image>标签
            >>> inputs = StdTemplateInputs(
            ...     messages=[{"role": "user", "content": "<image>描述第二张图"}],
            ...     images=["image1.jpg", "image2.jpg"]
            ... )
            >>> Template._add_default_tags(inputs)
            >>> print(inputs.messages[0]['content'])
            # 输出: "<image><image>描述第二张图"  # 自动在开头添加了缺失的<image>标签
            
            >>> # 示例2：用户提供了3张图像，但没有标注任何<image>标签
            >>> inputs = StdTemplateInputs(
            ...     messages=[{"role": "user", "content": "这些图片里有什么？"}],
            ...     images=["img1.jpg", "img2.jpg", "img3.jpg"]
            ... )
            >>> Template._add_default_tags(inputs)
            >>> print(inputs.messages[0]['content'])
            # 输出: "<image><image><image>这些图片里有什么？"  # 添加了3个<image>标签
            
            >>> # 示例3：用户标注的标签多于实际图像数量（会输出警告）
            >>> inputs = StdTemplateInputs(
            ...     messages=[{"role": "user", "content": "<image><image><image>描述"}],
            ...     images=["image.jpg"]  # 只有1张图像
            ... )
            >>> Template._add_default_tags(inputs)
            # 警告: num_media: 1, num_media_tags: 3, total_content: ...
            # 只会替换最前面的1个<image>标签，保留后续标签
            
            >>> # 示例4：混合多模态（图像+视频）
            >>> inputs = StdTemplateInputs(
            ...     messages=[{"role": "user", "content": "分析这些内容"}],
            ...     images=["img.jpg"],
            ...     videos=["video.mp4"]
            ... )
            >>> Template._add_default_tags(inputs)
            >>> print(inputs.messages[0]['content'])
            # 输出: "<image><video>分析这些内容"
            
            >>> # 示例5：结构化内容（非字符串）的用户消息会跳过处理
            >>> inputs = StdTemplateInputs(
            ...     messages=[{"role": "user", "content": [
            ...         {"type": "image", "image": "img.jpg"},
            ...         {"type": "text", "text": "描述图片"}
            ...     ]}],
            ...     images=["img.jpg"]
            ... )
            >>> Template._add_default_tags(inputs)
            # 不会修改，直接返回（因为content不是字符串）
        """
        # ===== 步骤1：收集所有消息内容 =====
        total_content = []  # 存储所有消息的文本内容
        for message in inputs.messages:  # 遍历所有对话消息
            content = message['content'] or ''  # 获取消息内容，若为None则使用空字符串
            if not isinstance(content, str):  # 若内容不是字符串（是结构化多模态内容）
                if message['role'] == 'user':  # 若是用户消息
                    # 结构化内容已经明确指定了多模态数据位置，无需添加默认标签
                    return  # 直接返回，放弃添加默认标签
                elif message['role'] == 'assistant':  # 若是助手消息
                    continue  # 跳过助手的结构化内容（助手通常不使用多模态标签）
            total_content.append(content)  # 将字符串内容添加到列表
        
        # ===== 步骤2：添加拒绝响应内容（RLHF训练用） =====
        if inputs.rejected_response:  # 若存在拒绝响应
            rejected_response = inputs.rejected_response  # 获取拒绝响应
            if isinstance(inputs.rejected_response, str):  # 若拒绝响应是单个字符串
                rejected_response = [rejected_response]  # 转换为列表格式
            total_content += rejected_response  # 将拒绝响应添加到总内容中
        
        # ===== 步骤3：拼接所有内容为一个字符串 =====
        total_content = '\n'.join(total_content)  # 用换行符连接所有内容
        if inputs.system:  # 若存在系统提示词
            total_content = f'{inputs.system}\n{total_content}'  # 将系统提示添加到开头
        
        # ===== 步骤4：遍历所有多模态类型，检查并添加缺失标签 =====
        for media_type in ['image', 'audio', 'video']:  # 遍历三种多模态类型
            # 构建字段名和标签名：'image' -> 'images' 和 '<image>'
            media_key, media_tag = f'{media_type}s', f'<{media_type}>'
            # 获取对应的多模态数据（如 inputs.images）
            medias = getattr(inputs, media_key)
            if not isinstance(medias, list):  # 若多模态数据不是列表
                medias = [medias]  # 转换为单元素列表
            if medias:  # 若多模态数据列表非空
                # 统计文本中该类型标签的数量（如<image>出现次数）
                num_media_tags = len(re.findall(media_tag, total_content))
                # 获取实际多模态数据的数量
                num_media = len(medias)
                # 计算需要新增的标签数量（数据数量 - 标签数量）
                num_new_tags = num_media - num_media_tags
                if num_new_tags > 0:  # 若标签数量不足（数据多于标签）
                    # 在第一条消息的内容前添加缺失的标签
                    # 例如：缺2个<image>标签，则添加 "<image><image>" + 原内容
                    inputs.messages[0]['content'] = media_tag * num_new_tags + inputs.messages[0]['content']
                elif num_new_tags < 0:  # 若标签数量过多（标签多于数据）
                    # 输出警告信息，提示用户标签数量与数据不匹配
                    logger.warning(
                        f'num_media: {num_media}, num_media_tags: {num_media_tags}, total_content: {total_content}. '
                        'We will only replace the frontmost media_tags while keeping the subsequent media_tags.')
                # 注意：num_new_tags == 0 时，标签数量与数据匹配，无需处理

    def _encode_context_list(
            self,
            context_list: List[Context],
            loss_scale_list: Optional[List[float]] = None) -> Tuple[List[int], List[int], List[float], Dict[str, Any]]:
        """
        函数功能：
            将上下文列表（Context List）编码为token序列，生成模型训练所需的input_ids、labels和
            loss_scale。该方法是模板编码的核心底层方法，负责将字符串或token列表转换为最终的token
            序列，并根据loss_scale_list决定每个位置是否参与loss计算。支持灵活的loss权重控制，
            实现对不同token段设置不同的训练权重。
        
        参数：
            context_list (List[Context]): 上下文列表，每个元素可以是：
                - str: 需要tokenize的字符串文本（如 "Hello, world!"）
                - List[int]: 已经tokenize的token ID列表（如 [15339, 11, 1879, 0]）
                Context类型定义为 Union[str, List[int]]，即 Word 类型
            loss_scale_list (Optional[List[float]]): loss权重列表，默认为None
                - 每个元素对应context_list中一个context的loss权重
                - 值为0.0：该段不参与loss计算（labels设为-100）
                - 值为1.0：该段正常参与loss计算（labels为实际token）
                - 值为其他浮点数：该段以指定权重参与loss计算（用于loss_scale）
                - None时：自动初始化为全0列表（所有context不参与loss计算）
        
        返回值：
            Tuple[List[int], List[int], List[float], Dict[str, Any]]: 包含四个元素的元组
                - input_ids (List[int]): 完整的输入token ID序列，拼接所有context的token
                    例如：[151644, 8948, 198, 151645, 872, 151643, 151644, ...]
                - labels (List[int]): 标签序列，与input_ids等长
                    参与loss计算的位置：对应的token ID
                    不参与loss计算的位置：-100（PyTorch中的ignore_index）
                    例如：[-100, -100, -100, -100, 872, 151643, 151644, ...]
                - loss_scale (Optional[List[float]]): loss权重序列，与input_ids等长
                    若所有权重为0或1：返回None（使用标准loss计算）
                    若包含其他值：返回权重列表（用于加权loss计算）
                    例如：None 或 [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, ...]
                - tokenizer_kwargs (Dict[str, Any]): tokenizer的额外参数（当前版本返回空字典）
        
        使用示例：
            >>> # 示例1：基础字符串编码
            >>> template = Template(...)
            >>> context_list = ["<|system|>", "You are a helpful assistant.", "<|user|>", "Hello!"]
            >>> loss_scale_list = [0.0, 0.0, 0.0, 0.0]  # 系统和用户消息不计算loss
            >>> input_ids, labels, loss_scale, kwargs = template._encode_context_list(
            ...     context_list, loss_scale_list
            ... )
            >>> print(len(input_ids))  # 例如: 15
            >>> print(labels[:5])  # [-100, -100, -100, -100, -100]
            >>> print(loss_scale)  # None (所有权重为0或1)
            
            >>> # 示例2：包含token列表的编码
            >>> context_list = ["<|system|>", [123, 456, 789], "<|assistant|>", "Hi there!"]
            >>> loss_scale_list = [0.0, 0.0, 0.0, 1.0]  # 只有助手回复参与loss计算
            >>> input_ids, labels, loss_scale, kwargs = template._encode_context_list(
            ...     context_list, loss_scale_list
            ... )
            >>> # labels前面部分全是-100，最后的"Hi there!"对应实际token ID
            
            >>> # 示例3：使用自定义loss权重
            >>> context_list = ["Question: ", "What is AI?", "Answer: ", "AI is..."]
            >>> loss_scale_list = [0.0, 0.0, 0.0, 0.8]  # 答案部分使用0.8权重
            >>> input_ids, labels, loss_scale, kwargs = template._encode_context_list(
            ...     context_list, loss_scale_list
            ... )
            >>> print(loss_scale[-5:])  # [0.8, 0.8, 0.8, 0.8, 0.8] (答案部分)
            
            >>> # 示例4：loss_scale_list为None的情况
            >>> context_list = ["Hello", " world"]
            >>> input_ids, labels, loss_scale, kwargs = template._encode_context_list(context_list)
            >>> # loss_scale_list自动设为[0.0, 0.0]，所有位置不参与loss计算
            >>> print(all(l == -100 for l in labels))  # True
        """
        # ===== 步骤1：初始化返回值 =====
        input_ids: List[int] = []  # 存储完整的输入token序列
        labels: List[int] = []  # 存储标签序列（-100表示不计算loss，否则为token ID）
        loss_scale: List[float] = []  # 存储loss权重序列
        tokenizer_kwargs = {}  # 存储tokenizer的额外参数（当前版本未使用）
        
        # ===== 步骤2：处理loss_scale_list默认值 =====
        if loss_scale_list is None:  # 若未提供loss权重列表
            # 初始化为全0列表，长度与context_list相同（所有context不参与loss计算）
            loss_scale_list = [0.] * len(context_list)
        
        # ===== 步骤3：判断是否需要保留loss_scale =====
        if self.loss_scale.keep_loss_scale:  # 若配置强制保留loss_scale
            ignore_loss_scale = False  # 不忽略loss_scale，始终返回权重列表
        else:  # 若未强制保留
            # 检查所有权重是否仅为0或1（标准二值权重）
            # 若是，则ignore_loss_scale=True，返回None而非权重列表（优化性能）
            ignore_loss_scale = all(loss_scale in {0, 1} for loss_scale in loss_scale_list)

        # ===== 步骤4：遍历context_list，逐个编码并拼接 =====
        for i, (context, loss_weight) in enumerate(zip(context_list, loss_scale_list)):
            # 处理当前context，获取其token列表
            if isinstance(context, str):  # 若context是字符串
                # 注释说明：tokenizer_kwargs是返回的参数字典，
                # 而curr_tokenizer_kwargs是当前context使用的tokenizer参数（未使用）
                token_list = self._tokenize(context)  # 使用tokenizer将字符串转为token列表
            else:  # 若context已经是token列表（List[int]）
                token_list = context  # 直接使用该token列表
            
            # 拼接input_ids（所有token都加入input_ids）
            input_ids += token_list
            
            # 根据loss权重决定labels的值
            if loss_scale_list[i] > 0.0:  # 若该段参与loss计算（权重>0）
                labels += token_list  # labels使用实际的token ID
            else:  # 若该段不参与loss计算（权重=0）
                labels += [-100] * len(token_list)  # labels全部设为-100（忽略该段loss）
            
            # 根据需要添加loss权重
            if not ignore_loss_scale:  # 若需要保留loss_scale（存在非0/1权重）
                # 将当前context的权重值扩展到每个token位置
                # 例如：loss_weight=0.8, len(token_list)=5 -> [0.8, 0.8, 0.8, 0.8, 0.8]
                loss_scale.extend([loss_weight] * len(token_list))
        
        # ===== 步骤5：根据ignore_loss_scale决定是否返回loss_scale =====
        if ignore_loss_scale:  # 若所有权重为0或1（标准情况）
            loss_scale = None  # 返回None，不返回权重列表（节省内存和计算）
        
        # ===== 步骤6：返回编码结果 =====
        return input_ids, labels, loss_scale, tokenizer_kwargs

    @staticmethod
    def _add_dynamic_eos(input_ids: List[int], labels: List[int], loss_scale: Optional[List[int]],
                         suffix_tokens_id: List[int]) -> None:
        """
        函数功能：
            动态添加EOS（End of Sentence）token到labels中。该方法用于在助手回复结束位置（从有效
            token转换到-100的边界）识别并激活suffix token（如<|im_end|>、ڕ 等）的loss计算。
            通过检测labels中从有效值（>=0）到-100的转换边界，判断该位置是否包含suffix token，
            若是则将对应的labels从-100改为实际token ID，使模型学习正确生成结束标记。
        
        参数：
            input_ids (List[int]): 完整的输入token序列
                包含所有token（系统提示、用户输入、助手回复、分隔符等）
                例如：[151644, 8948, 872, 151645, 151644, 882, 151643, 151644, ...]
            labels (List[int]): 标签序列，与input_ids等长
                有效值（>=0）：该位置参与loss计算，对应token ID
                -100：该位置不参与loss计算（如系统提示、用户输入部分）
                例如：[-100, -100, -100, -100, -100, 882, 151643, -100, ...]
            loss_scale (Optional[List[int]]): loss权重序列，与input_ids等长
                None：使用标准loss计算
                List[int]：每个位置的loss权重
            suffix_tokens_id (List[int]): 后缀token序列（通常是EOS token）
                从template_meta.suffix编码得到
                例如：[151643]（<|im_end|>）或 [2]（ڕ ）
        
        返回值：
            None: 该方法直接修改传入的labels和loss_scale列表（原地修改），不返回任何值。
                  修改后，suffix token位置的labels从-100变为实际token ID，
                  对应的loss_scale从0变为1（如果存在）。
        
        使用示例：
            >>> # 示例1：基础EOS token添加
            >>> input_ids = [1, 2, 3, 100, 200, 4, 5]  # 100是EOS token
            >>> labels = [1, 2, 3, -100, -100, 6, 7]  # 助手回复后的EOS被标记为-100
            >>> loss_scale = None
            >>> suffix_tokens_id = [100]  # EOS token ID
            >>> Template._add_dynamic_eos(input_ids, labels, loss_scale, suffix_tokens_id)
            >>> print(labels)
            # [1, 2, 3, 100, -100, 6, 7]  # 第4个位置从-100变为100
            
            >>> # 示例2：带loss_scale的情况
            >>> input_ids = [10, 20, 30, 151643, 151644, 40]  # 151643是<|im_end|>
            >>> labels = [10, 20, 30, -100, -100, 50]  # 助手回复结束，EOS位置是-100
            >>> loss_scale = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
            >>> suffix_tokens_id = [151643]
            >>> Template._add_dynamic_eos(input_ids, labels, loss_scale, suffix_tokens_id)
            >>> print(labels)
            # [10, 20, 30, 151643, -100, 50]  # EOS位置从-100变为151643
            >>> print(loss_scale)
            # [1.0, 1.0, 1.0, 1.0, 0.0, 1.0]  # EOS位置的loss_scale从0变为1
            
            >>> # 示例3：多token suffix（如"\nڕ "）
            >>> input_ids = [1, 2, 3, 10, 11, 4, 5]  # 10,11是两个suffix token
            >>> labels = [1, 2, 3, -100, -100, 6, 7]
            >>> loss_scale = None
            >>> suffix_tokens_id = [10, 11]  # 两个token的suffix
            >>> Template._add_dynamic_eos(input_ids, labels, loss_scale, suffix_tokens_id)
            >>> print(labels)
            # [1, 2, 3, 10, 11, 6, 7]  # 两个位置都从-100变为实际token
            
            >>> # 示例4：没有匹配suffix的情况（不修改）
            >>> input_ids = [1, 2, 3, 99, 98, 4, 5]  # 99,98不是suffix token
            >>> labels = [1, 2, 3, -100, -100, 6, 7]
            >>> suffix_tokens_id = [100]  # suffix是100，但input_ids中没有
            >>> Template._add_dynamic_eos(input_ids, labels, loss_scale, suffix_tokens_id)
            >>> print(labels)
            # [1, 2, 3, -100, -100, 6, 7]  # 不修改，因为没有匹配
        """
        # ===== 步骤1：初始化变量 =====
        suffix_len = len(suffix_tokens_id)  # 获取suffix的长度（通常为1，如<|im_end|>）
        start = 0  # 标记当前检测到的助手回复结束位置（从有效token到-100的边界）
        
        # ===== 步骤2：遍历labels，寻找需要添加EOS的位置 =====
        for i in range(1, len(labels)):  # 从索引1开始遍历（需要访问i-1）
            # 检测助手回复结束的边界：从有效token（>=0）转换到-100
            if labels[i - 1] >= 0 and labels[i] == -100:  # 发现边界（有效 -> -100）
                start = i  # 记录边界位置（-100开始的位置）
            
            # 检测下一轮助手回复的开始：从-100转换回有效token（>=0）
            if start > 0 and labels[i - 1] == -100 and labels[i] >= 0:  # 发现边界（-100 -> 有效）
                # 示例：[0, 1, 2, -100(start), -100, 3(i), 4]
                # start=3, i=5, length=2（中间有2个-100）
                length = i - start  # 计算中间-100区域的长度
                
                # 检查是否满足添加suffix的条件：
                # 1. 长度足够容纳suffix（length >= suffix_len）
                # 2. input_ids中对应位置确实是suffix token
                if length >= suffix_len and input_ids[start:start + suffix_len] == suffix_tokens_id:
                    # 将labels中suffix位置从-100改为实际token ID，使模型学习生成EOS
                    labels[start:start + suffix_len] = suffix_tokens_id
                    
                    # 如果存在loss_scale，同时更新对应位置的权重
                    if loss_scale and loss_scale[start:start + suffix_len] == [0] * suffix_len:
                        # 将suffix位置的loss_scale从0改为1，使该位置参与loss计算
                        loss_scale[start:start + suffix_len] = [1] * suffix_len

    @staticmethod
    def _get_std_messages(messages):
        if messages and messages[0]['role'] == 'assistant':
            messages.insert(0, {'role': 'user', 'content': ''})  # pretrain
        if len(messages) % 2 == 1:
            messages.append({'role': 'assistant', 'content': None})  # inference

    def _jinja_encode(self, inputs: StdTemplateInputs):
        """
        函数功能：
            使用Jinja模板后端编码对话消息，调用tokenizer的apply_chat_template方法将消息列表转换为
            格式化的文本。该方法是HuggingFace标准的模板应用方式，适用于支持Jinja模板的tokenizer（如
            Qwen、LLaMA等主流模型）。相比于Swift自定义后端，Jinja后端更简洁，直接使用tokenizer内置
            的对话模板，但灵活性较低。该方法会自动处理system消息的插入、空消息的移除、以及生成提示
            符的添加，返回格式化的文本字符串供后续分词使用。
        
        参数：
            inputs (StdTemplateInputs): 标准模板输入对象，包含对话消息、系统提示、工具定义等信息
                - inputs.messages: 对话消息列表，每条消息包含'role'和'content'字段
                - inputs.system: 可选的系统提示文本
                - inputs.tools: 可选的工具定义列表（用于function calling）
        
        返回值：
            Tuple[List[str], List[float], int]: (文本列表, loss_scale列表, 答案长度)
                - 第一个元素: [text]，包含单个格式化文本字符串的列表
                - 第二个元素: [1.]，loss_scale为1.0（Jinja后端下统一为1.0，不区分prompt和answer）
                - 第三个元素: answer_len，训练模式下为1，推理模式下为0
        
        使用示例：
            >>> # 示例1：基础对话编码（训练模式）
            >>> template = Template(..., template_backend='jinja')
            >>> template.mode = 'train'
            >>> inputs = StdTemplateInputs(
            ...     messages=[
            ...         {"role": "user", "content": "Hello"},
            ...         {"role": "assistant", "content": "Hi there!"}
            ...     ]
            ... )
            >>> text_list, loss_scale, answer_len = template._jinja_encode(inputs)
            >>> print(text_list)
            # ['<|user|>\nHello<|assistant|>\nHi there!']（具体格式取决于tokenizer的模板）
            >>> print(loss_scale)  # [1.]
            >>> print(answer_len)  # 1（训练模式）
            
            >>> # 示例2：带系统提示和工具的编码（推理模式）
            >>> template.mode = 'pt'
            >>> inputs = StdTemplateInputs(
            ...     system="You are a helpful assistant",
            ...     messages=[{"role": "user", "content": "What's the weather?"}],
            ...     tools=[{"type": "function", "function": {"name": "get_weather", ...}}]
            ... )
            >>> text_list, loss_scale, answer_len = template._jinja_encode(inputs)
            >>> # system消息被插入到messages开头，tools参数传递给apply_chat_template
            >>> print(answer_len)  # 0（推理模式）
        """
        # 复制消息列表，避免修改原始输入数据
        # NOTE: 浅拷贝，复制最外层容器（list 或 dict），但内部的元素（如字典、对象等）仍是原对象的引用。
        messages = inputs.messages.copy()
        
        # 如果提供了system提示，将其作为第一条消息插入到messages开头
        if inputs.system is not None:
            messages.insert(0, {'role': 'system', 'content': inputs.system})
        
        # 如果最后一条消息的content为None，则移除（通常是推理时的空assistant消息占位符）
        if messages[-1]['content'] is None:
            messages.pop()
        
        # 判断是否需要添加生成提示符：最后一条消息不是assistant时需要添加（推理模式）
        # 若最后一条是user消息，add_generation_prompt=True会添加assistant前缀提示生成
        add_generation_prompt = messages[-1]['role'] != 'assistant'
        
        # 初始化额外参数字典，用于传递给apply_chat_template
        kwargs = {}
        
        # 如果提供了tools（工具定义），添加到kwargs中（用于function calling场景）
        if inputs.tools:
            kwargs['tools'] = inputs.tools
        # 调用tokenizer的apply_chat_template方法，将消息列表转换为格式化文本
        # tokenize=False: 返回文本字符串而非token ID（后续会统一分词）
        # add_generation_prompt: 是否添加生成提示符（如'<rewritten_file>'）
        # **kwargs: 传递tools等额外参数
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs)
        
        # 计算答案长度：训练模式下为1（表示有答案部分），推理模式下为0（无答案）
        # Jinja后端下answer_len用于标识是否处于训练模式，但不精确计算实际答案token数
        answer_len = 1 if self.is_training else 0
        
        # 返回：[text]文本列表, [1.]统一的loss_scale, answer_len答案长度标识
        # Jinja后端下loss_scale统一为1.0，不像Swift后端那样区分prompt(0.0)和answer(1.0)
        return [text], [1.], answer_len

    def _get_system(self, inputs) -> Optional[str]:
        """
        函数功能：
            获取并处理系统提示（system prompt）。该方法负责确定最终使用的系统提示内容，处理逻辑包括：
            验证用户提供的系统提示是否符合模板要求、使用默认系统提示（如果用户未提供）、以及在使用
            工具（tools）时通过agent_template格式化工具定义并整合到系统提示中。该方法是编码流程中
            system prompt准备的核心环节，确保系统提示既符合模板规范又包含必要的工具信息。
        
        参数：
            inputs (StdTemplateInputs): 标准模板输入对象
                - inputs.system: 用户提供的系统提示文本（可能为None）
                - inputs.tools: 工具定义列表，用于function calling（可能为None）
                - inputs.messages: 对话消息列表，用于agent模板格式化时的上下文
        
        返回值：
            Optional[str]: 处理后的系统提示文本
                - 若用户提供了system且未使用tools，返回用户的system（经过验证）
                - 若用户未提供system，返回template_meta.default_system（可能为None）
                - 若使用了tools，返回格式化后的系统提示（包含工具定义和原system内容）
                - 可能返回None（当无默认系统提示且用户未提供时）
        
        使用示例：
            >>> # 示例1：使用默认系统提示
            >>> template = Template(...)
            >>> template.template_meta.default_system = "You are a helpful assistant"
            >>> inputs = StdTemplateInputs(messages=[...])  # 未提供system
            >>> system = template._get_system(inputs)
            >>> print(system)  # "You are a helpful assistant"
            
            >>> # 示例2：使用自定义系统提示和工具
            >>> inputs = StdTemplateInputs(
            ...     system="You are a weather assistant",
            ...     messages=[{"role": "user", "content": "What's the weather?"}],
            ...     tools=[{"type": "function", "function": {"name": "get_weather", ...}}]
            ... )
            >>> system = template._get_system(inputs)
            >>> # system包含格式化的工具定义和原系统提示，格式由agent_template决定
            >>> print(system)
            # "You are a weather assistant\n\n# Tools\n\n## get_weather\n..."
        """
        # 获取模板元数据对象，包含模板配置和默认值
        template_meta = self.template_meta
        
        # 获取用户提供的系统提示（可能为None）
        system = inputs.system
        
        # 获取工具定义列表（可能为None）
        tools = inputs.tools
        
        # 验证系统提示是否符合模板要求（某些模板不支持system或有特定格式要求）
        template_meta.check_system(system)
        
        # 如果用户未提供系统提示，使用模板的默认系统提示
        if system is None:
            system = template_meta.default_system

        # 如果提供了工具定义，使用agent模板格式化工具信息并整合到系统提示中
        if tools is not None:
            # _format_tools方法将工具列表格式化为文本描述，并与原系统提示合并
            # 参数：tools（工具列表）、system or ''（原系统提示或空串）、messages[0]（首条消息作为上下文）
            system = self.agent_template._format_tools(tools, system or '', inputs.messages[0])
        return system
    def _swift_prepare_inputs(self, inputs):
        """
        函数功能：
            预处理Swift后端的输入消息，规范化消息序列格式。该方法处理两类特殊情况：
            (1)合并assistant后的多个连续tool消息并格式化工具调用结果；
            (2)合并相同角色的连续消息（assistant连续或user连续）以避免模板解析错误。
            这些规范化操作确保消息序列符合对话模板的要求，支持function calling和多轮对话的正确编码。
        
        参数：
            inputs (StdTemplateInputs): 标准模板输入对象，包含messages消息列表，该方法会原地修改messages
        
        返回值：
            None: 该方法原地修改inputs.messages，无返回值
        
        使用示例：
            >>> # 示例1：合并连续的tool消息
            >>> inputs = StdTemplateInputs(messages=[
            ...     {"role": "assistant", "content": "调用工具"},
            ...     {"role": "tool", "content": "结果1"},
            ...     {"role": "tool", "content": "结果2"}
            ... ])
            >>> template._swift_prepare_inputs(inputs)
            >>> # messages变为：[assistant消息(格式化后), 合并的tool消息]
            
            >>> # 示例2：合并相同角色的连续消息
            >>> inputs = StdTemplateInputs(messages=[
            ...     {"role": "user", "content": "你好"},
            ...     {"role": "user", "content": "请回答"}
            ... ])
            >>> template._swift_prepare_inputs(inputs)
            >>> # messages变为：[{"role": "user", "content": "你好请回答"}]
        """
        messages = inputs.messages  # 获取消息列表
        if len(messages) < 2:  # 少于2条消息无需处理
            return
        
        i = 1  # 从第2条消息开始遍历（需要比较前后消息）
        while i < len(messages):
            pre_message, message = messages[i - 1], messages[i]  # 获取当前消息和前一条消息
            pre_role, pre_content = pre_message['role'], pre_message['content']  # 前一条消息的角色和内容
            role, content = message['role'], message['content']  # 当前消息的角色和内容
            if pre_role == 'assistant' and role == 'tool':  # 情况1：assistant后跟tool消息（工具调用场景）
                i_start = i  # 记录tool消息序列的起始位置
                # 收集所有连续的tool消息
                while i + 1 < len(messages) and messages[i + 1]['role'] == 'tool':
                    i += 1
                # 格式化assistant的工具调用和tool消息序列，返回更新后的assistant内容和合并的tool内容
                pre_message['content'], tool_content = self.agent_template._format_tool_responses(
                    pre_content, messages[i_start:i + 1])
                # 将多个tool消息替换为单个合并后的tool消息
                messages[i_start:i + 1] = [{'role': 'tool', 'content': tool_content}]
                i = i_start + 1  # 移动到合并后tool消息的下一位置
            elif pre_role == 'assistant' and role == 'assistant' or pre_role == 'user' and role == 'user':
                # 情况2：相同角色的连续消息（assistant连续或user连续），需要合并避免模板解析错误
                pre_message['content'] = pre_content + content  # 将当前消息内容追加到前一条消息
                messages.pop(i)  # 删除当前消息（已合并）
                # i不变，因为删除了当前位置的消息，下一个消息自动移到了当前位置
            else:  # 其他情况：正常的消息序列，继续遍历
                i += 1

    def _swift_encode(self, inputs: StdTemplateInputs):
        """
        函数功能：
            使用Swift自定义后端编码对话消息，构建完整的上下文列表和loss权重列表。该方法是Swift后端的核心
            编码逻辑，负责将结构化的对话消息转换为模板格式的上下文序列，处理包括：系统提示插入、BOS/EOS
            token添加、多轮对话拼接、工具调用格式化、后置系统提示、loss_scale权重分配等。
            相比Jinja后端，Swift后端提供更精细的控制（如区分prompt和answer的loss_scale），支持复杂的对话模板和多模态标签处理。
        
        参数：
            inputs (StdTemplateInputs): 标准模板输入对象，包含messages、system、tools等信息
        
        返回值：
            Tuple[List[Context], List[float], int]: (上下文列表, loss_scale列表, 答案长度)
                - 第一个元素: res_context_list，上下文列表，包含字符串和token ID列表
                - 第二个元素: loss_scale_list，与上下文列表对应的loss权重列表
                - 第三个元素: answer_len，答案部分的上下文元素数量，训练模式下计算，推理模式下为0
        
        使用示例：
            >>> # 示例1：单轮对话编码（训练模式）
            >>> template = Template(..., template_backend='swift')
            >>> template.mode = 'train'
            >>> inputs = StdTemplateInputs(
            ...     messages=[
            ...         {"role": "user", "content": "你好"},
            ...         {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
            ...     ]
            ... )
            >>> context_list, loss_scale, answer_len = template._swift_encode(inputs)
            >>> # context_list包含：[BOS token, system前缀, 用户消息, assistant消息, suffix]
            >>> # loss_scale对应每个context的权重，用户消息为0.0，assistant消息为1.0
            
            >>> # 示例2：多轮对话with工具调用（推理模式）
            >>> template.mode = 'pt'
            >>> inputs = StdTemplateInputs(
            ...     system="你是一个助手",
            ...     messages=[
            ...         {"role": "user", "content": "查询天气"},
            ...         {"role": "assistant", "content": "调用get_weather"},
            ...         {"role": "tool", "content": "晴天"},
            ...         {"role": "user", "content": "谢谢"}
            ...     ],
            ...     tools=[...]
            ... )
            >>> context_list, loss_scale, answer_len = template._swift_encode(inputs)
            >>> # 多轮对话被正确拼接，tool消息被格式化，answer_len=0（推理模式）
        """
        template_meta = self.template_meta  # 获取模板元数据
        self._swift_prepare_inputs(inputs)  # 预处理输入消息（合并连续tool消息和相同角色消息）
        system = self._get_system(inputs)  # 获取并处理系统提示

        self._get_std_messages(inputs.messages)  # 标准化消息格式（确保偶数条消息，补充空消息）
        n_round = len(inputs.messages) // 2  # 计算对话轮次
        if n_round > 1 and not self.template_meta.support_multi_round:  # 检查模板是否支持多轮对话
            logger.warning_once(
                'The template does not support multi-round chat. Only use the last round of the conversation.')
            inputs.messages = inputs.messages[-2:]  # 仅保留最后一轮对话

        res_context_list: List[Context] = []  # 初始化结果上下文列表
        res_context_types: List[ContextType] = []  # 初始化上下文类型列表
        sep_token = None  # 分隔token，用于auto_add_bos模式
        if template_meta.auto_add_bos:  # 如果需要自动添加BOS token
            # 通过编码单个字符'a'来提取BOS和EOS token
            all_tokens = self.tokenizer.encode('a')  # 完整编码（包含特殊token）
            single_token = self.tokenizer.encode('a', add_special_tokens=False)  # 仅字符编码
            assert len(single_token) == 1  # 确保单字符编码为单个token
            idx = all_tokens.index(single_token[0])  # 找到字符token在完整编码中的位置
            bos_token = all_tokens[:idx]  # 提取BOS token（字符前的token）
            sep_token = all_tokens[idx + 1:]  # 提取分隔/EOS token（字符后的token）
            if bos_token:  # 如果存在BOS token
                res_context_list.append(bos_token)  # 添加到上下文列表开头
                res_context_types.append(ContextType.OTHER)  # 标记为OTHER类型
        
        # 选择前缀类型：后置系统提示或无系统提示时使用普通prefix，否则使用system_prefix
        if self.template_meta.is_post_system or not system:
            prefix = template_meta.prefix
        else:
            prefix = template_meta.system_prefix
        self._concat_context_list(prefix, res_context_list, res_context_types, system=system)  # 添加前缀和系统提示

        n_round = len(inputs.messages) // 2  # 重新计算对话轮次（可能已被截断）
        # 遍历每一轮对话（user-assistant配对）
        for i, (query_message, response_message) in enumerate(zip(inputs.messages[::2], inputs.messages[1::2])):
            query_role, query = query_message['role'], query_message['content']  # 提取问题的角色和内容
            response_role, response = response_message['role'], response_message['content']  # 提取回复的角色和内容
            assert query_role in {'user', 'tool'}, f'query_role: {query_role}'  # 验证问题角色
            assert response_role in {'assistant'}, f'response_role: {response_role}'  # 验证回复角色
            
            # 根据不同情况选择prompt模板
            if query_role == 'tool':  # 工具调用结果作为输入
                prompt = query  # tool内容作为prompt（已格式化）
                query = ''  # query置空
            elif template_meta.is_post_system and i == n_round - 1:  # 后置系统提示且是最后一轮
                prompt = template_meta.system_prompt  # 使用system_prompt模板
            else:  # 普通对话轮次
                prompt = template_meta.prompt  # 使用标准prompt模板

            context_list = prompt.copy()  # 复制prompt模板作为当前轮次的上下文列表
            extra_context_list = []  # 额外的上下文列表（用于轮次间分隔符或后缀）
            extra_context_type = None  # 额外上下文的类型
            
            if i < n_round - 1:  # 非最后一轮对话
                context_list.append('{{RESPONSE}}')  # 添加响应占位符
                if inputs.messages[2 * (i + 1)]['role'] != 'tool':  # 下一轮不是tool消息
                    extra_context_list = template_meta.chat_sep  # 添加对话分隔符
                    extra_context_type = ContextType.OTHER
            elif response is not None:  # 最后一轮且存在响应（训练模式）
                context_list.append('{{RESPONSE}}')  # 添加响应占位符
                # 训练模式且非auto_add_bos，或embedding任务时添加后缀
                if self.is_training and not sep_token or self.task_type == 'embedding':
                    extra_context_list = template_meta.suffix  # 添加后缀（如EOS token）
                    extra_context_type = ContextType.SUFFIX
            elif template_meta.response_prefix:  # 最后一轮且推理模式
                context_list.append(template_meta.response_prefix)  # 添加响应前缀（引导生成）


            # 拼接当前轮次的上下文，替换占位符
            self._concat_context_list(
                context_list,
                res_context_list,
                res_context_types,
                query=query,
                response=response,
                system=system,
                round0=i)
            res_context_list += extra_context_list  # 添加额外上下文（分隔符或后缀）
            res_context_types += [extra_context_type] * len(extra_context_list)
        
        # 如果auto_add_bos且存在sep_token，在末尾添加（某些模型的特殊处理）
        if template_meta.auto_add_bos and sep_token:
            res_context_list.append(sep_token)
            res_context_types.append(ContextType.SUFFIX)
        
        # 根据上下文类型计算loss_scale权重列表
        res_context_list, loss_scale_list = self.loss_scale(res_context_list, res_context_types, inputs.messages)

        # 计算答案部分长度：训练模式下统计extra_context和response，推理模式下为0
        if self.is_training:
            answer_len = len(extra_context_list) + bool(response is not None)
        else:
            answer_len = 0
        
        return res_context_list, loss_scale_list, answer_len

    def _truncate(self, input_ids: List[int], labels: Optional[List[int]], loss_mask: Optional[List[float]],
                  truncation_strategy: Literal['left', 'right']):
        """
        函数功能：
            智能截断超长序列，将序列长度缩减至max_length以内。该方法的核心特点是在截断过程中
            保护占位符token（placeholder tokens）不被删除，确保多模态数据的占位标记（如<image>、
            <video>等）始终保留。通过先标记所有需要保护的token，再根据截断策略选择保留其他token，
            最后同步截断input_ids、labels和loss_mask，保持三者的一致性。
        
        参数：
            input_ids (List[int]): 输入token ID序列
                包含所有token（文本、占位符、特殊token等）
                例如：[1, 2, 100, 3, 4, 101, 5, 6, 7, 8, ...]（其中100、101是占位符token）
            labels (Optional[List[int]]): 标签序列，与input_ids等长
                None：推理模式，无标签
                List[int]：训练模式，包含实际token ID或-100
            loss_mask (Optional[List[float]]): loss权重序列，与input_ids等长
                None：使用标准loss计算
                List[float]：每个位置的loss权重
            truncation_strategy (Literal['left', 'right']): 截断策略
                'left': 左截断，删除序列左侧（开头）的token，保留右侧（结尾）
                    适用于推理场景，保留最新的上下文
                'right': 右截断，删除序列右侧（结尾）的token，保留左侧（开头）
                    适用于训练场景，保留对话开头部分
        
        返回值：
            Tuple[List[int], Optional[List[int]], Optional[List[float]]]: 截断后的三元组
                - input_ids (List[int]): 截断后的输入token序列，长度 <= max_length
                - labels (Optional[List[int]]): 截断后的标签序列（若原值非None）
                - loss_mask (Optional[List[float]]): 截断后的loss权重序列（若原值非None）
                注：所有占位符token都会保留，即使总长度超过max_length
        使用示例：
            >>> # 示例1：右截断（保留左侧），无占位符token
            >>> template = Template(...)
            >>> template.max_length = 10
            >>> template.placeholder_tokens = []  # 无占位符
            >>> input_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 长度15
            >>> labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            >>> loss_mask = None
            >>> result_ids, result_labels, _ = template._truncate(
            ...     input_ids, labels, loss_mask, truncation_strategy='right'
            ... )
            >>> print(len(result_ids))  # 10
            >>> print(result_ids)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]（保留左侧10个）
            
            >>> # 示例2：左截断（保留右侧），无占位符token
            >>> result_ids, result_labels, _ = template._truncate(
            ...     input_ids, labels, loss_mask, truncation_strategy='left'
            ... )
            >>> print(result_ids)  # [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]（保留右侧10个）
            
            >>> # 示例3：右截断，带占位符token（100和101是占位符）
            >>> template.placeholder_tokens = [100, 101]  # 设置占位符token
            >>> input_ids = [1, 2, 100, 3, 4, 5, 101, 6, 7, 8, 9, 10, 11, 12]  # 长度14，含2个占位符
            >>> labels = [1, 2, 100, 3, 4, 5, 101, 6, 7, 8, 9, 10, 11, 12]
            >>> result_ids, result_labels, _ = template._truncate(
            ...     input_ids, labels, loss_mask, truncation_strategy='right'
            ... )
            >>> print(len(result_ids))  # 10
            >>> print(100 in result_ids)  # True（占位符100必定保留）
            >>> print(101 in result_ids)  # True（占位符101必定保留）
            >>> # result_ids可能是：[1, 2, 100, 3, 4, 5, 101, 6, 7, 8]
            >>> # 保留了2个占位符 + 8个其他token（左侧的8个非占位符token）
            
            >>> # 示例4：占位符数量等于max_length（只保留占位符）
            >>> template.max_length = 3
            >>> input_ids = [1, 100, 2, 101, 3, 102, 4]  # 3个占位符：100, 101, 102
            >>> template.placeholder_tokens = [100, 101, 102]
            >>> result_ids, _, _ = template._truncate(
            ...     input_ids, None, None, truncation_strategy='right'
            ... )
            >>> print(result_ids)  # [100, 101, 102]（只保留占位符，其他全部删除）
            
            >>> # 示例5：带loss_mask的截断
            >>> template.max_length = 8
            >>> template.placeholder_tokens = [100]
            >>> input_ids = [1, 2, 100, 3, 4, 5, 6, 7, 8, 9, 10]  # 长度11
            >>> labels = [-100, -100, 100, 3, 4, 5, 6, 7, 8, 9, 10]
            >>> loss_mask = [0.0, 0.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
            >>> result_ids, result_labels, result_mask = template._truncate(
            ...     input_ids, labels, loss_mask, truncation_strategy='right'
            ... )
            >>> print(len(result_ids))  # 8
            >>> print(100 in result_ids)  # True
            >>> print(len(result_mask))  # 8（loss_mask同步截断）
        """
        # ===== 步骤1：准备tensor数据 =====
        # 将占位符token列表转换为tensor，形状：(num_placeholders,)
        # 例如：placeholder_tokens = [100, 101] -> tensor([100, 101])
        placeholder_tokens = torch.tensor(self.placeholder_tokens)
        
        # 将input_ids列表转换为tensor，形状：(seq_len,)
        # 例如：input_ids = [1, 2, 100, 3, 4, 101, 5] -> tensor([1, 2, 100, 3, 4, 101, 5])
        input_ids_tensor = torch.tensor(input_ids)
        
        # ===== 步骤2：标记所有占位符token的位置 =====
        # NOTE: 张量广播 + 布尔掩码
        # 使用广播机制比较input_ids和placeholder_tokens，生成一个布尔掩码 protected，标记哪些位置是占位符。
        # input_ids_tensor[:, None]：形状 (seq_len, 1)，在维度1上添加新维度
        # placeholder_tokens：形状 (num_placeholders,)
        # 广播后比较：形状 (seq_len, num_placeholders)，每个位置与每个占位符比较
        # .any(dim=-1)：沿最后一维（num_placeholders）求或，形状 (seq_len,)
        # 结果：布尔tensor，True表示该位置是占位符token
        # 例如：input_ids=[1, 2, 100, 3, 101], placeholders=[100, 101]
        #       -> protected = tensor([False, False, True, False, True])
        protected = (input_ids_tensor[:, None] == placeholder_tokens).any(dim=-1)
        
        # 计算需要保护的token总数（占位符数量）
        # .sum()：计算True的数量，.item()：转为Python整数
        # 例如：protected = [False, False, True, False, True] -> n_protected = 2
        n_protected = protected.sum().item()

        # ===== 步骤3：选择需要保留的非占位符token =====
        if n_protected < self.max_length:  # 若占位符数量少于max_length（还能保留部分其他token）
            # 获取所有非占位符token的索引
            # ~protected：取反，True变False，False变True
            # .nonzero(as_tuple=True)[0]：获取True位置的索引，as_tuple=True返回元组，取第一个元素
            # 例如：protected = [False, False, True, False, True]
            #       -> ~protected = [True, True, False, True, False]
            #       -> non_protected = tensor([0, 1, 3])（索引0, 1, 3是非占位符）
            non_protected = (~protected).nonzero(as_tuple=True)[0]
            
            # 根据截断策略选择保留哪些非占位符token
            if truncation_strategy == 'left':  # 左截断（删除左侧，保留右侧）
                # 计算能保留的非占位符数量：max_length - n_protected
                # 选择最右侧的这些非占位符token
                # 例如：max_length=8, n_protected=2 -> 能保留6个非占位符
                #       non_protected = [0, 1, 3, 5, 7, 8, 9] (7个) -> 选择最右侧6个：[1, 3, 5, 7, 8, 9]
                idx = non_protected[-(self.max_length - n_protected):]
            else:  # 右截断（删除右侧，保留左侧）
                # 选择最左侧的非占位符token
                # 例如：max_length=8, n_protected=2 -> 能保留6个非占位符
                #       non_protected = [0, 1, 3, 5, 7, 8, 9] (7个) -> 选择最左侧6个：[0, 1, 3, 5, 7, 8]
                idx = non_protected[:self.max_length - n_protected]
            
            # 将选中的非占位符token也标记为"保护"（保留）
            # 此时protected中包含：所有占位符 + 选中的非占位符
            # 例如：原protected = [False, False, True, False, True]
            #       idx = [0, 1, 3] -> 新protected = [True, True, True, True, True]（保留前4个位置）
            protected[idx] = True
        
        # ===== 步骤4：根据protected掩码提取保留的token =====
        # 使用布尔索引提取所有protected=True的token
        # input_ids_tensor[protected]：形状 (num_protected,)
        # .tolist()：转回Python列表
        # 例如：input_ids_tensor = [1, 2, 100, 3, 4, 101, 5]
        #       protected = [True, True, True, True, False, False, False]
        #       -> input_ids = [1, 2, 100, 3]
        input_ids = input_ids_tensor[protected].tolist()
        
        # ===== 步骤5：同步截断labels（若存在） =====
        if labels is not None:  # 若有labels字段
            # 将labels转为tensor，应用相同的protected掩码，再转回列表
            # 确保labels与input_ids保持同步
            labels = torch.tensor(labels)[protected].tolist()
        
        # ===== 步骤6：同步截断loss_mask（若存在） =====
        if loss_mask is not None:  # 若有loss_mask字段
            # 将loss_mask转为tensor，应用相同的protected掩码，再转回列表
            # 确保loss_mask与input_ids保持同步
            loss_mask = torch.tensor(loss_mask)[protected].tolist()
        
        # ===== 步骤7：返回截断后的三元组 =====
        return input_ids, labels, loss_mask

    def _encode_truncated(self, inputs: StdTemplateInputs):
        """函数功能：带截断策略的编码方法，调用_encode并根据max_length应用截断。
        
        主要步骤：1.添加默认多模态标签 2.调用_encode编码 3.计算长度 4.应用截断策略
        
        参数：inputs - 标准输入对象
        返回值：编码后的字典，包含input_ids, labels, loss_scale, length等
        """
        if inputs.is_multimodal:  # 若为多模态输入
            self._add_default_tags(inputs)  # 添加默认的多模态标签

        if self.mode in {'vllm', 'lmdeploy', 'sglang'}:  # 若为推理框架模式
            encoded = Template._encode(self, inputs)  # 调用基础_encode方法
            for key in ['images', 'audios', 'videos']:  # 遍历多模态字段
                value = getattr(inputs, key)  # 获取多模态数据
                if value:  # 若有数据
                    encoded[key] = value  # 添加到编码结果（推理框架需要）
        else:  # 标准训练或推理模式
            encoded = self._encode(inputs)  # 调用_encode方法

        input_ids = encoded.get('input_ids')  # 获取input_ids
        labels = encoded.get('labels')  # 获取labels
        loss_scale = encoded.get('loss_scale')  # 获取loss_scale

        # 计算序列长度（input_ids might be a tensor）
        lengths = [0]  # 初始化长度列表
        if input_ids is not None:  # 若有input_ids
            lengths.append(len(input_ids))  # 添加input_ids长度
        if labels is not None:  # 若有labels
            lengths.append(len(labels))  # 添加labels长度
        length = max(lengths)  # 取最大长度
        encoded['length'] = length  # 保存长度

        # 应用截断策略
        if self.max_length is not None and length > self.max_length:  # 若超过max_length
            if self.truncation_strategy in {'right', 'left'}:  # 若为截断策略
                input_ids, labels, loss_scale = self._truncate(
                    input_ids, labels, loss_scale, truncation_strategy=self.truncation_strategy)  # 执行截断
            elif self.truncation_strategy == 'raise':  # 若为抛异常策略
                raise MaxLengthError(f'Current length of row({length}) is larger'
                                     f' than the max_length({self.max_length}).')  # 抛出长度超限异常
        encoded['input_ids'] = input_ids  # 更新input_ids
        encoded['labels'] = labels  # 更新labels
        encoded['loss_scale'] = loss_scale  # 更新loss_scale
        return encoded  # 返回编码结果
    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """
        函数功能：
            Template类的核心编码实现方法。该方法根据配置的模板后端（swift或jinja）选择相应的
            编码策略，将标准化的输入数据转换为token序列。支持编码器-解码器架构和GKD模式的特殊
            处理（分离prompt和answer），并处理动态EOS、Megatron上下文并行等高级特性。最后根据
            训练/推理模式决定是否保留labels和loss_scale字段。
        
        参数：
            inputs (StdTemplateInputs): 标准化的模板输入对象，包含：
                - messages (List[Dict[str, Any]]): 对话消息列表
                - system (Optional[str]): 系统提示词
                - tools (Optional[List[Tool]]): 工具列表（Agent场景）
                - images/audios/videos: 多模态数据（如有）
                - 其他特定任务的字段
        
        返回值：
            Dict[str, Any]: 编码后的字典，包含以下字段：
                - input_ids (List[int]): 输入token序列
                - labels (Optional[List[int]]): 标签序列（训练模式下有效，推理模式为None）
                - loss_scale (Optional[List[float]]): loss权重序列（训练模式下有效，推理模式为None）
                - tokenizer_kwargs (Optional[Dict]): tokenizer的额外参数（如Qwen-Audio的prompt）
                - prompt_input_ids/answer_input_ids (List[int]): 编码器-解码器或GKD模式下的分段序列
                - prompt_labels/answer_labels (Optional[List[int]]): 分段标签序列
                - prompt_loss_scale/answer_loss_scale (Optional[List[float]]): 分段loss权重序列
        
        使用示例：
            >>> # 示例1：标准因果语言模型编码（decoder-only）
            >>> template = Template(...)
            >>> template.mode = 'train'
            >>> template.template_backend = 'swift'
            >>> inputs = StdTemplateInputs(
            ...     messages=[
            ...         {"role": "user", "content": "你好"},
            ...         {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
            ...     ]
            ... )
            >>> encoded = template._encode(inputs)
            >>> print(encoded.keys())
            # dict_keys(['input_ids', 'labels', 'loss_scale'])
            >>> print(len(encoded['input_ids']))  # 例如: 25
            >>> print(encoded['labels'][0])  # -100 (第一个token不计算loss)
            
            >>> # 示例2：编码器-解码器架构（如T5、Qwen-Audio）
            >>> template.is_encoder_decoder = True
            >>> encoded = template._encode(inputs)
            >>> print(encoded.keys())
            # dict_keys(['prompt_input_ids', 'prompt_labels', 'prompt_loss_scale',
            #           'answer_input_ids', 'answer_labels', 'answer_loss_scale',
            #           'input_ids', 'labels', 'loss_scale', 'tokenizer_kwargs'])
            >>> # prompt部分: 编码器输入（用户问题）
            >>> # answer部分: 解码器输入/输出（助手回答）
            
            >>> # 示例3：推理模式（不生成labels和loss_scale）
            >>> template.mode = 'pt'  # 推理模式
            >>> inputs = StdTemplateInputs(
            ...     messages=[{"role": "user", "content": "你好"}]
            ... )
            >>> encoded = template._encode(inputs)
            >>> print(encoded['labels'])  # None
            >>> print(encoded['loss_scale'])  # None
            >>> print(len(encoded['input_ids']))  # 例如: 10
            
            >>> # 示例4：GKD（Generalized Knowledge Distillation）模式
            >>> template.mode = 'gkd'
            >>> encoded = template._encode(inputs)
            >>> # 返回分离的prompt和answer，用于知识蒸馏训练
            
            >>> # 示例5：Jinja模板后端（使用tokenizer.apply_chat_template）
            >>> template.template_backend = 'jinja'
            >>> encoded = template._encode(inputs)
            >>> # 使用HuggingFace标准的chat template进行编码
        """
        # ===== 步骤1：确定模板后端（swift或jinja） =====
        template_backend = self.template_backend  # 获取配置的模板后端
        # 特殊情况：dummy模板在推理模式下自动切换到jinja后端
        if (self.template_meta.template_type == 'dummy' and self.use_chat_template and not self.is_training
                and self.task_type != 'seq_cls'):  # 若为dummy模板、启用chat_template、推理模式、非分类任务
            template_backend = 'jinja'  # 切换到jinja后端（使用tokenizer自带的chat template）
            logger.info_once(f'Setting template_backend: {template_backend}')  # 记录日志（仅输出一次）

        # ===== 步骤2：根据模板后端调用相应的编码方法 =====
        # 返回值：res_context_list（上下文列表）, loss_scale_list（loss权重列表）, answer_len（答案部分长度）
        res_context_list, loss_scale_list, answer_len = (
            self._swift_encode(inputs) if template_backend == 'swift' else self._jinja_encode(inputs))
        
        encoded = {}  # 初始化编码结果字典
        
        # ===== 步骤3：根据模型架构选择编码策略 =====
        if self.is_encoder_decoder or self.mode == 'gkd':  # 若为编码器-解码器架构或GKD模式
            # 需要分离prompt（编码器输入）和answer（解码器输入/输出）
            # 例如：T5模型的编码器处理问题，解码器生成答案
            # GKD模式需要分别处理teacher和student的prompt/answer
            
            total_len = len(res_context_list)  # 获取总上下文长度
            # 遍历prompt和answer两部分，分别进行编码
            for key, _slice in zip(['prompt', 'answer'],  # key: 'prompt' 或 'answer'
                                   [slice(0, total_len - answer_len),  # prompt部分的切片
                                    slice(total_len - answer_len, total_len)]):  # answer部分的切片
                # 简化上下文列表（合并相邻的同类型context）
                context_list, loss_scale = self._simplify_context_list(res_context_list[_slice],
                                                                       loss_scale_list[_slice], inputs)
                # 将context列表编码为token序列
                input_ids, labels, loss_scale, tokenizer_kwargs = self._encode_context_list(context_list, loss_scale)
                # 保存分段的编码结果
                encoded[f'{key}_input_ids'] = input_ids  # 'prompt_input_ids' 或 'answer_input_ids'
                encoded[f'{key}_labels'] = labels  # 'prompt_labels' 或 'answer_labels'
                encoded[f'{key}_loss_scale'] = loss_scale  # 'prompt_loss_scale' 或 'answer_loss_scale'
            
            # 拼接prompt和answer部分，生成完整序列（用于某些场景）
            input_ids = encoded['prompt_input_ids'] + encoded['answer_input_ids']
            labels = encoded['prompt_labels'] + encoded['answer_labels']
            loss_scale = None  # 默认不拼接loss_scale
            if isinstance(encoded['prompt_loss_scale'], list):  # 若loss_scale是列表（而非None）
                loss_scale = encoded['prompt_loss_scale'] + encoded['answer_loss_scale']  # 拼接loss_scale
        else:  # 标准的decoder-only架构（如GPT、Qwen、LLaMA等）
            # 简化上下文列表
            res_context_list, loss_scale_list = self._simplify_context_list(res_context_list, loss_scale_list, inputs)
            # 将完整的context列表一次性编码为token序列
            input_ids, labels, loss_scale, tokenizer_kwargs = self._encode_context_list(
                res_context_list, loss_scale_list)

        # ===== 步骤4：添加动态EOS token（如需要） =====
        # 某些模板可能需要在特定位置添加EOS token（如suffix部分）
        self._add_dynamic_eos(input_ids, labels, loss_scale, self._encode_context_list(self.template_meta.suffix)[0])

        # ===== 步骤5：保存tokenizer额外参数（如Qwen-Audio的prompt参数） =====
        if tokenizer_kwargs:  # 若有额外的tokenizer参数
            encoded['tokenizer_kwargs'] = tokenizer_kwargs  # 保存到编码结果中

        # ===== 步骤6：保存核心编码结果 =====
        encoded['input_ids'] = input_ids  # 输入token序列
        encoded['labels'] = labels  # 标签序列（训练时用于计算loss）
        encoded['loss_scale'] = loss_scale  # loss权重序列
        
        # ===== 步骤7：处理Megatron上下文并行（Context Parallelism） =====
        self._handle_megatron_cp(encoded)  # 为Megatron-LM的上下文并行添加padding
        # TODO: fix cp_size & cached_dataset

        # ===== 步骤8：将第一个token的labels设为-100（不计算loss） =====
        # 第一个token通常是BOS或系统提示的开始，不参与loss计算
        if encoded.get('labels') is not None:  # 若有labels
            encoded['labels'][0] = -100  # 设置为-100（PyTorch中ignore_index的默认值）
        if encoded.get('loss_scale') is not None:  # 若有loss_scale
            encoded['loss_scale'][0] = 0  # 设置为0（该位置loss权重为0）

        # ===== 步骤9：推理模式下移除labels和loss_scale =====
        if not self.is_training:  # 若为推理模式（而非训练模式）
            # 遍历所有键，移除训练相关的字段
            for k in list(encoded.keys()):  # 使用list()避免在迭代时修改字典
                if k.endswith('labels') or k.endswith('loss_scale'):  # 若键名以labels或loss_scale结尾
                    encoded[k] = None  # 设置为None（推理时不需要这些字段）
        return encoded
    def pre_forward_hook(self, model: nn.Module, args, kwargs):
        """
        函数功能：
            模型前向传播的预处理钩子函数。在模型forward方法执行前自动调用，负责对输入kwargs进行预处理和转换。
            主要用于多模态模型的特殊处理（如将input_ids转换为inputs_embeds）、设备迁移、参数兼容性检查等。
            通过_post_encode方法支持子类自定义输入转换逻辑（如图像特征编码、多模态融合等）。
        
        参数：
            model (nn.Module): PyTorch模型实例，可能是PeftModel或原始模型
            args (tuple): 位置参数（通常为空，未使用）
            kwargs (dict): 关键字参数字典，包含模型forward所需的输入数据
                - input_ids (torch.Tensor): 输入token序列，shape: (batch_size, seq_len)
                - attention_mask (torch.Tensor): 注意力掩码，shape: (batch_size, seq_len)
                - labels (torch.Tensor): 标签序列（训练时），shape: (batch_size, seq_len)
                - pixel_values (torch.Tensor): 图像像素值（多模态）
                - 其他模型特定的输入参数
        
        返回值：
            Tuple[tuple, dict]: (args, kwargs)元组
                - args: 位置参数（保持不变，原样返回）
                - kwargs: 处理后的关键字参数字典，可能包含：
                    - inputs_embeds (torch.Tensor): 嵌入向量（多模态模型），shape: (batch_size, seq_len, hidden_size)
                    - input_ids (torch.Tensor): token序列（若未使用inputs_embeds）
                    - attention_mask, labels等其他参数
                    - 已移除不兼容的参数（如模型不支持的position_ids）
        
        使用示例：
            >>> # 示例1：多模态模型的自动处理（通过register_forward_pre_hook注册）
            >>> template = QwenVLTemplate(...)
            >>> model = AutoModel.from_pretrained('Qwen-VL')
            >>> template.register_hook(model)  # 内部调用model.register_forward_pre_hook(template.pre_forward_hook)
            >>> # 当调用model(**kwargs)时，pre_forward_hook自动执行
            >>> outputs = model(input_ids=input_ids, pixel_values=pixel_values)
            >>> # pre_forward_hook将pixel_values编码并融合到inputs_embeds中
            
            >>> # 示例2：手动调用进行输入预处理
            >>> template = Template(...)
            >>> kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            >>> args, processed_kwargs = template.pre_forward_hook(model, (), kwargs)
            >>> outputs = model(**processed_kwargs)
        """
        from swift.llm import to_device  # 导入设备迁移工具函数
        
        # 将kwargs中所有tensor迁移到模型所在设备，保持数据类型不变
        # 例如：kwargs={'input_ids': tensor(cpu), 'labels': tensor(cpu)} -> old_kwargs={'input_ids': tensor(cuda:0), 'labels': tensor(cuda:0)}
        old_kwargs = to_device(kwargs, model.device)
        
        # 调用_post_encode进行模型特定的输入转换，然后再次迁移到模型设备
        # _post_encode由子类重写以实现特殊逻辑：如Qwen-VL将pixel_values编码为inputs_embeds
        # 例如：old_kwargs有pixel_values -> _post_encode提取图像特征并融合 -> kwargs={'inputs_embeds': ..., 'attention_mask': ...}
        kwargs = to_device(self._post_encode(model, old_kwargs), model.device)
        
        # 恢复_post_encode中未处理但需要保留的关键参数
        # 这些参数要么是模型forward必需的，要么是用于特定功能（如Megatron的序列并行）
        for k, v in old_kwargs.items():
            # 检查参数是否在白名单中，且_post_encode未返回该参数
            if k in {
                    'input_ids', 'attention_mask', 'labels', 'position_ids', 'output_hidden_states', 'logits_to_keep',
                    'cumulative_seqlens_q', 'cumulative_seqlens_k', 'max_length_q', 'max_length_k'
            } and k not in kwargs:
                # 将缺失的参数从old_kwargs恢复到kwargs
                # 例如：_post_encode返回{'inputs_embeds': ...}，但缺少attention_mask，则从old_kwargs恢复
                kwargs[k] = v
        
        # 若使用inputs_embeds（多模态融合后的嵌入），则移除input_ids避免冲突
        # transformers模型forward中，inputs_embeds优先级高于input_ids，两者不能同时存在
        if 'inputs_embeds' in kwargs:
            kwargs.pop('input_ids', None)  # 安全移除input_ids（若不存在也不报错）
        
        # 检查模型forward方法是否支持position_ids参数
        base_model = self.get_base_model(model)  # 获取底层模型（剥离PeftModel包装）
        parameters = inspect.signature(base_model.forward).parameters  # 获取forward方法的参数签名
        
        # 若模型不支持position_ids参数，则从kwargs中移除，避免传入时报错
        # 部分模型（如某些视觉模型）的forward不接受position_ids
        if 'position_ids' not in parameters:
            kwargs.pop('position_ids', None)
        
        # 返回处理后的参数
        # args保持不变（通常为空元组），kwargs已完成设备迁移、多模态处理、参数兼容性调整
        return args, kwargs

    @property
    def is_training(self):
        return self.mode not in {'pt', 'vllm', 'lmdeploy', 'sglang'}

    def set_mode(self, mode: Literal['pt', 'vllm', 'lmdeploy', 'sglang', 'train', 'rlhf', 'kto', 'gkd']) -> None:
        self.mode = mode

    def register_post_encode_hook(self, models: List[nn.Module]) -> None:
        """
        函数功能：
            为模型注册前向传播预处理钩子。
            该方法将`pre_forward_hook`注册到模型的forward方法执行前，对多模态训练至关重要，
            因为它会在模型前向传播前自动将input_ids转换为inputs_embeds（融合图像、视频、音频等多模态特征）。
            支持DeepSpeed Zero-3分布式训练的特殊处理，确保钩子在Zero-3初始化后仍保持正确的执行顺序。
            注册是幂等的，重复调用不会重复注册。
        
        参数：
            models (List[nn.Module]): 需要注册钩子的模型列表
                - 通常包含主模型及其子模块
                - 对于多模态模型（如Qwen-VL、InternVL），钩子会处理pixel_values等多模态输入
                - 示例：[model] 或 [model.base_model, model.lm_head]
        
        返回值：
            None: 该方法无返回值，通过副作用（注册钩子、修改_handles列表）完成工作
        
        使用示例：
            >>> # 示例1：多模态模型训练初始化
            >>> from swift.llm import Template
            >>> template = QwenVLTemplate(...)
            >>> model = AutoModelForCausalLM.from_pretrained('Qwen-VL')
            >>> template.register_post_encode_hook([model])  # 注册钩子
            >>> # 后续训练时，model.forward会自动调用pre_forward_hook处理多模态输入
            >>> outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            
            >>> # 示例2：DeepSpeed Zero-3分布式训练
            >>> template.register_post_encode_hook([model])
            >>> # 若启用Zero-3，会自动monkey patch deepspeed.initialize
            >>> # 确保钩子在Zero-3初始化后保持正确执行顺序
        """
        # 幂等性检查：若已注册钩子（_handles非空），直接返回，避免重复注册
        # _handles是(model, handle)元组的列表，存储已注册的钩子信息
        if self._handles:
            return

        # 遍历所有需要注册钩子的模型
        for model in models:
            # 注册前向传播预处理钩子（pre hook）到模型
            # self.pre_forward_hook: 钩子函数，在model.forward执行前被调用
            # with_kwargs=True: 允许钩子接收和修改关键字参数（kwargs），要求PyTorch>=2.0
            # 返回的handle用于后续移除钩子
            handle = model.register_forward_pre_hook(self.pre_forward_hook, with_kwargs=True)
            
            # 将(model, handle)元组添加到_handles列表
            # 保存这些信息以便后续通过remove_post_encode_hook移除钩子
            self._handles.append((model, handle))
        
        # DeepSpeed Zero-3特殊处理
        # Zero-3会在初始化时重新组织模型结构，可能导致钩子执行顺序错乱
        if is_deepspeed_zero3_enabled():
            import deepspeed  # 导入deepspeed库
            
            # 保存原始的deepspeed.initialize函数引用，用于后续恢复（在remove_post_encode_hook中）
            self._deepspeed_initialize = deepspeed.initialize
            
            # 使用装饰器保留原函数的元信息（函数名、文档字符串等）
            @wraps(self._deepspeed_initialize)
            def _initialize(*args, **kwargs):
                # 调用原始的deepspeed.initialize完成Zero-3初始化
                # res包含初始化后的模型、优化器等对象
                res = self._deepspeed_initialize(*args, **kwargs)
                
                # Zero-3初始化后，遍历所有已注册的钩子，调整钩子执行顺序
                for model, handle in self._handles:
                    # 将钩子移动到_forward_pre_hooks有序字典的末尾
                    # 确保pre_forward_hook在其他钩子之后执行（或根据需要调整顺序）
                    # handle.id: 钩子的唯一标识符
                    model._forward_pre_hooks.move_to_end(handle.id)
                
                # 返回初始化结果（模型、优化器等）
                return res
            
            # Monkey patch: 替换deepspeed.initialize为自定义的_initialize
            # 这样在外部调用deepspeed.initialize时，会自动执行钩子顺序调整逻辑
            # 确保在DeepSpeed Zero-3环境下，pre_forward_hook能正确执行
            deepspeed.initialize = _initialize

    def remove_post_encode_hook(self):
        models = []
        for model, handle in self._handles:
            models.append(model)
            handle.remove()
        self._handles = []

        if self._deepspeed_initialize is not None:
            import deepspeed
            deepspeed.initialize = self._deepspeed_initialize
        self._deepspeed_initialize = None
        return models

    def data_collator(self, batch: List[Dict[str, Any]],
                       *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """
        collator: 在机器学习 / 深度学习中，collator通常指数据整理器/批处理组装器，用于在 DataLoader 中把单个样本拼接成一个批次。
        功能：
        将批量的编码数据整理（collate）为模型可直接使用的批次张量。这是训练时的核心方法，
        负责对变长序列进行padding对齐、生成attention mask、拼接多模态数据等操作。

        具体职责：
        1. 根据task_type和mode选择相应的data collator子方法
        2. 对batch中的input_ids、labels等序列进行padding对齐
        3. 生成attention_mask（支持Megatron的4D causal mask）
        4. 拼接多模态数据（pixel_values、image_sizes等）
        5. 处理position_ids、loss_scale等辅助字段
        6. 可选保留extra_kwargs（非标准字段）

        核心流程：
        - causal_lm: 标准/_data_collator、RLHF、KTO、GKD
        - seq_cls: 标准/_seq_cls_data_collator、RLHF
        - prm: 使用标准_data_collator
        - embedding: _embedding_data_collator（处理anchor/positive/negative）
        - reranker: _reranker_data_collator（处理positive/negative pairs）

        参数：
        - batch (List[Dict[str, Any]]): 批量编码数据列表，每个元素是template.encode()返回的字典，
            通常包含{'input_ids': [...], 'labels': [...], 'loss_scale': [...]}等字段。
            batch的长度即为batch_size。
        - padding_to (Optional[int]): 可选的padding目标长度，默认None。
            None时会padding到batch中最长序列的长度；
            指定整数时会padding到该长度（用于固定长度训练或推理）。

        返回值：
        - Dict[str, Any]: 整理后的批次数据字典，通常包含：
            * input_ids (torch.Tensor): shape=[batch_size, seq_len]，padded token ID序列
            * attention_mask (torch.Tensor): shape=[batch_size, seq_len]或[batch_size, 1, seq_len, seq_len]，
              标记有效token位置（1为有效，0为padding）。Megatron模式下为4D causal mask。
            * labels (Optional[torch.Tensor]): shape=[batch_size, seq_len]，训练标签（推理时为None）
            * position_ids (Optional[torch.Tensor]): shape=[batch_size, seq_len]，位置ID序列
            * loss_scale (Optional[torch.Tensor]): shape=[batch_size, seq_len]，loss权重
            * pixel_values (Optional[torch.Tensor]): 多模态图像数据（如有）
            * image_sizes (Optional[List]): 图像尺寸列表（如有）
            * 其他任务特定字段（如RLHF的chosen_*/rejected_*，embedding的anchor_*/positive_*/negative_*）

        使用示例：
        >>> # 示例1：标准训练场景
        >>> template.mode = 'train'
        >>> template.task_type = 'causal_lm'
        >>> batch_encoded = [template.encode(inp) for inp in inputs_list]
        >>> batch = template.data_collator(batch_encoded)
        >>> # batch: {'input_ids': tensor([[...]]), 'attention_mask': tensor([[...]]), 'labels': tensor([[...]])}
        >>> output = model(**batch)  # 可直接送入模型
        >>> 
        >>> # 示例2：固定长度padding
        >>> batch = template.data_collator(batch_encoded, padding_to=2048)
        >>> # 所有序列都会padding到长度2048
        >>> 
        >>> # 示例3：RLHF场景
        >>> template.mode = 'rlhf'
        >>> batch = template.data_collator(batch_encoded)
        >>> # batch包含: chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask等
        >>> 
        >>> # 示例4：多模态训练
        >>> batch = template.data_collator(batch_encoded)
        >>> # batch额外包含: pixel_values, image_sizes等多模态数据
        """
        from swift.llm import RowPreprocessor  # 导入行预处理器：用于将行格式数据转换为批次格式
        
        # 1> 根据task_type和mode选择相应的data collator子方法
        if self.task_type == 'causal_lm':  # 若任务类型为因果语言模型
            if self.mode in {'pt', 'train'}:  # 标准推理或训练模式
                res = self._data_collator(batch, padding_to=padding_to)  # 调用标准data collator：padding、生成attention_mask等
            elif self.mode == 'rlhf':  # RLHF模式
                res = self._rlhf_data_collator(batch, padding_to=padding_to)  # 调用RLHF data collator：分别处理chosen和rejected
            elif self.mode == 'kto':  # KTO模式
                res = self._kto_data_collator(batch, padding_to=padding_to)  # 调用KTO data collator：处理KTO特定格式
            elif self.mode == 'gkd':  # GKD模式
                res = self._gkd_data_collator(batch, padding_to=padding_to)  # 调用GKD data collator：处理prompt和answer分离
        elif self.task_type == 'prm':  # 若任务类型为过程奖励模型
            res = self._data_collator(batch, padding_to=padding_to)  # 使用标准data collator（PRM与causal_lm处理类似）
        elif self.task_type == 'seq_cls':  # 若任务类型为序列分类
            if self.mode == 'rlhf':  # 序列分类的RLHF模式（奖励模型）
                res = self._rlhf_data_collator(batch, padding_to=padding_to)  # 使用RLHF data collator
            else:  # 标准序列分类模式
                res = self._seq_cls_data_collator(batch, padding_to=padding_to)  # 调用序列分类data collator：处理分类标签
        elif self.task_type == 'embedding':  # 若任务类型为文本向量化
            res = self._embedding_data_collator(batch, padding_to=padding_to)  # 调用embedding data collator：处理anchor/positive/negative三元组
        elif self.task_type in {'reranker', 'generative_reranker'}:  # 若任务类型为文档排序
            res = self._reranker_data_collator(batch, padding_to=padding_to)  # 调用reranker data collator：处理positive/negative文档对
        
        # 2> 如果不移除未使用列，则配置为保留额外参数，处理额外参数
        if not self.remove_unused_columns:
            extra_kwargs = [b['_extra_kwargs'] for b in batch if b.get('_extra_kwargs') is not None]  # 收集所有batch元素中的_extra_kwargs
            extra_kwargs = RowPreprocessor.rows_to_batched(extra_kwargs)  # 将行格式的extra_kwargs转换为批次格式（如将列表合并为单个列表）
            res.update({k: v for k, v in extra_kwargs.items() if k not in res})  # 将extra_kwargs合并到结果中（不覆盖已有键）
        
        return res  # 返回整理后的批次数据

    @staticmethod
    def _fetch_inputs_startswith(batch: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
        new_batch = []
        for inputs in batch:
            new_inputs = {}
            for k, v in inputs.items():
                if k.startswith(prefix):
                    new_inputs[k[len(prefix):]] = v
            new_batch.append(new_inputs)
        return new_batch

    @staticmethod
    def fetch_inputs(batch: List[Dict[str, Any]], keys: Optional[List[str]] = None) -> Dict[str, Any]:
        from swift.llm import RowPreprocessor
        keys = keys or []
        rows = RowPreprocessor.rows_to_batched(batch)
        return {k: rows[k] for k in keys if rows.get(k) is not None}

    @staticmethod
    def gather_list(batch: List[Dict[str, Any]], attr_name: str) -> Optional[List[Any]]:
        """
        功能：
            从批次样本 `batch` 中收集名为 `attr_name` 的列表型字段，按顺序扁平化拼接为一个单一列表；
            同时会从各样本字典中移除该字段（使用 pop），以避免后续重复处理。
            原注释含义：List[Tensor] -> List[Tensor]（聚合外层结构，不改变内部元素的形状/类型）。

        参数：
            batch (List[Dict[str, Any]]): 编码后的样本列表，每个样本为一个字典；
                其中 `attr_name` 对应的值应为 List[Any]，常见为 List[Tensor] 或 List[int] 等。
            attr_name (str): 需要收集的字段名（如 'input_ids'、'labels'、'position_ids' 等）。

        返回：
            Optional[List[Any]]: 扁平化后的聚合列表；若所有样本均无该字段则返回空列表 []。
                - 若元素为 Tensor，本函数不更改其 shape（常见形状：(seq_len,) 或 (seq_len, hidden_size)）。

        示例：
            >>> batch = [
            ...     {'input_ids': [torch.tensor([1, 2])]},
            ...     {'input_ids': [torch.tensor([3])]},
            ... ]
            >>> Template.gather_list(batch, 'input_ids')
            [tensor([1, 2]), tensor([3])]
            # 注意：调用后对应键已从各样本中移除，避免后续被再次聚合。
        """
        res = []  # 初始化结果列表，用于累积各样本中该字段的所有元素
        for b in batch:  # 逐样本遍历
            # 仅在样本中存在该字段且不为 None 时处理
            if b.get(attr_name) is not None:
                # 取出并删除该字段（副作用：从样本字典中移除键），再将其列表元素按顺序扩展到 res
                # 若元素为 Tensor，不改变其数值与形状，仅改变外层聚合结构
                res += b.pop(attr_name)
        return res  # 返回聚合后的扁平列表

    @staticmethod
    def concat_tensor(batch: List[Dict[str, Any]], attr_name: str, dim: int) -> Optional[torch.Tensor]:
        res = []
        for b in batch:
            if b.get(attr_name) is not None:
                res.append(b.pop(attr_name))
        return torch.concat(res, dim=dim) if res else None
    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """
        功能：
            批量数据整理器的核心实现。这是训练和推理中DataLoader的关键组件。
            将一个batch中的多个样本（每个样本由encode方法生成）整理为模型forward所需的统一格式，主要工作包括：
                1> 序列padding（对齐到相同长度）；
                2> 数据类型转换（list→tensor）；
                3> attention_mask和position_ids生成；
                4> 支持特殊训练模式（packing、padding_free、Megatron）；
                5> 多模态数据整理；
                6> 序列并行数据处理。
        
        参数：
            batch (List[Dict[str, Any]]): 批量编码数据，每个元素是encode()返回的字典
                - 每个dict包含：input_ids/inputs_embeds, labels, loss_scale, position_ids等字段
                - 示例：[{'input_ids': [1,2,3], 'labels': [-100,2,3]}, {'input_ids': [4,5], 'labels': [-100,5]}]
            
            padding_to (Optional[int]): 固定padding长度，若为None则padding到batch内最长序列
                - 用于确定性训练（固定batch大小）或Megatron的特殊要求
                - 示例：padding_to=512表示所有序列padding到512长度
        
        返回：
            Dict[str, Any]: 整理后的批量数据字典，可直接传入model.forward
                - input_ids (torch.Tensor): shape (batch_size, max_seq_len)，padding后的token序列
                - attention_mask (torch.Tensor): shape (batch_size, max_seq_len) 或 (batch_size, 1, max_seq_len, max_seq_len)
                - labels (torch.Tensor): shape (batch_size, max_seq_len)，padding位置填充-100
                - loss_scale (torch.Tensor): shape (batch_size, max_seq_len)，padding位置填充0
                - position_ids (torch.Tensor): shape (batch_size, max_seq_len)，位置索引
                - pixel_values, image_sizes等多模态字段（如有）
        
        示例：
            >>> # 示例1：标准训练场景
            >>> template = Template(...)
            >>> batch = [template.encode(inputs1), template.encode(inputs2)]
            >>> # batch = [{'input_ids': [1,2,3], 'labels': [-100,2,3]}, {'input_ids': [4,5,6,7], 'labels': [-100,5,6,7]}]
            >>> collated = template._data_collator(batch)
            >>> # collated = {'input_ids': tensor([[1,2,3,pad], [4,5,6,7]]), 'labels': tensor([[-100,2,3,-100], [-100,5,6,7]]), ...}
            
            >>> # 示例2：固定长度padding
            >>> collated = template._data_collator(batch, padding_to=512)
            >>> # 所有序列padding到512长度
        """
        # 1> 前置检查和配置初始化
        assert self.tokenizer.pad_token_id is not None  # 确保tokenizer已配置pad_token_id
        padding_side = self.padding_side if self.is_training else 'left'  # 训练用配置的padding_side，推理用left
        padding_right = padding_side == 'right'  # 标志是否右padding（True=右，False=左）
        
        # 2> 特殊训练模式的预处理
        if self.padding_free:  # padding_free模式：将batch内所有样本packing为单个长序列
            batch[:] = [self.packing_row(batch)]  # 原地替换batch为单个packing后的样本
        elif self.use_megatron:  # Megatron模式：手动生成position_ids
            for encoded in batch:
                encoded['position_ids'] = list(range(len(encoded['labels'])))  # position_ids=[0,1,2,...,seq_len-1]
        
        if self._packing:  # packing模式下必须有position_ids（用于区分不同样本边界）
            assert 'position_ids' in batch[0], f'batch[0]: {batch[0]}'
        
        # 3> 收集批量数据到结果字典
        res = {}
        if self._packing:  # packing模式：所有样本已合并为单序列
            for k in ['input_ids', 'labels', 'position_ids', 'loss_scale', 'channel']:
                v = self.gather_list(batch, k)  # 收集该字段的所有值并flatten（packing模式下batch只有1个元素）
                if v:
                    if k == 'channel':
                        res[k] = v  # channel保持列表格式
                    else:
                        res[k] = [v]  # 其他字段包装为单元素列表（兼容后续统一处理）
        else:  # 非packing模式：收集各样本的字段到列表
            inputs_embeds = [b['inputs_embeds'] for b in batch if b.get('inputs_embeds') is not None]
            input_ids = [b['input_ids'] for b in batch if b.get('input_ids') is not None]
            channel = [b['channel'] for b in batch if b.get('channel') is not None]
            
            if inputs_embeds:  # 多模态场景使用inputs_embeds（融合后的嵌入）
                res['inputs_embeds'] = inputs_embeds
            if input_ids:  # 标准场景使用input_ids
                res['input_ids'] = input_ids
            if channel:  # 数据来源渠道标识
                res['channel'] = channel
            
            for key in ['labels', 'loss_scale', 'position_ids', 'token_type_ids']:
                val = [b[key] for b in batch if b.get(key) is not None]
                if val:
                    res[key] = val
        
        # 4> 数据类型转换：list→tensor，并移除冗余维度
        keys = ['input_ids', 'inputs_embeds', 'attention_mask', 'labels', 'loss_scale', 'position_ids', 'token_type_ids']
        pad_values = [self.tokenizer.pad_token_id, 0., 0, -100, 0., 0., 0]  # 各字段对应的padding值
        
        seq_lens = None  # 记录batch内各序列的长度
        for key in keys:
            if key not in res:
                continue
            for i, val in enumerate(res[key]):
                if isinstance(val, (list, tuple)):  # list/tuple转为tensor
                    val = torch.tensor(val)
                elif key == 'inputs_embeds' and val.ndim == 3 or key != 'inputs_embeds' and val.ndim == 2:
                    # 移除冗余batch维度：inputs_embeds (1,seq_len,hidden)->（seq_len,hidden），其他(1,seq_len)->(seq_len)
                    val = val[0]
                res[key][i] = val
            if not seq_lens:  # 仅需计算一次（所有字段的seq_len相同）
                seq_lens = [seq.shape[0] for seq in res[key]]  # 获取每个样本的序列长度
        
        # 5> 生成attention_mask和position_ids（非packing模式）
        if not self._packing and seq_lens and ('input_ids' in res or 'inputs_embeds' in res):
            if not self.use_megatron:  # 标准模式：生成全1的attention_mask
                # torch.ones
                # 功能：用于创建一个所有元素都为1的张量
                # 原型：torch.ones(size, dtype=None, device=None)
                res['attention_mask'] = [torch.ones(seq_len, dtype=torch.int64) for seq_len in seq_lens]
            if self.is_training and self.padding_side == 'left':  # 训练+左padding：需手动生成position_ids
                # torch.arange
                # 功能：用于创建一个从起始值到终止值的等差序列张量（左闭右开）
                # 原型：torch.arange(start=0, end, step=1, dtype=None, device=None)
                res['position_ids'] = [torch.arange(seq_len, dtype=torch.int64) for seq_len in seq_lens]
        
        # 6> Megatron特殊处理：生成4D causal attention mask
        if self.use_megatron:
            if padding_to is not None:  # 向上取整到padding_to的倍数
                padding_to = math.ceil(max(seq_lens) / padding_to) * padding_to
            
            if self._packing:  # packing模式：处理序列并行的position_ids
                cp_size = self.sequence_parallel_size
                if cp_size > 1:
                    padding_len = padding_to - seq_lens[0]
                    position_ids = res['position_ids'][0].tolist()
                    position_ids += list(range(cp_size * 2)) * (padding_len // (cp_size * 2))  # 循环填充position_ids
                    res['position_ids'] = [torch.tensor(position_ids)]
            else:  # 非packing模式：生成4D下三角causal mask
                seq_len = max(seq_lens) if padding_to is None else padding_to
                # shape: (batch_size, 1, seq_len, seq_len)，下三角为True（可见），上三角为False（不可见）
                res['attention_mask'] = torch.tril(torch.ones(
                    (len(seq_lens), seq_len, seq_len), dtype=torch.bool)).view(len(seq_lens), 1, seq_len, seq_len)
                assert res['attention_mask'].dtype is torch.bool, f'attention_mask.dtype: {res["attention_mask"].dtype}'
                for i, seq_len in enumerate(seq_lens):  # 将padding部分的mask置为0（不可见）
                    res['attention_mask'][i, :, seq_len:] = 0

        # 7> 执行padding操作
        for key, pad_value in zip(keys, pad_values):
            if key not in res:
                continue
            if self.use_megatron and not self._packing and key == 'attention_mask':
                continue  # Megatron的attention_mask已在上面生成，跳过padding
            
            # 若指定padding_to且不是特殊情况，先padding到padding_to长度
            if padding_to is not None and not (self._packing and key == 'position_ids' and self.sequence_parallel_size > 1):
                padding_len = padding_to - seq_lens[0]
                if padding_len > 0:
                    # F.pad参数：(0, 右padding) for 右padding模式，(右padding, 0) for 左padding模式
                    res[key][0] = F.pad(res[key][0], (0, padding_len) if padding_right else (padding_len, 0),
                                        'constant', pad_value)
            
            # 将列表中的多个tensor padding到相同长度并stack为单个tensor
            # _pad_sequence会根据padding_side选择左或右padding
            res[key] = self._pad_sequence(res[key], pad_value)

        # 8> 整理多模态数据（pixel_values, image_sizes等）
        res.update(self._data_collator_mm_data(batch))
        
        # 9> 序列并行数据处理（非Megatron场景）
        if not self.use_megatron and self.sequence_parallel_size > 1:
            res = self._sp_data_collator(res, padding_to, self.tokenizer, padding_side)
        
        return res

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        功能：
            汇总批次样本中的视觉模态张量（图像/视频像素值及尺寸），沿批次维拼接为统一输入，
            便于多模态模型直接读取；只返回实际存在的字段。

        参数：
            batch (List[Dict[str, Any]]): encode() 输出的样本列表，元素可能含有 pixel_values、
                image_sizes、pixel_values_videos 等字段。

        返回：
            Dict[str, torch.Tensor]: 拼接后的多模态字典，如 {'pixel_values': tensor(...)}。

        示例：
            >>> batch = [
            ...     {'pixel_values': torch.randn(2, 3, 224, 224)},
            ...     {'pixel_values': torch.randn(1, 3, 224, 224)}
            ... ]
            >>> mm = template._data_collator_mm_data(batch)
            >>> mm['pixel_values'].shape  # torch.Size([3, 3, 224, 224])
        """
        res: Dict[str, torch.Tensor] = {}

        # 1> 图像像素：收集所有非 None 的 pixel_values，并沿批次维拼接
        pixel_values = [b['pixel_values'] for b in batch if b.get('pixel_values') is not None]
        if pixel_values:
            res['pixel_values'] = torch.concat(pixel_values)

            # 2> 图像尺寸：与像素值数量一一对应，常见形状 (num_images, 2)
            image_sizes = [b['image_sizes'] for b in batch if b.get('image_sizes') is not None]
            if image_sizes:
                res['image_sizes'] = torch.concat(image_sizes)

        # 3> 视频像素：concat 后得到连续帧序列
        pixel_values_videos = [b['pixel_values_videos'] for b in batch if b.get('pixel_values_videos') is not None]
        if pixel_values_videos:
            res['pixel_values_videos'] = torch.concat(pixel_values_videos)

        return res

    def _sp_data_collator(self, res, padding_to, tokenizer, padding_side):
        """
        功能：
            序列并行（sequence parallel）场景下，对批次字典 `res` 进行额外整理：
            - 若启用序列并行且存在 input_ids，则确保 position_ids 就绪（左 padding 时需显式生成）。
            - 将函数内部计算出的 input_ids／attention_mask／labels／loss_scale 回写到 `res`。
            该函数通常在 `_data_collator` 末尾调用，用于 Megatron 以外的序列并行配置。

        参数：
            res (Dict[str, Any]): `_data_collator` 已整理出的结果字典，包含张量或 None。
            padding_to (Optional[int]): 目标 padding 长度；此处仅透传，不直接使用（为保持接口一致）。
            tokenizer (PreTrainedTokenizerBase): tokenizer 对象，调试或回退时可用；本函数中未直接调用。
            padding_side (str): 当前使用的 padding 方向，'left' 或 'right'。

        返回：
            Dict[str, Any]: 更新后的结果字典 `res`，包含序列并行所需的 position_ids 以及同步过的键值。

        示例：
            >>> res = {'input_ids': torch.ones(2, 10, dtype=torch.long)}
            >>> res = template._sp_data_collator(res, padding_to=None, tokenizer=tokenizer, padding_side='right')
            >>> res['position_ids'].shape  # torch.Size([2, 10])
        """
        input_ids = res.get('input_ids')
        attention_mask = res.get('attention_mask')
        labels = res.get('labels')
        loss_scale = res.get('loss_scale')
        if self.sequence_parallel_size > 1 and input_ids is not None:
            # 1> 序列并行场景：input_ids 形状通常为 (batch_size, seq_len)
            bs, seq_len = input_ids.shape
            # 若结果中尚未提供 position_ids，则按顺序生成 [0, 1, ..., seq_len-1]
            if 'position_ids' not in res:
                position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)
            else:
                position_ids = res['position_ids']
            # 序列并行要求右 padding（或 batch_size==1）；违背时抛出断言提示
            assert padding_side == 'right' or bs == 1, 'Sequence parallel only support padding_side=right'
            res['position_ids'] = position_ids
        _local_var = locals()
        for key in ['input_ids', 'attention_mask', 'labels', 'loss_scale']:
            value = _local_var[key]
            # 依次回写可能在调用链中被更新的键，避免外部引用旧数据
            if value is not None:
                res[key] = value
        return res

    def print_inputs(self, inputs: Dict[str, Any], tokenizer_kwargs: Optional[Dict[str, Any]] = None) -> None:
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        # Base keys to check
        base_keys = [
            'input', 'labels', 'generate', 'chosen_input', 'chosen_labels', 'rejected_input', 'rejected_labels'
        ]

        # For reranker/embedding modes, also check prefixed keys
        if self.task_type in {'reranker', 'generative_reranker', 'embedding'}:
            prefixes = []
            if self.task_type in {'reranker', 'generative_reranker'}:
                prefixes = ['positive_', 'negative_']
            elif self.task_type == 'embedding':
                prefixes = ['anchor_', 'positive_', 'negative_']

            # Add prefixed keys for reranker/embedding modes
            extended_keys = base_keys.copy()
            for prefix in prefixes:
                for base_key in ['input', 'labels']:
                    extended_keys.append(f'{prefix}{base_key}')

            # Also check for numbered negative keys (negative0_, negative1_, etc.)
            input_keys = list(inputs.keys())
            for key in input_keys:
                if any(key.startswith(f'{prefix}') for prefix in prefixes):
                    # Extract the base key after removing prefix
                    for prefix in prefixes:
                        if key.startswith(prefix):
                            base_key = key[len(prefix):]
                            if base_key in ['input_ids', 'labels'
                                            ] or base_key.rstrip('0123456789_') in ['input', 'labels']:
                                extended_keys.append(key.replace('_ids', ''))
                            break

            keys_to_check = list(set(extended_keys))
        else:
            keys_to_check = base_keys

        for key in keys_to_check:
            # Skip labels completely for certain modes
            if key.endswith('labels') and self.task_type in {'reranker', 'generative_reranker'}:
                continue

            val = inputs.get(key)  # fix val is a tensor
            if val is None:
                val = inputs.get(f'{key}_ids')
            if val is not None:
                key_upper = key.upper()
                logger.info(f'[{key_upper}_IDS] {val}')
                if key.endswith('labels') and self.task_type in {'seq_cls', 'embedding'}:
                    continue
                if isinstance(val, (list, tuple, torch.Tensor)):
                    # Handle nested lists (e.g., for reranker negative samples)
                    if isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], (list, tuple)):
                        val_str = [self.safe_decode(sub_val, **tokenizer_kwargs) for sub_val in val]
                    else:
                        val_str = self.safe_decode(val, **tokenizer_kwargs)
                    logger.info(f'[{key_upper}] {val_str}')
        if inputs.get('loss_scale') is not None:
            val = inputs['loss_scale']
            logger.info(f'[LOSS_SCALE] {val}')

    async def prepare_lmdeploy_pytorch_inputs(self, inputs) -> None:
        """
        功能：
            针对 lmdeploy PyTorch 推理接口，将 input_ids 中的图像占位符 (-100) 替换为实际图像 token，
            并记录每张图像在新序列中的起始偏移，最终把图像信息写入 inputs['multimodal']。

        参数：
            inputs (Dict[str, Any]): 推理输入字典，包含：
                - input_ids (List[int]): 可能带有 -100 占位符的 token 序列；
                - images (List[Dict[str, Any]]): 图像元信息，需包含 image_token_id、image_tokens 等字段。

        返回：
            None: 原地修改 inputs，不返回值。

        示例：
            >>> inputs = {
            ...     'input_ids': [1, -100, 2, -100, 3],
            ...     'images': [
            ...         {'image_token_id': 32000, 'image_tokens': 4},
            ...         {'image_token_id': 32001, 'image_tokens': 2},
            ...     ]
            ... }
            >>> await template.prepare_lmdeploy_pytorch_inputs(inputs)
            >>> inputs['input_ids']  # [1, 32000, 32000, 32000, 32000, 2, 32001, 32001, 3]
            >>> inputs['multimodal'][0]['offset']  # 1，表示第一张图像的 token 起始索引
        """
        images = inputs.pop('images', None) or []  # 取出图像列表，若无图像则返回空列表
        if len(images) == 0:  # 没有图像无需处理
            return

        input_ids = inputs['input_ids']
        idx_list = findall(input_ids, -100)  # 找到所有占位符索引（-100）
        assert len(idx_list) == len(images), f'len(idx_list): {len(idx_list)}, len(images): {len(images)}'
        idx_list.insert(0, -1)  # 在开头插入 -1，方便处理首段 token
        new_input_ids: List[int] = []

        for i in range(len(idx_list) - 1):
            # 复制当前占位符前的原始 token 片段
            new_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]]
            # 记录当前图像在新序列中的起始位置 offset
            images[i]['offset'] = len(new_input_ids)
            # 6> 用图像 token 填充：重复 image_token_id，数量为 image_tokens
            new_input_ids += [images[i]['image_token_id']] * images[i]['image_tokens']

        # 追加最后一个占位符之后的 token
        new_input_ids += input_ids[idx_list[-1] + 1:]

        # 回写更新后的 input_ids 和图像信息
        inputs['input_ids'] = new_input_ids
        inputs['multimodal'] = images

    async def prepare_lmdeploy_turbomind_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        功能：
            为 lmdeploy TurboMind 推理引擎准备多模态输入：将 input_ids 中的图像占位符 (-100) 替换为
            IMAGE_DUMMY_TOKEN_INDEX，并记录每张图像在新序列中的起止区间 [start, end)，同时将图像
            embedding 张量移至 CPU。TurboMind 通过这些区间将图像 token 与文本 token 对齐。

        参数：
            inputs (Dict[str, Any]): 推理输入字典，需包含：
                - input_ids (List[int]): 带 -100 占位符的 token 序列；
                - images (List[torch.Tensor]): 图像特征张量列表，shape 通常为 (num_visual_tokens, hidden_size)。

        返回：
            None: 原地修改 inputs，不返回值。

        示例：
            >>> inputs = {
            ...     'input_ids': [1, -100, 2, -100, 3],
            ...     'images': [torch.randn(8, 4096), torch.randn(4, 4096)]  # 两张图像，8 和 4 个视觉 token
            ... }
            >>> await template.prepare_lmdeploy_turbomind_inputs(inputs)
            >>> inputs['input_ids']  # [1, IMAGE_DUMMY_TOKEN_INDEX*8, 2, IMAGE_DUMMY_TOKEN_INDEX*4, 3]
            >>> inputs['input_embedding_ranges']  # [[1, 9], [10, 14]]，表示图像 token 区间
        """
        images = inputs.pop('images', None) or []  # 取出图像列表，若无则为空
        if len(images) == 0:  # 无图像直接返回
            return
        from lmdeploy.vl.constants import IMAGE_DUMMY_TOKEN_INDEX  # TurboMind 图像占位 token
        input_ids = inputs['input_ids']
        idx_list = findall(input_ids, -100)  # 找出所有 -100 占位符的索引位置
        assert len(idx_list) == len(images), f'len(idx_list): {len(idx_list)}, len(images): {len(images)}'
        idx_list.insert(0, -1)  # 在开头插入 -1，便于处理第一段文本 token
        new_input_ids = []  # 新的 token 序列
        ranges = []  # 记录每张图像的 [start, end) 区间
        
        for i in range(len(idx_list) - 1):
            _range = []  # 当前图像的区间
            # 复制当前占位符前的原始 token 片段
            new_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]]
            # 记录图像 token 起始位置（当前 new_input_ids 的长度）
            _range.append(len(new_input_ids))
            # 用 IMAGE_DUMMY_TOKEN_INDEX 填充视觉 token
            # images[i].shape[0] 是该图像的视觉 token 数量（如 8 个 token）
            # 例如：images[0].shape=(8, 4096) → 填充 8 个 IMAGE_DUMMY_TOKEN_INDEX
            new_input_ids += [IMAGE_DUMMY_TOKEN_INDEX] * images[i].shape[0]
            # 记录图像 token 结束位置（当前 new_input_ids 的长度）
            _range.append(len(new_input_ids))
            ranges.append(_range)  # 保存区间 [start, end)
        
        # 3> 追加最后一个占位符之后的所有 token
        new_input_ids += input_ids[idx_list[-1] + 1:]
        
        # 4> 将图像 embedding 移至 CPU（TurboMind 要求）并回写结果
        inputs['input_embeddings'] = [image.to('cpu') for image in images]
        inputs['input_embedding_ranges'] = ranges  # 每张图像的 token 区间
        inputs['input_ids'] = new_input_ids  # 更新后的 token 序列


    def _pad_sequence(self, sequences: List[torch.Tensor], padding_value: float = 0.) -> torch.Tensor:
        """
        功能：
            对一批张量序列执行统一的 padding，使其对齐到相同长度。训练阶段遵循模板配置的
            `padding_side`（left/right），推理阶段统一使用左 padding。右 padding 直接调用
            `torch.nn.utils.rnn.pad_sequence`，左 padding 通过 `torch.nn.functional.pad` 手动实现。

        参数：
            sequences (List[torch.Tensor]): 待对齐的序列张量列表，常见形状：
                - (seq_len,) —— 纯 token 序列
                - (seq_len, hidden_size) —— embedding 序列
            padding_value (float): padding 使用的数值；文本通常为0，labels常用-100以忽略loss。

        返回：
            torch.Tensor: 对齐后的批量张量，shape = (len(sequences), max_seq_len, ...)。

        示例：
            >>> seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
            >>> Template()._pad_sequence(seqs, padding_value=0)
            tensor([[1, 2, 3],
                    [0, 4, 5]])  # 默认左padding

            >>> seqs = [torch.tensor([[1., 1.], [2., 2.]]), torch.tensor([[3., 3.]])]
            >>> template.padding_side = 'right'
            >>> template._pad_sequence(seqs, padding_value=0.)
            tensor([[[1., 1.], [2., 2.]],
                    [[3., 3.], [0., 0.]]])  # 右padding示例
        """
        padding_side = self.padding_side if self.is_training else 'left'
        padding_right = padding_side == 'right'
        if padding_right:
            # 1> 右 padding：直接调用 pad_sequence，内部自动在序列尾部补齐到最大长度
            return pad_sequence(sequences, batch_first=True, padding_value=padding_value)

        # 2> 左 padding：需手动在序列前补齐到相同长度
        max_len = max(seq.shape[0] for seq in sequences)  # 批次中最长序列长度

        padded_sequences = []
        for seq in sequences:
            pad_length = max_len - seq.shape[0]  # 当前序列需要补齐的时间步数
            # pad_tuple 构造过程：
            #   a) F.pad 需要为每一维提供成对的"左/右补齐"长度，顺序为 (dim_n_left, dim_n_right, ..., dim1_left, dim1_right)。
            #   b) [0] * ((seq.dim() - 1) * 2) 生成除时间维以外所有维度的 padding 配置（全 0 表示不补齐）。
            #   c) 最后拼接 [pad_length, 0]，只在时间维左侧补 pad_length 个 padding_value，右侧补 0。
            #   d) 例1：seq.shape=(3,) 且 pad_length=2 ⇒ pad_tuple=[2, 0] ⇒ F.pad 后得到 [0, 0, x1, x2, x3]。
            #      例2：seq.shape=(3, hidden) ⇒ pad_tuple=[0, 0, pad_length, 0] ⇒ 隐藏维保持原状，时间维左补 pad_length 行 0，输出 shape=(3+pad_length, hidden)。
            pad_tuple = [0] * ((seq.dim() - 1) * 2) + [pad_length, 0]
            padded_seq = F.pad(seq, tuple(pad_tuple), 'constant', padding_value)  # 应用左 padding，shape 变为 (max_len, ...)
            padded_sequences.append(padded_seq)

        # 3> 堆叠为批量张量，最终 shape=(batch_size, max_len, ...)
        return torch.stack(padded_sequences)

    def safe_decode(self, input_ids: List[int], **tokenizer_kwargs) -> str:
        """
        功能：
            安全解码 token 序列为可读文本，跳过特殊 token（如图像占位符 -100、多模态占位 token）。
            将连续的普通 token 解码为文本，连续的特殊 token 表示为 `[token_id * count]` 格式。
            用于调试和日志输出，避免 tokenizer 对无效 token 解码失败。

        参数：
            input_ids (List[int]): 待解码的 token ID 序列，可能包含特殊 token（负数、占位符等）。
            **tokenizer_kwargs: 传递给 tokenizer.decode 的额外参数（如 skip_special_tokens）。

        返回：
            str: 解码后的字符串，特殊 token 用 `[token_id * count]` 表示。

        示例：
            >>> input_ids = [1, 2, -100, -100, 3, 4]
            >>> template.safe_decode(input_ids)
            'Hello[-100 * 2] world'  # 假设 [1,2]→'Hello', [3,4]→' world'
        """
        # 1> 兼容性处理：支持直接在 tokenizer 上调用（用于工具方法）
        if isinstance(self, Template):
            tokenizer = self.tokenizer
            placeholder_tokens = self.placeholder_tokens  # 多模态占位 token 列表
        else:
            tokenizer = self
            placeholder_tokens = []

        # 内部函数：判断是否为特殊 token
        def _is_special(token: int) -> bool:
            if isinstance(token, float) or token < 0:  # 负数或浮点数视为特殊
                return True
            return token in placeholder_tokens  # 多模态占位符

        # 2> 输入预处理
        if isinstance(input_ids, torch.Tensor):  # 支持 tensor 输入
            input_ids = input_ids.tolist()
        if len(input_ids) == 0:  # 空序列直接返回空字符串
            return ''
        
        # 3> 状态追踪变量初始化
        result_str = ''  # 累积结果字符串
        # s: 特殊 token 连续段的起始索引
        # e: 普通 token 连续段的起始索引
        
        # 4> 遍历 token 序列，按普通/特殊分段解码
        for i in range(len(input_ids)):
            if i == 0:  # 初始化：根据首个 token 类型设置起始索引
                if _is_special(input_ids[i]):
                    s = 0  # 首个是特殊 token，记录特殊段起点
                else:
                    e = 0  # 首个是普通 token，记录普通段起点
                continue
            
            # 检测从普通→特殊的边界：解码之前累积的普通 token
            if _is_special(input_ids[i]) and not _is_special(input_ids[i - 1]):
                s = i  # 记录特殊段起点
                result_str += tokenizer.decode(input_ids[e:s], **tokenizer_kwargs)  # 解码 [e, s)
            
            # 检测从特殊→普通的边界：输出之前累积的特殊 token 统计
            if not _is_special(input_ids[i]) and _is_special(input_ids[i - 1]):
                e = i  # 记录普通段起点
                result_str += f'[{input_ids[i - 1]} * {e - s}]'  # 格式：[token_id * 数量]

        # 5> 处理序列末尾：根据最后一个 token 类型收尾
        if _is_special(input_ids[i]):  # 末尾是特殊 token
            result_str += f'[{input_ids[i]} * {len(input_ids) - s}]'
        else:  # 末尾是普通 token
            result_str += tokenizer.decode(input_ids[e:], **tokenizer_kwargs)
        
        return result_str

    @staticmethod
    @contextmanager
    def _patch_flash_attention_forward(modeling_module, position_ids, use_new_func: bool = False):
        """
        功能：
            临时 patch（猴子补丁）模型的 `_flash_attention_forward` 方法，注入自定义 position_ids。
            作为上下文管理器，进入时替换方法，退出时恢复原方法，确保不污染全局状态。
            用于在推理或特殊训练场景下，向 Flash Attention 传递自定义位置编码。

        参数：
            modeling_module: 模型模块对象，需包含 `_flash_attention_forward` 方法（如 transformers 模型）。
            position_ids: 自定义的位置编码张量，将注入到 Flash Attention 调用中。
            use_new_func (bool): 是否使用 transformers 新版 Flash Attention 实现（默认 False 使用原方法）。

        返回：
            生成器（上下文管理器）：进入时 patch 生效，退出时自动恢复。

        示例：
            >>> with Template._patch_flash_attention_forward(model, position_ids):
            ...     outputs = model(**inputs)  # 内部使用自定义 position_ids
            >>> # 退出上下文后，model._flash_attention_forward 已恢复
        """
        # 1> 保存原始方法引用，用于退出时恢复
        _origin_flash_attention_forward = modeling_module._flash_attention_forward

        # 2> 定义替换方法：包装原方法并注入 position_ids
        def _flash_attention_forward(*args, **kwargs):
            if use_new_func:  # 使用 transformers 新版实现
                from transformers.modeling_flash_attention_utils import (_flash_attention_forward as
                                                                         flash_attention_forward)
                # 兼容性处理：部分调用会传 self，需移除
                if args and isinstance(args[0], nn.Module):
                    args = args[1:]
                # 设置默认 causal mask 行为
                if 'is_causal' not in kwargs:
                    kwargs['is_causal'] = True
            else:  # 使用原方法
                flash_attention_forward = _origin_flash_attention_forward
            
            # 注入自定义 position_ids 到参数中
            kwargs['position_ids'] = position_ids
            return flash_attention_forward(*args, **kwargs)

        # 3> 替换模型方法为包装后的版本
        modeling_module._flash_attention_forward = _flash_attention_forward
        try:
            yield  # 上下文管理器：暂停执行，返回控制权给 with 块
        finally:
            # 4> 退出时恢复原方法，确保不影响后续调用
            modeling_module._flash_attention_forward = _origin_flash_attention_forward
