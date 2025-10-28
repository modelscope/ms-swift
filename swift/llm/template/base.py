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
        # NOTE: repr(obj) 是 Python 的内置函数，用于获取对象的“官方字符串表示形式”，生成一个尽可能准确、可用于调试的字符串
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
        problem_type = config.problem_type
        if problem_type is not None:
            return problem_type
        if labels is not None:
            if isinstance(labels, (list, tuple)):
                if labels and isinstance(labels[0], float):
                    problem_type = 'regression'
                else:
                    problem_type = 'multi_label_classification'
            else:
                problem_type = 'single_label_classification'
                assert config.num_labels >= labels + 1
        if logits is not None:
            if logits.shape[-1] == 1:
                problem_type = 'regression'
            else:
                problem_type = 'single_label_classification'  # compatible with older versions
        assert problem_type is not None
        config.problem_type = problem_type
        return problem_type

    def decode_seq_cls(self, logits: torch.Tensor, top_logprobs: int):
        assert isinstance(logits, torch.Tensor)
        problem_type = self._get_problem_type(self.config, logits=logits)
        if problem_type == 'regression':
            preds = logits.squeeze(dim=-1).tolist()
            logprobs = [None] * len(preds)
        else:
            if problem_type == 'single_label_classification':
                preds = torch.argmax(logits, dim=-1).tolist()
                logprobs = torch.log_softmax(logits, -1)
            else:
                preds = [(logprob >= 0.5).nonzero(as_tuple=True)[0].tolist() for logprob in torch.sigmoid(logits)]
                logprobs = F.logsigmoid(logits)
            logprobs = [self._get_seq_cls_logprobs(pred, logprobs[i], top_logprobs) for i, pred in enumerate(preds)]
        return preds, logprobs

    def decode(self,
               generate_ids: List[int],  # 生成的token ID序列：模型生成的完整token列表
               *,  # 以下参数必须使用关键字传递
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
        origin_mode = self.mode
        if self.mode in {'train', 'rlhf', 'kto', 'gkd'}:
            self.set_mode('pt')
        is_multimodal = self.model_meta.is_multimodal
        if is_multimodal:
            models = self.remove_post_encode_hook()
        try:
            yield
        finally:
            if is_multimodal:
                self.register_post_encode_hook(models)
            self.set_mode(origin_mode)

    def generate(self, model, *args, **kwargs):
        base_model = self.get_base_model(model)
        signature = inspect.signature(base_model.generate)
        if 'use_model_defaults' in signature.parameters and 'use_model_defaults' not in kwargs:
            kwargs['use_model_defaults'] = False
        return model.generate(*args, **kwargs)

    def skip_stop_tokens(self, generate_ids: List[int], is_finished: bool = True) -> List[int]:
        # Do not print template_meta.suffix[-1] and eos_token.
        # However, other stop_words will be printed.
        tokenizer = self.tokenizer

        if len(generate_ids) > 0 and generate_ids[-1] == tokenizer.eos_token_id:
            generate_ids = generate_ids[:-1]
        # skip suffix and eos_token
        template_suffix = self.template_meta.suffix[-1]
        if isinstance(template_suffix, str):
            # [-1:]: fix OpenGVLab/Mini-InternVL-Chat-4B-V1-5
            template_suffix = tokenizer.encode(template_suffix, add_special_tokens=False)[-1:]

        len_tokens = len(template_suffix)
        if is_finished and generate_ids[-len_tokens:] == template_suffix:
            generate_ids = generate_ids[:-len_tokens]
        elif not is_finished:
            for i in range(len_tokens, 0, -1):
                if generate_ids[-i:] == template_suffix[:i]:
                    generate_ids = generate_ids[:-i]
                    break
        return generate_ids

    def prepare_generate_kwargs(self, generate_kwargs: Dict[str, Any], *, model=None) -> Dict[str, Any]:
        generation_config = generate_kwargs['generation_config']
        stop_words = getattr(generation_config, 'stop_words', None) or self.template_meta.stop_words
        generate_kwargs['stopping_criteria'] = StoppingCriteriaList([StopWordsCriteria(self.tokenizer, stop_words)])
        return generate_kwargs

    @staticmethod
    def _save_pil_image(image: Image.Image) -> str:
        img_bytes = image.tobytes()
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        tmp_dir = os.path.join(get_cache_dir(), 'tmp', 'images')
        logger.info_once(f'create tmp_dir: {tmp_dir}')
        os.makedirs(tmp_dir, exist_ok=True)
        img_path = os.path.join(tmp_dir, f'{img_hash}.png')
        if not os.path.exists(img_path):
            image.save(img_path)
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
        """Concat context list and replace placeholder"""
        round1 = None
        if round0 is not None:
            round1 = str(round0 + 1)
            round0 = str(round0)
        for context in context_list:
            if isinstance(context, str):
                if '{{RESPONSE}}' == context:
                    assert response is not None
                    res_context_list.append(response)
                    res_context_type.append(ContextType.RESPONSE)
                    continue
                old_str_list = ['{{SYSTEM}}', '{{QUERY}}', '{{ROUND0}}', '{{ROUND1}}']
                new_str_list = [system, query, round0, round1]
                for (old_str, new_str) in zip(old_str_list, new_str_list):
                    if new_str is not None and old_str in context:
                        assert isinstance(new_str, str), f'new_str: {new_str}'
                        context = context.replace(old_str, new_str)
            if len(context) == 0:
                continue
            res_context_list.append(context)
            res_context_type.append(ContextType.OTHER)

    def _simplify_context_list(self, context_list: List[Context], loss_scale_list: List[float],
                               inputs: StdTemplateInputs) -> Tuple[List[Context], List[float]]:
        """Merge anything in the context to simplify the inputs"""
        context_list, loss_scale_list = self._split_special_tokens(context_list, loss_scale_list)
        context_list, loss_scale_list = self._pre_tokenize(context_list, loss_scale_list, inputs)

        res: List[Context] = []  # result of context_list
        res_loss_scale: List[float] = []  # result of loss_scale_list
        temp: List[str] = []
        temp_loss_scale = 0.
        for i, (context, loss_scale) in enumerate(zip(context_list, loss_scale_list)):
            if isinstance(context, str) and (loss_scale == temp_loss_scale):
                temp.append(context)
            else:
                if len(temp) > 0:
                    res.append(''.join(temp))
                    res_loss_scale.append(temp_loss_scale)
                    temp.clear()
                if isinstance(context, str):  # loss_scale diff
                    temp.append(context)
                else:
                    res.append(context)
                    res_loss_scale.append(loss_scale)
                temp_loss_scale = loss_scale
        if len(temp) > 0:
            res.append(''.join(temp))
            res_loss_scale.append(temp_loss_scale)

        return res, res_loss_scale

    @staticmethod
    def _split_special_tokens(context_list: List[Context],
                              loss_scale_list: List[float]) -> Tuple[List[Context], List[float]]:
        """Split special tokens, for example `<image>`, `<video>`, this will help the replace_tag operation"""
        res: List[Context] = []
        loss_scale_res: List[float] = []
        for context, loss_scale in zip(context_list, loss_scale_list):
            contexts = []
            if isinstance(fetch_one(context), str):
                for d in split_str_parts_by(context, Template.special_tokens):
                    contexts.extend([d['key'], d['content']])
                contexts = [c for c in contexts if c]
                res.extend(contexts)
                loss_scale_res.extend([loss_scale] * len(contexts))
            else:
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
        # https://github.com/modelscope/ms-swift/issues/3407
        # Fix the bounding box position offset issue in the Qwen2.5-VL grounding task.
        res: List[Context] = []
        res_loss_scale: List[float] = []
        inputs.image_idx = 0

        for context, loss_scale in zip(context_list, loss_scale_list):
            if context == '<image>' and inputs.is_multimodal and inputs.image_idx < len(inputs.images):
                c_list = self.replace_tag('image', inputs.image_idx, inputs)
                inputs.image_idx += 1
                loss_scale = 0. if self.template_backend == 'swift' else 1.
            else:
                c_list = [context]
            res += c_list
            res_loss_scale += [loss_scale] * len(c_list)
        return res, res_loss_scale

    def _pre_tokenize(self, context_list: List[Context], loss_scale_list: List[float],
                      inputs: StdTemplateInputs) -> Tuple[List[Context], List[float]]:
        """This method happens before tokenization, replace standard tags to the contents or input_ids needed by
        the model.

        Args:
            context_list: The content list
            loss_scale_list: The loss scale list
        Returns:
            The context_list and loss_scale_list after replacement.
        """
        context_list, loss_scale_list = self._pre_tokenize_images(context_list, loss_scale_list, inputs)
        if inputs.images and inputs.objects:
            self.normalize_bbox(inputs)
        # replace tag/object/box
        res: List[Context] = []  # result of context_list
        res_loss_scale: List[float] = []  # result of loss_scale_list

        # reset
        for k in ['video', 'audio', 'object', 'box']:
            setattr(inputs, f'{k}_idx', 0)

        for context, loss_scale in zip(context_list, loss_scale_list):
            for k in ['video', 'audio']:
                if context == f'<{k}>' and inputs.is_multimodal and getattr(inputs, f'{k}_idx') < len(
                        getattr(inputs, f'{k}s')):
                    c_list = self.replace_tag(k, getattr(inputs, f'{k}_idx'), inputs)
                    setattr(inputs, f'{k}_idx', getattr(inputs, f'{k}_idx') + 1)
                    loss_scale = 0.
                    break
            else:
                ref = inputs.objects.get('ref') or []
                bbox = inputs.objects.get('bbox') or []
                if context == '<ref-object>' and inputs.ref_idx < len(ref):
                    idx = inputs.ref_idx
                    c_list = self.replace_ref(ref[idx], idx, inputs)
                    inputs.ref_idx += 1
                elif context == '<bbox>' and inputs.bbox_idx < len(bbox):
                    idx = inputs.bbox_idx
                    c_list = self.replace_bbox(bbox[idx], idx, inputs)
                    inputs.bbox_idx += 1
                elif context == '<cot-process>' and self.task_type == 'prm':
                    c_list = self.replace_cot_process(inputs)
                else:
                    c_list = [context]
            res += c_list
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
            token转换到-100的边界）识别并激活suffix token（如<|im_end|>、</s>等）的loss计算。
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
                例如：[151643]（<|im_end|>）或 [2]（</s>）
        
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
            
            >>> # 示例3：多token suffix（如"\n</s>"）
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
        messages = inputs.messages.copy()
        if inputs.system is not None:
            messages.insert(0, {'role': 'system', 'content': inputs.system})
        if messages[-1]['content'] is None:
            messages.pop()
        add_generation_prompt = messages[-1]['role'] != 'assistant'
        kwargs = {}
        if inputs.tools:
            kwargs['tools'] = inputs.tools
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs)
        answer_len = 1 if self.is_training else 0
        return [text], [1.], answer_len

    def _get_system(self, inputs) -> Optional[str]:
        template_meta = self.template_meta
        system = inputs.system
        tools = inputs.tools
        template_meta.check_system(system)
        if system is None:
            system = template_meta.default_system

        if tools is not None:
            system = self.agent_template._format_tools(tools, system or '', inputs.messages[0])
        return system

    def _swift_prepare_inputs(self, inputs):
        messages = inputs.messages
        if len(messages) < 2:
            return
        i = 1
        while i < len(messages):
            pre_message, message = messages[i - 1], messages[i]
            pre_role, pre_content = pre_message['role'], pre_message['content']
            role, content = message['role'], message['content']
            if pre_role == 'assistant' and role == 'tool':
                i_start = i
                while i + 1 < len(messages) and messages[i + 1]['role'] == 'tool':
                    i += 1
                pre_message['content'], tool_content = self.agent_template._format_tool_responses(
                    pre_content, messages[i_start:i + 1])
                messages[i_start:i + 1] = [{'role': 'tool', 'content': tool_content}]
                i = i_start + 1
            elif pre_role == 'assistant' and role == 'assistant' or pre_role == 'user' and role == 'user':
                # Consecutive messages from the assistant/user role need to be merged to prevent errors.
                pre_message['content'] = pre_content + content
                messages.pop(i)
            else:
                i += 1

    def _swift_encode(self, inputs: StdTemplateInputs):
        template_meta = self.template_meta
        self._swift_prepare_inputs(inputs)
        system = self._get_system(inputs)

        self._get_std_messages(inputs.messages)
        n_round = len(inputs.messages) // 2
        if n_round > 1 and not self.template_meta.support_multi_round:
            logger.warning_once(
                'The template does not support multi-round chat. Only use the last round of the conversation.')
            inputs.messages = inputs.messages[-2:]

        res_context_list: List[Context] = []
        res_context_types: List[ContextType] = []
        sep_token = None
        if template_meta.auto_add_bos:
            all_tokens = self.tokenizer.encode('a')
            single_token = self.tokenizer.encode('a', add_special_tokens=False)
            assert len(single_token) == 1
            idx = all_tokens.index(single_token[0])
            bos_token = all_tokens[:idx]
            sep_token = all_tokens[idx + 1:]
            if bos_token:
                res_context_list.append(bos_token)
                res_context_types.append(ContextType.OTHER)

        if self.template_meta.is_post_system or not system:
            prefix = template_meta.prefix
        else:
            prefix = template_meta.system_prefix
        self._concat_context_list(prefix, res_context_list, res_context_types, system=system)

        n_round = len(inputs.messages) // 2
        for i, (query_message, response_message) in enumerate(zip(inputs.messages[::2], inputs.messages[1::2])):
            query_role, query = query_message['role'], query_message['content']
            response_role, response = response_message['role'], response_message['content']
            # TODO: Optimize the Template mechanism.
            assert query_role in {'user', 'tool'}, f'query_role: {query_role}'
            assert response_role in {'assistant'}, f'response_role: {response_role}'
            if query_role == 'tool':
                prompt = query
                query = ''
            elif template_meta.is_post_system and i == n_round - 1:
                prompt = template_meta.system_prompt
            else:
                prompt = template_meta.prompt

            context_list = prompt.copy()
            extra_context_list = []
            extra_context_type = None
            if i < n_round - 1:
                # Not the last round.
                context_list.append('{{RESPONSE}}')
                if inputs.messages[2 * (i + 1)]['role'] != 'tool':
                    extra_context_list = template_meta.chat_sep
                    extra_context_type = ContextType.OTHER
            elif response is not None:
                # It is the final round, and the response exists (during training).
                context_list.append('{{RESPONSE}}')
                # self.is_training needed because we may want to continue generation from
                # the current response
                if self.is_training and not sep_token or self.task_type == 'embedding':
                    extra_context_list = template_meta.suffix
                    extra_context_type = ContextType.SUFFIX
            elif template_meta.response_prefix:
                # final round and during inference.
                context_list.append(template_meta.response_prefix)

            self._concat_context_list(
                context_list,
                res_context_list,
                res_context_types,
                query=query,
                response=response,
                system=system,
                round0=i)
            res_context_list += extra_context_list
            res_context_types += [extra_context_type] * len(extra_context_list)
        if template_meta.auto_add_bos and sep_token:
            res_context_list.append(sep_token)
            res_context_types.append(ContextType.SUFFIX)
        res_context_list, loss_scale_list = self.loss_scale(res_context_list, res_context_types, inputs.messages)
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
        
        # ===== 步骤10：返回编码结果 =====
        return encoded

    def _handle_megatron_cp(self, encoded: Dict[str, Any]) -> None:
        """
        函数功能：
            处理Megatron-LM的上下文并行（Context Parallelism，简称CP）所需的序列padding。在使用
            Megatron-LM进行序列并行训练时，序列长度必须是 `cp_size * 2` 的整数倍，以便将序列均匀
            分割到多个设备上。该方法通过在序列末尾添加padding token，确保序列长度满足这一要求，
            同时更新对应的labels和loss_scale，使padding部分不参与loss计算。
        
        参数：
            encoded (Dict[str, Any]): 编码后的数据字典，包含以下字段（会被原地修改）：
                - input_ids (List[int]): 输入token序列
                - labels (List[int]): 标签序列
                - loss_scale (Optional[List[float]]): loss权重序列（如有）
        
        返回值：
            None: 该方法直接修改传入的encoded字典（原地修改），不返回任何值。
                  修改后，input_ids、labels和loss_scale（如有）的长度都会变为 `cp_size * 2` 的整数倍，
                  padding部分使用pad_token_id填充input_ids，-100填充labels，0填充loss_scale。
        
        使用示例：
            >>> # 示例1：基础场景，序列长度不满足要求
            >>> template = Template(...)
            >>> template.use_megatron = True
            >>> template.sequence_parallel_size = 4  # cp_size = 4，要求长度为8的倍数
            >>> encoded = {
            ...     'input_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 长度10
            ...     'labels': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ...     'loss_scale': None
            ... }
            >>> template._handle_megatron_cp(encoded)
            >>> print(len(encoded['input_ids']))  # 16（向上取整到8的倍数：16）
            >>> print(encoded['input_ids'][-6:])  # [10, pad_id, pad_id, pad_id, pad_id, pad_id]
            >>> print(encoded['labels'][-6:])  # [10, -100, -100, -100, -100, -100]
            
            >>> # 示例2：序列长度已经满足要求（不需要padding）
            >>> encoded = {
            ...     'input_ids': [1, 2, 3, 4, 5, 6, 7, 8],  # 长度8，已经是8的倍数
            ...     'labels': [1, 2, 3, 4, 5, 6, 7, 8],
            ... }
            >>> template._handle_megatron_cp(encoded)
            >>> print(len(encoded['input_ids']))  # 8（不变，无需padding）
            
            >>> # 示例3：带loss_scale的场景
            >>> encoded = {
            ...     'input_ids': [10, 20, 30, 40, 50],  # 长度5
            ...     'labels': [10, 20, 30, 40, 50],
            ...     'loss_scale': [1.0, 1.0, 0.8, 0.8, 1.0]
            ... }
            >>> template.sequence_parallel_size = 2  # cp_size = 2，要求长度为4的倍数
            >>> template._handle_megatron_cp(encoded)
            >>> print(len(encoded['input_ids']))  # 8（向上取整到4的倍数：8）
            >>> print(encoded['loss_scale'])
            # [1.0, 1.0, 0.8, 0.8, 1.0, 0, 0, 0]  # padding部分添加了3个0
            
            >>> # 示例4：未启用Megatron或cp_size=1（直接返回，不处理）
            >>> template.use_megatron = False  # 未启用Megatron
            >>> encoded = {'input_ids': [1, 2, 3], 'labels': [1, 2, 3]}
            >>> template._handle_megatron_cp(encoded)
            >>> print(len(encoded['input_ids']))  # 3（不变，直接返回）
        """
        # ===== 步骤1：获取上下文并行大小 =====
        cp_size = self.sequence_parallel_size  # 获取序列并行的分片数量
        
        # ===== 步骤2：检查是否需要处理 =====
        if not self.use_megatron or cp_size == 1:  # 若未启用Megatron或不使用序列并行
            return  # 直接返回，无需处理（cp_size=1时序列长度无特殊要求）
        
        # ===== 步骤3：计算需要padding的长度 =====
        input_ids = encoded['input_ids']  # 获取输入token序列
        # 计算padding长度，使序列长度成为 cp_size * 2 的整数倍
        # 公式：向上取整到 (cp_size * 2) 的倍数，然后减去当前长度
        # 例如：len=10, cp_size=4 -> (cp_size*2)=8 -> ceil(10/8)*8=16 -> padding_len=16-10=6
        padding_len = math.ceil(len(input_ids) / (cp_size * 2)) * (cp_size * 2) - len(input_ids)
        
        # ===== 步骤4：为input_ids添加padding =====
        # 在input_ids末尾添加 padding_len 个 pad_token_id
        # pad_token_id通常是特殊的padding token（如0或其他tokenizer指定的值）
        input_ids += [self.tokenizer.pad_token_id] * padding_len
        
        # ===== 步骤5：为labels添加padding =====
        # 在labels末尾添加 padding_len 个 -100
        # -100是PyTorch中ignore_index的默认值，表示这些位置不参与loss计算
        encoded['labels'] += [-100] * padding_len
        
        # ===== 步骤6：为loss_scale添加padding（如果存在） =====
        if encoded.get('loss_scale') is not None:  # 若loss_scale字段存在且不为None
            # 在loss_scale末尾添加 padding_len 个 0
            # 0表示这些padding位置的loss权重为0（不参与loss计算）
            encoded['loss_scale'] += [0] * padding_len

    def debug_logger(self, inputs):
        if not strtobool(os.getenv('SWIFT_DEBUG', 'false')):
            return
        if 'input_ids' in inputs:
            k = 'input_ids'
            val = inputs['input_ids']
        elif 'generate_ids' in inputs:
            k = 'generate_ids'
            val = inputs['generate_ids']
        for v in val:
            self.print_inputs({k: v.tolist()})

    @staticmethod
    def _split_list(inputs: List[int], x: int) -> List[List[int]]:
        idxs = findall(inputs, x)
        idxs.append(len(inputs))
        res = []
        lo = 0
        for idx in idxs:
            res.append(inputs[lo:idx])
            lo = idx + 1
        return res

    def replace_video2image(self, load_video_func, inputs, replace_tag: Callable) -> List[Context]:
        context_list = []
        if self.mode in {'vllm', 'lmdeploy'}:
            video = inputs.videos.pop(inputs.video_idx)
            inputs.video_idx -= 1
        else:
            video = inputs.videos[inputs.video_idx]
        images = inputs.images
        new_images = load_video_func(video)
        inputs.images = images[:inputs.image_idx] + new_images + images[inputs.image_idx:]
        for i in range(len(new_images)):
            context_list += replace_tag(i)
        inputs.image_idx += len(new_images)
        return context_list

    def get_generate_ids(self, generate_ids: Union[torch.Tensor, List[int]],
                         num_prompt_tokens: int) -> Union[torch.Tensor, List[int]]:
        if self.skip_prompt:
            generate_ids = generate_ids[..., num_prompt_tokens:]
        return generate_ids

    def post_process_generate_response(self, response: str, inputs: StdTemplateInputs) -> str:
        return response

    def pre_forward_hook(self, model: nn.Module, args, kwargs):
        from swift.llm import to_device
        old_kwargs = to_device(kwargs, model.device)
        kwargs = to_device(self._post_encode(model, old_kwargs), model.device)
        for k, v in old_kwargs.items():
            if k in {
                    'input_ids', 'attention_mask', 'labels', 'position_ids', 'output_hidden_states', 'logits_to_keep',
                    'cumulative_seqlens_q', 'cumulative_seqlens_k', 'max_length_q', 'max_length_k'
            } and k not in kwargs:
                kwargs[k] = v
        if 'inputs_embeds' in kwargs:
            kwargs.pop('input_ids', None)

        base_model = self.get_base_model(model)
        parameters = inspect.signature(base_model.forward).parameters
        if 'position_ids' not in parameters:
            kwargs.pop('position_ids', None)
        return args, kwargs

    @property
    def is_training(self):
        return self.mode not in {'pt', 'vllm', 'lmdeploy', 'sglang'}

    def set_mode(self, mode: Literal['pt', 'vllm', 'lmdeploy', 'sglang', 'train', 'rlhf', 'kto', 'gkd']) -> None:
        self.mode = mode

    def register_post_encode_hook(self, models: List[nn.Module]) -> None:
        """This function is important for multi-modal training, as it registers the post_encode method
            as a forward hook, converting input_ids into inputs_embeds.
        """
        if self._handles:
            return

        for model in models:
            # please use torch>=2.0
            handle = model.register_forward_pre_hook(self.pre_forward_hook, with_kwargs=True)
            self._handles.append((model, handle))

        if is_deepspeed_zero3_enabled():
            import deepspeed
            self._deepspeed_initialize = deepspeed.initialize

            @wraps(self._deepspeed_initialize)
            def _initialize(*args, **kwargs):
                res = self._deepspeed_initialize(*args, **kwargs)
                for model, handle in self._handles:
                    model._forward_pre_hooks.move_to_end(handle.id)
                return res

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

    def data_collator(self, batch: List[Dict[str, Any]],  # 批量编码数据：每个元素是encode()返回的字典
                       *, padding_to: Optional[int] = None) -> Dict[str, Any]:  # 可选的padding目标长度
        """函数功能：
        将批量的编码数据整理（collate）为模型可直接使用的批次张量。这是训练时的核心方法，
        负责对变长序列进行padding对齐、生成attention mask、拼接多模态数据等操作。

        主要职责：
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
        
        # ===== 根据task_type和mode选择相应的data collator子方法 =====
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
        
        # ===== 处理额外参数（如果不移除未使用列） =====
        if not self.remove_unused_columns:  # 若配置为保留额外参数
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
        # List[Tensor] ->  List[Tensor]
        res = []
        for b in batch:
            if b.get(attr_name) is not None:
                res += b.pop(attr_name)
        return res

    @staticmethod
    def concat_tensor(batch: List[Dict[str, Any]], attr_name: str, dim: int) -> Optional[torch.Tensor]:
        res = []
        for b in batch:
            if b.get(attr_name) is not None:
                res.append(b.pop(attr_name))
        return torch.concat(res, dim=dim) if res else None

    def _rlhf_data_collator(self,
                            batch: List[Dict[str, Any]],
                            *,
                            chosen_prefix: str = 'chosen_',
                            rejected_prefix: str = 'rejected_',
                            padding_to: Optional[int] = None) -> Dict[str, Any]:
        new_batch = []
        for prefix in [chosen_prefix, rejected_prefix]:
            new_batch += self._fetch_inputs_startswith(batch, prefix)
        res = self._data_collator(new_batch, padding_to=padding_to)

        # reward modeling
        margin = [b['margin'] for b in batch if b.get('margin') is not None]
        if margin:
            res['margin'] = torch.tensor(margin, dtype=torch.float)

        return res

    def _kto_data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        new_batch = self._fetch_inputs_startswith(batch, 'chosen_')
        kl_batch = self._fetch_inputs_startswith(batch, 'rejected_')

        res = self._data_collator(new_batch, padding_to=padding_to)
        kl_res = self._data_collator(kl_batch, padding_to=padding_to)
        res = {
            **{f'completion_{k}': v
               for k, v in res.items()},
            **{f'KL_completion_{k}': v
               for k, v in kl_res.items()},
        }
        label = [b['label'] for b in batch if b.get('label') is not None]
        if label:
            res['label'] = label
        return res

    def _gkd_data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = self._data_collator(batch, padding_to=padding_to)
        prompts_batch = [{'input_ids': b['prompts']} for b in batch if b.get('prompts') is not None]
        if prompts_batch:
            prompts_res = self._data_collator(prompts_batch, padding_to=padding_to)
            res['prompts'] = prompts_res.pop('input_ids')
            res.update({f'prompt_{k}': v for k, v in prompts_res.items()})
        return res

    def _embedding_data_collator(self,
                                 batch: List[Dict[str, Any]],
                                 *,
                                 padding_to: Optional[int] = None) -> Dict[str, Any]:
        labels = []
        new_batch = []
        for b in batch:
            if 'input_ids' in b:
                new_batch += [b]
            else:
                keys = [key for key in b.keys() if 'negative' in key]
                max_neg = None
                for key in keys:
                    value_list = b[key]
                    suffix = key[len('negative_'):]
                    max_neg = len(value_list)
                    for i, value in enumerate(value_list):
                        b[f'negative{i}_{suffix}'] = value
                    b.pop(key)

                indexes = ['anchor_', 'positive_']
                if max_neg is not None:
                    for i in range(0, max_neg):
                        indexes.append(f'negative{i}_')
                for prefix in indexes:
                    new_batch += self._fetch_inputs_startswith([b], prefix)
            labels.extend(b.get('labels', []))
        res = self._data_collator(new_batch, padding_to=padding_to)
        if labels:
            res['labels'] = torch.tensor(labels, dtype=torch.float32)
        return res

    def _reranker_data_collator(self,
                                batch: List[Dict[str, Any]],
                                *,
                                padding_to: Optional[int] = None) -> Dict[str, Any]:
        import os
        max_negative_samples = int(os.environ.get('MAX_NEGATIVE_SAMPLES', 7))
        labels = []
        new_batch = []
        for b in batch:
            keys = [key for key in b.keys() if 'negative' in key]
            max_neg = None
            for key in keys:
                value_list = b[key]
                suffix = key[len('negative_'):]
                max_neg = min(max_negative_samples, len(value_list))
                for i, value in enumerate(value_list):
                    b[f'negative{i}_{suffix}'] = value
                b.pop(key)

            indexes = ['positive_']
            if max_neg is not None:
                for i in range(0, max_neg):
                    indexes.append(f'negative{i}_')
            for prefix in indexes:
                new_batch += self._fetch_inputs_startswith([b], prefix)
            labels.extend(b.get('labels', None)[:max_negative_samples + 1])
        res = self._data_collator(new_batch, padding_to=padding_to)
        if labels:
            res['labels'] = torch.tensor(labels, dtype=torch.long)
        return res

    def _seq_cls_data_collator(self,
                               batch: List[Dict[str, Any]],
                               *,
                               padding_to: Optional[int] = None) -> Dict[str, Any]:
        labels = [b.pop('labels') for b in batch if b.get('labels') is not None]
        res = self._data_collator(batch, padding_to=padding_to)
        if labels:
            problem_type = self._get_problem_type(self.config)
            if problem_type == 'regression':
                labels = torch.tensor(labels, dtype=torch.float32)
            elif problem_type == 'multi_label_classification':
                one_hot_labels = torch.zeros((len(labels), self.config.num_labels), dtype=torch.float32)
                for i, label in enumerate(labels):
                    one_hot_labels[i, label] = 1
                labels = one_hot_labels
            else:
                labels = torch.tensor(labels, dtype=torch.long)
            res['labels'] = labels
        return res

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """
        Args:
            batch(`List[Dict[str, Any]]`): The input data in batch
            padding_to(`int`, optional): Whether padding the batch to a fixed length, if none, the batch
                will be padded to the `longest`
        """
        assert self.tokenizer.pad_token_id is not None
        padding_side = self.padding_side if self.is_training else 'left'
        padding_right = padding_side == 'right'
        if self.padding_free:
            batch[:] = [self.packing_row(batch)]
        elif self.use_megatron:
            for encoded in batch:
                encoded['position_ids'] = list(range(len(encoded['labels'])))
        if self._packing:
            assert 'position_ids' in batch[0], f'batch[0]: {batch[0]}'
        res = {}
        if self._packing:
            # only support llm
            for k in ['input_ids', 'labels', 'position_ids', 'loss_scale', 'channel']:
                v = self.gather_list(batch, k)
                if v:
                    if k == 'channel':
                        res[k] = v
                    else:
                        res[k] = [v]
        else:
            inputs_embeds = [b['inputs_embeds'] for b in batch if b.get('inputs_embeds') is not None]
            input_ids = [b['input_ids'] for b in batch if b.get('input_ids') is not None]
            channel = [b['channel'] for b in batch if b.get('channel') is not None]

            if inputs_embeds:
                res['inputs_embeds'] = inputs_embeds
            if input_ids:
                res['input_ids'] = input_ids
            if channel:
                res['channel'] = channel

            for key in ['labels', 'loss_scale', 'position_ids', 'token_type_ids']:
                val = [b[key] for b in batch if b.get(key) is not None]
                if val:
                    res[key] = val

        keys = [
            'input_ids', 'inputs_embeds', 'attention_mask', 'labels', 'loss_scale', 'position_ids', 'token_type_ids'
        ]
        pad_values = [self.tokenizer.pad_token_id, 0., 0, -100, 0., 0., 0]
        # Convert to tensor and remove unnecessary dimensions.
        seq_lens = None
        for key in keys:
            if key not in res:
                continue
            for i, val in enumerate(res[key]):
                if isinstance(val, (list, tuple)):
                    val = torch.tensor(val)
                elif key == 'inputs_embeds' and val.ndim == 3 or key != 'inputs_embeds' and val.ndim == 2:
                    val = val[0]
                res[key][i] = val
            if not seq_lens:
                seq_lens = [seq.shape[0] for seq in res[key]]
        if not self._packing and seq_lens and ('input_ids' in res or 'inputs_embeds' in res):
            if not self.use_megatron:
                res['attention_mask'] = [torch.ones(seq_len, dtype=torch.int64) for seq_len in seq_lens]
            if self.is_training and self.padding_side == 'left':
                res['position_ids'] = [torch.arange(seq_len, dtype=torch.int64) for seq_len in seq_lens]

        if self.use_megatron:
            # For code simplicity, only the attention_backend 'flash' is supported here.
            if padding_to is not None:
                padding_to = math.ceil(max(seq_lens) / padding_to) * padding_to
            if self._packing:
                cp_size = self.sequence_parallel_size
                if cp_size > 1:
                    padding_len = padding_to - seq_lens[0]
                    position_ids = res['position_ids'][0].tolist()
                    position_ids += list(range(cp_size * 2)) * (padding_len // (cp_size * 2))
                    res['position_ids'] = [torch.tensor(position_ids)]
            else:
                seq_len = max(seq_lens) if padding_to is None else padding_to
                res['attention_mask'] = torch.tril(torch.ones(
                    (len(seq_lens), seq_len, seq_len), dtype=torch.bool)).view(len(seq_lens), 1, seq_len, seq_len)
                assert res['attention_mask'].dtype is torch.bool, f'attention_mask.dtype: {res["attention_mask"].dtype}'
                for i, seq_len in enumerate(seq_lens):
                    res['attention_mask'][i, :, seq_len:] = 0

        for key, pad_value in zip(keys, pad_values):
            if key not in res:
                continue
            if self.use_megatron and not self._packing and key == 'attention_mask':
                continue
            if padding_to is not None and not (self._packing and key == 'position_ids'
                                               and self.sequence_parallel_size > 1):
                padding_len = padding_to - seq_lens[0]
                if padding_len > 0:
                    res[key][0] = F.pad(res[key][0], (0, padding_len) if padding_right else (padding_len, 0),
                                        'constant', pad_value)
            res[key] = self._pad_sequence(res[key], pad_value)

        # multimodal
        res.update(self._data_collator_mm_data(batch))
        if not self.use_megatron and self.sequence_parallel_size > 1:
            res = self._sp_data_collator(res, padding_to, self.tokenizer, padding_side)

        return res

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        函数功能：
            整理和拼接批次中的多模态数据（multimodal data）。该方法专门处理视觉模态的数据，
            包括图像的像素值、图像尺寸信息和视频的像素值。通过将批次中所有样本的多模态张量
            沿批次维度拼接，为多模态模型（如视觉-语言模型）提供统一格式的输入数据。
        
        参数：
            batch (List[Dict[str, Any]]): 批量数据列表，每个元素是一个字典，可能包含以下多模态字段：
                - pixel_values (torch.Tensor, optional): 图像的像素值张量
                    形状通常为 (num_images, channels, height, width) 或模型特定格式
                    如 Qwen2-VL 的 (num_patches, channels) 格式
                - image_sizes (torch.Tensor, optional): 图像尺寸信息张量
                    用于记录原始图像的宽高，便于模型进行位置编码或后处理
                    形状通常为 (num_images, 2)，存储 [height, width]
                - pixel_values_videos (torch.Tensor, optional): 视频的像素值张量
                    形状通常为 (num_frames, channels, height, width) 或模型特定格式
                    用于处理视频输入的多模态任务
        
        返回值：
            Dict[str, torch.Tensor]: 拼接后的多模态数据字典，可能包含以下字段：
                - pixel_values (torch.Tensor): 批次中所有图像的像素值拼接结果
                    形状为 (total_num_images, channels, height, width) 或模型特定格式
                - image_sizes (torch.Tensor): 批次中所有图像的尺寸信息拼接结果
                    形状为 (total_num_images, 2)
                - pixel_values_videos (torch.Tensor): 批次中所有视频的像素值拼接结果
                    形状为 (total_num_frames, channels, height, width) 或模型特定格式
            注：只返回批次中实际存在的字段，不存在的字段不会出现在返回字典中
        
        使用示例：
            >>> # 假设批次中有2个样本，第1个包含图像，第2个只有文本
            >>> batch = [
            ...     {
            ...         'input_ids': [1, 2, 3],
            ...         'pixel_values': torch.randn(2, 3, 224, 224),  # 2张图像
            ...         'image_sizes': torch.tensor([[224, 224], [224, 224]])
            ...     },
            ...     {
            ...         'input_ids': [4, 5, 6],
            ...         'pixel_values': torch.randn(1, 3, 224, 224),  # 1张图像
            ...         'image_sizes': torch.tensor([[224, 224]])
            ...     }
            ... ]
            >>> template = Template(...)
            >>> mm_data = template._data_collator_mm_data(batch)
            >>> print(mm_data['pixel_values'].shape)  # torch.Size([3, 3, 224, 224])
            >>> print(mm_data['image_sizes'].shape)   # torch.Size([3, 2])
            
            >>> # 视频数据示例
            >>> batch = [
            ...     {
            ...         'input_ids': [1, 2],
            ...         'pixel_values_videos': torch.randn(16, 3, 224, 224)  # 16帧视频
            ...     }
            ... ]
            >>> mm_data = template._data_collator_mm_data(batch)
            >>> print(mm_data['pixel_values_videos'].shape)  # torch.Size([16, 3, 224, 224])
        """
        # ===== 步骤1：初始化结果字典 =====
        res = {}  # 存储拼接后的多模态数据
        
        # ===== 步骤2：处理图像像素值数据 =====
        # 从批次中提取所有非空的 pixel_values 字段（过滤掉 None 值）
        pixel_values = [b['pixel_values'] for b in batch if b.get('pixel_values') is not None]
        if len(pixel_values) > 0:  # 如果批次中至少有一个样本包含图像数据
            # 沿批次维度（通常是第0维）拼接所有图像的像素值张量
            # 例如：[tensor(2,3,224,224), tensor(1,3,224,224)] -> tensor(3,3,224,224)
            res['pixel_values'] = torch.concat(pixel_values)

            # ===== 步骤3：处理图像尺寸信息 =====
            # 从批次中提取所有非空的 image_sizes 字段
            image_sizes = [b['image_sizes'] for b in batch if b.get('image_sizes') is not None]
            if len(image_sizes) > 0:  # 如果批次中至少有一个样本包含图像尺寸信息
                # 拼接所有图像的尺寸信息张量
                # 例如：[tensor([[224,224],[224,224]]), tensor([[224,224]])] -> tensor([[224,224],[224,224],[224,224]])
                res['image_sizes'] = torch.concat(image_sizes)

        # ===== 步骤4：处理视频像素值数据 =====
        # 从批次中提取所有非空的 pixel_values_videos 字段（过滤掉 None 值）
        pixel_values_videos = [b['pixel_values_videos'] for b in batch if b.get('pixel_values_videos') is not None]
        if len(pixel_values_videos) > 0:  # 如果批次中至少有一个样本包含视频数据
            # 沿批次维度拼接所有视频的像素值张量
            # 例如：[tensor(16,3,224,224), tensor(8,3,224,224)] -> tensor(24,3,224,224)
            res['pixel_values_videos'] = torch.concat(pixel_values_videos)
        
        # ===== 步骤5：返回拼接后的多模态数据字典 =====
        return res

    def _sp_data_collator(self, res, padding_to, tokenizer, padding_side):
        input_ids = res.get('input_ids')
        attention_mask = res.get('attention_mask')
        labels = res.get('labels')
        loss_scale = res.get('loss_scale')
        if self.sequence_parallel_size > 1 and input_ids is not None:
            bs, seq_len = input_ids.shape
            if 'position_ids' not in res:
                position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)
            else:
                position_ids = res['position_ids']
            assert padding_side == 'right' or bs == 1, 'Sequence parallel only support padding_side=right'
            res['position_ids'] = position_ids
        _local_var = locals()
        for key in ['input_ids', 'attention_mask', 'labels', 'loss_scale']:
            value = _local_var[key]
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
        images = inputs.pop('images', None) or []
        if len(images) == 0:
            return
        input_ids = inputs['input_ids']
        idx_list = findall(input_ids, -100)
        assert len(idx_list) == len(images), f'len(idx_list): {len(idx_list)}, len(images): {len(images)}'
        idx_list.insert(0, -1)
        new_input_ids = []
        for i in range(len(idx_list) - 1):
            new_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]]
            images[i]['offset'] = len(new_input_ids)
            new_input_ids += [images[i]['image_token_id']] * images[i]['image_tokens']
        new_input_ids += input_ids[idx_list[-1] + 1:]
        inputs['input_ids'] = new_input_ids
        inputs['multimodal'] = images

    async def prepare_lmdeploy_turbomind_inputs(self, inputs: Dict[str, Any]) -> None:
        images = inputs.pop('images', None) or []
        if len(images) == 0:
            return
        from lmdeploy.vl.constants import IMAGE_DUMMY_TOKEN_INDEX
        input_ids = inputs['input_ids']
        idx_list = findall(input_ids, -100)
        assert len(idx_list) == len(images), f'len(idx_list): {len(idx_list)}, len(images): {len(images)}'
        idx_list.insert(0, -1)
        new_input_ids = []
        ranges = []
        for i in range(len(idx_list) - 1):
            _range = []
            new_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]]
            _range.append(len(new_input_ids))
            new_input_ids += [IMAGE_DUMMY_TOKEN_INDEX] * images[i].shape[0]
            _range.append(len(new_input_ids))
            ranges.append(_range)
        new_input_ids += input_ids[idx_list[-1] + 1:]
        inputs['input_embeddings'] = [image.to('cpu') for image in images]
        inputs['input_embedding_ranges'] = ranges
        inputs['input_ids'] = new_input_ids

    def _pad_sequence(self, sequences: List[torch.Tensor], padding_value: float = 0.) -> torch.Tensor:
        """Pad sequence by some side

        Args:
            sequences: The input sequences in tensor.
            padding_value: The padding value

        Returns:
            A tensor after padding
        """
        padding_side = self.padding_side if self.is_training else 'left'
        padding_right = padding_side == 'right'
        if padding_right:
            return pad_sequence(sequences, batch_first=True, padding_value=padding_value)

        max_len = max([s.shape[0] for s in sequences])

        padded_sequences = []
        for seq in sequences:
            pad_length = max_len - seq.shape[0]
            pad_tuple = [0] * ((seq.dim() - 1) * 2) + [pad_length, 0]
            padded_seq = F.pad(seq, tuple(pad_tuple), 'constant', padding_value)
            padded_sequences.append(padded_seq)

        return torch.stack(padded_sequences)

    def safe_decode(self, input_ids: List[int], **tokenizer_kwargs) -> str:
        if isinstance(self, Template):
            tokenizer = self.tokenizer
            placeholder_tokens = self.placeholder_tokens
        else:
            tokenizer = self
            placeholder_tokens = []

        def _is_special(token: int) -> bool:
            if isinstance(token, float) or token < 0:
                return True
            return token in placeholder_tokens

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        if len(input_ids) == 0:
            return ''
        result_str = ''
        for i in range(len(input_ids)):
            if i == 0:
                if _is_special(input_ids[i]):
                    s = 0
                else:
                    e = 0
                continue
            if _is_special(input_ids[i]) and not _is_special(input_ids[i - 1]):
                s = i
                result_str += tokenizer.decode(input_ids[e:s], **tokenizer_kwargs)
            if not _is_special(input_ids[i]) and _is_special(input_ids[i - 1]):
                e = i
                result_str += f'[{input_ids[i - 1]} * {e - s}]'
        if _is_special(input_ids[i]):
            result_str += f'[{input_ids[i]} * {len(input_ids) - s}]'
        else:
            result_str += tokenizer.decode(input_ids[e:], **tokenizer_kwargs)
        return result_str

    @staticmethod
    @contextmanager
    def _patch_flash_attention_forward(modeling_module, position_ids, use_new_func: bool = False):
        _origin_flash_attention_forward = modeling_module._flash_attention_forward

        def _flash_attention_forward(*args, **kwargs):
            if use_new_func:
                from transformers.modeling_flash_attention_utils import (_flash_attention_forward as
                                                                         flash_attention_forward)
                if args and isinstance(args[0], nn.Module):
                    args = args[1:]
                if 'is_causal' not in kwargs:
                    kwargs['is_causal'] = True
            else:
                flash_attention_forward = _origin_flash_attention_forward
            kwargs['position_ids'] = position_ids
            return flash_attention_forward(*args, **kwargs)

        modeling_module._flash_attention_forward = _flash_attention_forward
        try:
            yield
        finally:
            modeling_module._flash_attention_forward = _origin_flash_attention_forward
