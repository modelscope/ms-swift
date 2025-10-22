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
            if images and not load_images_origin:  # fix pt & qwen-vl
                for i, image in enumerate(images):
                    if isinstance(image, Image.Image):
                        images[i] = self._save_pil_image(image)
            setattr(inputs, img_field, images)

        if self.mode == 'vllm' and inputs.audios:
            sampling_rate = get_env_args('sampling_rate', int, None)
            inputs.audios = load_batch(
                inputs.audios, load_func=partial(load_audio, sampling_rate=sampling_rate, return_sr=True))

    @staticmethod
    def _replace_image_tags(inputs: StdTemplateInputs):
        # compat
        if inputs.images:
            return
        images = []
        pattern = r'<img>(.+?)</img>'
        for message in inputs.messages:
            content = message['content']
            if not isinstance(content, str):
                continue
            for image in re.findall(pattern, content):
                # only support local_path
                if os.path.isfile(image):
                    images.append(image)
                else:
                    logger.warning_once(f'Failed to parse image path: `{content}`.', hash_id='<img></img>')
            message['content'] = re.sub(pattern, '<image>', content)
        inputs.images = images

    @staticmethod
    def _replace_start_image_tags(inputs: StdTemplateInputs):
        # compat
        generate_mode = False
        message = inputs.messages[-1]
        content = message['content']
        if message['role'] == 'user' and content.endswith('<start-image>'):
            generate_mode = True
            message['content'] = message['content'][:-len('<start-image>')]  # remove the <start-image>
        inputs.generate_mode = generate_mode

    @staticmethod
    def _extend_tokens(
            input_ids: List[int], labels: Optional[List[int]], loss_scale: Optional[List[float]],
            replace_idx_list: List[int],
            get_new_tokens: Callable[[int], List[int]]) -> Tuple[List[int], Optional[List[int]], Optional[List[float]]]:
        added_tokens_len = 0
        for i, idx in enumerate(replace_idx_list):
            new_tokens = get_new_tokens(i)
            token_len = len(new_tokens)
            input_ids = input_ids[:idx + added_tokens_len] + new_tokens + input_ids[added_tokens_len + idx + 1:]
            if labels:
                labels = labels[:idx + added_tokens_len] + [-100] * token_len + labels[added_tokens_len + idx + 1:]
            if loss_scale:
                scale_idx = loss_scale[idx + added_tokens_len]
                loss_scale = loss_scale[:idx + added_tokens_len] + [scale_idx] * token_len + loss_scale[added_tokens_len
                                                                                                        + idx + 1:]
            added_tokens_len += token_len - 1
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
        margin = inputs.margin
        chosen_inputs, rejected_inputs = inputs, deepcopy(inputs)

        assert chosen_inputs.rejected_response or chosen_inputs.rejected_images, f'inputs: {inputs}'
        if chosen_inputs.rejected_response:
            rejected_inputs.messages[-1]['content'] = chosen_inputs.rejected_response
        if chosen_inputs.rejected_images:
            rejected_inputs.images = chosen_inputs.rejected_images
        chosen_encoded = self._encode_truncated(chosen_inputs)
        rejected_encoded = self._encode_truncated(rejected_inputs)

        encoded = {}
        for prefix in ['chosen', 'rejected']:
            data = locals()[f'{prefix}_encoded']
            for k, v in data.items():
                encoded[f'{prefix}_{k}'] = v
        if margin:
            encoded['margin'] = float(margin)
        return encoded

    def _kto_encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        label, inputs.label = inputs.label, None
        encoded = self._rlhf_encode(inputs)
        encoded['label'] = bool(label)
        return encoded

    def _gkd_encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = self._encode_truncated(inputs)
        encoded['prompts'] = encoded['input_ids'][:-len(encoded.pop('answer_input_ids'))]
        for k in list(encoded.keys()):
            if k.startswith('prompt_') or k.endswith('answer_'):
                encoded.pop(k, None)
        return encoded

    def _embedding_encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        _encoded = {}
        labels = []
        inference = len(inputs.messages) == 1
        if inference:
            inputs.messages.append({'role': 'assistant', 'content': ''})

        def split_multi_medias(_inputs):
            _content = _inputs.messages[-2]['content']
            image_size = len(re.findall('<image>', _content))
            video_size = len(re.findall('<video>', _content))
            audio_size = len(re.findall('<audio>', _content))
            _inputs.images = inputs.images[:image_size]
            assert len(_inputs.images) == image_size
            inputs.images = inputs.images[image_size:]
            _inputs.videos = inputs.videos[:video_size]
            assert len(_inputs.videos) == video_size
            inputs.videos = inputs.videos[video_size:]
            _inputs.audios = inputs.audios[:audio_size]
            assert len(_inputs.audios) == audio_size
            inputs.audios = inputs.audios[audio_size:]

        if not inference:
            anchor = deepcopy(inputs)
            anchor.messages[-1]['content'] = ''
            anchor.rejected_response = []
            split_multi_medias(anchor)
            anchor_encoded = self._encode_truncated(anchor)
            for key in anchor_encoded:
                _encoded[f'anchor_{key}'] = anchor_encoded[key]
            positive = deepcopy(inputs)
            positive.messages[-2]['content'] = positive.messages[-1]['content']
            positive.messages[-1]['content'] = ''
            positive.rejected_response = []
            split_multi_medias(positive)
            positive_encoded = self._encode_truncated(positive)
            for key in positive_encoded:
                _encoded[f'positive_{key}'] = positive_encoded[key]
                _encoded[f'negative_{key}'] = []
            labels.append(float(inputs.label) if inputs.label is not None else 1.0)

            rejected_len = len(inputs.rejected_response) if inputs.rejected_response else 0
            for i in range(rejected_len):
                negative = deepcopy(inputs)
                negative.messages[-2]['content'] = negative.rejected_response[i]
                negative.messages[-1]['content'] = ''
                negative.rejected_response = []
                split_multi_medias(negative)
                negative_encoded = self._encode_truncated(negative)
                for key in negative_encoded:
                    _encoded[f'negative_{key}'].append(negative_encoded[key])
                labels.append(0.0)

            _encoded['labels'] = labels
        else:
            anchor = deepcopy(inputs)
            anchor.messages[-1]['content'] = ''
            anchor.rejected_response = []
            split_multi_medias(anchor)
            _encoded = self._encode_truncated(anchor)
            _encoded.pop('labels', None)
        return _encoded

    def _reranker_encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        _encoded = {}
        labels = []

        positive = deepcopy(inputs)
        positive.rejected_response = []
        if '{doc}' in positive.messages[-2]['content']:
            positive.messages[-2]['content'] = positive.messages[-2]['content'].replace(
                '{doc}', inputs.messages[-1]['content'])
            positive.messages.pop(-1)
        positive_encoded = self._encode_truncated(positive)
        for key in positive_encoded:
            _encoded[f'positive_{key}'] = positive_encoded[key]
            _encoded[f'negative_{key}'] = []
        labels.append(1)

        rejected_len = len(inputs.rejected_response) if inputs.rejected_response else 0
        for i in range(rejected_len):
            negative = deepcopy(inputs)
            if '{doc}' in negative.messages[-2]['content']:
                negative.messages[-2]['content'] = negative.messages[-2]['content'].replace(
                    '{doc}', negative.rejected_response[i])
                negative.messages.pop(-1)
            else:
                negative.messages[-1]['content'] = negative.rejected_response[i]
            negative.rejected_response = []
            negative_encoded = self._encode_truncated(negative)
            for key in negative_encoded:
                _encoded[f'negative_{key}'].append(negative_encoded[key])
            labels.append(0)

        _encoded['labels'] = labels
        return _encoded

    def _seq_cls_encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        encoded = self._encode_truncated(inputs)
        encoded.pop('labels', None)
        if inputs.label is not None:
            labels = inputs.label
            problem_type = self._get_problem_type(self.config, labels=labels)
            if problem_type == 'single_label_classification':
                labels = int(labels)
            encoded['labels'] = labels
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
        >>> template.mode = 'pt'
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
            elif self.mode == 'rlhf':  # RLHF模式（强化学习从人类反馈）
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
        packed = {}
        keys = set()
        length = []
        for r in row:
            keys.update(r.keys())
            length.append(r['length'])
        for key in keys:
            if key in {'input_ids', 'labels', 'loss_scale'}:
                packed[key] = sum((x[key] for x in row), start=[])
            elif key == 'length':
                packed[key] = sum((x[key] for x in row))
            elif key == 'channel':
                packed[key] = [x[key] for x in row]
        if 'position_ids' not in packed:
            packed['position_ids'] = sum((list(range(x)) for x in length), start=[])

        packed.update(self._data_collator_mm_data(row))
        return packed

    def _post_encode(self, model: nn.Module, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    @staticmethod
    def _get_seq_cls_logprobs(pred: int, logprobs: torch.Tensor, top_logprobs: int):
        idxs = logprobs.argsort(descending=True, dim=-1)[:top_logprobs].tolist()
        logprobs = logprobs.tolist()
        return {
            'content': [{
                'index': pred,
                'logprobs': [logprobs[p] for p in pred] if isinstance(pred, (list, tuple)) else logprobs[pred],
                'top_logprobs': [{
                    'index': idx,
                    'logprob': logprobs[idx]
                } for idx in idxs]
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
        total_content = []
        for message in inputs.messages:
            content = message['content'] or ''
            if not isinstance(content, str):
                if message['role'] == 'user':
                    # Give up adding the default tag
                    return
                elif message['role'] == 'assistant':
                    continue
            total_content.append(content)
        if inputs.rejected_response:
            rejected_response = inputs.rejected_response
            if isinstance(inputs.rejected_response, str):
                rejected_response = [rejected_response]
            total_content += rejected_response
        total_content = '\n'.join(total_content)
        if inputs.system:
            total_content = f'{inputs.system}\n{total_content}'
        for media_type in ['image', 'audio', 'video']:
            media_key, media_tag = f'{media_type}s', f'<{media_type}>'
            medias = getattr(inputs, media_key)
            if not isinstance(medias, list):
                medias = [medias]
            if medias:
                num_media_tags = len(re.findall(media_tag, total_content))
                num_media = len(medias)
                num_new_tags = num_media - num_media_tags
                if num_new_tags > 0:
                    inputs.messages[0]['content'] = media_tag * num_new_tags + inputs.messages[0]['content']
                elif num_new_tags < 0:
                    logger.warning(
                        f'num_media: {num_media}, num_media_tags: {num_media_tags}, total_content: {total_content}. '
                        'We will only replace the frontmost media_tags while keeping the subsequent media_tags.')

    def _encode_context_list(
            self,
            context_list: List[Context],
            loss_scale_list: Optional[List[float]] = None) -> Tuple[List[int], List[int], List[float], Dict[str, Any]]:
        """return: input_ids, labels, tokenizer_kwargs"""
        input_ids: List[int] = []
        labels: List[int] = []
        loss_scale: List[float] = []
        tokenizer_kwargs = {}
        if loss_scale_list is None:
            loss_scale_list = [0.] * len(context_list)
        if self.loss_scale.keep_loss_scale:
            ignore_loss_scale = False
        else:
            ignore_loss_scale = all(loss_scale in {0, 1} for loss_scale in loss_scale_list)
        for i, (context, loss_weight) in enumerate(zip(context_list, loss_scale_list)):
            if isinstance(context, str):
                # tokenizer_kwargs is the returned tokenizer_kwargs,
                # while curr_tokenizer_kwargs is the tokenizer_kwargs for the current context.
                token_list = self._tokenize(context)
            else:
                token_list = context
            input_ids += token_list
            if loss_scale_list[i] > 0.0:
                labels += token_list
            else:
                labels += [-100] * len(token_list)
            if not ignore_loss_scale:
                loss_scale.extend([loss_weight] * len(token_list))
        if ignore_loss_scale:
            loss_scale = None
        return input_ids, labels, loss_scale, tokenizer_kwargs

    @staticmethod
    def _add_dynamic_eos(input_ids: List[int], labels: List[int], loss_scale: Optional[List[int]],
                         suffix_tokens_id: List[int]) -> None:
        suffix_len = len(suffix_tokens_id)
        start = 0
        for i in range(1, len(labels)):
            if labels[i - 1] >= 0 and labels[i] == -100:
                start = i
            if start > 0 and labels[i - 1] == -100 and labels[i] >= 0:
                # [0, 1, 2, -100(start), -100, 3(i), 4]
                length = i - start
                if length >= suffix_len and input_ids[start:start + suffix_len] == suffix_tokens_id:
                    labels[start:start + suffix_len] = suffix_tokens_id
                    if loss_scale and loss_scale[start:start + suffix_len] == [0] * suffix_len:
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
        """函数功能：截断超长序列，同时保护占位符token不被截断。
        
        参数：
        - input_ids: 输入token ID列表
        - labels: 标签列表（可选）
        - loss_mask: loss权重列表（可选）
        - truncation_strategy: 截断策略（'left'左截断/'right'右截断）
        
        返回值：截断后的(input_ids, labels, loss_mask)元组
        """
        placeholder_tokens = torch.tensor(self.placeholder_tokens)  # 转换占位符token为tensor
        input_ids_tensor = torch.tensor(input_ids)  # 转换input_ids为tensor
        protected = (input_ids_tensor[:, None] == placeholder_tokens).any(dim=-1)  # 标记所有占位符token位置（需要保护）
        n_protected = protected.sum().item()  # 计算需要保护的token数量
        if n_protected < self.max_length:  # 若保护的token数量小于max_length（需要保留部分非保护token）
            non_protected = (~protected).nonzero(as_tuple=True)[0]  # 获取所有非保护token的索引
            if truncation_strategy == 'left':  # 若左截断（保留序列右侧）
                idx = non_protected[-(self.max_length - n_protected):]  # 选择右侧的非保护token
            else:  # 若右截断（保留序列左侧）
                idx = non_protected[:self.max_length - n_protected]  # 选择左侧的非保护token
            protected[idx] = True  # 将选中的非保护token也标记为保护（保留）
        input_ids = input_ids_tensor[protected].tolist()  # 提取所有保护的token（截断后的input_ids）
        if labels is not None:  # 若有labels
            labels = torch.tensor(labels)[protected].tolist()  # 同步截断labels
        if loss_mask is not None:  # 若有loss_mask
            loss_mask = torch.tensor(loss_mask)[protected].tolist()  # 同步截断loss_mask
        return input_ids, labels, loss_mask  # 返回截断后的三元组

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
        """函数功能：核心编码方法，根据template_backend选择swift或jinja编码。
        
        参数：inputs - 标准输入对象
        返回值：编码后的字典，包含input_ids, labels, loss_scale等
        """
        template_backend = self.template_backend  # 获取模板后端
        if (self.template_meta.template_type == 'dummy' and self.use_chat_template and not self.is_training
                and self.task_type != 'seq_cls'):  # 若为dummy模板且推理模式
            template_backend = 'jinja'  # 使用jinja后端
            logger.info_once(f'Setting template_backend: {template_backend}')  # 记录日志
        res_context_list, loss_scale_list, answer_len = (
            self._swift_encode(inputs) if template_backend == 'swift' else self._jinja_encode(inputs))  # 调用相应的编码方法
        encoded = {}
        if self.is_encoder_decoder or self.mode == 'gkd':
            # tokenizer_kwargs: use prompt (qwen-audio)
            total_len = len(res_context_list)
            for key, _slice in zip(['prompt', 'answer'],
                                   [slice(0, total_len - answer_len),
                                    slice(total_len - answer_len, total_len)]):
                context_list, loss_scale = self._simplify_context_list(res_context_list[_slice],
                                                                       loss_scale_list[_slice], inputs)
                input_ids, labels, loss_scale, tokenizer_kwargs = self._encode_context_list(context_list, loss_scale)
                encoded[f'{key}_input_ids'] = input_ids
                encoded[f'{key}_labels'] = labels
                encoded[f'{key}_loss_scale'] = loss_scale
            input_ids = encoded['prompt_input_ids'] + encoded['answer_input_ids']
            labels = encoded['prompt_labels'] + encoded['answer_labels']
            loss_scale = None
            if isinstance(encoded['prompt_loss_scale'], list):
                loss_scale = encoded['prompt_loss_scale'] + encoded['answer_loss_scale']
        else:
            res_context_list, loss_scale_list = self._simplify_context_list(res_context_list, loss_scale_list, inputs)
            input_ids, labels, loss_scale, tokenizer_kwargs = self._encode_context_list(
                res_context_list, loss_scale_list)
        self._add_dynamic_eos(input_ids, labels, loss_scale, self._encode_context_list(self.template_meta.suffix)[0])

        if tokenizer_kwargs:
            encoded['tokenizer_kwargs'] = tokenizer_kwargs

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        self._handle_megatron_cp(encoded)  # TODO: fix cp_size & cached_dataset
        if encoded.get('labels') is not None:
            encoded['labels'][0] = -100
        if encoded.get('loss_scale') is not None:
            encoded['loss_scale'][0] = 0
        if not self.is_training:
            for k in list(encoded.keys()):
                if k.endswith('labels') or k.endswith('loss_scale'):
                    encoded[k] = None
        return encoded

    def _handle_megatron_cp(self, encoded: Dict[str, Any]) -> None:
        cp_size = self.sequence_parallel_size
        if not self.use_megatron or cp_size == 1:
            return
        input_ids = encoded['input_ids']
        padding_len = math.ceil(len(input_ids) / (cp_size * 2)) * (cp_size * 2) - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_len
        encoded['labels'] += [-100] * padding_len
        if encoded.get('loss_scale') is not None:
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
        # multimodal
        res = {}
        pixel_values = [b['pixel_values'] for b in batch if b.get('pixel_values') is not None]
        if len(pixel_values) > 0:
            res['pixel_values'] = torch.concat(pixel_values)

            image_sizes = [b['image_sizes'] for b in batch if b.get('image_sizes') is not None]
            if len(image_sizes) > 0:
                res['image_sizes'] = torch.concat(image_sizes)

        pixel_values_videos = [b['pixel_values_videos'] for b in batch if b.get('pixel_values_videos') is not None]
        if len(pixel_values_videos) > 0:
            res['pixel_values_videos'] = torch.concat(pixel_values_videos)
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
