# Copyright (c) Alibaba, Inc. and its affiliates.

"""
模板输入数据模块 (Template Inputs Module)

本模块定义了大语言模型模板系统的输入数据结构，用于封装推理请求和训练数据。
通过数据类封装消息、多模态数据（图像、音频、视频）、工具调用等信息，为模板
处理提供统一的数据接口。

主要功能：
    - 定义推理请求的数据结构（InferRequest）
    - 支持多模态输入（图像、音频、视频）
    - 处理工具调用和 Agent 交互
    - 提供训练数据的扩展支持（TemplateInputs、StdTemplateInputs）
    - 实现消息格式的标准化和转换

核心类：
    - InferRequest: 基础推理请求数据类，支持多模态和工具调用
    - RolloutInferRequest: 用于 POST 请求的推理请求（字符串化图像）
    - TemplateInputs: 扩展推理请求，添加训练功能支持
    - StdTemplateInputs: 标准化模板输入，支持系统提示和拒绝响应
"""

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import json
from PIL import Image

from swift.utils import get_logger
from ..utils import Messages, Tool, messages_to_history

logger = get_logger()


@dataclass
class InferRequest:
    """
    推理请求数据类。
    
    该类封装了模型推理所需的所有输入数据，包括对话消息、多模态数据（图像、音频、视频）
    和工具调用信息。支持两种消息格式：简单的字符串内容或结构化的多模态内容。
    
    类功能：
        封装推理请求的输入数据，统一处理文本、多模态和工具调用
    
    继承关系：
        使用 @dataclass 装饰器，自动生成 __init__、__repr__ 等方法
    
    应用场景：
        - 单轮或多轮对话推理
        - 多模态理解任务（图文、音视频问答）
        - Agent 工具调用
        - 目标检测等 grounding 任务
    
    使用示例：
        # 示例1：纯文本对话
        request = InferRequest(
            messages=[
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"}
            ]
        )
        
        # 示例2：图文多模态（结构化格式）
        request = InferRequest(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://example.com/image.jpg"},
                    {"type": "text", "text": "Please describe the picture."}
                ]
            }]
        )
        
        # 示例3：图文多模态（简化格式 + 额外传递图像）
        request = InferRequest(
            messages=[{"role": "user", "content": "<image>Please describe the picture."}],
            images=["https://example.com/image.jpg"]
        )
        # 注意：示例2和示例3是等价的
        
        # 示例4：工具调用
        request = InferRequest(
            messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {"city": {"type": "string"}}
                }
            }]
        )
        
        # 示例5：视频理解
        request = InferRequest(
            messages=[{"role": "user", "content": "<video>Describe this video."}],
            videos=["path/to/video.mp4"]
        )
    
    字段说明：
        messages: 对话消息列表，支持两种内容格式：
            1. 字符串格式：{"role": "user", "content": "文本内容"}
            2. 结构化格式：{"role": "user", "content": [
                {"type": "text", "text": "文本"},
                {"type": "image", "image": "图像路径/URL/base64"}
            ]}
        
        tools: 工具列表，会被组织成 agent_template 格式添加到系统提示中，
               例如使用 'react_en' 格式
    """
    # ========== 必需字段 ==========
    # 对话消息列表，每个消息包含 role 和 content
    # role 可以是：user, assistant, system, tool
    # content 可以是字符串或结构化的多模态内容列表
    messages: Messages

    # ========== 多模态数据字段 ==========
    # 图像列表，支持 URL、本地路径、Base64 编码或 PIL.Image 对象
    images: List[Union[str, Image.Image]] = field(default_factory=list)
    
    # 音频文件列表，支持 URL 或本地路径
    audios: List[str] = field(default_factory=list)
    
    # 视频文件列表，支持 URL 或本地路径
    videos: List[str] = field(default_factory=list)

    # ========== 工具和对象字段 ==========
    # 工具列表，用于 Agent 任务，会被格式化为系统提示
    tools: Optional[List[Tool]] = None
    
    # 对象字典，用于 grounding 任务（如目标检测）的通用格式
    # 键为对象类型，值为对象列表
    objects: Dict[str, List[Any]] = field(default_factory=dict)

    def __post_init__(self):
        """
        数据类初始化后的后处理方法。
        在 dataclass 的 __init__ 方法执行后自动调用，用于标准化多模态数据字段
        的格式，确保它们都是列表类型，并验证消息字段的有效性。
        
        功能：
            标准化多模态字段格式，验证消息字段
        
        副作用：
            - 将字符串类型的 images/audios/videos 字段转换为单元素列表
            - 验证 messages 字段是列表类型
        
        异常：
            AssertionError: 当 messages 不是列表类型时抛出
        
        使用示例：
            # 示例1：自动转换字符串为列表
            request = InferRequest(
                messages=[{"role": "user", "content": "Hello"}],
                images="image.jpg"  # 字符串
            )
            # __post_init__ 自动调用后：
            # request.images == ["image.jpg"]  # 转换为列表
            
            # 示例2：列表保持不变
            request = InferRequest(
                messages=[{"role": "user", "content": "Hello"}],
                images=["img1.jpg", "img2.jpg"]
            )
            # __post_init__ 自动调用后：
            # request.images == ["img1.jpg", "img2.jpg"]  # 保持不变
            
            # 示例3：验证失败
            # request = InferRequest(messages="invalid")  # 抛出 AssertionError
        """
        # 遍历多模态数据字段（images, audios, videos）
        for key in ['images', 'audios', 'videos']:
            # 获取当前字段的值
            val = getattr(self, key)
            
            # 如果值是字符串（单个文件），转换为单元素列表
            if isinstance(val, str):
                setattr(self, key, [val])
        
        # 验证 messages 字段必须是列表类型
        assert isinstance(self.messages, list), f'messages: {self.messages}'

    @staticmethod
    def remove_response(messages: Messages) -> Optional[str]:
        """
        功能：
            移除并返回消息列表中最后一条助手响应，并返回其内容。具体地，
            该方法检查消息列表的最后一条消息是否为助手角色，如果是则移除并返回其内容。
            通常用于提取目标响应或在训练时分离输入和标签。
        
        参数：
            messages (List[Dict[str, Any]]): 消息列表，每个消息包含 role 和 content
                例如：[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        
        返回值：
            Optional[str]: 如果最后一条消息是助手角色，返回其内容；否则返回 None
        
        副作用：
            如果最后一条消息是助手角色，会从 messages 列表中移除该消息（就地修改）
        
        使用示例：
            # 示例1：移除助手响应
            messages = [
                {"role": "user", "content": "What's 1+1?"},
                {"role": "assistant", "content": "2"}
            ]
            response = InferRequest.remove_response(messages)
            # response == "2"
            # messages == [{"role": "user", "content": "What's 1+1?"}]
            
            # 示例2：最后一条不是助手消息
            messages = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"}
            ]
            response = InferRequest.remove_response(messages)
            # response == None
            # messages 保持不变
            
            # 示例3：空消息列表
            messages = []
            response = InferRequest.remove_response(messages)
            # response == None
        """
        # 获取最后一条消息的角色（如果消息列表不为空）
        last_role = messages[-1]['role'] if messages else None
        
        # 如果最后一条消息的角色是 assistant，移除并返回其内容
        if last_role == 'assistant':
            return messages.pop()['content']

    @staticmethod
    def _to_printable(obj, key: Optional[str] = None):
        """
        功能：
            内部辅助方法，将对象转换为适合打印的格式，截断过长的 Base64 数据。具体地，
            该方法递归处理对象，将过长的字符串（如 Base64 编码的数据）截断为摘要形式，
            以便日志输出和调试。对于嵌套的列表和字典，递归处理所有元素。
        
        参数：
            obj (Any): 要转换的对象，可以是字符串、列表、字典或其他类型
            key (Optional[str]): 当前对象在父字典中的键名，用于判断是否需要截断
                'content' 和 'text' 键的值不会被截断
        
        返回值：
            Any: 转换后的对象，结构与输入相同，但长字符串被截断
        
        使用示例：
            # 示例1：截断长 Base64 字符串
            long_str = "iVBORw0KGgo..." + "x" * 1000
            result = InferRequest._to_printable(long_str, key='image')
            # result == '<<<base64:iVBORw0KGgo...>>>'  # 只保留前 50 个字符
            
            # 示例2：保留 content 和 text 的长字符串
            long_text = "This is a long content..." + "x" * 1000
            result = InferRequest._to_printable(long_text, key='content')
            # result == long_text  # 不截断
            
            # 示例3：递归处理列表
            data = ["short", "x" * 1000]
            result = InferRequest._to_printable(data)
            # result == ["short", "<<<base64:xxxx...>>>"]
            
            # 示例4：递归处理字典
            data = {
                "text": "Hello",
                "image": "base64..." + "x" * 1000
            }
            result = InferRequest._to_printable(data)
            # result == {"text": "Hello", "image": "<<<base64:base64...>>>"}
        """
        # 情况1：字符串类型，且不是 content/text 字段，且长度 >= 1000
        if isinstance(obj, str) and key not in {'content', 'text'} and len(obj) >= 1000:
            # 截断为前 50 个字符 + 省略号，标记为 base64 数据
            return f'<<<base64:{obj[:50]}..>>>'
        
        # 情况2：列表类型，递归处理每个元素
        elif isinstance(obj, list):
            res = []
            for item in obj:
                # 递归调用自身处理列表中的每个元素
                res.append(InferRequest._to_printable(item))
            return res
        
        # 情况3：字典类型，递归处理每个键值对
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                # 递归调用自身处理字典的值，传递键名用于判断是否截断
                res[k] = InferRequest._to_printable(v, key=k)
            return res
        
        # 情况4：其他类型（数字、布尔值等），直接返回
        return obj

    def to_printable(self):
        """
         功能：
        将当前请求对象转换为可打印的字典格式。具体地，
        该方法将 InferRequest 对象转换为字典，并截断其中的长字符串（如 Base64 数据），
        便于日志输出和调试，避免日志过长。
        
        返回值：
            Dict[str, Any]: 可打印的字典，长字符串已被截断
        
        使用示例：
            # 示例1：打印纯文本请求
            request = InferRequest(
                messages=[{"role": "user", "content": "Hello"}]
            )
            printable = request.to_printable()
            print(printable)
            # 输出：{'messages': [{'role': 'user', 'content': 'Hello'}], 'images': [], ...}
            
            # 示例2：打印包含长 Base64 图像的请求
            request = InferRequest(
                messages=[{"role": "user", "content": "<image>Describe it."}],
                images=["data:image/png;base64," + "x" * 2000]
            )
            printable = request.to_printable()
            # printable['images'][0] == '<<<base64:data:image/png;base64,xxxx...>>>'
            
            # 示例3：用于日志记录
            import logging
            logger.info(f"Request: {request.to_printable()}")
        """
        # 将数据类转换为字典，然后调用 _to_printable 处理长字符串
        return InferRequest._to_printable(asdict(self))


@dataclass
class RolloutInferRequest(InferRequest):
    """
    类功能：
        专门用于通过 HTTP POST 请求传递的推理请求。
        将 images 字段限制为字符串列表（URL 或 Base64 编码），以便于序列化和网络传输。
    
    继承关系：
        继承自 InferRequest，重写了 images 字段类型
    
    应用场景：
        - 通过 HTTP API 进行远程推理
        - Rollout 数据收集
        - 分布式推理系统
        - 需要序列化推理请求的场景
    
    使用示例：
        # 示例1：使用图像 URL
        request = RolloutInferRequest(
            messages=[{"role": "user", "content": "<image>Describe this."}],
            images=["https://example.com/image.jpg"]
        )
        
        # 示例2：使用 Base64 编码的图像
        request = RolloutInferRequest(
            messages=[{"role": "user", "content": "<image>What's in the picture?"}],
            images=["data:image/png;base64,iVBORw0KGgo..."]
        )
        
        # 示例3：包含额外数据字典
        request = RolloutInferRequest(
            messages=[{"role": "user", "content": "Hello"}],
            data_dict={"request_id": "12345", "user_id": "user_001"}
        )
        
        # 示例4：序列化为 JSON 用于 POST 请求
        import json
        request_json = json.dumps(asdict(request))
    """
    # 图像列表，仅支持字符串类型（URL 或 Base64 编码）
    # 覆盖父类的 images 字段，不再支持 PIL.Image 对象
    images: List[str] = field(default_factory=list)
    
    # 额外的数据字典，用于存储请求相关的元数据
    # 例如：request_id, user_id, timestamp 等
    data_dict: Dict = field(default_factory=dict)


@dataclass
class TemplateInputs(InferRequest):
    """
    类功能：
        模板输入数据类（训练扩展版）。
        在 InferRequest 的基础上扩展训练相关功能，包括拒绝响应（用于 RLHF）和标签（用于分类任务）。
    
    继承关系：
        继承自 InferRequest，添加了训练相关字段。
    
    应用场景：
        - RLHF (Reinforcement Learning from Human Feedback) 训练
        - DPO (Direct Preference Optimization) 训练
        - 分类任务训练
    
    使用示例：
        # 示例1：RLHF/DPO 训练数据
        inputs = TemplateInputs(
            messages=[{"role": "user", "content": "Write a poem about spring."}],
            rejected_response="Spring is a season. It has flowers."  # 拒绝的响应
        )
        # 用于训练时比较 accepted response（从 messages 提取）和 rejected_response
        
        # 示例2：分类任务
        inputs = TemplateInputs(
            messages=[{"role": "user", "content": "This movie is great!"}],
            label=True  # 正面情感
        )
    """
    # 拒绝的响应内容，用于 RLHF/DPO 等偏好学习任务
    # 与消息中的 accepted response 形成对比对
    rejected_response: Optional[str] = None
    
    # 标签，用于分类任务（如情感分析、意图识别等）
    # True/False 表示二分类，也可扩展为多分类
    label: Optional[bool] = None


@dataclass
class StdTemplateInputs:
    """
    类功能：
        标准化模板输入数据类。
        提供更标准化的模板输入格式，将系统提示词从消息列表中分离出来，
        仅在消息中保留 user/tool/assistant 角色。支持训练、RLHF 和奖励建模等多种场景。
    
    继承关系：
        使用 @dataclass 装饰器，独立的数据类（不继承 InferRequest）
    
    应用场景：
        - 模型训练和微调
        - RLHF 和 DPO 训练
        - 奖励模型训练
        - 多模态任务
        - 标准化的推理请求
    
    使用示例：
        # 示例1：基本对话（使用默认系统提示）
        inputs = StdTemplateInputs(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            system=None  # 使用默认系统提示
        )
        
        # 示例2：自定义系统提示
        inputs = StdTemplateInputs(
            messages=[{"role": "user", "content": "What's 1+1?"}],
            system="You are a math teacher."
        )
        
        # 示例3：不使用系统提示
        inputs = StdTemplateInputs(
            messages=[{"role": "user", "content": "Hi"}],
            system=""  # 空字符串表示不使用系统提示
        )
        
        # 示例4：RLHF 训练数据
        inputs = StdTemplateInputs(
            messages=[{"role": "user", "content": "Tell me a joke."}],
            rejected_response="I don't know any jokes.",
            margin=0.5  # 奖励差值
        )
        
        # 示例5：工具调用
        inputs = StdTemplateInputs(
            messages=[
                {"role": "user", "content": "What's the weather?"},
                {"role": "tool", "content": '{"city": "Beijing", "temp": 25}'}
            ],
            tools=[{"type": "function", "function": {"name": "get_weather"}}]
        )
        
        # 示例6：多模态分类
        inputs = StdTemplateInputs(
            messages=[{"role": "user", "content": "<image>What's this?"}],
            images=["cat.jpg"],
            label=1,  # 类别标签
            channel="vision"  # 数据来源渠道
        )
    
    字段说明：
        messages: 仅包含 user/tool/assistant 角色的消息列表
        system: 系统提示词设置
            - None: 使用默认系统提示
            - '': 不使用系统提示
            - 其他字符串: 使用自定义系统提示
        margin: 用于奖励建模的边际值，表示 accepted 和 rejected 响应的奖励差异
    """
    # ========== 核心消息字段 ==========
    # 消息列表，仅包含 user/tool/assistant 角色
    # 系统提示词已被分离到 system 字段
    messages: List[Dict[str, str]]
    
    # 系统提示词
    # None: 使用模板的默认系统提示
    # '': 不使用系统提示
    # 其他字符串: 使用指定的系统提示
    system: Optional[str] = None
    
    # 工具列表，用于 Agent 任务
    tools: Optional[List[Tool]] = None

    # ========== 训练相关字段 ==========
    # 拒绝的响应内容，用于 RLHF/DPO
    rejected_response: Optional[str] = None
    
    # 标签，用于分类任务（整数类型，支持多分类）
    label: Optional[int] = None
    
    # 数据来源渠道标识
    channel: Optional[str] = None

    # ========== 多模态数据字段 ==========
    # 图像列表，支持 URL、本地路径、Base64 或 PIL.Image
    images: List[Union[str, Image.Image]] = field(default_factory=list)
    
    # 音频文件列表
    audios: List[str] = field(default_factory=list)
    
    # 视频文件列表
    videos: List[str] = field(default_factory=list)
    
    # 对象字典，用于 grounding 任务
    objects: Dict[str, List[Any]] = field(default_factory=dict)
    
    # 拒绝响应对应的图像列表（用于多模态 RLHF）
    rejected_images: List[Union[str, Image.Image]] = field(default_factory=list)

    # ========== 奖励建模字段 ==========
    # 边际值，用于奖励模型训练
    # 表示 accepted response 和 rejected response 的奖励差异
    margin: Optional[float] = None

    def __post_init__(self):
        """
        功能：
        数据类初始化后的后处理方法。
        该方法在 dataclass 的 __init__ 方法执行后自动调用，用于初始化索引计数器和标准化多模态数据字段的格式。
        
        副作用：
            - 初始化 image_idx, audio_idx, video_idx, ref_idx, bbox_idx 为 0
            - 将非列表/元组的多模态字段转换为单元素列表
        
        使用示例：
            # 示例1：自动转换单个图像为列表
            inputs = StdTemplateInputs(
                messages=[{"role": "user", "content": "Hi"}],
                images="image.jpg"  # 单个字符串
            )
            # __post_init__ 自动调用后：
            # inputs.images == ["image.jpg"]
            # inputs.image_idx == 0
            
            # 示例2：列表保持不变
            inputs = StdTemplateInputs(
                messages=[{"role": "user", "content": "Hi"}],
                images=["img1.jpg", "img2.jpg"]
            )
            # __post_init__ 自动调用后：
            # inputs.images == ["img1.jpg", "img2.jpg"]
        """
        # 初始化各类多模态数据的索引计数器
        self.image_idx = 0      # 图像索引
        self.audio_idx = 0      # 音频索引
        self.video_idx = 0      # 视频索引
        self.ref_idx = 0        # 引用索引
        self.bbox_idx = 0       # 边界框索引
        
        # 标准化 images 字段：如果不是列表或元组，转换为单元素列表
        if self.images and not isinstance(self.images, (list, tuple)):
            self.images = [self.images]
        
        # 标准化 videos 字段
        if self.videos and not isinstance(self.videos, (list, tuple)):
            self.videos = [self.videos]
        
        # 标准化 audios 字段
        if self.audios and not isinstance(self.audios, (list, tuple)):
            self.audios = [self.audios]
        
        # 标准化 rejected_images 字段
        if self.rejected_images and not isinstance(self.rejected_images, (list, tuple)):
            self.rejected_images = [self.rejected_images]

    def to_history(self) -> Optional[List]:
        """
        功能：
        该方法将标准消息格式转换为历史对话格式（通常是 query-response 对的列表）。

        返回值：
            Optional[List]: 历史对话列表，如果消息为空则返回 None
                格式通常为：[[query1, response1], [query2, response2], ...]
        
        使用示例：
            # 示例1：多轮对话
            inputs = StdTemplateInputs(
                messages=[
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "How are you?"}
                ]
            )
            history = inputs.to_history()
            # history == [["Hi", "Hello!"], ["How are you?", None]]
            
            # 示例2：空消息
            inputs = StdTemplateInputs(messages=[])
            history = inputs.to_history()
            # history == None
        """
        # 如果消息列表为空，返回 None
        if not self.messages:
            return None
        
        # 调用工具函数将消息转换为历史对话格式
        return messages_to_history(self.messages)

    # NOTE: @property 装饰器的作用是，
    # @property 是 Python 的内置装饰器，用于将类的方法伪装成只读属性，从而可以通过“属性访问”的方式调用方法，而无需加括号 ()。
    # 它常用于封装属性、提供安全的访问方式。
    @property
    def is_multimodal(self) -> bool:
        """
        功能：
            判断当前输入是否包含多模态数据。具体地，
            该属性检查是否存在图像、音频、视频或对象数据，用于快速判断是否为多模态任务。
        
        返回值：
            bool: 如果包含图像、音频、视频或对象数据，返回 True；否则返回 False
        
        使用示例：
            # 示例1：纯文本
            inputs = StdTemplateInputs(
                messages=[{"role": "user", "content": "Hello"}]
            )
            print(inputs.is_multimodal)  # False
            
            # 示例2：包含图像
            inputs = StdTemplateInputs(
                messages=[{"role": "user", "content": "<image>What's this?"}],
                images=["cat.jpg"]
            )
            print(inputs.is_multimodal)  # True
            
            # 示例3：包含音频
            inputs = StdTemplateInputs(
                messages=[{"role": "user", "content": "Transcribe this."}],
                audios=["audio.mp3"]
            )
            print(inputs.is_multimodal)  # True
        """
        # 返回是否存在任何多模态数据
        return bool(self.images or self.audios or self.videos or self.objects)

    @classmethod
    def from_dict(cls, inputs: Dict[str, Any]) -> Tuple['StdTemplateInputs', Dict[str, Any]]:
        """
        功能：
            类方法，从字典创建 StdTemplateInputs 对象，处理各种格式的输入数据。
            具体地，将字典格式的输入数据解析并转换为 StdTemplateInputs 对象。处理包括：
            提取系统提示词、标准化角色名称、提取多模态数据、分离额外参数等。
        
        参数：
            inputs (Dict[str, Any]): 输入数据字典，至少包含 'messages' 字段
                可选字段：tools, objects, rejected_response, label, channel, margin,
                         images, audios, videos 等
        
        返回值：
            Tuple['StdTemplateInputs', Dict[str, Any]]: 
                - 第一个元素：创建的 StdTemplateInputs 对象
                - 第二个元素：额外的未识别字段字典
        
        副作用：
            会修改 inputs['messages']，提取系统消息和多模态数据
        
        使用示例：
            # 示例1：基本对话
            data = {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"}
                ]
            }
            inputs, extra = StdTemplateInputs.from_dict(data)
            # inputs.system == "You are helpful."
            # inputs.messages == [{"role": "user", "content": "Hi"}]
            
            # 示例2：多模态数据（结构化格式）
            data = {
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "cat.jpg"},
                        {"type": "text", "text": "What's this?"}
                    ]
                }]
            }
            inputs, extra = StdTemplateInputs.from_dict(data)
            # inputs.messages == [{"role": "user", "content": "<image>What's this?"}]
            # inputs.images == ["cat.jpg"]
            
            # 示例3：工具调用
            data = {
                "messages": [
                    {"role": "user", "content": "Get weather"},
                    {"role": "tool_response", "content": {"temp": 25}}
                ],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}]
            }
            inputs, extra = StdTemplateInputs.from_dict(data)
            # inputs.messages[1]["role"] == "tool"
            # inputs.messages[1]["content"] == '{"temp": 25}'
            
            # 示例4：带额外字段
            data = {
                "messages": [{"role": "user", "content": "Hi"}],
                "custom_field": "value"
            }
            inputs, extra = StdTemplateInputs.from_dict(data)
            # extra == {"custom_field": "value"}
        """
        # ========== 提取可选的训练相关字段 ==========
        kwargs = {}
        for key in ['rejected_response', 'label', 'channel', 'margin']:
            if key in inputs:
                kwargs[key] = inputs[key]
        
        # ========== 提取核心字段 ==========
        messages = inputs['messages']
        tools = inputs.get('tools')
        objects = inputs.get('objects') or {}

        # ========== 处理系统提示词 ==========
        # 如果第一条消息是系统角色，提取并移除
        if messages and messages[0]['role'] == 'system':
            message = messages.pop(0)
            system = message['content']
        else:
            system = None

        # ========== 标准化角色和内容格式 ==========
        for message in messages:
            # 将 'tool_response' 角色标准化为 'tool'
            if message['role'] == 'tool_response':
                message['role'] = 'tool'
            
            # 将工具调用和工具响应的非字符串内容转换为 JSON 字符串
            if message['role'] in {'tool_call', 'tool'} and not isinstance(message['content'], str):
                message['content'] = json.dumps(message['content'], ensure_ascii=False)

        # ========== 提取和合并多模态数据 ==========
        # 从消息中提取多模态数据（结构化格式）
        media_kwargs = StdTemplateInputs.remove_messages_media(messages)

        for k in list(media_kwargs.keys()):
            # 从消息中提取的多模态数据
            mm_data = media_kwargs[k]

            # 从 inputs 字典中获取的多模态数据
            inputs_mm_data = inputs.get(k)
            if isinstance(inputs_mm_data, str):
                inputs_mm_data = [inputs_mm_data]
            inputs_mm_data = (inputs_mm_data or []).copy()
            
            # 如果消息中已有多模态数据，则不允许在 inputs 中再次提供
            if mm_data:
                assert not inputs_mm_data, f'self.{k}: {inputs_mm_data}'
            else:
                # 否则使用 inputs 中提供的多模态数据
                media_kwargs[k] = inputs_mm_data

        # ========== 分离额外字段 ==========
        # 获取 StdTemplateInputs 的所有字段名
        all_keys = set(f.name for f in fields(StdTemplateInputs))
        # 提取不属于 StdTemplateInputs 的额外字段
        extra_kwargs = {k: v for k, v in inputs.items() if k not in all_keys}
        
        # ========== 创建对象并返回 ==========
        return cls(
            messages=messages, system=system, tools=tools, objects=objects, **kwargs, **media_kwargs), extra_kwargs

    @staticmethod
    def remove_messages_media(messages: Messages) -> Dict[str, Any]:
        """
        功能：
            静态方法，从消息列表中提取多模态数据并转换为占位符格式。
            具体地，该方法解析消息列表中的结构化内容（List[Dict]），提取图像、音频、视频数据，
            并将消息内容转换为包含占位符的字符串格式（如 "<image>描述图片"）。
        
        参数：
            messages (Messages): 消息列表，可能包含结构化的多模态内容
                例如：[{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "cat.jpg"},
                            {"type": "text", "text": "What's this?"}
                        ]
                    }]
        
        返回值：
            Dict[str, Any]: 提取的多模态数据字典，包含以下键：
                - 'images': 图像列表
                - 'audios': 音频列表
                - 'videos': 视频列表
                - 'rejected_images': 拒绝响应的图像列表
        
        副作用：
            会修改 messages 中的 content 字段，将结构化内容转换为字符串
        
        使用示例：
            # 示例1：提取图像数据
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": "cat.jpg"},
                    {"type": "text", "text": "What's this?"}
                ]
            }]
            media = StdTemplateInputs.remove_messages_media(messages)
            # media == {'images': ['cat.jpg'], 'audios': [], 'videos': [], 'rejected_images': []}
            # messages[0]['content'] == '<image>What\'s this?'
            
            # 示例2：处理 URL 格式
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                    {"type": "text", "text": "Describe it."}
                ]
            }]
            media = StdTemplateInputs.remove_messages_media(messages)
            # media['images'] == ['https://example.com/img.jpg']
            # messages[0]['content'] == '<image>Describe it.'
            
            # 示例3：纯文本（不处理）
            messages = [{"role": "user", "content": "Hello"}]
            media = StdTemplateInputs.remove_messages_media(messages)
            # media == {'images': [], 'audios': [], 'videos': [], 'rejected_images': []}
            # messages[0]['content'] == 'Hello'  # 保持不变
            
            # 示例4：混合多模态
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": "img.jpg"},
                    {"type": "text", "text": "Look at this "},
                    {"type": "audio", "audio": "sound.mp3"},
                    {"type": "text", "text": " and this."}
                ]
            }]
            media = StdTemplateInputs.remove_messages_media(messages)
            # media == {'images': ['img.jpg'], 'audios': ['sound.mp3'], 'videos': [], 'rejected_images': []}
            # messages[0]['content'] == '<image>Look at this <audio> and this.'
        """
        # 初始化结果字典，存储各类多模态数据
        res = {'images': [], 'audios': [], 'videos': [], 'rejected_images': []}
        
        # 遍历所有消息
        for message in messages:
            content = message['content']
            
            # 情况1：内容是字符串，无需处理
            if isinstance(content, str):
                continue
            
            # 情况2：内容是 token ID 列表或包含 token_ids 的字典，无需处理
            elif (isinstance(content, list) and content
                  and isinstance(content[0], int)) or (isinstance(content, dict) and 'token_ids' in content):
                continue
            
            # 情况3：内容是结构化的多模态列表 List[Dict[str, Any]]
            new_content = ''
            for item in content:
                # 获取项目类型（text, image, audio, video 等）
                key: str = item['type']
                # 获取对应的值
                value = item.get(key)
                
                # 处理文本类型：直接拼接到内容中
                if key == 'text':
                    new_content += value
                    continue
                
                # 处理多模态类型（image/audio/video）
                # 兼容 image_url/audio_url/video_url 格式
                if key.endswith('_url'):
                    # 去除 '_url' 后缀，统一为 image/audio/video
                    key = key[:-len('_url')]
                
                # 添加占位符标记（如 <image>, <audio>, <video>）
                new_content += f'<{key}>'
                
                # 如果值是字典（如 {"url": "..."}），提取 URL
                if isinstance(value, dict):
                    value = value['url']
                
                # 如果有值，添加到对应的多模态数据列表
                if value:
                    res[f'{key}s'].append(value)

            # 将消息内容替换为转换后的字符串
            message['content'] = new_content
        
        # 返回提取的多模态数据
        return res
