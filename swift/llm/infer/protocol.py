"""
模块功能概述：
本模块定义了推理服务所使用的请求/响应协议数据结构，兼容 OpenAI 风格的 Chat/Completion/Embedding 接口，包含：
- 基础模型信息与列表（Model/ModelList）
- 生成配置（RequestConfig）
- 各类请求 Mixin（文本补全、嵌入、聊天、多模态）与对应请求体（CompletionRequest、EmbeddingRequest、ChatCompletionRequest）
- 响应与流式响应的数据结构（ChatCompletionResponse/Stream、CompletionResponse/Stream、EmbeddingResponse）
- 工具函数：随机ID、二进制数据到base64转换、多模态字段自动转base64等
"""

# 版权信息：阿里巴巴及其附属公司保留所有权利
# Copyright (c) Alibaba, Inc. and its affiliates.
# 导入base64库：用于将二进制（如图片）编码为Base64字符串
import base64
# 导入io：用于内存字节流（图片转内存字节后再编码）
import io
# 导入os：用于判断路径/文件是否存在、提取后缀等
import os
# 导入time：用于生成时间戳（created字段）
import time
# 导入uuid：用于生成唯一ID
import uuid
# 从copy导入deepcopy：用于响应转换时深拷贝对象，避免修改原对象
from copy import deepcopy
# 从dataclasses导入常用工具：asdict（对象转字典）、dataclass（数据类装饰器）、field/fields（字段定义/反射）
from dataclasses import asdict, dataclass, field, fields
# 导入类型注解：用于声明数据结构字段与函数签名
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# 导入json库：用于序列化函数参数或返回
import json
# 从PIL导入Image：处理内存图像对象（保存/转BytesIO）
from PIL import Image
# 从pydantic导入BaseModel：用于定义少数校验模型
from pydantic import BaseModel

# 从上级模板模块导入InferRequest：作为parse解析的目标数据结构之一
from ..template import InferRequest
# 从上级工具模块导入消息/工具结构体：用于聊天消息与函数调用工具
from ..utils import Messages, Tool


# 工具函数：生成随机十六进制uuid字符串
def random_uuid() -> str:
    """
    函数功能：
        生成一个随机UUID（版本4），并返回其十六进制字符串表示。

    入参：
        无

    返回值：
        str: uuid4的hex字符串（不含连接符）。

    示例：
        >>> random_uuid()  # 'e3b0c44298fc1c149afbf4c8996fb924'
    """
    # 使用uuid.uuid4生成随机UUID，并取其hex字段（32位十六进制字符串）
    return str(uuid.uuid4().hex)


# 使用dataclass定义模型信息结构：用于/v1/models返回的数据构成
@dataclass
class Model:
    """
    类功能：
        表示一个可用的模型对象，兼容OpenAI /v1/models 接口中单个模型的字段。

    字段：
        id (str): 模型的唯一标识符（通常对应模型类型/后缀）。
        object (str): 固定为'model'，表明对象类型。
        created (int): 创建时间戳（秒）。
        owned_by (str): 模型所有者标识。
    """
    # 模型唯一标识（展示/切换用）
    id: str  # model_type
    # 固定对象类型：'model'
    object: str = 'model'
    # 创建时间：默认当前时间戳（秒）
    created: int = field(default_factory=lambda: int(time.time()))
    # 所有者：默认'ms-swift'
    owned_by: str = 'ms-swift'


# 模型列表结构：包含若干Model与对象类型
@dataclass
class ModelList:
    """
    类功能：
        表示模型列表对象，兼容OpenAI /v1/models 的返回结构。
    字段：
        data (List[Model]): 模型对象列表。
        object (str): 固定为'list'。
    """
    # 模型列表
    data: List[Model]
    # 固定对象类型：'list'
    object: str = 'list'


# 生成配置数据结构：控制生成长度、采样策略、流式与logprobs等
@dataclass
class RequestConfig:
    """NOTE: The following behavior is inconsistent with the OpenAI API.
    Default values for OpenAI:
        temperature = 1.
        top_k = -1
        top_p = 1.
        repetition_penalty = 1.
    """
    # 最大生成token数；为None时由模型最大长度-输入长度决定
    max_tokens: Optional[int] = None  # None: max_model_len - num_tokens
    # 采样温度；None表示使用服务端默认/部署参数
    temperature: Optional[float] = None
    # top_k采样；None表示使用服务端默认
    top_k: Optional[int] = None
    # 核采样概率；None表示使用服务端默认
    top_p: Optional[float] = None
    # 重复惩罚；None表示使用服务端默认
    repetition_penalty: Optional[float] = None
    # beam search的beam数量（默认1表示不启用beam搜索）
    num_beams: int = 1
    # 停止词列表：匹配任一项则停止生成
    stop: Optional[List[str]] = field(default_factory=list)
    # 随机种子（可选）
    seed: Optional[int] = None
    # 是否使用SSE流式返回
    stream: bool = False
    # 是否返回logprobs（对每个token给出概率）
    logprobs: bool = False
    # 返回top_k个logprobs（需与logprobs配合）
    top_logprobs: Optional[int] = None
    # 生成的候选数n（默认生成一个）
    n: int = 1
    # best_of策略（可选）
    best_of: Optional[int] = None
    # 出现惩罚（OpenAI兼容字段，默认0）
    presence_penalty: float = 0.
    # 频率惩罚（OpenAI兼容字段，默认0）
    frequency_penalty: float = 0.
    # 长度惩罚（beam search相关）
    length_penalty: float = 1.
    # Return token_ids additionally (non-stream)
    # 是否在非流式时额外返回token_ids（实现可选）
    return_details: bool = False

    def __post_init__(self):
        """
        函数功能：
            在数据类初始化后进行字段规范化；将stop为None的情况统一为[]，便于后续处理。
        """
        # 若stop为None，统一转为[]
        if self.stop is None:
            self.stop = []


@dataclass
class CompletionRequestMixin:
    """
    类功能：
        文本补全请求的核心字段，包含模型名与prompt文本。
    """
    # 目标模型ID
    model: str
    # 用户输入的prompt
    prompt: str


@dataclass
class EmbeddingRequestMixin:
    """
    类功能：
        嵌入请求的核心字段，包含输入文本、模型名与编码格式。
    """
    # 待编码的输入文本
    input: str
    # 目标模型ID
    model: str
    # 返回向量的编码格式（'float' 或 'base64'）
    encoding_format: Literal['float', 'base64'] = 'float'


@dataclass
class ChatCompletionRequestMixin:
    """
    类功能：
        聊天补全请求的核心字段，包含模型名、消息列表与可选工具/工具选择。
    """
    # 目标模型ID
    model: str
    # 聊天消息列表（system/user/assistant等）
    messages: Messages
    # 可选的工具定义列表（函数调用）
    tools: Optional[List[Tool]] = None
    # 工具选择策略：'none'/'auto'或指定函数名的字典
    tool_choice: Optional[Union[str, Dict]] = None

    def __post_init__(self):
        """
        函数功能：
            在初始化后，根据tools与tool_choice关系规范化字段：
            - 未指定tool_choice时：若tools为空则设为'none'，否则设为'auto'
            - 指定为'none'则清空tools；若为具体函数名，则仅保留该函数
        """
        # 默认工具策略：无工具->'none'，存在工具->'auto'
        if self.tool_choice is None:
            self.tool_choice = 'none' if self.tools is None else 'auto'

        # 若提供了工具
        if self.tools:
            # tool_choice为'none'时清空tools
            if self.tool_choice == 'none':
                self.tools = None
            # tool_choice为具体函数名（字典）时，仅保留该函数对应的工具
            elif isinstance(self.tool_choice, dict):
                name = self.tool_choice['function']['name']
                tool = next(tool for tool in self.tools if tool['function']['name'] == name)
                if tool is None:
                    raise ValueError(f"Tool choice '{name}' not found in tools.")
                self.tools = [tool]


# 多模态请求混入：统一管理images/audios/videos/objects，并提供to_base64工具
@dataclass
class MultiModalRequestMixin:
    """
    类功能：
        提供多模态字段（图像/音频/视频/对象）与辅助方法，将本地路径或PIL图像等转换为base64字符串。
    """
    # 图片列表（可为本地路径/base64/url），初始化后将转为base64字符串
    images: List[str] = field(default_factory=list)
    # 音频列表
    audios: List[str] = field(default_factory=list)
    # 视频列表
    videos: List[str] = field(default_factory=list)
    # 其他对象：自定义键到对象数组的映射
    objects: Dict[str, List[Any]] = field(default_factory=dict)

    @staticmethod
    def to_base64(mm_data: Union[str, Image.Image, bytes]) -> str:
        """
        函数功能：
            将多模态数据转换为base64字符串：
            - 若为{'bytes': b'...'}或{'path': '...'}字典，优先取'bytes'否则取'path'
            - 若为字符串且不是本地文件，认为已经是base64或url，直接返回
            - 若为字符串且是本地路径，读取字节
            - 若为PIL.Image，写入内存字节（PNG）
            - 否则认为已是字节序列

        入参：
            mm_data (Union[str, Image.Image, bytes]): 待转换的多模态数据。

        返回值：
            str: base64编码的字符串（不含data:前缀）。
        """
        # 若为带bytes/path的字典，则优先取bytes，否则取path
        if isinstance(mm_data, dict) and 'bytes' in mm_data:
            mm_data = mm_data['bytes'] or mm_data['path']
        # 若是字符串且非本地文件，视为base64或URL，直接返回
        if isinstance(mm_data, str) and not os.path.isfile(mm_data):
            # base64 or url
            return mm_data
        # 若是本地路径字符串：读取文件字节
        if isinstance(mm_data, str):
            # local_path
            with open(mm_data, 'rb') as f:
                bytes_ = f.read()
        # 若是PIL图片对象：保存到内存字节流后取字节
        elif isinstance(mm_data, Image.Image):
            bytes_io = io.BytesIO()
            mm_data.save(bytes_io, format='png')
            bytes_ = bytes_io.getvalue()
        else:
            # 否则认为已是字节序列
            bytes_ = mm_data
        # 执行base64编码并转utf-8字符串
        img_base64: str = base64.b64encode(bytes_).decode('utf-8')
        return img_base64

    def __post_init__(self):
        """
        函数功能：
            在初始化后，将images/audios/videos字段规范化为列表，并统一将其元素转换为base64字符串。
        """
        # 遍历三类多模态字段
        for key in ['images', 'audios', 'videos']:
            values = getattr(self, key)
            # 若传入为单个字符串，转为单元素列表
            if isinstance(values, str):
                values = [values]
                setattr(self, key, values)
            # 将每个元素转换为base64
            for i, val in enumerate(values):
                values[i] = self.to_base64(val)


# 文本补全完整请求：继承生成配置与多模态/补全字段
@dataclass
class CompletionRequest(RequestConfig, MultiModalRequestMixin, CompletionRequestMixin):
    """
    类功能：
        表示文本补全请求体；在初始化时同时处理生成配置与多模态字段规范化。
    """

    def __post_init__(self):
        """
        函数功能：
            初始化后依次调用父类__post_init__以规范化字段（生成配置与多模态）。
        """
        RequestConfig.__post_init__(self)
        MultiModalRequestMixin.__post_init__(self)


# 嵌入完整请求：继承生成配置与多模态/嵌入字段
@dataclass
class EmbeddingRequest(RequestConfig, MultiModalRequestMixin, EmbeddingRequestMixin):
    """
    类功能：
        表示嵌入请求体；在初始化时处理生成配置与多模态字段。
    """

    def __post_init__(self):
        """
        函数功能：
            初始化后依次调用父类__post_init__以规范化字段（生成配置与多模态）。
        """
        RequestConfig.__post_init__(self)
        MultiModalRequestMixin.__post_init__(self)

    def parse(self) -> Tuple['InferRequest', 'RequestConfig']:
        """
        函数功能：
            将当前请求数据类拆分为InferRequest与RequestConfig两个对象，便于下游推理引擎使用。

        返回值：
            Tuple[InferRequest, RequestConfig]: 对应的数据类实例。
        """
        # 转为字典，便于筛选各自字段
        data = asdict(self)
        res = []
        # 依次构造InferRequest与RequestConfig实例
        for cls_type in [InferRequest, RequestConfig]:
            parameters = set(f.name for f in fields(cls_type))
            _data = {k: v for k, v in data.items() if k in parameters}
            res.append(cls_type(**_data))
        return tuple(res)


# 聊天完整请求：继承生成配置、多模态与聊天字段，并进行多模态内容转base64
@dataclass
class ChatCompletionRequest(RequestConfig, MultiModalRequestMixin, ChatCompletionRequestMixin):
    """
    类功能：
        表示聊天补全请求体；在初始化时处理生成配置、多模态与工具选择，并将消息中的多模态内容转为base64。
    """

    def __post_init__(self):
        """
        函数功能：
            依次调用各父类的__post_init__进行字段规范化，并在最后将消息中的多模态内容转为base64。
        """
        RequestConfig.__post_init__(self)
        MultiModalRequestMixin.__post_init__(self)
        ChatCompletionRequestMixin.__post_init__(self)
        self.convert_to_base64()

    def convert_to_base64(self):
        """
        函数功能：
            遍历messages中每条消息的content；若包含多模态项（如image_url等），
            则将其本地路径或PIL图片转换为data URI（形如'data:image/jpeg;base64,...'）并回写。
        """
        # 遍历所有消息
        for message in self.messages:
            content = message['content']
            # 纯文本直接跳过
            if isinstance(content, str):
                continue
            # content为由多个多模态片段组成的列表
            for item in content:
                key: str = item['type']
                # 文本片段无需处理
                if key == 'text':
                    continue
                # 记录原始键名（可能是 image_url/audio_url 等）
                key_origin = key
                value = item[key]
                # *_url 形式去掉后缀，得到真实的模态键名
                if key.endswith('_url'):
                    key = key[:-len('_url')]
                # 标记值是否为字典（{'url': ...}）以便回写保持结构
                is_dict = False
                if isinstance(value, dict):
                    is_dict = True
                    value = value['url']
                # 若已经是data:或http开头或超长（疑似base64），则跳过
                if isinstance(value, str) and (value.startswith('data:') or value.startswith('http')
                                               or len(value) > 200):
                    continue
                # 判断后缀：本地路径则取文件后缀；PIL图片默认用jpeg
                # local_path / PIL.Image
                if isinstance(value, str) and os.path.isfile(value):
                    suffix = os.path.splitext(value)[1][1:].lower()
                elif isinstance(value, Image.Image):
                    suffix = 'jpeg'
                else:
                    raise ValueError(f'value: {value}')
                # 执行base64转换并拼接为data URI
                mm_data_base64 = self.to_base64(value)
                new_value = f'data:{key}/{suffix};base64,{mm_data_base64}'
                # 若原值为字典，则保持字典结构
                if is_dict:
                    new_value = {'url': new_value}
                # 写回到原键位
                item[key_origin] = new_value

    def parse(self) -> Tuple['InferRequest', 'RequestConfig']:
        """
        函数功能：
            将聊天请求拆分为InferRequest与RequestConfig两个数据类实例。
        """
        data = asdict(self)
        res = []
        for cls_type in [InferRequest, RequestConfig]:
            parameters = set(f.name for f in fields(cls_type))
            _data = {k: v for k, v in data.items() if k in parameters}
            res.append(cls_type(**_data))
        return tuple(res)

    @classmethod
    def from_cmpl_request(cls, cmpl_request: Union[CompletionRequest, EmbeddingRequest]) -> 'ChatCompletionRequest':
        """
        函数功能：
            将Completion或Embedding请求转换为ChatCompletion请求：
            - 将prompt或input映射为单条user消息的content
            - 移除encoding_format等无关字段
        """
        # 将原请求转为字典以便修改
        cmpl_request = asdict(cmpl_request)
        # prompt优先；若无prompt则取input
        if 'prompt' in cmpl_request:
            prompt = cmpl_request.pop('prompt')
        else:
            prompt = cmpl_request.pop('input')
        # 组装messages为单条user消息
        cmpl_request['messages'] = [{'role': 'user', 'content': prompt}]
        # 移除embedding特有字段
        if 'encoding_format' in cmpl_request:
            cmpl_request.pop('encoding_format')
        # 构造并返回ChatCompletionRequest实例
        return cls(**cmpl_request)


# 使用信息：记录prompt/生成token数
@dataclass
class UsageInfo:
    """
    类功能：
        记录一次请求中提示、生成与总token数量，用于计费/统计。
    """
    # 提示词token数
    prompt_tokens: int
    # 生成的token数
    completion_tokens: int
    # 总token数
    total_tokens: int


# 函数描述数据结构：用于工具调用结果（function call）
@dataclass
class Function:
    """
    类功能：
        表示一个可调用函数（工具），包含函数名与JSON字符串化的参数。
    """
    # 函数名
    name: str
    # 参数（可为任意对象，初始化后将被转为JSON字符串）
    arguments: Optional[str]

    def __post_init__(self):
        """
        函数功能：
            在初始化后规范化字段：
            - 将arguments转换为JSON字符串
            - 去除name与arguments两端空白
        """
        # 非字符串的arguments转为JSON字符串
        if not isinstance(self.arguments, str):
            self.arguments = json.dumps(self.arguments)
        # 去除函数名与参数字符串的首尾空白
        self.name = self.name.strip()
        self.arguments = self.arguments.strip()


# 工具调用结构：包含函数实体、类型与自动生成的ID
@dataclass
class ChatCompletionMessageToolCall:
    """
    类功能：
        表示一次消息的工具调用项，包含函数与自动生成的唯一ID。
    """
    # 函数实体（名称+参数）
    function: Function
    # 类型固定为'function'
    type: str = 'function'
    # 自动生成工具调用ID
    id: str = field(default_factory=lambda: f'toolcall-{random_uuid()}')


# 聊天消息数据结构：角色+内容+可选工具调用
@dataclass
class ChatMessage:
    """
    类功能：
        表示一条聊天消息，包含角色、内容与可选工具调用（仅assistant可带tool_calls）。
    """
    # 角色：system/user/assistant
    role: Literal['system', 'user', 'assistant']
    # 内容：字符串或多模态列表（或数值信息）
    content: Union[str, List[Dict[str, Any]], int, float]
    # 可选工具调用列表
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


# 聊天响应单项：包含message与finish_reason、logprobs等
@dataclass
class ChatCompletionResponseChoice:
    """
    类功能：
        表示聊天响应的一个choice项，包含消息、结束原因与可选的logprobs信息。
    """
    # 序号索引
    index: int
    # 消息内容（ChatMessage）
    message: ChatMessage
    # 结束原因：'stop'/'length'/None
    finish_reason: Literal['stop', 'length', None]
    # 可选logprobs信息
    logprobs: Optional[Dict[str, List[Dict[str, Any]]]] = None
    # 可选返回的token id列表
    token_ids: Optional[List[int]] = None

    # NOTE: 返回值类型上加引号是为了 前向引用（forward reference），避免类未定义时报错
    def to_cmpl_choice(self) -> 'CompletionResponseChoice':
        """
        函数功能：
            将聊天响应choice转换为文本补全响应choice（去除tool_calls）。
        """
        # 深拷贝避免修改原对象
        self = deepcopy(self)
        # 确保无tool_calls方可转换
        assert not self.message.tool_calls, f'message: {self.message}'
        # 返回补全响应choice
        return CompletionResponseChoice(self.index, self.message.content, self.finish_reason, self.logprobs)


# 嵌入响应数据单项
@dataclass
class EmbeddingResponseData:
    """
    类功能：
        表示一次嵌入响应中的单条数据，包括对象类型、索引与向量本体。
    """
    # 对象类型固定为'embedding'
    object: str = 'embedding'
    # 索引序号
    index: int = 0
    # 向量本体（列表），可为float或base64字符串，具体由encoding_format决定
    embedding: List[str] = field(default_factory=lambda: [])


# 嵌入响应整体：包含模型名、数据列表与usage
@dataclass
class EmbeddingResponse:
    """
    类功能：
        嵌入接口的响应对象，包含模型名、数据列表与使用信息。
    """
    # 模型名
    model: str
    # 数据列表
    data: List[EmbeddingResponseData]
    # 使用信息（token数）
    usage: UsageInfo
    # 自动生成的响应ID
    id: str = field(default_factory=lambda: f'chatcmpl-{random_uuid()}')
    # 对象类型固定为'list'
    object: str = 'list'
    # 创建时间戳
    created: int = field(default_factory=lambda: int(time.time()))


# Rollout 响应扩展：在choice上增加多轮消息、图片与多轮信息
@dataclass
class RolloutResponseChoice(ChatCompletionResponseChoice):
    """
    类功能：
        在标准聊天响应choice基础上增加强化学习/多轮采样相关信息（messages/images/multi_turn_infos）。
    """
    # 还原的上下文消息（可选）
    messages: Optional[Messages] = None
    # 关联的图片（可选）
    images: Optional[List[str]] = None
    # 多轮对话的附加信息
    multi_turn_infos: Dict[str, Any] = field(default_factory=dict)


# Gym环境的Rollout扩展：包含轨迹信息与奖励
@dataclass
class GymRolloutResponseChoice(RolloutResponseChoice):
    """
    类功能：
        在RolloutResponseChoice上扩展强化学习环境信息，如轨迹ID与奖励等。
    """
    # 轨迹ID
    trajectory_id: str = None
    # 总奖励
    total_reward: float = 0.0
    # 每步奖励列表
    step_rewards: List[float] = None
    # 轨迹附加信息
    trajectory_info: List[Dict[str, Any]] = None


# 文本补全响应单项
@dataclass
class CompletionResponseChoice:
    """
    类功能：
        表示文本补全响应的一个choice项，包含文本、结束原因与可选logprobs。
    """
    # 序号索引
    index: int
    # 生成的文本内容
    text: str
    # 结束原因
    finish_reason: Literal['stop', 'length', None]
    # 可选logprobs
    logprobs: Optional[Dict[str, List[Dict[str, Any]]]] = None


# 聊天响应整体：包含多个choice、usage与元信息
@dataclass
class ChatCompletionResponse:
    """
    类功能：
        标准聊天响应结构，包含模型名、choices、使用信息与可选提示token ids。
    """
    # 模型名
    model: str
    # choice列表（可为多类型扩展项）
    choices: List[Union[ChatCompletionResponseChoice, RolloutResponseChoice, GymRolloutResponseChoice]]
    # 使用信息
    usage: UsageInfo
    # 自动生成ID
    id: str = field(default_factory=lambda: f'chatcmpl-{random_uuid()}')
    # 对象类型固定为'chat.completion'
    object: str = 'chat.completion'
    # 创建时间戳
    created: int = field(default_factory=lambda: int(time.time()))
    # 可选：提示词对应的token id列表
    prompt_token_ids: Optional[List[int]] = None

    def to_cmpl_response(self) -> 'CompletionResponse':
        """
        函数功能：
            将聊天响应整体转换为文本补全响应整体（映射choice并转换ID前缀）。
        """
        # 深拷贝避免副作用
        self = deepcopy(self)
        # 将每个choice转换为completion choice
        choices = [choice.to_cmpl_choice() for choice in self.choices]
        # 将chatcmpl-前缀替换为cmpl-保持风格一致
        id_ = f'cmpl{self.id[len("chatcmpl"):]}'
        # 构造CompletionResponse并返回
        return CompletionResponse(self.model, choices, self.usage, id_, created=self.created)


# 文本补全响应整体
@dataclass
class CompletionResponse:
    """
    类功能：
        标准文本补全响应结构，包含模型名、choices与使用信息等。
    """
    # 模型名
    model: str
    # choice列表
    choices: List[CompletionResponseChoice]
    # 使用信息
    usage: UsageInfo
    # 自动生成ID
    id: str = field(default_factory=lambda: f'cmpl-{random_uuid()}')
    # 对象类型固定为'text_completion'
    object: str = 'text_completion'
    # 创建时间戳
    created: int = field(default_factory=lambda: int(time.time()))


# 流式增量消息结构：仅包含变更delta
@dataclass
class DeltaMessage:
    """
    类功能：
        表示流式响应中的增量消息片段，包含可选角色、增量文本与工具调用。
    """
    # 可选角色（通常只在首帧出现）
    role: Literal['system', 'user', 'assistant', None] = None
    # 增量文本内容
    content: Optional[str] = None
    # 可选工具调用（仅assistant）
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


# 聊天流式响应单项
@dataclass
class ChatCompletionResponseStreamChoice:
    """
    类功能：
        表示聊天流式响应的一个choice增量，包含delta与结束原因。
    """
    # 序号索引
    index: int
    # 增量消息片段
    delta: DeltaMessage
    # 结束原因
    finish_reason: Literal['stop', 'length', None]
    # 可选logprobs
    logprobs: Optional[Dict[str, List[Dict[str, Any]]]] = None

    def to_cmpl_choice(self) -> 'CompletionResponseStreamChoice':
        """
        函数功能：
            将聊天流式choice转换为文本流式choice（取delta.content）。
        """
        # 深拷贝避免副作用
        self = deepcopy(self)
        # 确保无工具调用
        assert not self.delta.tool_calls
        # 返回文本流式choice
        return CompletionResponseStreamChoice(self.index, self.delta.content, self.finish_reason, self.logprobs)


# 文本流式响应单项
@dataclass
class CompletionResponseStreamChoice:
    """
    类功能：
        表示文本流式响应的一个choice增量，包含增量文本与结束原因。
    """
    # 序号索引
    index: int
    # 增量文本
    text: str
    # 结束原因
    finish_reason: Literal['stop', 'length', None]
    # 可选logprobs
    logprobs: Optional[Dict[str, List[Dict[str, Any]]]] = None


# 聊天流式响应整体
@dataclass
class ChatCompletionStreamResponse:
    """
    类功能：
        表示聊天流式响应整体对象，包含choices增量列表与可选usage。
    """
    # 模型名
    model: str
    # choices增量列表
    choices: List[ChatCompletionResponseStreamChoice]
    # 可选使用信息
    usage: Optional[UsageInfo] = None
    # 自动生成ID
    id: str = field(default_factory=lambda: f'chatcmpl-{random_uuid()}')
    # 对象类型固定为'chat.completion.chunk'
    object: str = 'chat.completion.chunk'
    # 创建时间戳
    created: int = field(default_factory=lambda: int(time.time()))

    def to_cmpl_response(self) -> 'CompletionStreamResponse':
        """
        函数功能：
            将聊天流式整体转换为文本流式整体（映射choices并替换ID前缀）。
        """
        # 深拷贝避免副作用
        self = deepcopy(self)
        # 将每个choice转换为文本流式choice
        choices = [choice.to_cmpl_choice() for choice in self.choices]
        # 替换ID前缀
        id_ = f'cmpl{self.id[len("chatcmpl"):]}'
        # 返回文本流式整体
        return CompletionStreamResponse(self.model, choices, self.usage, id_, created=self.created)


# 文本流式响应整体
@dataclass
class CompletionStreamResponse:
    """
    类功能：
        文本流式响应整体对象。
    """
    # 模型名
    model: str
    # choices增量列表
    choices: List[CompletionResponseStreamChoice]
    # 可选使用信息
    usage: Optional[UsageInfo] = None
    # 自动生成ID
    id: str = field(default_factory=lambda: f'cmpl-{random_uuid()}')
    # 对象类型固定为'text_completion.chunk'
    object: str = 'text_completion.chunk'
    # 创建时间戳
    created: int = field(default_factory=lambda: int(time.time()))


class InitCommunicatorRequest(BaseModel):
    """
    类功能：
        初始化分布式通信的请求体（pydantic模型），定义初始化通信所需的基本参数，如host/port/world_size与可选客户端设备UUID。
    """
    # 主机地址
    host: str
    # 端口
    port: int
    # 全局并行规模
    world_size: int
    # 可选客户端设备UUID
    client_device_uuid: Optional[str] = None


class UpdateWeightsRequest(BaseModel):
    """
    类功能：
        更新权重的请求体（pydantic模型），定义一次权重更新的元信息：张量名称、dtype与shape。
    """
    # 张量名称
    name: str
    # 张量数据类型（字符串表示）
    dtype: str
    # 张量形状（列表）
    shape: list[int]
