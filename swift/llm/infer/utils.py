"""
模块功能概述：
    本模块为推理交互与模型准备提供通用工具方法与数据结构。
    主要包含：
    - 命令行交互状态 `InferCliState`（记录对话消息、系统提示、多模态资源与输入模式）
    - 适配器/LoRA 等微调权重的装配 `prepare_adapter`
    - 组合模型与模板并更新生成配置的便捷函数 `prepare_model_template`

使用场景：
    - 交互式 CLI 推理：缓存会话消息、支持用户输入多行文本/多模态资源占位标签并收集实际路径/URL
    - 模型部署/推理前准备：加载增量权重、修正生成配置终止符、返回模板对象

注意：
    - 每一行代码均附有注释，说明其用途与上下文含义，便于快速上手与维护。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.
import re  # 正则表达式库：用于在查询文本中查找多模态占位标签
from copy import deepcopy  # 深拷贝工具：避免就地修改原始对象
from dataclasses import dataclass, field  # 数据类与字段工厂：定义轻量对象与默认值
from typing import List, Literal, Optional  # 类型注解：列表、字面量与可选类型

from swift.llm.utils import update_generation_config_eos_token  # 更新生成配置的 eos token
from swift.plugin import extra_tuners  # 可扩展的外部调优器注册表：按键选择特定 Tuner
from swift.tuners import Swift  # 默认 Tuner：当未匹配到外部调优器时使用
from swift.utils import get_logger  # 统一的日志记录器工厂
from ..utils import Messages  # 消息类型别名：兼容 OpenAI 风格的多轮消息结构

logger = get_logger()  # 模块级日志器：打印交互提示与状态信息


@dataclass  # 声明数据类：自动生成 __init__/__repr__/__eq__ 等方法
class InferCliState:
    """类功能：
    维护交互式推理（CLI）时的上下文状态，包括系统提示、对话消息、多模态资源与输入模式。

    关键字段：
    - system: 系统提示词（None 表示使用默认 system，'' 表示不使用 system）
    - messages: 当前会话消息（不含 system 条目，入参/输出时可临时插入）
    - images/audios/videos: 多模态资源列表（路径或 URL）
    - multiline_mode: 是否启用多行输入模式（以 `#` 结尾结束输入）
    - input_system: 下次输入是否采集 system 内容（用于动态切换 system）
    """
    # None: use default-system. '': not use system.
    system: Optional[str] = None  # 系统提示词；None=使用默认；''=不使用 system
    messages: Messages = field(default_factory=list)  # not including system（对话消息缓存，不含 system）

    images: List[str] = field(default_factory=list)  # 图片资源列表：存储路径或 URL
    audios: List[str] = field(default_factory=list)  # 音频资源列表：存储路径或 URL
    videos: List[str] = field(default_factory=list)  # 视频资源列表：存储路径或 URL

    multiline_mode: bool = False  # 是否启用多行输入（以 `#` 结束）
    input_system: bool = False  # 是否在下一次输入时采集/设置 system 提示

    def clear(self):  # 重置对话与多模态缓存
        """函数功能：清空当前会话消息与多模态资源缓存。

        示例：
            >>> state.clear()  # 清空 messages/images/audios/videos
        """
        self.messages = []  # 清空消息列表
        self.images = []  # 清空图片列表
        self.audios = []  # 清空音频列表
        self.videos = []  # 清空视频列表

    def add_query(self, query: str) -> None:  # 追加用户/工具消息
        """函数功能：将一条查询文本追加为消息（支持以 `tool:` 前缀标记工具消息）。

        入参：
            query (str): 原始输入文本；以 `tool:` 开头视为工具消息。

        返回值：
            None

        示例：
            >>> state.add_query('Hello')
            >>> state.add_query('tool:{"result": 1}')
        """
        role = 'user'  # 默认视为用户消息
        if query.startswith('tool:'):  # 若以 tool: 开头则标记为工具消息
            role = 'tool'  # 角色改为 tool
            query = query[len('tool:'):]  # 去除前缀，保留真实内容
        self.messages.append({'role': role, 'content': query})  # 追加到消息列表

    def add_response(self, response: str) -> None:  # 追加助手回复
        """函数功能：将助手生成的回复追加到消息列表。

        入参：
            response (str): 助手文本回复。

        返回值：
            None

        示例：
            >>> state.add_response('好的，我来解释……')
        """
        self.messages.append({'role': 'assistant', 'content': response})  # 追加 assistant 消息

    def to_dict(self):  # 状态序列化为请求字典
        """函数功能：将状态对象转换为推理请求需要的字典结构（如 OpenAI Chat 格式）。

        入参：
            无

        返回值：
            dict: 包含 messages/images/audios/videos 的请求字典；如设置了 system 会头插入。

        示例：
            >>> payload = state.to_dict()
        """    
        infer_state = deepcopy(self)  # 深拷贝，避免修改原状态对象
        if infer_state.system is not None:  # 若指定了 system，则在首位插入 system 消息
            infer_state.messages.insert(0, {'role': 'system', 'content': infer_state.system})  # 头插入 system
        return {
            'messages': infer_state.messages,  # 对话消息序列
            'images': infer_state.images,  # 图片列表
            'audios': infer_state.audios,  # 音频列表
            'videos': infer_state.videos  # 视频列表
        }

    def input_mm_data(self) -> None:  # 交互式输入多模态数据（根据占位标签提示用户）
        """函数功能：
        根据最近一条消息中的多模态占位标签（<image>/<video>/<audio>），提示用户输入对应的路径或 URL，
        并追加到各自的资源列表中。

        示例：
            >>> state.add_query('请分析这张图片 <image> 和这段音频 <audio>')
            >>> state.input_mm_data()  # 按提示输入路径或 URL
        """

        def _input_mm_file(mm_type: Literal['image', 'video', 'audio']) -> str:  # 子函数：读取单个模态输入
            a_an = 'an' if mm_type[0] in {'i', 'a'} else 'a'  # 英语语法：以元音开头用 an，否则用 a
            return input(f'Input {a_an} {mm_type} path or URL <<< ')  # 提示用户输入路径或 URL

        mm_types = ['image', 'video', 'audio']  # 支持的多模态占位标签类型
        query = self.messages[-1]['content']  # 获取最近一条用户输入文本
        mm_tags = re.findall('|'.join(f'<{mm_type}>' for mm_type in mm_types), query)  # 在文本中查找占位标签
        # mm_tag -> mm_type/mm_key
        mm_mapping = {f'<{mm_type}>': (mm_type, f'{mm_type}s') for mm_type in mm_types}  # 映射：占位→(类型, 属性名)
        for mm_tag in mm_tags:  # 遍历每个占位标签
            mm_type, mm_key = mm_mapping[mm_tag]  # 解析模态类型与在状态中的属性名
            mm_val = getattr(self, mm_key)  # 取得对应的资源列表引用
            mm_val.append(_input_mm_file(mm_type))  # 追加用户输入的路径或 URL

    @staticmethod  # 声明静态方法：不依赖实例状态
    def _input_multiline(prompt: str) -> str:  # 多行输入助手：用 # 换行结束
        """函数功能：读取多行输入，直至用户以 `#` 独占一行结尾为止。

        入参：
            prompt (str): 首行提示符。

        返回值：
            str: 拼接后的完整多行文本（去除结尾的 `#` 行）。

        示例：
            >>> InferCliState._input_multiline('<<< ')
        """
        query = ''  # 聚合最终输出的多行文本
        stop_words = '#\n'  # 结束标记：单行 `#` 并回车
        while True:  # 循环读取每一行
            text = f'{input(prompt)}\n'  # 读取一行并补上换行符
            prompt = ''  # 第二行及之后不再重复提示符
            if text.endswith(stop_words):  # 若以结束标记结尾
                query += text[:-len(stop_words)]  # 追加除去结束标记的内容
                break  # 结束循环
            query += text  # 累加当前行
        return query  # 返回聚合文本

    def input_text(self) -> str:  # 读取一段文本输入（单行或多行）
        """函数功能：根据当前模式（单行/多行）读取用户文本输入。

        入参：
            无

        返回值：
            str: 用户输入的文本。

        示例：
            >>> state.input_text()
        """
        if self.multiline_mode:  # 多行模式：使用专用读取器
            addi_prompt = '[MS]' if self.input_system else '[M]'  # 提示符：M=multiline，MS=multiline+system
            text = InferCliState._input_multiline(f'<<<{addi_prompt} ')  # 读取多行输入
        else:  # 单行模式：直接 input
            addi_prompt = '[S]' if self.input_system else ''  # 提示符：S=本次输入为 system
            text = input(f'<<<{addi_prompt} ')  # 读取单行输入
        return text  # 返回文本

    def check_query(self, query: str) -> Optional[str]:  # 解析并响应特殊命令
        """函数功能：解析用户输入的特殊指令，更新状态并决定是否继续作为对话内容。

        入参：
            query (str): 原始用户输入。

        返回值：
            Optional[str]: 返回标准化后的文本；若处理了控制命令则返回 None。

        已支持命令：
            - 'clear'：清空会话与多模态资源
            - ''（空行）：忽略
            - 'reset-system'：切换到设置 system 的输入状态
            - 'multi-line' / 'single-line'：切换多行/单行输入模式
            - 当 input_system=True 时：输入 'default-system' 使用默认 system；其他则设置为自定义 system
        """
        query_std = query.strip().lower()  # 标准化输入：去空白并小写
        if self.input_system:  # 若当前要采集/设置 system
            if query == 'default-system':  # 特殊关键词：回退到默认 system
                self.system = None  # 使用默认 system
            else:
                self.system = query  # 设置自定义 system 内容
            self.input_system = False  # 关闭设置 system 模式
            query_std = 'clear'  # 设置完成后清空消息，避免旧上下文影响
        if query_std == 'clear':  # 清空对话
            self.clear()  # 执行清空
            return  # 不返回对话内容
        if query_std == '':  # 空输入：忽略
            return  # 不追加消息
        if query_std == 'reset-system':  # 切换到设置 system 的输入模式
            self.input_system = True  # 下一次输入将写入 system
            return  # 不作为对话
        if query_std == 'multi-line':  # 开启多行模式
            self.multiline_mode = True  # 标记为多行
            logger.info('End multi-line input with `#`.')  # 提示结束方式
            logger.info('Input `single-line` to switch to single-line input mode.')  # 提示切换命令
            return  # 不作为对话
        if query_std == 'single-line':  # 切回单行模式
            self.multiline_mode = False  # 关闭多行
            return  # 不作为对话
        return query  # 普通文本：返回用于后续追加到消息


def prepare_adapter(args, model, adapters=None):  # 准备微调适配器/权重并装载到模型
    """函数功能：根据训练/部署参数，为模型注入对应的适配器（如 LoRA/Unsloth 等）。

    入参：
        args: 包含 `tuner_backend`、`train_type`、`adapters`、`model_meta` 等配置的对象。
        model: 基础模型实例（transformers 模型等）。
        adapters: 可选的适配器路径/标识列表；不提供则回退到 `args.adapters`。

    返回值：
        Any: 注入适配器后的模型实例。

    示例：
        >>> model = prepare_adapter(args, model)
    """
    if args.tuner_backend == 'unsloth':  # 若采用 Unsloth 推理加速后端
        if args.model_meta.is_multimodal:  # 多模态模型使用视觉模型封装
            from unsloth import FastVisionModel as UnslothModel  # 延迟导入，避免非必要依赖
        else:  # 纯文本模型使用语言模型封装
            from unsloth import FastLanguageModel as UnslothModel  # 延迟导入
        UnslothModel.for_inference(model)  # 将模型包装为推理优化版本（权重不修改）
        return model  # 直接返回处理后的模型
    if args.train_type in extra_tuners:  # 若 train_type 在外部可扩展 Tuner 注册表
        tuner = extra_tuners[args.train_type]  # 选择对应的 Tuner 类
    else:
        tuner = Swift  # 默认使用 Swift Tuner
    # compat deploy
    adapters = adapters or args.adapters  # 适配器列表：优先使用入参，否则使用全局配置
    for adapter in adapters:  # 逐个装载适配器权重
        model = tuner.from_pretrained(model, adapter)  # 从适配器路径加载并注入权重
    if args.train_type == 'bone':  # 针对 Bone 训练类型的兼容（float32/bfloat16 matmul 问题）
        # Bone has a problem of float32 matmul with bloat16 in `peft==0.14.0`
        model.to(model.dtype)  # 将模型移动到其自身 dtype，避免混合精度 matmul 问题
    return model  # 返回完成适配器注入的模型实例


def prepare_model_template(args, **kwargs):  # 准备模型与模板，并同步更新生成配置
    """函数功能：构造模型与处理器，装载适配器，生成模板，并更新模型的终止符配置。

    入参：
        args: 含有 `get_model_processor`、`get_template` 等方法的参数对象。
        **kwargs: 透传给 `get_model_processor` 的可选配置（如 revision、device 等）。

    返回值：
        Tuple[ModelLike, Any]: (模型实例, 模板对象)

    示例：
        >>> model, template = prepare_model_template(args)
    """
    model, processor = args.get_model_processor(**kwargs)  # 获取基础模型与对应处理器（可能是分词器/图像处理器）
    model = prepare_adapter(args, model)  # 注入适配器/增量权重，得到可推理模型
    template = args.get_template(processor)  # 基于处理器推断模板（如聊天/补全模板）
    update_generation_config_eos_token(model.generation_config, template)  # 根据模板同步生成配置中的 eos token
    return model, template  # 返回模型与模板
