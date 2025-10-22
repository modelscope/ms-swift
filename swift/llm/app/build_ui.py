"""模块功能概述：
该模块实现了 Swift 框架的 Gradio Web UI 界面构建功能，负责创建和配置可视化对话界面。

核心功能：
1. build_ui 函数：构建完整的 Gradio 界面，包括聊天框、文本输入、按钮等组件；
2. 会话管理函数：clear_session、modify_system_session，用于清空和修改对话会话；
3. 消息处理函数：_history_to_messages，将 Gradio 历史记录转换为标准消息格式；
4. 文本解析函数：_parse_text，对特殊字符进行转义以正确显示在 HTML 中；
5. 模型对话函数：model_chat，异步调用推理服务并更新界面；
6. 用户输入处理函数：add_text、add_file，处理文本和文件上传。

应用场景：
- 为 LLM 模型创建可视化对话界面；
- 支持多模态输入（文本、图像、音频、视频等）；
- 支持流式和非流式响应；
- 提供系统提示词配置、历史记录管理等功能。

典型使用：
    # 基本用法
    >>> from swift.llm.app.build_ui import build_ui
    >>> demo = build_ui(base_url='http://localhost:8000/v1', model='qwen/Qwen-7B-Chat')
    >>> demo.launch()  # 启动 Gradio 服务器
    
    # 自定义配置
    >>> from swift.llm import RequestConfig
    >>> config = RequestConfig(temperature=0.7, stream=True)
    >>> demo = build_ui(
    ...     base_url='http://localhost:8000/v1',
    ...     model='qwen/Qwen-VL-Chat',
    ...     request_config=config,
    ...     is_multimodal=True,
    ...     studio_title='我的智能助手',
    ...     lang='zh',
    ...     default_system='你是一个有帮助的AI助手'
    ... )
"""
# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，标注代码版权所有者
from functools import partial  # 引入 partial 函数，用于固定函数的部分参数（如固定 model_chat 的 client 和 model 参数）
from typing import Literal, Optional  # 引入类型注解，Literal 用于限定字符串字面量类型（如 'en' 或 'zh'），Optional 表示可选参数

import gradio as gr  # 引入 Gradio 库，用于构建和管理 Web UI 界面组件

from swift.utils import get_file_mm_type  # 引入多模态文件类型判断函数，用于识别文件是图像、音频还是视频
from ..utils import History  # 引入 History 类型别名，表示对话历史记录的数据结构（通常是 List[List[str]]）
from .locale import locale_mapping  # 引入本地化映射字典，用于根据语言（中文/英文）显示不同的按钮文本


def clear_session():
    """函数功能：
    清空当前对话会话，重置所有界面状态（文本框、聊天历史、真实历史）。
    
    参数：
        无
    
    返回值：
        tuple: 包含三个元素的元组 (textbox, chatbot, history_state)
            - textbox (str): 空字符串，用于清空输入框
            - chatbot (list): 空列表，用于清空聊天显示区域
            - history_state (list): 空列表，用于清空内部历史记录状态
    
    实际使用示例：
        示例 1：在 Gradio 按钮点击事件中使用
        >>> clear_history_button.click(clear_session, [], [textbox, chatbot, history_state])
        # 用户点击"清空历史"按钮后，输入框、聊天框、历史记录全部被清空
        
        示例 2：手动调用清空会话
        >>> new_textbox, new_chatbot, new_history = clear_session()
        >>> print(new_textbox)  # 输出: ''
        >>> print(new_chatbot)  # 输出: []
        >>> print(new_history)  # 输出: []
    """
    return '', [], []  # 返回空字符串（清空输入框）、空列表（清空聊天显示）、空列表（清空历史状态）


def modify_system_session(system: str):
    """函数功能：
    修改系统提示词并重置对话会话，确保新的系统提示词生效。
    
    参数：
        system (str): 新的系统提示词内容（如 'You are a helpful assistant'），若为空则使用空字符串
    
    返回值：
        tuple: 包含四个元素的元组 (system_state, textbox, chatbot, history_state)
            - system_state (str): 更新后的系统提示词（空值会被转换为空字符串）
            - textbox (str): 空字符串，用于清空输入框
            - chatbot (list): 空列表，用于清空聊天显示
            - history_state (list): 空列表，用于清空历史记录
    
    实际使用示例：
        示例 1：更新系统提示词
        >>> modify_system.click(modify_system_session, [system_input], 
        ...                     [system_state, textbox, chatbot, history_state])
        # 用户在系统输入框输入"你是一个专业的翻译助手"，点击"修改系统"按钮
        # 系统提示词更新，对话历史被清空，下一轮对话将使用新的系统提示词
        
        示例 2：手动调用
        >>> new_system, new_textbox, new_chatbot, new_history = modify_system_session('You are a poet')
        >>> print(new_system)  # 输出: 'You are a poet'
        >>> print(len(new_history))  # 输出: 0（历史已清空）
    """
    system = system or ''  # 若传入的 system 为 None 或空字符串，则使用空字符串（避免 None 值导致的错误）
    return system, '', [], []  # 返回更新后的系统提示词、空输入框、空聊天显示、空历史记录


def _history_to_messages(history: History, system: Optional[str]): 
    """函数功能：
    私有方法，将 Gradio 的对话历史记录（History 格式）转换为标准的消息列表格式（OpenAI API 格式）。
    支持文本、图像、视频、音频等多模态内容的转换。
    
    参数：
        history (History): Gradio 对话历史记录，格式为 List[List[Union[str, tuple]]]
            - 单轮对话示例：[['用户问题', '助手回答'], ...]
            - 多模态示例：[[('image.jpg',), None], ['文本问题', '助手回答']]
        system (Optional[str]): 系统提示词，若不为 None 则添加到消息列表开头
    
    返回值：
        list: 标准消息列表，每个消息为字典，包含 'role' 和 'content' 字段
            示例：[
                {'role': 'system', 'content': '你是助手'},
                {'role': 'user', 'content': [{'type': 'text', 'text': '你好'}]},
                {'role': 'assistant', 'content': '你好！有什么可以帮助你的吗？'}
            ]
    
    实际使用示例：
        示例 1：纯文本对话转换
        >>> history = [['你好', '你好！'], ['今天天气如何', '今天天气很好']]
        >>> messages = _history_to_messages(history, 'You are helpful')
        >>> print(messages)
        # [{'role': 'system', 'content': 'You are helpful'},
        #  {'role': 'user', 'content': [{'type': 'text', 'text': '你好'}]},
        #  {'role': 'assistant', 'content': '你好！'},
        #  {'role': 'user', 'content': [{'type': 'text', 'text': '今天天气如何'}]},
        #  {'role': 'assistant', 'content': '今天天气很好'}]
        
        示例 2：多模态对话转换（包含图像）
        >>> history = [[('cat.jpg',), None], ['这是什么动物', '这是一只猫']]
        >>> messages = _history_to_messages(history, None)
        >>> print(messages[0]['content'])
        # [{'type': 'image', 'image': 'cat.jpg'}, {'type': 'text', 'text': '这是什么动物'}]
    """
    messages = []  # 初始化空的消息列表，用于存储转换后的标准消息
    if system is not None:  # 若提供了系统提示词（不为 None）
        messages.append({'role': 'system', 'content': system})  # 将系统提示词作为第一条消息添加到列表中，角色为 'system'
    content = []  # 初始化空的内容列表，用于累积同一轮用户消息的多个部分（文本、图像等）
    for h in history:  # 遍历对话历史中的每一轮对话（h 为 [用户输入, 助手回复] 的列表或元组）
        assert isinstance(h, (list, tuple))  # 断言 h 必须是列表或元组类型（确保数据格式正确）
        if isinstance(h[0], tuple):  # 若用户输入部分是元组（表示这是一个文件上传，如 ('image.jpg',)）
            assert h[1] is None  # 断言助手回复部分必须为 None（文件上传消息后面不应立即有回复）
            file_path = h[0][0]  # 从元组中提取文件路径字符串（元组的第一个元素）
            try:  # 尝试识别文件的多模态类型（图像、音频、视频）
                mm_type = get_file_mm_type(file_path)  # 调用工具函数获取文件的多模态类型（返回 'image'、'audio' 或 'video'）
                content.append({'type': mm_type, mm_type: file_path})  # 将多模态内容添加到内容列表，格式为 {'type': 'image', 'image': 'path/to/file.jpg'}
            except ValueError:  # 若文件类型无法识别（不是图像、音频、视频），作为文本文件处理
                with open(file_path, 'r', encoding='utf-8') as f:  # 以 UTF-8 编码打开文件进行读取
                    content.append({'type': 'text', 'text': f.read()})  # 读取文件内容并作为文本添加到内容列表
        else:  # 否则（用户输入是字符串，表示这是普通文本消息）
            content.append({'type': 'text', 'text': h[0]})  # 将用户文本添加到内容列表，格式为 {'type': 'text', 'text': '用户问题'}
            messages.append({'role': 'user', 'content': content})  # 将累积的用户内容（可能包含文本、图像等）作为一条用户消息添加到消息列表
            if h[1] is not None:  # 若助手回复部分不为 None（表示有助手的回复内容）
                messages.append({'role': 'assistant', 'content': h[1]})  # 将助手回复添加到消息列表，角色为 'assistant'，内容为纯字符串
            content = []  # 重置内容列表为空，准备累积下一轮用户消息
    return messages  # 返回转换后的标准消息列表


def _parse_text(text: str) -> str:
    """函数功能：
    文本解析函数，对文本中的特殊 HTML 字符进行转义，防止在 Gradio 界面中被误解析为 HTML 标签。
    
    参数：
        text (str): 原始文本字符串（可能包含 <、>、* 等特殊字符）
    
    返回值：
        str: 转义后的文本字符串，特殊字符被替换为 HTML 实体
            - '<' 转换为 '&lt;'
            - '>' 转换为 '&gt;'
            - '*' 转换为 '&ast;'
    
    实际使用示例：
        示例 1：转义 HTML 标签
        >>> text = "使用 <div> 标签创建容器"
        >>> parsed = _parse_text(text)
        >>> print(parsed)  # 输出: "使用 &lt;div&gt; 标签创建容器"
        # 在 Gradio 中显示时，会正确显示为 "使用 <div> 标签创建容器"，而不是解析为 HTML
        
        示例 2：转义星号（Markdown 语法）
        >>> text = "这是**粗体**文本"
        >>> parsed = _parse_text(text)
        >>> print(parsed)  # 输出: "这是&ast;&ast;粗体&ast;&ast;文本"
        
        示例 3：混合转义
        >>> text = "比较: 5 > 3 且 2 < 4"
        >>> parsed = _parse_text(text)
        >>> print(parsed)  # 输出: "比较: 5 &gt; 3 且 2 &lt; 4"
    """
    mapping = {'<': '&lt;', '>': '&gt;', '*': '&ast;'}  # 定义字符映射字典：将特殊字符映射到对应的 HTML 实体编码
    for k, v in mapping.items():  # 遍历映射字典的每个键值对（k 为原字符，v 为 HTML 实体）
        text = text.replace(k, v)  # 将文本中的所有原字符替换为对应的 HTML 实体
    return text  # 返回转义后的文本字符串


async def model_chat(history: History, real_history: History, system: Optional[str], *, client, model: str,
                     request_config: Optional['RequestConfig']):
    """函数功能：
    异步调用推理服务进行对话生成，支持流式和非流式两种模式，并实时更新界面显示。
    
    参数：
        history (History): Gradio 显示用的对话历史（包含转义后的文本），格式为 List[List[str]]
        real_history (History): 实际的对话历史（未转义的原始文本），用于发送给推理服务
        system (Optional[str]): 系统提示词，若为 None 则不使用系统提示
        client: InferClient 推理客户端实例，用于向推理服务发送请求
        model (str): 模型名称或 ID（如 'qwen/Qwen-7B-Chat'）
        request_config (Optional[RequestConfig]): 请求配置对象，包含 temperature、stream 等参数
    
    返回值：
        AsyncGenerator[tuple, None]: 异步生成器，每次 yield 返回更新后的 (history, real_history)
            - 流式模式：每收到一个 token 就 yield 一次，实现逐字显示效果
            - 非流式模式：只 yield 一次完整的响应
    
    实际使用示例：
        示例 1：流式对话（逐字显示）
        >>> from swift.llm import InferClient, RequestConfig
        >>> client = InferClient(base_url='http://localhost:8000/v1')
        >>> config = RequestConfig(stream=True, temperature=0.7)
        >>> history = [['你好', None]]
        >>> real_history = [['你好', None]]
        >>> async for h, rh in model_chat(history, real_history, None, 
        ...                               client=client, model='qwen', request_config=config):
        ...     print(h[-1][1])  # 逐字打印助手回复
        # 输出: '你'
        # 输出: '你好'
        # 输出: '你好！'
        # 输出: '你好！有什么'
        # ...（逐步更新）
        
        示例 2：非流式对话（一次性返回）
        >>> config = RequestConfig(stream=False)
        >>> async for h, rh in model_chat(history, real_history, 'You are helpful',
        ...                               client=client, model='qwen', request_config=config):
        ...     print(h[-1][1])  # 只打印一次完整回复
        # 输出: '你好！有什么可以帮助你的吗？'
    """
    if history:  # 若对话历史不为空（存在待处理的用户输入）
        from swift.llm import InferRequest  # 延迟导入 InferRequest 类（避免循环导入，减少启动时间）

        messages = _history_to_messages(real_history, system)  # 将真实历史记录（未转义文本）转换为标准消息格式，包含系统提示词
        resp_or_gen = await client.infer_async(  # 异步调用推理客户端的 infer_async 方法，发送推理请求
            InferRequest(messages=messages), request_config=request_config, model=model)  # 创建推理请求对象，传入消息列表、请求配置和模型名称
        if request_config and request_config.stream:  # 若请求配置存在且启用了流式模式（stream=True）
            response = ''  # 初始化空字符串，用于累积流式响应的所有 token
            async for resp in resp_or_gen:  # 异步迭代流式响应生成器，resp 为 ChatCompletionStreamResponse 对象
                if resp is None:  # 若响应对象为 None（可能是心跳或空数据）
                    continue  # 跳过本次循环，继续等待下一个响应
                response += resp.choices[0].delta.content  # 从响应对象中提取增量内容（新生成的 token）并累加到 response 字符串
                history[-1][1] = _parse_text(response)  # 更新显示历史的最后一条消息的助手回复部分（转义后的文本，用于 Gradio 显示）
                real_history[-1][-1] = response  # 更新真实历史的最后一条消息的助手回复部分（未转义的原始文本）
                yield history, real_history  # 向外部 yield 更新后的历史记录，触发 Gradio 界面的实时更新

        else:  # 否则（非流式模式，一次性返回完整响应）
            response = resp_or_gen.choices[0].message.content  # 从响应对象中提取完整的助手回复内容（字符串）
            history[-1][1] = _parse_text(response)  # 更新显示历史的最后一条消息的助手回复部分（转义后的文本）
            real_history[-1][-1] = response  # 更新真实历史的最后一条消息的助手回复部分（未转义的原始文本）
            yield history, real_history  # 向外部 yield 更新后的完整历史记录

    else:  # 否则（对话历史为空，没有用户输入）
        yield [], []  # 返回空的历史记录（不执行任何推理）


def add_text(history: History, real_history: History, query: str):
    """函数功能：
    将用户输入的文本消息添加到对话历史中，准备发送给模型进行推理。
    
    参数：
        history (History): Gradio 显示用的对话历史（包含转义后的文本）
        real_history (History): 实际的对话历史（未转义的原始文本）
        query (str): 用户输入的查询文本（如 '今天天气如何？'）
    
    返回值：
        tuple: 包含三个元素的元组 (updated_history, updated_real_history, empty_textbox)
            - updated_history (History): 更新后的显示历史（添加了新的用户消息，助手回复为 None）
            - updated_real_history (History): 更新后的真实历史（添加了新的用户消息，助手回复为 None）
            - empty_textbox (str): 空字符串，用于清空输入框
    
    实际使用示例：
        示例 1：用户提交文本消息
        >>> history = [['你好', '你好！']]  # 已有一轮对话
        >>> real_history = [['你好', '你好！']]
        >>> new_h, new_rh, empty = add_text(history, real_history, '今天天气如何？')
        >>> print(new_h)
        # [['你好', '你好！'], ['今天天气如何？', None]]  # 新消息已添加，等待模型回复
        >>> print(empty)  # 输出: ''  # 输入框被清空
        
        示例 2：首次提交消息（空历史）
        >>> new_h, new_rh, empty = add_text([], [], '你是谁？')
        >>> print(new_h)
        # [['你是谁？', None]]  # 创建了第一轮对话
    """
    history = history or []  # 若 history 为 None 或空值，初始化为空列表（防止 None 值导致的错误）
    real_history = real_history or []  # 若 real_history 为 None 或空值，初始化为空列表
    history.append([_parse_text(query), None])  # 将用户查询添加到显示历史：对文本进行转义处理，助手回复初始化为 None（等待生成）
    real_history.append([query, None])  # 将用户查询添加到真实历史：保持原始文本不转义，助手回复初始化为 None
    return history, real_history, ''  # 返回更新后的两个历史记录和空字符串（用于清空输入框）


def add_file(history: History, real_history: History, file):
    """函数功能：
    将用户上传的文件添加到对话历史中，用于多模态对话（如图像理解、语音识别等）。
    
    参数：
        history (History): Gradio 显示用的对话历史
        real_history (History): 实际的对话历史
        file: Gradio 的 File 对象，包含上传文件的信息（主要使用 file.name 获取文件路径）
    
    返回值：
        tuple: 包含两个元素的元组 (updated_history, updated_real_history)
            - updated_history (History): 更新后的显示历史（添加了文件，助手回复为 None）
            - updated_real_history (History): 更新后的真实历史（添加了文件，助手回复为 None）
    
    实际使用示例：
        示例 1：上传图像文件
        >>> history = []
        >>> real_history = []
        >>> # 假设用户上传了一个图像文件 'cat.jpg'
        >>> class FakeFile:
        ...     name = '/tmp/gradio/cat.jpg'
        >>> file = FakeFile()
        >>> new_h, new_rh = add_file(history, real_history, file)
        >>> print(new_h)
        # [[('/tmp/gradio/cat.jpg',), None]]  # 文件路径被包装为元组
        
        示例 2：连续上传多个文件
        >>> # 先上传图像
        >>> new_h, new_rh = add_file([], [], FakeFile1())  # FakeFile1.name = 'image.jpg'
        >>> # 再上传文本输入
        >>> new_h, new_rh, _ = add_text(new_h, new_rh, '这是什么？')
        >>> print(new_h)
        # [[('image.jpg',), None], ['这是什么？', None]]
        # 多模态消息：图像 + 文本问题
    """
    history = history or []  # 若 history 为 None 或空值，初始化为空列表（防止 None 值导致的错误）
    real_history = real_history or []  # 若 real_history 为 None 或空值，初始化为空列表
    history.append([(file.name, ), None])  # 将文件路径包装为单元素元组并添加到显示历史：格式为 [('path/to/file.jpg',), None]，助手回复为 None
    real_history.append([(file.name, ), None])  # 将文件路径包装为单元素元组并添加到真实历史：格式相同，保持一致性
    return history, real_history  # 返回更新后的两个历史记录（不清空输入框，因为用户可能继续添加文本）


def build_ui(base_url: str,  # 接受推理服务的基础 URL
             model: Optional[str] = None,  # 可选参数：模型名称，若为 None 则自动选择第一个可用模型
             *,  # 仅限关键字参数分隔符，后续参数必须以关键字形式传入
             request_config: Optional['RequestConfig'] = None,  # 可选参数：推理请求配置（温度、流式等）
             is_multimodal: bool = True,  # 是否为多模态模型（控制文件上传按钮的可见性）
             studio_title: Optional[str] = None,  # UI 标题，若为 None 则使用模型名称
             lang: Literal['en', 'zh'] = 'en',  # 界面语言，支持英文（'en'）和中文（'zh'）
             default_system: Optional[str] = None):  # 默认系统提示词
    """函数功能：
    定义构建 UI 的主函数，构建完整的 Gradio Web UI 界面，包括聊天框、输入框、按钮、状态管理等所有组件。
    
    参数：
        base_url (str): 推理服务的基础 URL（如 'http://localhost:8000/v1'）
        model (Optional[str]): 模型名称，若为 None 则使用推理服务返回的第一个可用模型
        request_config (Optional[RequestConfig]): 推理配置对象，包含 temperature、top_p、stream 等参数
        is_multimodal (bool): 是否为多模态模型，若为 True 则显示文件上传按钮，默认 True
        studio_title (Optional[str]): UI 页面标题，若为 None 则使用模型名称作为标题
        lang (Literal['en', 'zh']): 界面语言，'en' 为英文，'zh' 为中文，默认 'en'
        default_system (Optional[str]): 默认系统提示词，初始化系统输入框和状态
    
    返回值：
        gr.Blocks: Gradio Blocks 对象，可调用 .launch() 方法启动服务器
    
    实际使用示例：
        示例 1：基本用法（使用默认配置）
        >>> demo = build_ui(base_url='http://localhost:8000/v1')
        >>> demo.launch(server_port=7860)
        # 启动英文界面，自动选择第一个模型，支持多模态输入
        
        示例 2：中文界面 + 自定义标题
        >>> demo = build_ui(
        ...     base_url='http://localhost:8000/v1',
        ...     model='qwen/Qwen-7B-Chat',
        ...     studio_title='智能对话助手',
        ...     lang='zh',
        ...     default_system='你是一个有帮助的AI助手'
        ... )
        >>> demo.launch()
        # 启动中文界面，显示"智能对话助手"标题，系统提示词已预设
        
        示例 3：流式响应 + 仅文本模式
        >>> from swift.llm import RequestConfig
        >>> config = RequestConfig(stream=True, temperature=0.8)
        >>> demo = build_ui(
        ...     base_url='http://localhost:8000/v1',
        ...     request_config=config,
        ...     is_multimodal=False,  # 隐藏文件上传按钮
        ...     lang='zh'
        ... )
        # 启动流式对话界面，不支持文件上传
    """
    from swift.llm import InferClient  # 延迟导入 InferClient 类（推理客户端），避免循环导入
    client = InferClient(base_url=base_url)  # 创建推理客户端实例，连接到指定的推理服务 URL
    model = model or client.models[0]  # 确定使用的模型：若未指定模型名称，则从推理服务获取可用模型列表并选择第一个
    studio_title = studio_title or model  # 确定 UI 标题：若未指定标题，则使用模型名称作为标题
    with gr.Blocks() as demo:  # 创建 Gradio Blocks 上下文，demo 为 Blocks 实例，用于组织和管理所有 UI 组件
        gr.Markdown(f'<center><font size=8>{studio_title}</center>')  # 创建 Markdown 组件显示标题：使用 HTML 标签设置居中、大字体（size=8）的标题文本
        with gr.Row():  # 创建一个水平布局行，用于放置系统提示词输入框和修改按钮
            with gr.Column(scale=3):  # 创建一个列容器，占据行宽度的 3/4（scale=3）
                system_input = gr.Textbox(value=default_system, lines=1, label='System')  # 创建文本输入框组件：初始值为 default_system，单行输入（lines=1），标签为 'System'
            with gr.Column(scale=1):  # 创建一个列容器，占据行宽度的 1/4（scale=1）
                modify_system = gr.Button(locale_mapping['modify_system'][lang], scale=2)  # 创建按钮组件：文本从本地化映射中获取（中文为"修改系统"，英文为"Modify System"），scale=2 控制按钮高度
        chatbot = gr.Chatbot(label='Chatbot')  # 创建聊天框组件，用于显示对话历史，标签为 'Chatbot'
        textbox = gr.Textbox(lines=1, label='Input')  # 创建文本输入框组件，用于用户输入查询，单行输入（lines=1），标签为 'Input'

        with gr.Row():  # 创建一个水平布局行，用于放置操作按钮（上传、提交、重新生成、清空历史）
            upload = gr.UploadButton(locale_mapping['upload'][lang], visible=is_multimodal)  # 创建文件上传按钮：文本从本地化映射获取，可见性由 is_multimodal 控制（多模态时显示，否则隐藏）
            submit = gr.Button(locale_mapping['submit'][lang])  # 创建提交按钮：文本从本地化映射获取（中文为"提交"，英文为"Submit"）
            regenerate = gr.Button(locale_mapping['regenerate'][lang])  # 创建重新生成按钮：文本从本地化映射获取（中文为"重新生成"，英文为"Regenerate"）
            clear_history = gr.Button(locale_mapping['clear_history'][lang])  # 创建清空历史按钮：文本从本地化映射获取（中文为"清空历史"，英文为"Clear History"）

        system_state = gr.State(value=default_system)  # 创建状态组件，用于存储当前的系统提示词（初始值为 default_system），不在界面中显示
        history_state = gr.State(value=[])  # 创建状态组件，用于存储真实的对话历史（未转义的原始文本），初始值为空列表
        model_chat_ = partial(model_chat, client=client, model=model, request_config=request_config)  # 使用 partial 固定 model_chat 函数的部分参数（client、model、request_config），创建新函数 model_chat_

        upload.upload(add_file, [chatbot, history_state, upload], [chatbot, history_state])  # 绑定上传按钮的 upload 事件：触发时调用 add_file 函数，输入为 [chatbot, history_state, upload]，输出更新 [chatbot, history_state]
        textbox.submit(add_text, [chatbot, history_state, textbox],  # 绑定输入框的 submit 事件（用户按回车键）：第一步调用 add_text 函数，输入为 [chatbot, history_state, textbox]
                       [chatbot, history_state, textbox]).then(model_chat_, [chatbot, history_state, system_state],  # 输出更新 [chatbot, history_state, textbox]，然后（.then）调用 model_chat_ 函数
                                                               [chatbot, history_state])  # model_chat_ 的输入为 [chatbot, history_state, system_state]，输出更新 [chatbot, history_state]
        submit.click(add_text, [chatbot, history_state, textbox],  # 绑定提交按钮的 click 事件：第一步调用 add_text 函数，输入为 [chatbot, history_state, textbox]
                     [chatbot, history_state, textbox]).then(model_chat_, [chatbot, history_state, system_state],  # 输出更新 [chatbot, history_state, textbox]，然后（.then）调用 model_chat_ 函数
                                                             [chatbot, history_state])  # model_chat_ 的输入为 [chatbot, history_state, system_state]，输出更新 [chatbot, history_state]
        regenerate.click(model_chat_, [chatbot, history_state, system_state], [chatbot, history_state])  # 绑定重新生成按钮的 click 事件：直接调用 model_chat_ 函数重新生成最后一条回复，输入为 [chatbot, history_state, system_state]，输出更新 [chatbot, history_state]
        clear_history.click(clear_session, [], [textbox, chatbot, history_state])  # 绑定清空历史按钮的 click 事件：调用 clear_session 函数，无输入参数（[]），输出更新 [textbox, chatbot, history_state]（全部清空）
        modify_system.click(modify_system_session, [system_input], [system_state, textbox, chatbot, history_state])  # 绑定修改系统按钮的 click 事件：调用 modify_system_session 函数，输入为 [system_input]，输出更新 [system_state, textbox, chatbot, history_state]（更新系统提示词并清空历史）
    return demo  # 返回构建好的 Gradio Blocks 对象，调用方可调用 demo.launch() 启动服务器
