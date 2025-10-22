# Copyright (c) Alibaba, Inc. and its affiliates.

"""
模板元数据模块 (Template Metadata Module)

本模块定义了大语言模型对话模板的元数据结构，用于描述和管理不同模型的对话格式。
通过 TemplateMeta 数据类，可以灵活配置模型的对话前缀、提示词、分隔符、后缀等组件，
支持系统提示、多轮对话、自动添加 BOS 标记等功能。

主要功能：
    - 定义对话模板的基本结构（前缀、提示词、分隔符、后缀）
    - 支持系统提示词的配置（前置或后置）
    - 管理停止词和停止 token ID
    - 提供模板验证和初始化机制
    - 支持生成专用模板的创建

核心类：
    - TemplateMeta: 对话模板元数据的数据类，包含模板配置的所有必要信息
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Type, Union

from transformers import PreTrainedTokenizerBase

from .base import Template
from .utils import Prompt, Word


@dataclass
class TemplateMeta:
    """
    大语言模型对话模板元数据类。
    
    该类用于定义和管理大语言模型的对话格式模板，包括对话的前缀、提示词、分隔符、
    后缀等组件。支持系统提示词、多轮对话、停止词配置等功能。通过灵活组合这些组件，
    可以适配不同模型的对话格式要求。
    
    类功能：
        封装对话模板的所有元数据信息，提供模板的初始化、验证和转换功能
    
    继承关系：
        使用 @dataclass 装饰器，自动生成 __init__、__repr__ 等方法
    
    应用场景：
        - 定义不同大语言模型的对话格式（如 ChatML、Llama、Qwen 等）
        - 配置模型的系统提示词和用户交互格式
        - 管理多轮对话的分隔符和停止条件
        - 为模型推理和训练提供统一的模板接口
    
    使用示例：
        # 示例1：定义 ChatML 格式模板（带 BOS）
        chatml_meta = TemplateMeta(
            template_type='chatml',
            prefix=['<s>'],
            prompt=['<|im_start|>user\\n{{QUERY}}<|im_end|>\\n<|im_start|>assistant\\n'],
            chat_sep=['<|im_end|>\\n'],
            suffix=['<|im_end|>'],
            system_prefix=['<s><|im_start|>system\\n{{SYSTEM}}<|im_end|>\\n']
        )
        
        # 示例2：定义简单的问答模板
        simple_meta = TemplateMeta(
            template_type='simple',
            prefix=[],
            prompt=['User: {{QUERY}}\\nAssistant: '],
            chat_sep=['\\n'],
            suffix=[['eos_token_id']],
            auto_add_bos=True
        )
        
        # 示例3：初始化模板（绑定 tokenizer）
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B-Chat')
        chatml_meta.init(tokenizer)
        
        # 示例4：生成用于推理的模板
        generate_meta = chatml_meta.to_generate_template_meta()
    
    模板结构示例（ChatML 格式）：
        <s><|im_start|>system      # prefix 或 system_prefix
        {{SYSTEM}}<|im_end|>
        <|im_start|>user            # prompt
        {{QUERY}}<|im_end|>
        <|im_start|>assistant
        {{RESPONSE}}<|im_end|>      # chat_sep
        <|im_start|>user            # prompt
        {{QUERY}}<|im_end|>
        <|im_start|>assistant
        {{RESPONSE}}<|im_end|>      # suffix
    """
    # ========== 必需字段 ==========
    
    # 模板类型标识符，如 'chatml'、'qwen'、'llama' 等
    template_type: str
    
    # 对话前缀，添加在整个对话的开头（不包含系统提示时）
    # 例如：['<s>'] 或 []
    prefix: Prompt
    
    # 用户提示词格式，包含 {{QUERY}} 占位符
    # 例如：['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n']
    prompt: Prompt
    
    # 多轮对话的分隔符，添加在每轮对话的响应之后
    # 如果为 None，表示不支持多轮对话
    # 例如：['<|im_end|>\n'] 或 None
    chat_sep: Optional[Prompt]
    
    # ========== 可选字段（带默认值）==========
    
    # 对话后缀，添加在整个对话的末尾
    # 默认为 [['eos_token_id']]，表示使用 tokenizer 的 EOS token
    suffix: Prompt = field(default_factory=lambda: [['eos_token_id']])
    
    # 模板类的类型，用于实例化具体的模板对象
    # 默认为 Template 基类
    template_cls: Type[Template] = Template
    
    # 系统提示词前缀，包含 {{SYSTEM}} 占位符
    # 如果为 None，表示不支持系统提示或系统提示在 prefix 中定义
    # 例如：['<s><|im_start|>system\n{{SYSTEM}}<|im_end|>\n']
    system_prefix: Optional[Prompt] = None
    
    # 默认的系统提示词内容
    # 例如：'You are a helpful assistant.'
    default_system: Optional[str] = None
    
    # 响应前缀，添加在模型响应之前（通常为空字符串）
    response_prefix: str = ''
    
    # ========== 配置选项 ==========
    
    # 是否自动添加 BOS (Beginning of Sequence) token
    # 用于某些需要显式添加开始标记的模型
    auto_add_bos: bool = False
    
    # 停止词列表，用于控制生成停止的特殊词或 token
    # 例如：['<|im_end|>', '\n\n']
    stop_words: List[Word] = field(default_factory=list)
    
    # Agent 模板类型，用于指定 Agent 任务的模板格式
    # 默认为 'react_en'（英文 ReAct 格式）
    agent_template: str = 'react_en'

    def to_generate_template_meta(self) -> 'TemplateMeta':
        """
        创建用于生成（推理）的简化模板元数据。
        
        该方法将当前对话模板转换为适用于模型推理生成的简化版本。生成模板移除了
        对话格式的复杂结构，只保留最基本的查询输入，适用于单次生成场景。
        
        功能：
            基于当前模板创建一个简化的生成专用模板，移除多轮对话和系统提示等复杂结构
        
        参数：
            无
        
        返回值：
            TemplateMeta: 简化的生成模板元数据对象，具有以下特点：
                - prefix 为空列表（无前缀）
                - prompt 简化为 ['{{QUERY}}']（仅保留查询占位符）
                - chat_sep 为 None（不支持多轮对话）
                - auto_add_bos 设置为 True（自动添加 BOS token）
                - 保留原模板的 template_cls 和 stop_words
        
        使用示例：
            # 创建对话模板
            chat_meta = TemplateMeta(
                template_type='chatml',
                prefix=['<s>'],
                prompt=['<|im_start|>user\\n{{QUERY}}<|im_end|>\\n<|im_start|>assistant\\n'],
                chat_sep=['<|im_end|>\\n'],
                suffix=['<|im_end|>']
            )
            
            # 转换为生成模板
            gen_meta = chat_meta.to_generate_template_meta()
            # gen_meta.prefix == []
            # gen_meta.prompt == ['{{QUERY}}']
            # gen_meta.chat_sep == None
            # gen_meta.auto_add_bos == True
        """
        # 深拷贝当前对象，避免修改原对象
        self = deepcopy(self)
        
        # 创建并返回简化的生成模板
        return TemplateMeta(
            self.template_type,              # 保留模板类型标识
            prefix=[],                        # 清空前缀，不需要对话格式
            prompt=['{{QUERY}}'],            # 简化为纯查询占位符
            chat_sep=None,                    # 移除多轮对话分隔符
            template_cls=self.template_cls,  # 保留模板类
            auto_add_bos=True,               # 启用自动添加 BOS token
            stop_words=self.stop_words,      # 保留停止词配置
        )

    @staticmethod
    def _has_system(prefix_or_prompt: Prompt) -> bool:
        """
        检查提示词列表中是否包含系统提示占位符。
        
        该方法用于判断给定的提示词列表（prefix 或 prompt）中是否包含 {{SYSTEM}} 占位符，
        以确定该部分是否支持系统提示词。
        
        功能：
            检测提示词列表中是否存在 {{SYSTEM}} 占位符
        
        参数：
            prefix_or_prompt (Prompt): 提示词列表，可以是 prefix 或 prompt
                例如：['<s><|im_start|>system\n{{SYSTEM}}<|im_end|>\n']
        
        返回值：
            bool: 如果提示词列表中至少有一个元素包含 '{{SYSTEM}}'，返回 True；
                  否则返回 False
        
        使用示例：
            # 包含系统占位符的情况
            prefix = ['<s><|im_start|>system\\n{{SYSTEM}}<|im_end|>\\n']
            has_sys = TemplateMeta._has_system(prefix)  # 返回 True
            
            # 不包含系统占位符的情况
            prefix = ['<s>']
            has_sys = TemplateMeta._has_system(prefix)  # 返回 False
            
            # 空列表
            prefix = []
            has_sys = TemplateMeta._has_system(prefix)  # 返回 False
        """
        # 使用 any() 函数检查列表中是否有任何元素包含 '{{SYSTEM}}' 字符串
        # 遍历 prefix_or_prompt 中的每个元素 p，检查 '{{SYSTEM}}' 是否在 p 中
        return any(['{{SYSTEM}}' in p for p in prefix_or_prompt])

    @staticmethod
    def _replace_system(prefix: Prompt) -> Prompt:
        """
        移除提示词列表中的系统提示占位符。
        
        该方法用于从 prefix 中移除 {{SYSTEM}} 占位符，生成不包含系统提示的前缀版本。
        只处理字符串类型的元素，非字符串元素会被过滤掉。
        
        功能：
            从提示词列表中移除 {{SYSTEM}} 占位符，生成纯前缀版本
        
        参数：
            prefix (Prompt): 原始前缀列表，可能包含 {{SYSTEM}} 占位符
                例如：['<s><|im_start|>system\n{{SYSTEM}}<|im_end|>\n']
        
        返回值：
            Prompt: 移除了 {{SYSTEM}} 占位符的新列表，只包含字符串类型的元素
                例如：['<s><|im_start|>system\n<|im_end|>\n']
        
        使用示例：
            # 移除系统占位符
            prefix = ['<s><|im_start|>system\\n{{SYSTEM}}<|im_end|>\\n']
            new_prefix = TemplateMeta._replace_system(prefix)
            # new_prefix == ['<s><|im_start|>system\\n<|im_end|>\\n']
            
            # 处理混合类型列表
            prefix = ['<s>', ['eos_token_id'], 'system: {{SYSTEM}}']
            new_prefix = TemplateMeta._replace_system(prefix)
            # new_prefix == ['<s>', 'system: ']  # 非字符串元素被过滤
            
            # 无系统占位符的情况
            prefix = ['<s>']
            new_prefix = TemplateMeta._replace_system(prefix)
            # new_prefix == ['<s>']
        """
        # 遍历 prefix 列表，对每个字符串类型的元素 p，将其中的 '{{SYSTEM}}' 替换为空字符串
        # 非字符串类型的元素（如 token ID 列表）会被过滤掉
        return [p.replace('{{SYSTEM}}', '') for p in prefix if isinstance(p, str)]

    def _check_template_meta(self):
        """
        验证模板元数据字段的类型正确性。
        
        该方法检查模板元数据的关键字段是否符合类型要求，确保必需字段为列表类型，
        可选字段为 None 或列表类型。
        
        功能：
            验证模板元数据字段的类型，确保配置正确
        
        参数：
            无
        
        返回值：
            无（如果验证失败会触发 AssertionError）
        
        异常：
            AssertionError: 当字段类型不符合要求时抛出
        
        使用示例：
            # 正确的模板配置
            meta = TemplateMeta(
                template_type='test',
                prefix=['<s>'],          # 列表类型 ✓
                prompt=['{{QUERY}}'],    # 列表类型 ✓
                chat_sep=None,           # None 或列表 ✓
                suffix=[['eos_token_id']]  # 列表类型 ✓
            )
            meta._check_template_meta()  # 通过验证
            
            # 错误的配置示例（会触发断言错误）
            # meta.prefix = '<s>'  # 字符串而非列表 ✗
            # meta._check_template_meta()  # 抛出 AssertionError
        """
        # 检查必需字段：prefix, prompt, suffix 必须是列表类型
        for x in [self.prefix, self.prompt, self.suffix]:
            assert isinstance(x, list)
        
        # 检查可选字段：chat_sep, system_prefix 可以是 None 或列表类型
        for x in [self.chat_sep, self.system_prefix]:
            assert x is None or isinstance(x, list)

    def __post_init__(self):
        """
        数据类初始化后的后处理方法。
        
        该方法在 dataclass 的 __init__ 方法执行后自动调用，用于处理系统提示词的位置、
        验证模板配置、设置模板的功能支持标志（如是否支持系统提示、多轮对话等）。
        
        功能：
            自动检测和配置系统提示词位置，设置模板支持特性标志
        
        参数：
            无（由 dataclass 自动调用）
        
        返回值：
            无
        
        副作用：
            设置以下实例属性：
                - system_prefix: 系统提示前缀（如果 prefix 包含 {{SYSTEM}}）
                - prefix: 移除系统占位符后的前缀
                - is_post_system: 标记系统提示是否在 prompt 中（后置系统提示）
                - system_prompt: 包含系统提示的 prompt（如果是后置系统提示）
                - support_system: 标记是否支持系统提示
                - support_multi_round: 标记是否支持多轮对话
        
        异常：
            AssertionError: 当 prefix 已包含 {{SYSTEM}} 但 system_prefix 也被设置时
            AssertionError: 当 default_system 不为 None 但模板不支持系统提示时
        
        使用示例：
            # 示例1：前置系统提示（在 prefix 中）
            meta = TemplateMeta(
                template_type='test',
                prefix=['<s><|im_start|>system\\n{{SYSTEM}}<|im_end|>\\n'],
                prompt=['<|im_start|>user\\n{{QUERY}}<|im_end|>\\n'],
                chat_sep=['\\n']
            )
            # __post_init__ 自动调用后：
            # meta.system_prefix == ['<s><|im_start|>system\\n{{SYSTEM}}<|im_end|>\\n']
            # meta.prefix == ['<s><|im_start|>system\\n<|im_end|>\\n']
            # meta.support_system == True
            # meta.support_multi_round == True
            
            # 示例2：后置系统提示（在 prompt 中，如 Mistral Nemo）
            meta = TemplateMeta(
                template_type='mistral',
                prefix=['<s>'],
                prompt=['[INST]{{SYSTEM}}\\n{{QUERY}}[/INST]'],
                chat_sep=None
            )
            # __post_init__ 自动调用后：
            # meta.is_post_system == True
            # meta.system_prompt == ['[INST]{{SYSTEM}}\\n{{QUERY}}[/INST]']
            # meta.prompt == []  # 移除了包含 {{SYSTEM}} 的部分
            # meta.support_system == True
            # meta.support_multi_round == False
            
            # 示例3：不支持系统提示
            meta = TemplateMeta(
                template_type='simple',
                prefix=[],
                prompt=['{{QUERY}}'],
                chat_sep=None
            )
            # __post_init__ 自动调用后：
            # meta.support_system == False
            # meta.support_multi_round == False
        """
        # ========== 处理系统提示词位置 ==========
        
        # 情况1：检查 prefix 是否包含 {{SYSTEM}} 占位符（前置系统提示）
        if self._has_system(self.prefix):
            # 断言：如果 prefix 已包含系统占位符，则 system_prefix 不应该被单独设置
            assert self.system_prefix is None, 'The prefix already contains {{SYSTEM}}.'
            
            # 将包含系统占位符的 prefix 保存为 system_prefix
            self.system_prefix = self.prefix
            
            # 从 prefix 中移除系统占位符，生成无系统提示的版本
            self.prefix = self._replace_system(self.prefix)

        # 情况2：检查 prompt 是否包含 {{SYSTEM}} 占位符（后置系统提示，如 Mistral Nemo）
        self.is_post_system = self._has_system(self.prompt)
        if self.is_post_system:
            # 保存包含系统提示的原始 prompt
            self.system_prompt = self.prompt
            
            # 从 prompt 中过滤掉包含 {{SYSTEM}} 的部分
            self.prompt = [context for context in self.prompt if '{{SYSTEM}}' not in context]

        # ========== 设置功能支持标志 ==========
        
        # 判断是否支持系统提示：system_prefix 存在或使用后置系统提示
        if self.system_prefix is None and not self.is_post_system:
            self.support_system = False
        else:
            self.support_system = True
        
        # 验证默认系统提示词的配置是否合法
        self.check_system(self.default_system)

        # 判断是否支持多轮对话：chat_sep 不为 None 时支持
        self.support_multi_round = self.chat_sep is not None

    @staticmethod
    def _token_attr_to_id(tokenizer: PreTrainedTokenizerBase, value: Optional[Prompt]) -> Optional[Prompt]:
        """
        将 tokenizer 属性名转换为实际的 token ID。
        
        该方法用于将提示词列表中的 tokenizer 属性引用（如 'eos_token_id'、'bos_token_id'）
        转换为对应的 token ID 值。这样可以在模板定义时使用语义化的属性名，在运行时再
        绑定到具体的 token ID。
        
        功能：
            将提示词列表中的字符串形式的 tokenizer 属性名替换为实际的 token ID 值
        
        参数：
            tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer 对象，
                提供 eos_token_id、bos_token_id 等属性
            value (Optional[Prompt]): 提示词列表，可能包含 tokenizer 属性引用
                例如：[['eos_token_id'], '<s>', ['bos_token_id']]
        
        返回值：
            Optional[Prompt]: 转换后的提示词列表，属性名被替换为实际值
                例如：[[2], '<s>', [1]]
                如果输入为 None，则返回 None
        
        使用示例：
            from transformers import AutoTokenizer
            
            # 加载 tokenizer
            tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B-Chat')
            # 假设 tokenizer.eos_token_id = 151643
            # 假设 tokenizer.bos_token_id = 151643
            
            # 示例1：转换 EOS token
            value = [['eos_token_id']]
            result = TemplateMeta._token_attr_to_id(tokenizer, value)
            # result == [[151643]]
            
            # 示例2：混合类型列表
            value = ['<s>', ['eos_token_id'], 'text', ['bos_token_id']]
            result = TemplateMeta._token_attr_to_id(tokenizer, value)
            # result == ['<s>', [151643], 'text', [151643]]
            
            # 示例3：嵌套列表中的多个属性
            value = [['eos_token_id', 'bos_token_id']]
            result = TemplateMeta._token_attr_to_id(tokenizer, value)
            # result == [[151643, 151643]]
            
            # 示例4：None 值
            result = TemplateMeta._token_attr_to_id(tokenizer, None)
            # result == None
            
            # 示例5：已经是 ID 的情况
            value = [[2], [1]]
            result = TemplateMeta._token_attr_to_id(tokenizer, value)
            # result == [[2], [1]]  # 保持不变
        """
        # 如果输入为 None，直接返回 None
        if value is None:
            return None
        
        # 初始化结果列表
        res_value = []
        
        # 遍历提示词列表中的每个元素
        for v in value:
            # 如果当前元素是列表（通常是 token ID 列表）
            if isinstance(v, list):
                # 遍历列表中的每个子元素
                # 如果子元素是字符串，则从 tokenizer 获取对应属性值（如 eos_token_id）
                # 如果子元素不是字符串（已经是数字 ID），则保持不变
                v = [getattr(tokenizer, sub_v) if isinstance(sub_v, str) else sub_v for sub_v in v]
            
            # 将处理后的元素添加到结果列表
            res_value.append(v)
        
        # 返回转换后的提示词列表
        return res_value

    def init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """
        使用 tokenizer 初始化模板元数据。
        
        该方法将模板中的 tokenizer 属性引用转换为实际的 token ID，并配置停止词和
        停止 token ID。这是模板在实际使用前必须执行的初始化步骤。
        
        功能：
            绑定 tokenizer 到模板，转换属性引用为实际 token ID，配置停止条件
        
        参数：
            tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer 对象，
                用于提供 token ID 和停止 token 信息
        
        返回值：
            无（就地修改当前对象的属性）
        
        副作用：
            - 转换 prefix, prompt, chat_sep, suffix, system_prefix 中的属性引用为 token ID
            - 将 suffix 的最后一个元素添加到 stop_words（如果尚未存在）
            - 将 tokenizer.eos_token 添加到 stop_words（如果尚未存在）
            - 设置 stop_token_id（优先使用 suffix 的最后一个 token，否则使用 eos_token_id）
        
        使用示例：
            from transformers import AutoTokenizer
            
            # 创建模板
            meta = TemplateMeta(
                template_type='chatml',
                prefix=['<s>'],
                prompt=['<|im_start|>user\\n{{QUERY}}<|im_end|>\\n<|im_start|>assistant\\n'],
                chat_sep=['<|im_end|>\\n'],
                suffix=[['eos_token_id']]  # 使用属性引用
            )
            
            # 加载 tokenizer
            tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B-Chat')
            
            # 初始化模板（绑定 tokenizer）
            meta.init(tokenizer)
            
            # 初始化后：
            # meta.suffix == [[151643]]  # 'eos_token_id' 被转换为实际 ID
            # meta.stop_words 包含 [151643] 和 tokenizer.eos_token
            # meta.stop_token_id == 151643
            
            # 示例2：自定义后缀 token
            meta2 = TemplateMeta(
                template_type='custom',
                prefix=[],
                prompt=['{{QUERY}}'],
                chat_sep=None,
                suffix=['<|endoftext|>']  # 字符串形式的后缀
            )
            meta2.init(tokenizer)
            # meta2.stop_token_id 将是 '<|endoftext|>' 对应的 token ID
        """
        # ========== 转换 tokenizer 属性引用为实际 token ID ==========
        
        # 遍历需要转换的关键字段
        for key in ['prefix', 'prompt', 'chat_sep', 'suffix', 'system_prefix']:
            # 获取当前字段的值
            value = getattr(self, key)
            
            # 将属性引用（如 'eos_token_id'）转换为实际的 token ID
            value = self._token_attr_to_id(tokenizer, value)
            
            # 更新字段值
            setattr(self, key, value)

        # ========== 配置停止词列表 ==========
        
        # 如果 suffix 存在且其最后一个元素不在停止词列表中，则添加
        if self.suffix and self.suffix[-1] not in self.stop_words:
            self.stop_words.append(self.suffix[-1])
        
        # 如果 tokenizer 的 EOS token 不在停止词列表中，则添加
        if tokenizer.eos_token not in self.stop_words:
            self.stop_words.append(tokenizer.eos_token)

        # ========== 设置停止 token ID ==========
        
        # 默认使用 tokenizer 的 EOS token ID
        self.stop_token_id = tokenizer.eos_token_id
        
        # 如果有自定义的 suffix，尝试从中提取停止 token ID
        if self.suffix:
            # 获取 suffix 的最后一个元素
            suffix_tokens = self.suffix[-1]
            
            # 情况1：suffix 最后一个元素是字符串（token 文本）
            if isinstance(suffix_tokens, str):
                # 将 token 文本转换为 token ID
                stop_token_id = tokenizer.convert_tokens_to_ids(suffix_tokens)
            
            # 情况2：suffix 最后一个元素是只包含一个 ID 的列表
            elif isinstance(suffix_tokens, list) and len(suffix_tokens) == 1:
                # 直接使用该 ID
                stop_token_id = suffix_tokens[0]
            
            # 情况3：其他情况（多个 token 或空列表）
            else:
                stop_token_id = None
            
            # 如果成功提取到停止 token ID，则使用它替代默认的 EOS token ID
            if stop_token_id is not None:
                self.stop_token_id = stop_token_id

    def check_system(self, system: Optional[str]) -> None:
        """
        验证系统提示词的配置是否合法。
        
        该方法检查当前模板是否支持系统提示词。如果提供了系统提示词但模板不支持，
        则会抛出断言错误。
        
        功能：
            验证系统提示词配置与模板支持能力的一致性
        
        参数：
            system (Optional[str]): 系统提示词内容，如果为 None 则无需验证
                例如：'You are a helpful assistant.'
        
        返回值：
            无（如果验证失败会触发 AssertionError）
        
        异常：
            AssertionError: 当提供了系统提示词但模板不支持时抛出，
                错误信息包含模板类型和提供的系统提示词内容
        
        使用示例：
            # 示例1：支持系统提示的模板
            meta = TemplateMeta(
                template_type='chatml',
                prefix=['<s>'],
                prompt=['<|im_start|>user\\n{{QUERY}}<|im_end|>\\n'],
                chat_sep=['\\n'],
                system_prefix=['<|im_start|>system\\n{{SYSTEM}}<|im_end|>\\n']
            )
            # meta.support_system == True
            meta.check_system('You are a helpful assistant.')  # 验证通过
            meta.check_system(None)  # 验证通过
            
            # 示例2：不支持系统提示的模板
            meta2 = TemplateMeta(
                template_type='simple',
                prefix=[],
                prompt=['{{QUERY}}'],
                chat_sep=None
            )
            # meta2.support_system == False
            meta2.check_system(None)  # 验证通过
            # meta2.check_system('You are a helpful assistant.')  # 抛出 AssertionError
            
            # 示例3：在初始化时自动调用（__post_init__）
            meta3 = TemplateMeta(
                template_type='test',
                prefix=[],
                prompt=['{{QUERY}}'],
                chat_sep=None,
                default_system='You are an AI.'  # 设置了默认系统提示
            )
            # __post_init__ 会自动调用 check_system(default_system)
            # 由于模板不支持系统提示，会在初始化时抛出 AssertionError
        """
        # 如果提供了系统提示词（不为 None）
        if system is not None:
            # 断言：模板必须支持系统提示
            # 如果不支持，抛出断言错误，提示模板类型和尝试使用的系统提示词
            assert self.support_system, (
                f'The template does not support `system`, template_type: {self.template_type}, system: {system}')
