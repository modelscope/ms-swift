"""模块功能概述：
该模块实现了用于推理引擎的猴子补丁（Monkey Patch）工具函数，主要用于临时替换
Transformers 库中的 AutoTokenizer 和 AutoConfig 类的 from_pretrained 方法。

核心功能：
1. patch_auto_tokenizer: 在上下文中将 AutoTokenizer.from_pretrained 替换为返回指定 tokenizer 实例的函数；
2. patch_auto_config: 在上下文中将 AutoConfig.from_pretrained 替换为返回指定 config 实例的函数。

应用场景：
- 避免重复从磁盘加载已存在于内存中的 tokenizer 或 config；
- 在推理引擎初始化时注入自定义的 tokenizer 或 config 实例；
- 兼容某些第三方库（如 vLLM）内部调用 from_pretrained 的场景。
"""
# Copyright (c) Alibaba, Inc. and its affiliates.  # 版权声明，标注代码版权所有者
from contextlib import contextmanager  # 引入上下文管理器装饰器，用于实现临时补丁的自动恢复
from functools import wraps
from sre_parse import Tokenizer  # 引入 wraps 装饰器，用于保留被包装函数的元信息（如 __name__、__doc__）

from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase  # 引入 Transformers 库的自动类和基类类型，用于类型注解和方法替换


@contextmanager  # 使用上下文管理器装饰器，确保补丁在退出时自动恢复
def patch_auto_tokenizer(tokenizer: PreTrainedTokenizerBase):
    """函数功能：
    临时替换 AutoTokenizer.from_pretrained 方法，使其直接返回指定的 tokenizer 实例，
    而不是从磁盘重新加载。补丁在上下文结束后自动恢复原始方法。
    
    参数：
        tokenizer (PreTrainedTokenizerBase): 预先加载的 tokenizer 实例，将在上下文中被重复使用
    
    返回值：
        ContextManager: 上下文管理器，可用于 with 语句
    
    实现逻辑：
        1. 保存原始的 AutoTokenizer.from_pretrained 方法；
        2. 创建新的 _from_pretrained 函数，忽略所有参数并直接返回传入的 tokenizer；
        3. 使用 @wraps 装饰器保留原始方法的元信息（如函数签名、文档字符串）；
        4. 替换 AutoTokenizer.from_pretrained 为新函数；
        5. 在 finally 块中恢复原始方法，确保即使发生异常也能恢复。
    
    应用场景：
        在使用 vLLM 或 LMDeploy 等推理引擎时，避免内部重复调用 from_pretrained 导致的性能损失。
    
    示例：
        >>> tokenizer = AutoTokenizer.from_pretrained('qwen/Qwen-7B')
        >>> with patch_auto_tokenizer(tokenizer):
        ...     # 在此上下文中，任何 AutoTokenizer.from_pretrained() 调用都会返回同一个 tokenizer
        ...     new_tokenizer = AutoTokenizer.from_pretrained('any/path')  # 返回的是传入的 tokenizer
        >>> # 上下文结束后，AutoTokenizer.from_pretrained 恢复原始行为
    """
    _old_from_pretrained = AutoTokenizer.from_pretrained  # 保存原始的 from_pretrained 类方法到局部变量，用于后续恢复

    @wraps(_old_from_pretrained)  # 使用 wraps 装饰器，将原始方法的 __name__、__doc__ 等元信息复制到新函数
    def _from_pretrained(*args, **kwargs):
        """定义新的 from_pretrained 函数，接受任意位置参数和关键字参数（但实际忽略），直接返回预先传入的 tokenizer 实例。"""
        return tokenizer

    AutoTokenizer.from_pretrained = _from_pretrained  # 将 AutoTokenizer 的 from_pretrained 类方法替换为新函数（应用补丁）
    try:  # 开始异常处理块，确保即使上下文代码抛出异常也能恢复原始方法
        yield  # 将控制权交给 with 语句块内的代码（此时补丁已生效）
    finally:  # 无论 with 块是否正常结束或抛出异常，都会执行此块
        AutoTokenizer.from_pretrained = _old_from_pretrained  # 恢复原始的 from_pretrained 方法（移除补丁）


@contextmanager  # 使用上下文管理器装饰器，确保补丁在退出时自动恢复
def patch_auto_config(config: PretrainedConfig):
    """函数功能：
    临时替换 AutoConfig.from_pretrained 方法，使其直接返回指定的 config 实例，
    而不是从磁盘重新加载。补丁在上下文结束后自动恢复原始方法。
    
    参数：
        config (PretrainedConfig): 预先加载的 config 实例，将在上下文中被重复使用
    
    返回值：
        ContextManager: 上下文管理器，可用于 with 语句
    
    实现逻辑：
        1. 保存原始的 AutoConfig.from_pretrained 方法；
        2. 创建新的 _from_pretrained 函数，根据 return_unused_kwargs 参数决定返回格式：
           - 若 kwargs 中包含 'return_unused_kwargs'，返回元组 (config, {})；
           - 否则，仅返回 config。
        3. 使用 @wraps 装饰器保留原始方法的元信息；
        4. 替换 AutoConfig.from_pretrained 为新函数；
        5. 在 finally 块中恢复原始方法，确保即使发生异常也能恢复。
    
    应用场景：
        在推理引擎初始化时，避免重复加载已存在于内存中的模型配置，提升性能。
    
    示例：
        >>> config = AutoConfig.from_pretrained('qwen/Qwen-7B')
        >>> with patch_auto_config(config):
        ...     # 在此上下文中，任何 AutoConfig.from_pretrained() 调用都会返回同一个 config
        ...     new_config = AutoConfig.from_pretrained('any/path')  # 返回的是传入的 config
        ...     # 测试 return_unused_kwargs 参数
        ...     cfg, unused = AutoConfig.from_pretrained('any/path', return_unused_kwargs=True)
        >>> # 上下文结束后，AutoConfig.from_pretrained 恢复原始行为
    """
    _old_from_pretrained = AutoConfig.from_pretrained  # 保存原始的 from_pretrained 类方法到局部变量，用于后续恢复

    @wraps(_old_from_pretrained)  # 使用 wraps 装饰器，将原始方法的 __name__、__doc__ 等元信息复制到新函数
    def _from_pretrained(*args, **kwargs):
        """替换后的 from_pretrained 方法，接受任意位置参数和关键字参数，根据参数决定返回 config 或 (config, {})。"""
        return (config, {}) if 'return_unused_kwargs' in kwargs else config  # 若 kwargs 中包含 'return_unused_kwargs' 参数，返回元组 (config, {})（空字典表示没有未使用的参数）；否则仅返回 config

    AutoConfig.from_pretrained = _from_pretrained  # 将 AutoConfig 的 from_pretrained 类方法替换为新函数（应用补丁）
    try:  # 开始异常处理块，确保即使上下文代码抛出异常也能恢复原始方法
        yield  # 将控制权交给 with 语句块内的代码（此时补丁已生效）
    finally:  # 无论 with 块是否正常结束或抛出异常，都会执行此块
        AutoConfig.from_pretrained = _old_from_pretrained  # 恢复原始的 from_pretrained 方法（移除补丁）
