"""模块文档注释：
该脚本定义了文本生成相关的参数数据类 `GenerationArguments`，
用于配置最大生成长度、采样参数（temperature/top_k/top_p）、
重复惩罚、束搜索数量、流式输出、停用词、以及 logprobs 等。
并提供了初始化流式开关的辅助方法与将参数映射为请求配置的功能。
"""

# 版权声明：本文件版权归 Alibaba 及其关联方所有
# Copyright (c) Alibaba, Inc. and its affiliates.

# 导入 dataclass 与 field：用于声明数据类与字段默认值/工厂
from dataclasses import dataclass, field
# 导入类型注解：List 列表类型、Optional 可选类型
from typing import List, Optional

# 导入日志工具：用于创建模块级日志记录器
from swift.utils import get_logger

# 初始化模块级日志记录器
logger = get_logger()


# 使用 dataclass 声明文本生成参数容器类
@dataclass
class GenerationArguments:
    """文本生成参数容器。

    该数据类承载生成阶段的关键参数，覆盖最大新生成长度、随机采样控制、
    束搜索、停用词与流式输出、以及 logprobs 返回等配置；
    同时提供初始化流式输出开关与构建请求配置的辅助方法。

    参数：
        max_new_tokens (Optional[int]): 最大新生成 token 数；None 表示不限制（受模型最大长度约束）。
        temperature (Optional[float]): 采样温度；为 0 时等价于关闭采样（贪心/束搜索）。
        top_k (Optional[int]): Top-K 采样阈值。
        top_p (Optional[float]): Top-P（核采样）阈值。
        repetition_penalty (Optional[float]): 重复惩罚系数。
        num_beams (int): 束搜索的束数。
        stream (Optional[bool]): 是否启用流式输出；None 表示待初始化。
        stop_words (List[str]): 触发提前停止的停用词列表。
        logprobs (bool): 是否返回 token 级对数概率。
        top_logprobs (Optional[int]): 返回前 n 个最高对数概率条目。
    """

    # 生成配置分组说明
    # 最大新生成 token 数；None 表示不限制（仍受模型总长度限制）
    max_new_tokens: Optional[int] = None  # Unlimited, constrained by max_model_len.
    # 采样温度；None 表示沿用 generation_config；为 0 等价于 do_sample=False
    temperature: Optional[float] = None  # Set to 0, which means do_sample is False.
    # Top-K 采样阈值；None 表示沿用 generation_config
    top_k: Optional[int] = None
    # Top-P（核采样）阈值；None 表示沿用 generation_config
    top_p: Optional[float] = None
    # 重复惩罚系数；None 表示沿用 generation_config
    repetition_penalty: Optional[float] = None
    # 束搜索束数；默认为 1
    num_beams: int = 1

    # 是否启用流式输出；None 表示延后由 _init_stream 判定
    stream: Optional[bool] = None
    # 停用词列表；用于在生成中途提前停止
    stop_words: List[str] = field(default_factory=list)
    # 是否返回 token 级对数概率
    logprobs: bool = False
    # 返回前 n 个对数概率条目；None 表示不限制或由上层决定
    top_logprobs: Optional[int] = None

    # 初始化流式输出开关
    def _init_stream(self):
        """若 `stream` 未显式设置，则默认关闭流式输出。"""
        # 当 stream 为 None 时，设置为 False
        if self.stream is None:
            self.stream = False

    # 构建请求配置对象（仅对 causal_lm 任务生效）
    def get_request_config(self):
        """基于当前生成参数构造请求配置。

        仅当 `task_type == 'causal_lm'` 时返回配置对象，否则返回 None。

        返回：
            Optional[RequestConfig]: 文本生成请求配置；非因果语言建模任务返回 None。
        """
        # 若任务类型不是因果语言建模，则不返回配置
        if getattr(self, 'task_type') != 'causal_lm':
            return
        # 延迟导入以避免不必要的依赖开销
        from swift.llm import RequestConfig

        # 将参数映射到请求配置对象并返回
        return RequestConfig(
            # 最大生成 token 数
            max_tokens=self.max_new_tokens,
            # 采样温度
            temperature=self.temperature,
            # Top-P（核采样）
            top_p=self.top_p,
            # Top-K 采样
            top_k=self.top_k,
            # 束搜索束数
            num_beams=self.num_beams,
            # 停用词列表
            stop=self.stop_words,
            # 是否流式输出
            stream=self.stream,
            # 重复惩罚系数
            repetition_penalty=self.repetition_penalty,
            # 是否返回 token 对数概率
            logprobs=self.logprobs,
            # 返回前 n 个对数概率条目
            top_logprobs=self.top_logprobs)
