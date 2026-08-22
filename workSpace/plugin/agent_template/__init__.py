# Copyright (c) Alibaba, Inc. and its affiliates.
from .base import BaseAgentTemplate
from .extra import ReactGRPOAgentTemplate
from .glm4 import GLM4_0414AgentTemplate, GLM4AgentTemplate
from .hermes import HermesAgentTemplate
from .llama import Llama3AgentTemplate, Llama4AgentTemplate
from .qwen import QwenEnAgentTemplate, QwenEnParallelAgentTemplate, QwenZhAgentTemplate, QwenZhParallelAgentTemplate
from .react import ReactEnAgentTemplate, ReactZnAgentTemplate
from .toolbench import ToolBenchAgentTemplate

agent_templates = {
    # ref: https://qwen.readthedocs.io/zh-cn/latest/framework/function_call.html#function-calling-templates
    'react_en': ReactEnAgentTemplate,
    'react_zh': ReactZnAgentTemplate,
    # ref: https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/fncall_prompts/qwen_fncall_prompt.py
    'qwen_en': QwenEnAgentTemplate,
    'qwen_zh': QwenZhAgentTemplate,
    'qwen_en_parallel': QwenEnParallelAgentTemplate,
    'qwen_zh_parallel': QwenZhParallelAgentTemplate,
    'hermes': HermesAgentTemplate,
    'toolbench': ToolBenchAgentTemplate,  # ref: https://modelscope.cn/datasets/swift/ToolBench
    'glm4': GLM4AgentTemplate,
    'glm4_0414': GLM4_0414AgentTemplate,  # ref: https://modelscope.cn/models/ZhipuAI/GLM-4-9B-0414
    'llama3': Llama3AgentTemplate,
    'llama4': Llama4AgentTemplate,
    # extra
    'react_grpo': ReactGRPOAgentTemplate
}
