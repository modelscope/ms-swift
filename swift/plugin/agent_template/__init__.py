# Copyright (c) Alibaba, Inc. and its affiliates.
from .base import BaseAgentTemplate
from .extra import ReactGRPOAgentTemplate
from .glm4 import GLM4AgentTemplate
from .hermes import HermesAgentTemplate
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
    #
    'toolbench': ToolBenchAgentTemplate,
    'glm4': GLM4AgentTemplate,
    # extra
    'react_grpo': ReactGRPOAgentTemplate
}
