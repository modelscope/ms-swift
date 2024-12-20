# Copyright (c) Alibaba, Inc. and its affiliates.
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import json


@dataclass
class AgentKeyword:
    action: str = 'Action:'
    action_input: str = 'Action Input:'
    observation: str = 'Observation:'


def format_react_en(tool_names, tool_descs):
    REACT_PROMPT = """Answer the following questions as best as you can. You have access to the following tools:

{tool_list}

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Final Answer: the final answer to the original input question

Begin!
"""
    tool_descs = [json.dumps(t) if not isinstance(t, str) else t for t in tool_descs]
    return REACT_PROMPT.format(tool_list='\n\n'.join(tool_descs), tool_names=','.join(tool_names))


def format_react_zh(tool_names, tool_descs):
    REACT_ZH_PROMPT = """尽你所能回答以下问题。你拥有如下工具：

{tool_list}

使用以下格式回答：

Thought: 思考你应该做什么
Action: 工具的名称，必须是[{tool_names}]之一
Action Input: 工具的输入
Observation: 工具返回的结果
... (Thought/Action/Action Input/Observation的过程可以重复零次或多次)
Final Answer: 对输入问题的最终答案

开始！
"""
    tool_descs = [json.dumps(t) if not isinstance(t, str) else t for t in tool_descs]
    return REACT_ZH_PROMPT.format(tool_list='\n\n'.join(tool_descs), tool_names=','.join(tool_names))


def format_glm4(tool_names, tool_descs):
    GLM4_PROMPT = """你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。

# 可用工具

{tool_list}"""
    tool_descs = [json.dumps(t) if not isinstance(t, str) else t for t in tool_descs]
    tool_list = ''
    for name, tool in zip(tool_names, tool_descs):
        tool_list += f'## {name}\n\n{tool}\n\n'
    return GLM4_PROMPT.format(tool_list=tool_list)


def format_toolbench(tool_names, tool_descs):
    TOOLBENCH_PROMPT = """You can use many tools(functions) to do the following task.
First I will give you the task description, and your task start.
At each step, you need to give your thought to analyze the status now and what to do next, \
with a function call to actually excute your step. Your output should follow this format:
Thought:
Action:
Action Input:

After the call, you will get the call result, and you are now in a new state.
Then you will analyze your status now, then decide what to do next...
After many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.
Remember:
1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, \
say \"I give up and restart\".
2.All the thought is short, at most in 5 sentence.
3.You can do more then one trys, so if your plan is to continusly try some conditions, \
you can do one of the conditions per try.
Let's Begin!
Task description: You should use functions to help handle the real time user querys. Remember:
1.ALWAYS call \"Finish\" function at the end of the task. And the final answer should contain enough information \
to show to the user,If you can't handle the task, \
or you find that function calls always fail(the function is not valid now), \
use function Finish->give_up_and_restart.
2.Do not use origin tool names, use only subfunctions' names.
Specifically, you have access to the following APIs: {tool_list}"""
    tool_descs = [json.dumps(t) if not isinstance(t, str) else t for t in tool_descs]
    return TOOLBENCH_PROMPT.format(tool_list='\n\n'.join(tool_descs))


def format_qwen(tool_names, tool_descs):
    PROMPT = '''You are a helpful assistant.

当前时间：{date}

# 工具

## 你拥有如下工具：

{tool_list}

## 你可以在回复中插入以下命令以调用这些工具：

{format_list}
    '''
    # 定义星期映射
    weekdays = {0: '星期一', 1: '星期二', 2: '星期三', 3: '星期四', 4: '星期五', 5: '星期六', 6: '星期日'}
    now = dt.datetime.now()
    year = now.year
    month = now.month
    day = now.day
    weekday = weekdays[now.weekday()]
    formatted_date = f'{year}年{month:02d}月{day:02d}日，{weekday}'
    PROMPT = PROMPT.replace('{date}', formatted_date)
    tool_list = ''
    for name, tool in zip(tool_names, tool_descs):
        tool_list += f'### {name} \n{name}: {tool["description"]} 输入参数: {json.dumps(tool["parameters"])}\n'

    PROMPT = PROMPT.replace('{tool_list}', tool_list)

    format_list = ''
    for i, _ in enumerate(tool_names):
        format_list += f'✿FUNCTION✿:工具{i+1}的名称\n✿ARGS✿:工具{i + 1}的输入\n✿RESULT✿:工具{i + 1}的结果\n'
    PROMPT = PROMPT.replace('{format_list}', format_list)
    return PROMPT


def format_custom(tool_names, tool_descs):
    PROMPT = '''你是一个人工智能助手。你的任务是针对用户的问题和要求提供适当的答复和支持。

    # 可用工具

    {tool_list}'''
    tool_list = ''
    tool_descs = [json.dumps(t) if not isinstance(t, str) else t for t in tool_descs]
    for name, tool in zip(tool_names, tool_descs):
        tool_list += f'## {name}\n\n{tool}\n\n'
    return PROMPT.format(tool_list=tool_list)


# Add your prompt here, use --tools_prompt to train
tools_prompt = {
    'react_en': (format_react_en, AgentKeyword().__dict__),
    'react_zh': (format_react_zh, AgentKeyword().__dict__),
    'glm4': (format_glm4, AgentKeyword().__dict__),
    'toolbench': (format_toolbench, AgentKeyword().__dict__),
    'qwen': (format_qwen, AgentKeyword(
        action='✿FUNCTION✿:',
        action_input='✿ARGS✿:',
        observation='✿RESULT✿:',
    ).__dict__),
    'custom': (format_custom, AgentKeyword().__dict__),
}


def get_tools_prompt(tools: List[Dict[str, Union[str, Dict]]], prompt_format: str = 'react_en') -> Optional[str]:
    tool_names: List[str] = []
    tool_descs: List[str] = []
    for info in tools:  # info: Dict[str, Union[str, dict]]
        try:
            if isinstance(info, dict) and 'function' in info:
                info = info['function']
            tool_names.append(info['name'])
            tool_descs.append(info)  # info: dict
        except KeyError:
            print('invalid tools format, please check'
                  'https://github.com/modelscope/swift/blob/main/docs/source_en/LLM/Agent-deployment-best-practice.md')
            return None
    prompt_format = tools_prompt.get(prompt_format, (None, None))[0] or format_toolbench
    return prompt_format(tool_names, tool_descs)


def get_tools_keyword(prompt_format: str = 'react_en') -> Dict[str, str]:
    keyword = tools_prompt.get(prompt_format, (None, None))[1] or AgentKeyword().__dict__
    return keyword
