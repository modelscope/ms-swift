# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple, Union

from swift.utils import get_logger
from swift.utils.utils import split_str_parts_by

logger = get_logger()

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

TOOLBENCH_PROMPT = '''You can use many tools(functions) to do the following task.
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
Specifically, you have access to the following APIs: {tool_list}'''

GLM4_PROMPT = '''你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。

# 可用工具

{tool_list}'''


def calculate_loss_scale(query: str,
                         response: str,
                         use_loss_scale=False,
                         response_loss_scale_map: Optional[Dict[str, list]] = None,
                         query_loss_scale_map: Optional[Dict[str, list]] = None) -> Tuple[List[str], List[float]]:
    """Calculate the loss scale by splitting the agent response.

    This algorithm comes from paper: https://arxiv.org/pdf/2309.00986.pdf

    Agent response format:

    ```text
        Thought: you should always think about what to do
        Action: the action to take, should be one of the above tools[fire_recognition,
            fire_alert, call_police, call_fireman]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
    ```

    Args:
        response: The response text
        use_loss_scale: Use weighted loss. With this, some part of the loss will be enhanced to improve performance.

    Returns:
        A tuple of agent response parts and their weights.
    """
    if use_loss_scale:
        # query loss scale map
        if query_loss_scale_map is not None:
            for key in query_loss_scale_map.keys():
                if key in query:
                    if isinstance(query_loss_scale_map[key], (float, int)):
                        query_loss_scale_map[key] = [query_loss_scale_map[key]]
                    loss_scale_value = query_loss_scale_map[key][0]
                    return [response], [float(loss_scale_value)]
        delimiters = list(k for k in response_loss_scale_map.keys() if len(response_loss_scale_map[k]) == 2)
        agent_parts = split_str_parts_by(response, delimiters)
        regex_delimiters = {k: v for k, v in response_loss_scale_map.items() if len(v) == 1}
        if len(regex_delimiters):
            split_parts_by_regex(agent_parts, regex_delimiters)
        weights = []
        agent_content = []
        for c in agent_parts:
            if isinstance(c['key'], (float, int)):
                weights += [c['key']]
                agent_content.append(c['content'])
            else:
                if c['key'] in response_loss_scale_map:
                    weights += [response_loss_scale_map[c['key']][0]]
                    weights += [response_loss_scale_map[c['key']][1]]
                    agent_content.append(c['key'])
                    agent_content.append(c['content'])
                else:
                    weights += [1.0]
                    agent_content.append(c['content'])
        return agent_content, weights
    else:
        return [response], [1.0]


def split_action_action_input(response: str) -> Tuple[Optional[str], Optional[str]]:
    agent_keyword = [
        'action:', 'Action:', 'ACTION:', 'action input:', 'Action Input:', 'Action input:', 'ACTION INPUT:', 'Thought:',
        'Final Answer:', 'Observation:'
    ]
    agent_parts = split_str_parts_by(response, agent_keyword)
    action = None
    action_input = None
    for c in agent_parts:
        if c['key'].lower() == 'action:':
            action = c['content']
        elif c['key'].lower() == 'action input:':
            action_input = c['content']
    if action:
        action = action.strip().replace('\n', '')
    if action_input:
        action_input.strip().replace('\n', '')
    return action, action_input


def split_parts_by_regex(text_list: list, regex_delimiters: Dict[str, List[float]]) -> None:
    import re
    compiled_patterns = [(re.compile(pattern), scale) for pattern, scale in regex_delimiters.items()]
    for i in range(len(text_list) - 1, -1, -1):
        item = text_list[i]
        if item.get('key') == '':
            res_text = item['content']
            last_idx = 0
            segments = []

            for pattern, scale in compiled_patterns:
                matches = list(re.finditer(pattern, res_text))
                for match in matches:
                    if match.start() > last_idx:
                        segments.append({'key': '', 'content': res_text[last_idx:match.start()]})
                    segments.append({'key': scale[0], 'content': match.group(0)})
                    last_idx = match.end()

            if last_idx < len(res_text):
                segments.insert(0, {'key': '', 'content': res_text[last_idx:]})

            if segments:
                text_list[i:i + 1] = segments


def get_tools_prompt(TOOLS: List[Dict[str, Union[str, dict]]], prompt_format: str = 'react_en') -> Optional[str]:
    tool_descs = []
    tool_names = []
    for info in TOOLS:  # info: Dict[str, Union[str, dict]]
        try:
            if 'function' in info:
                info = info['function']
            tool_names.append(info['name'])
            tool_descs.append(str(info))  # info: dict
        except KeyError:
            print('invalid tools format, please check'
                  'https://github.com/modelscope/swift/blob/main/docs/source_en/LLM/Agent-deployment-best-practice.md')
            return None
    if prompt_format == 'react_en':
        return REACT_PROMPT.format(tool_list='\n\n'.join(tool_descs), tool_names=','.join(tool_names))
    elif prompt_format == 'react_zh':
        return REACT_ZH_PROMPT.format(tool_list='\n\n'.join(tool_descs), tool_names=','.join(tool_names))
    elif prompt_format == 'glm4':
        tool_list = ''
        for name, tool in zip(tool_names, tool_descs):
            tool_list += f'## {name}\n\n{tool}\n\n'
        return GLM4_PROMPT.format(tool_list=tool_list)
    return TOOLBENCH_PROMPT.format(tool_list='\n\n'.join(tool_descs))
