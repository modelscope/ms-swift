# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List, Tuple, Union

from swift.utils import get_logger
from swift.utils.utils import split_str_parts_by

logger = get_logger()


def calculate_loss_scale(response: str, use_loss_scale=False) -> Tuple[List[str], List[float]]:
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
    if 'Action:' in response and 'Observation:' in response and use_loss_scale:
        agent_keyword = ['Action:', 'Action Input:', 'Thought:', 'Final Answer:', 'Observation:']
        agent_parts = split_str_parts_by(response, agent_keyword)
        weights = []
        agent_content = []
        for c in agent_parts:
            if c['key'] in ('Action:', 'Action Input:'):
                weights += [2.0]
                weights += [2.0]
            elif c['key'] in ('Thought:', 'Final Answer:', ''):
                weights += [1.0]
                weights += [1.0]
            elif c['key'] in ('Observation:', ):
                weights += [2.0]
                weights += [0.0]
            agent_content.append(c['key'])
            agent_content.append(c['content'])
        return agent_content, weights
    elif ('Action:' in response or 'Next:' in response) and use_loss_scale:  # alpha-umi
        agent_keyword = ['Next:', 'Action:', 'Action Input:']
        agent_parts = split_str_parts_by(response, agent_keyword)
        weights = []
        agent_content = []
        for c in agent_parts:
            if c['key'] in ('Action:', 'Action Input:', 'Next:'):
                weights += [2.0]
                weights += [2.0]
            elif c['key'] in ('Thought:', 'Final Answer:', ''):
                weights += [1.0]
                weights += [1.0]
            elif c['key'] in ('Observation:', ):
                weights += [2.0]
                weights += [0.0]
            agent_content.append(c['key'])
            agent_content.append(c['content'])
        return agent_content, weights
    else:
        return [response], [1.0]


def split_action_action_input(response):
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
    return action, action_input


def get_tools_prompt(TOOLS: list[dict[str, Union[str, dict]]], prompt_format: str = 'default') -> Tuple[str, str]:
    # ref:
    TOOL_DESC = """{tool_name}: Call this tool to interact with the {tool_name} API. \
    What is the {tool_name} API useful for? {tool_desc} \
    Parameters: {paras} Format the arguments as a JSON object."""

    DEFAULT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

    {tool_descs}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!
    """

    TOOLBENCH_PROMPT = '''You are AutoGPT, you can use many tools(functions) to do the following task.
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
    Specifically, you have access to the following APIs: {api_list}'''

    tool_descs = []
    tool_names = []
    for info in TOOLS:  # info: dict[str, Union[str, dict]]
        try:
            if 'function' in info:
                info = info['function']
            if prompt_format == 'default':
                tool_descs.append(
                    TOOL_DESC.format(tool_name=info['name'], tool_desc=info['description'], paras=info['parameters']))
                tool_names.append(info['name'])
            else:  # toolbench
                tool_descs.append(str(info))  # info: dict
        except KeyError:
            print('invalid tools format, please check the Agent.md document')
    tool_descs = '\n\n'.join(tool_descs)
    tool_names = ','.join(tool_names)
    if prompt_format == 'default':
        return DEFAULT_PROMPT.format(tool_descs=tool_descs,tool_names=tool_names)
    return TOOLBENCH_PROMPT.format(api_list=tool_descs)
