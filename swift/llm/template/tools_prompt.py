from typing import List, Dict, Union, Optional


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
    return REACT_ZH_PROMPT.format(tool_list='\n\n'.join(tool_descs), tool_names=','.join(tool_names))


def format_glm4(tool_names, tool_descs):
    GLM4_PROMPT = '''你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。

    # 可用工具

    {tool_list}'''
    tool_list = ''
    for name, tool in zip(tool_names, tool_descs):
        tool_list += f'## {name}\n\n{tool}\n\n'
    return GLM4_PROMPT.format(tool_list=tool_list)


def format_toolbench(tool_names, tool_descs):
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
    return TOOLBENCH_PROMPT.format(tool_list='\n\n'.join(tool_descs))


tools_prompt = {
    'react_en': format_react_en,
    'react_zh': format_react_zh,
    'glm4': format_glm4,
    'toolbench': format_toolbench,
}


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
    prompt_format = tools_prompt.get(prompt_format) or format_toolbench
    return prompt_format(tool_names, tool_descs)


