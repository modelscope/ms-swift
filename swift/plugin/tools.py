from swift.llm.template.agent.tools import tools_prompt


def format_custom(tool_names, tool_descs):
    PROMPT = '''你是一个人工智能助手。你的任务是针对用户的问题和要求提供适当的答复和支持。

    # 可用工具

    {tool_list}'''
    tool_list = ''
    for name, tool in zip(tool_names, tool_descs):
        tool_list += f'## {name}\n\n{tool}\n\n'
    return PROMPT.format(tool_list=tool_list)


tools_prompt['custom'] = format_custom
