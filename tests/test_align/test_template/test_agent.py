import os

from swift.plugin import agent_templates

os.environ['SWIFT_DEBUG'] = '1'

system = 'You are a helpful assistant.'

tools = [{
    'name': 'get_current_weather',
    'description': 'Get the current weather in a given location',
    'parameters': {
        'type': 'object',
        'properties': {
            'location': {
                'type': 'string',
                'description': 'The city and state, e.g. San Francisco, CA'
            },
            'unit': {
                'type': 'string',
                'enum': ['celsius', 'fahrenheit']
            }
        },
        'required': ['location']
    }
}, {
    'name_for_model': 'tool2',
    'name_for_human': '工具2',
    'description': 'Tool2的描述',
}]

glm4_tools = [{
    'type': 'function',
    'function': {
        'name': 'realtime_aqi',
        'description': '天气预报。获取实时空气质量。当前空气质量，PM2.5，PM10信息',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'description': '城市名'
                }
            },
            'required': ['city']
        }
    }
}]
glm4_tool_messasges = [
    {
        'role': 'tool',
        'content': '{"city": "北京", "aqi": "10", "unit": "celsius"}'
    },
    {
        'role': 'tool',
        'content': '{"city": "上海", "aqi": "72", "unit": "fahrenheit"}'
    },
]
glm4_query = '北京和上海今天的天气情况'


def _infer(model, agent_template, num_tools: int = 1, agent_tools=None, tool_messages=None, query=None):
    if agent_tools is None:
        agent_tools = tools
    if tool_messages is None:
        tool_messages = []
        for _ in range(num_tools):
            tool_messages.append({
                'role': 'tool',
                'content': '{"temperature": 32, "condition": "Sunny", "humidity": 50}'
            })
    engine = PtEngine(model)
    engine.default_template.agent_template = agent_template
    stop = [agent_template.keyword.observation]
    query = query or "How's the weather in Beijing today?"
    infer_request = InferRequest([{'role': 'user', 'content': query}], tools=agent_tools)
    request_config = RequestConfig(max_tokens=512, stop=stop)
    resp_list = engine.infer([infer_request], request_config=request_config)
    response = resp_list[0].choices[0].message.content
    toolcall = resp_list[0].choices[0].message.tool_calls[0].function
    print(f'response: {response}')
    print(f'toolcall: {toolcall}')
    assert toolcall is not None
    infer_request.messages.append({'role': 'assistant', 'content': response})
    infer_request.messages += tool_messages
    resp_list = engine.infer([infer_request], request_config=request_config)
    response2 = resp_list[0].choices[0].message.content
    print(f'response2: {response2}')
    return response2


def test_react_en():
    agent_template = agent_templates['react_en']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 1144
    _infer('Qwen/Qwen2.5-7B-Instruct', agent_template)


def test_react_zh():
    agent_template = agent_templates['react_zh']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 712
    _infer('Qwen/Qwen2.5-7B-Instruct', agent_template)


def test_qwen_en():
    agent_template = agent_templates['qwen_en']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 879
    _infer('Qwen/Qwen2.5-7B-Instruct', agent_template)


def test_qwen_zh():
    agent_template = agent_templates['qwen_zh']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 577
    _infer('Qwen/Qwen2.5-7B-Instruct', agent_template)


def test_qwen_en_parallel():
    agent_template = agent_templates['qwen_en_parallel']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 1012
    _infer('Qwen/Qwen2.5-7B-Instruct', agent_template, num_tools=2)


def test_qwen_zh_parallel():
    agent_template = agent_templates['qwen_zh_parallel']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 688
    _infer('Qwen/Qwen2.5-7B-Instruct', agent_template, num_tools=2)


def test_hermes():
    agent_template = agent_templates['hermes']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 875
    _infer('Qwen/Qwen2.5-7B-Instruct', agent_template, num_tools=2)


def test_toolbench():
    agent_template = agent_templates['toolbench']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 1833
    _infer('Qwen/Qwen2.5-7B-Instruct', agent_template)


def test_glm4():
    agent_template = agent_templates['glm4']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 846
    _infer(
        'ZhipuAI/glm-4-9b-chat',
        agent_template,
        agent_tools=glm4_tools,
        tool_messages=glm4_tool_messasges,
        query=glm4_query)


def test_glm4_0414():
    agent_template = agent_templates['glm4_0414']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 769
    response = _infer(
        'ZhipuAI/GLM-4-9B-0414',
        agent_template,
        agent_tools=glm4_tools,
        tool_messages=glm4_tool_messasges,
        query=glm4_query)
    assert response == '根据天气预报工具，北京今天的空气质量指数为10，属于良好水平；上海今天的空气质量指数为72，属于轻度污染水平。'


if __name__ == '__main__':
    from swift.llm import PtEngine, InferRequest, RequestConfig
    # test_react_en()
    # test_react_zh()
    # test_qwen_en()
    # test_qwen_zh()
    # test_qwen_en_parallel()
    # test_qwen_zh_parallel()
    # test_hermes()
    # test_toolbench()
    # test_glm4()
    test_glm4_0414()
