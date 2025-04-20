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


def _infer(model, agent_template, num_tools: int = 1):
    engine = PtEngine(model)
    engine.default_template.agent_template = agent_template
    stop = [agent_template.keyword.observation]
    query = "How's the weather in Beijing today?"
    infer_request = InferRequest([{'role': 'user', 'content': query}], tools=tools)
    request_config = RequestConfig(max_tokens=512, stop=stop)
    resp_list = engine.infer([infer_request], request_config=request_config)
    response = resp_list[0].choices[0].message.content
    toolcall = resp_list[0].choices[0].message.tool_calls[0].function
    print(f'response: {response}')
    print(f'toolcall: {toolcall}')
    assert toolcall is not None
    tool_result = '{"temperature": 32, "condition": "Sunny", "humidity": 50}'
    infer_request.messages.append({'role': 'assistant', 'content': response})
    for _ in range(num_tools):
        infer_request.messages.append({'role': 'tool', 'content': tool_result})
    resp_list = engine.infer([infer_request], request_config=request_config)
    response2 = resp_list[0].choices[0].message.content
    print(f'response2: {response2}')


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


def test_glm4():
    agent_template = agent_templates['glm4']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 846


def test_glm4_0414():
    agent_template = agent_templates['glm4_0414']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 769


if __name__ == '__main__':
    from swift.llm import PtEngine, InferRequest, RequestConfig
    # test_react_en()
    # test_react_zh()
    # test_qwen_en()
    # test_qwen_zh()
    # test_qwen_en_parallel()
    # test_qwen_zh_parallel()
    test_hermes()
    # test_toolbench()
    # test_glm4()
    # test_glm4_0414()
