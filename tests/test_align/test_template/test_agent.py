from swift.plugin import agent_templates

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


def test_react_en():
    agent_template = agent_templates['react_en']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 1144


def test_react_zh():
    agent_template = agent_templates['react_zh']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 712


def test_qwen_en():
    agent_template = agent_templates['qwen_en']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 879


def test_qwen_zh():
    agent_template = agent_templates['qwen_zh']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 577


def test_qwen_en_parallel():
    agent_template = agent_templates['qwen_en_parallel']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 1012


def test_qwen_zh_parallel():
    agent_template = agent_templates['qwen_zh_parallel']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 688


def test_hermes():
    agent_template = agent_templates['hermes']()
    new_system = agent_template.format_system(tools, system)
    assert len(new_system) == 875


if __name__ == '__main__':
    test_react_en()
    test_react_zh()
    test_qwen_en()
    test_qwen_zh()
    test_qwen_en_parallel()
    test_qwen_zh_parallel()
    test_hermes()
