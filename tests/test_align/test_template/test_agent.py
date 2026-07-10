import json
import os
import pytest
from types import SimpleNamespace

os.environ['SWIFT_DEBUG'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0,1,2,3'

system = 'You are a helpful assistant.'

tools = [{
    'type': 'function',
    'function': {
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


def _infer(engine, num_tools: int = 1, agent_tools=None, tool_messages=None, query=None):
    from swift import InferRequest, RequestConfig

    if agent_tools is None:
        agent_tools = tools
    if tool_messages is None:
        tool_messages = []
        for _ in range(num_tools):
            tool_messages.append({
                'role': 'tool',
                'content': '{"temperature": 32, "condition": "Sunny", "humidity": 50}'
            })
    stop = [engine.template.agent_template.keyword.observation]
    query = query or "How's the weather in Beijing today?"
    infer_request = InferRequest([{'role': 'user', 'content': query}], tools=agent_tools)
    request_config = RequestConfig(max_tokens=512, stop=stop, temperature=0)
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
    infer_request.messages.append({'role': 'assistant', 'content': response2})
    return infer_request.messages


def test_react_en():
    agent_template = agent_template_map['react_en']()
    new_system = agent_template._format_tools(tools, system)
    assert len(new_system) == 1144
    engine = TransformersEngine('Qwen/Qwen2.5-7B-Instruct')
    template = engine.template
    template._agent_template = 'react_en'
    messages = _infer(engine)
    assert messages[-1]['content'] == (
        'Thought: The current temperature in Beijing is 32 degrees Celsius, and the condition is sunny '
        'with a humidity of 50%.\nFinal Answer: The current temperature in Beijing is 32 degrees Celsius,'
        ' and the condition is sunny with a humidity of 50%.')
    template.set_mode('train')
    encoded = template.encode({'messages': messages})
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')

    dataset = load_dataset('AI-ModelScope/function-calling-chatml')[0]
    data = dataset[6]
    data['messages'].insert(1, data['messages'][1])
    data['messages'].insert(3, data['messages'][3])
    template.template_backend = 'swift'
    encoded = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')


def test_react_zh():
    agent_template = agent_template_map['react_zh']()
    new_system = agent_template._format_tools(tools, system)
    assert len(new_system) == 712
    engine = TransformersEngine('Qwen/Qwen2.5-7B-Instruct')
    template = engine.template
    template._agent_template = 'react_zh'
    _infer(engine)


def test_qwen_en():
    agent_template = agent_template_map['qwen_en']()
    new_system = agent_template._format_tools(tools, system)
    assert len(new_system) == 879
    engine = TransformersEngine('Qwen/Qwen2.5-7B-Instruct')
    template = engine.template
    template._agent_template = 'qwen_en'
    messages = _infer(engine)
    assert messages[-1]['content'] == (
        '✿RETURN✿: Today in Beijing, the temperature is 32°C with sunny conditions and the humidity '
        'is at 50%. Enjoy the nice weather!')
    template.set_mode('train')
    encoded = template.encode({'messages': messages})
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')

    dataset = load_dataset('AI-ModelScope/function-calling-chatml')[0]
    data = dataset[6]
    data['messages'].insert(1, data['messages'][1])
    data['messages'].insert(3, data['messages'][3])
    template.template_backend = 'swift'
    encoded = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')


def test_qwen_zh():
    agent_template = agent_template_map['qwen_zh']()
    new_system = agent_template._format_tools(tools, system)
    assert len(new_system) == 577
    engine = TransformersEngine('Qwen/Qwen2.5-7B-Instruct')
    template = engine.template
    template._agent_template = 'qwen_zh'
    _infer(engine)


def test_qwen_en_parallel():
    agent_template = agent_template_map['qwen_en_parallel']()
    new_system = agent_template._format_tools(tools, system)
    assert len(new_system) == 1012
    engine = TransformersEngine('Qwen/Qwen2.5-7B-Instruct')
    template = engine.template
    template._agent_template = 'qwen_en_parallel'
    messages = _infer(engine, num_tools=2)
    assert messages[-1]['content'] == (
        '✿RETURN✿: Today in Beijing, the temperature is 32 degrees Celsius with sunny conditions '
        'and the humidity is at 50%. Enjoy the nice weather!')
    template.set_mode('train')
    encoded = template.encode({'messages': messages})
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')

    dataset = load_dataset('AI-ModelScope/function-calling-chatml')[0]
    data = dataset[6]
    data['messages'].insert(1, data['messages'][1])
    data['messages'].insert(3, data['messages'][3])
    template.template_backend = 'swift'
    encoded = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')


def test_qwen_zh_parallel():
    agent_template = agent_template_map['qwen_zh_parallel']()
    new_system = agent_template._format_tools(tools, system)
    assert len(new_system) == 688
    engine = TransformersEngine('Qwen/Qwen2.5-7B-Instruct')
    template = engine.template
    template._agent_template = 'qwen_zh_parallel'
    _infer(engine, num_tools=2)


def test_hermes():
    agent_template = agent_template_map['hermes']()
    new_system = agent_template._format_tools(tools, system)
    assert len(new_system) == 875
    engine = TransformersEngine('Qwen/Qwen2.5-7B-Instruct')
    template = engine.template
    template._agent_template = 'hermes'
    messages = _infer(engine, num_tools=2)
    template.template_backend = 'jinja'
    messages2 = _infer(engine, num_tools=2)
    assert messages[-1]['content'] == messages2[-1]['content'] == (
        'Today in Beijing, the temperature is 32 degrees Celsius with sunny conditions '
        'and the humidity is at 50%. Enjoy the nice weather!')
    template.set_mode('train')
    encoded = template.encode({'messages': messages})
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')

    dataset = load_dataset('AI-ModelScope/function-calling-chatml')[0]
    data = dataset[6]
    data['messages'].insert(1, data['messages'][1])
    data['messages'].insert(3, data['messages'][3])
    template.template_backend = 'swift'
    encoded = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')
    template.template_backend = 'jinja'
    encoded2 = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded2["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded2["labels"])}')
    assert encoded['input_ids'] == encoded2['input_ids']


def test_toolbench():
    agent_template = agent_template_map['toolbench']()
    new_system = agent_template._format_tools(tools, system)
    assert len(new_system) == 1833
    engine = TransformersEngine('Qwen/Qwen2.5-7B-Instruct')
    template = engine.template
    template._agent_template = 'toolbench'
    _infer(engine)


def test_chatglm4():
    agent_template = agent_template_map['chatglm4']()
    new_system = agent_template._format_tools(tools, system)
    assert len(new_system) == 846
    engine = TransformersEngine('ZhipuAI/glm-4-9b-chat')
    template = engine.template
    template._agent_template = 'chatglm4'
    _infer(engine, agent_tools=glm4_tools, tool_messages=glm4_tool_messasges, query=glm4_query)


def test_glm4():
    agent_template = agent_template_map['glm4']()
    new_system = agent_template._format_tools(tools, system)
    assert len(new_system) == 769
    engine = TransformersEngine('ZhipuAI/GLM-4-9B-0414')
    template = engine.template
    template._agent_template = 'glm4'
    messages = _infer(engine, agent_tools=glm4_tools, tool_messages=glm4_tool_messasges, query=glm4_query)
    assert messages[-1]['content'] == '根据天气预报工具，北京今天的空气质量指数为10，属于良好水平；上海今天的空气质量指数为72，属于轻度污染水平。'
    template.set_mode('train')
    encoded = template.encode({'messages': messages})
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')

    dataset = load_dataset('AI-ModelScope/function-calling-chatml')[0]
    data = dataset[6]
    data['messages'].insert(1, data['messages'][1])
    data['messages'].insert(3, data['messages'][3])
    template.template_backend = 'swift'
    encoded = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')


def test_llama3():
    engine = TransformersEngine('LLM-Research/Llama-3.2-3B-Instruct')
    template = engine.template
    template._agent_template = 'llama3'
    messages = _infer(engine)

    template.set_mode('train')
    encoded = template.encode({'messages': messages})
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')

    dataset = load_dataset('AI-ModelScope/function-calling-chatml')[0]
    data = dataset[6]
    data['messages'].insert(1, data['messages'][1])
    data['messages'].insert(3, data['messages'][3])
    template.template_backend = 'swift'
    encoded = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')


def test_llama4():
    engine = TransformersEngine('LLM-Research/Llama-4-Scout-17B-16E-Instruct')
    template = engine.template
    messages = _infer(engine)
    template.set_mode('train')
    encoded = template.encode({'messages': messages})
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')


def test_hunyuan():
    engine = TransformersEngine('Tencent-Hunyuan/Hunyuan-1.8B-Instruct')
    template = engine.template
    template.template_backend = 'jinja'
    _infer(engine, num_tools=2)

    dataset = load_dataset('AI-ModelScope/function-calling-chatml')[0]
    data = dataset[6]
    data['messages'].insert(1, data['messages'][1])
    data['messages'].insert(3, data['messages'][3])
    template.template_backend = 'swift'
    template.set_mode('train')
    encoded = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')
    template.template_backend = 'jinja'
    encoded2 = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded2["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded2["labels"])}')
    assert encoded['input_ids'][:-1] == encoded2['input_ids']


def test_glm4_5():
    engine = TransformersEngine('ZhipuAI/GLM-4.5-Air')
    template = engine.template
    template.template_backend = 'jinja'
    _infer(engine, num_tools=2)

    dataset = load_dataset('AI-ModelScope/function-calling-chatml')[0]
    data = dataset[6]
    data['messages'].insert(1, data['messages'][1])
    data['messages'].insert(3, data['messages'][3])
    template.template_backend = 'swift'
    template.set_mode('train')
    encoded = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')
    template.template_backend = 'jinja'
    encoded2 = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded2["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded2["labels"])}')
    assert encoded['input_ids'][:-1] == encoded2['input_ids']


def test_glm4_7():
    engine = TransformersEngine('ZhipuAI/GLM-4.7-FP8', load_model=False)
    template = engine.template

    dataset = load_dataset('AI-ModelScope/function-calling-chatml')[0]
    data = dataset[6]
    data['messages'].insert(1, data['messages'][1])
    data['messages'].insert(3, data['messages'][3])
    template.template_backend = 'swift'
    template.set_mode('train')
    encoded = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')
    template.template_backend = 'jinja'
    encoded2 = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded2["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded2["labels"])}')
    assert encoded['input_ids'][:-1] == encoded2['input_ids']


def test_qwen3_coder():
    engine = TransformersEngine('Qwen/Qwen3-Coder-30B-A3B-Instruct')
    template = engine.template
    template.template_backend = 'jinja'
    _infer(engine, num_tools=2)

    dataset = load_dataset('AI-ModelScope/function-calling-chatml')[0]
    data = dataset[6]
    data['messages'].insert(1, data['messages'][1])
    data['messages'].insert(3, data['messages'][3])
    template.template_backend = 'swift'
    template.set_mode('train')
    encoded = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')
    template.template_backend = 'jinja'
    encoded2 = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded2["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded2["labels"])}')
    assert encoded['input_ids'] == encoded2['input_ids']


def test_qwen3_5():
    engine = TransformersEngine('Qwen/Qwen3.5-35B-A3B')
    template = engine.template
    template.template_backend = 'jinja'
    _infer(engine, num_tools=2)

    dataset = load_dataset('AI-ModelScope/function-calling-chatml')[0]
    data = dataset[6]
    data['messages'].insert(1, data['messages'][1])
    data['messages'].insert(3, data['messages'][3])
    data['messages'].insert(0, {'role': 'system', 'content': 'You are a helpful assistant.'})
    template.template_backend = 'swift'
    template.set_mode('train')
    encoded = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')
    template.template_backend = 'jinja'
    encoded2 = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded2["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded2["labels"])}')
    assert encoded['input_ids'] == encoded2['input_ids']


def test_deepseek_v3_1():
    engine = TransformersEngine('deepseek-ai/DeepSeek-V3.1', load_model=False)
    template = engine.template

    dataset = load_dataset('AI-ModelScope/function-calling-chatml')[0]
    data = dataset[6]
    # To test multiple tool calls and responses, we duplicate some messages.
    data['messages'].insert(1, data['messages'][1])
    data['messages'].insert(3, data['messages'][3])
    template.template_backend = 'swift'
    template.set_mode('train')
    encoded = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')
    template.template_backend = 'jinja'
    encoded2 = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded2["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded2["labels"])}')

    expected_input_ids = (
        '<｜begin▁of▁sentence｜>\n\n## Tools\n'
        'You have access to the following tools:\n\n'
        '### convert_temperature\n'
        'Description: Convert temperature from one unit to another\n\n'
        "Parameters: {\"type\": \"object\", \"properties\": {\"temperature\": {\"type\": \"number\", "
        "\"description\": \"The temperature value\"}, \"from_unit\": {\"type\": \"string\", \"description\": "
        "\"The unit to convert from\"}, \"to_unit\": {\"type\": \"string\", \"description\": \"The unit "
        "to convert to\"}}, \"required\": [\"temperature\", \"from_unit\", \"to_unit\"]}\n\n"
        '### get_current_date\n'
        'Description: Get the current date\n\n'
        'Parameters: {}\n\n'
        'IMPORTANT: ALWAYS adhere to this exact format for tool use:\n'
        '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>'
        '{additional_tool_calls}<｜tool▁calls▁end｜>\n\n'
        'Where:\n'
        '- `tool_call_name` must be an exact match to one of the available tools\n'
        "- `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema\n"
        '- For multiple tool calls, chain them directly without separators or spaces<｜User｜>'
        'Hi, I need to convert a temperature from Celsius to Fahrenheit. The temperature is 30 degrees Celsius.'
        '<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>convert_temperature<｜tool▁sep｜>'
        "{\"temperature\": 30, \"from_unit\": \"Celsius\", \"to_unit\": \"Fahrenheit\"}<｜tool▁call▁end｜>"
        '<｜tool▁call▁begin｜>convert_temperature<｜tool▁sep｜>'
        "{\"temperature\": 30, \"from_unit\": \"Celsius\", \"to_unit\": \"Fahrenheit\"}<｜tool▁call▁end｜>"
        '<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'
        "<｜tool▁output▁begin｜>{\"converted_temperature\": 86}<｜tool▁output▁end｜>"
        "<｜tool▁output▁begin｜>{\"converted_temperature\": 86}<｜tool▁output▁end｜>"
        'The converted temperature from 30 degrees Celsius to Fahrenheit is 86 degrees Fahrenheit.<｜end▁of▁sentence｜>')

    # Expected labels string
    expected_labels = (
        '[-100 * 239]</think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>convert_temperature<｜tool▁sep｜>'
        "{\"temperature\": 30, \"from_unit\": \"Celsius\", \"to_unit\": \"Fahrenheit\"}<｜tool▁call▁end｜>"
        '<｜tool▁call▁begin｜>convert_temperature<｜tool▁sep｜>'
        "{\"temperature\": 30, \"from_unit\": \"Celsius\", \"to_unit\": \"Fahrenheit\"}<｜tool▁call▁end｜>"
        '<｜tool▁calls▁end｜><｜end▁of▁sentence｜>[-100 * 22]'
        'The converted temperature from 30 degrees Celsius to Fahrenheit is 86 degrees Fahrenheit.<｜end▁of▁sentence｜>')

    assert template.safe_decode(encoded['input_ids']) == expected_input_ids
    assert template.safe_decode(encoded['labels']) == expected_labels
    assert encoded['input_ids'][-122:] == encoded2['input_ids'][1:]


def test_youtu():
    agent_template = agent_template_map['youtu']()
    new_system = agent_template._format_tools(tools, system)
    assert len(new_system) == 883
    engine = TransformersEngine('Tencent-YouTu-Research/Youtu-LLM-2B')
    template = engine.template
    template._agent_template = 'youtu'

    stop = [template.agent_template.keyword.observation]
    query = "How's the weather in Beijing today?"
    tool_messages = [{'role': 'tool', 'content': '{"temperature": 32, "condition": "Sunny", "humidity": 50}'}]
    infer_request = InferRequest([{'role': 'user', 'content': query}], tools=tools)
    request_config = RequestConfig(max_tokens=2048, stop=stop, temperature=0)

    # First inference: get tool call
    resp_list = engine.infer([infer_request], request_config=request_config)
    response = resp_list[0].choices[0].message.content
    toolcall = resp_list[0].choices[0].message.tool_calls
    print(f'response: {response}')
    print(f'toolcall: {toolcall}')
    assert toolcall is not None, 'No tool_call generated'
    infer_request.messages.append({'role': 'assistant', 'content': response})
    infer_request.messages += tool_messages

    # Second inference: get final response
    resp_list = engine.infer([infer_request], request_config=request_config)
    response2 = resp_list[0].choices[0].message.content
    print(f'response2: {response2}')
    infer_request.messages.append({'role': 'assistant', 'content': response2})
    messages = infer_request.messages

    template.set_mode('train')
    encoded = template.encode({'messages': messages})
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')

    dataset = load_dataset('AI-ModelScope/function-calling-chatml')[0]
    data = dataset[6]
    data['messages'].insert(1, data['messages'][1])
    data['messages'].insert(3, data['messages'][3])
    template.template_backend = 'swift'
    encoded = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')
    template.template_backend = 'jinja'
    encoded2 = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded2["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded2["labels"])}')
    assert encoded['input_ids'] == encoded2['input_ids']


def test_deepseek_v4():
    engine = TransformersEngine('deepseek-ai/DeepSeek-V4-Flash', load_model=False)
    template = engine.template

    tools = [{
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get the weather for a specific location',
            'parameters': {
                'type': 'object',
                'properties': {
                    'location': {
                        'type': 'string',
                        'description': 'The city name'
                    },
                    'unit': {
                        'type': 'string',
                        'enum': ['celsius', 'fahrenheit'],
                        'description': 'Temperature unit'
                    }
                },
                'required': ['location']
            }
        }
    }, {
        'type': 'function',
        'function': {
            'name': 'search',
            'description': 'Search the web for information',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Search query'
                    },
                    'num_results': {
                        'type': 'integer',
                        'description': 'Number of results to return'
                    }
                },
                'required': ['query']
            }
        }
    }]
    data = {
        'tools':
        tools,
        'messages': [{
            'role': 'system',
            'content': 'You are a helpful assistant.'
        }, {
            'role': 'user',
            'content': "What's the weather in Beijing?"
        }, {
            'role':
            'assistant',
            'content':
            '<think>The user wants to know the weather in Beijing. I should use the get_weather tool.</think>\n\n'
        }, {
            'role':
            'tool_call',
            'content':
            '{"name": "get_weather", "arguments": "{\\"location\\": \\"Beijing\\", \\"unit\\": \\"celsius\\"}"}'
        }, {
            'role': 'tool_response',
            'content': '{"temperature": 22, "condition": "sunny", "humidity": 45}'
        }, {
            'role':
            'assistant',
            'content': ('<think>Got the weather data. Let me format a nice response.</think>'
                        'The weather in Beijing is currently sunny with a temperature of 22°C and 45% humidity.')
        }]
    }

    template.template_backend = 'swift'
    template.set_mode('train')
    encoded = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')

    expected_input_ids = (
        '<｜begin▁of▁sentence｜>You are a helpful assistant.\n\n## Tools\n\n'
        'You have access to a set of tools to help answer the user\'s question. '
        'You can invoke tools by writing a "<｜DSML｜tool_calls>" block like the following:\n\n'
        '<｜DSML｜tool_calls>\n'
        '<｜DSML｜invoke name="$TOOL_NAME">\n'
        '<｜DSML｜parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜parameter>\n'
        '...\n'
        '</｜DSML｜invoke>\n'
        '<｜DSML｜invoke name="$TOOL_NAME2">\n'
        '...\n'
        '</｜DSML｜invoke>\n'
        '</｜DSML｜tool_calls>\n\n'
        'String parameters should be specified as is and set `string="true"`. '
        'For all other types (numbers, booleans, arrays, objects), '
        'pass the value in JSON format and set `string="false"`.\n\n'
        'If thinking_mode is enabled (triggered by <think>), '
        'you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.'
        '\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n'
        '### Available Tool Schemas\n\n'
        '{"name": "get_weather", "description": "Get the weather for a specific location", '
        '"parameters": {"type": "object", "properties": {"location": {"type": "string", '
        '"description": "The city name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], '
        '"description": "Temperature unit"}}, "required": ["location"]}}\n'
        '{"name": "search", "description": "Search the web for information", '
        '"parameters": {"type": "object", "properties": {"query": {"type": "string", '
        '"description": "Search query"}, "num_results": {"type": "integer", '
        '"description": "Number of results to return"}}, "required": ["query"]}}\n\n'
        'You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n'
        '<｜User｜>What\'s the weather in Beijing?<｜Assistant｜>'
        '<think>The user wants to know the weather in Beijing. I should use the get_weather tool.</think>\n\n'
        '<｜DSML｜tool_calls>\n'
        '<｜DSML｜invoke name="get_weather">\n'
        '<｜DSML｜parameter name="location" string="true">Beijing</｜DSML｜parameter>\n'
        '<｜DSML｜parameter name="unit" string="true">celsius</｜DSML｜parameter>\n'
        '</｜DSML｜invoke>\n'
        '</｜DSML｜tool_calls>'
        '<｜end▁of▁sentence｜>'
        '<｜User｜><tool_result>{"temperature": 22, "condition": "sunny", "humidity": 45}</tool_result>'
        '<｜Assistant｜>'
        '<think>Got the weather data. Let me format a nice response.</think>'
        'The weather in Beijing is currently sunny with a temperature of 22°C and 45% humidity.'
        '<｜end▁of▁sentence｜>')

    assert template.safe_decode(encoded['input_ids']) == expected_input_ids


telechat3_tools = [{
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get weather',
        'parameters': {
            'type': 'object',
            'properties': {
                'city': {
                    'type': 'string'
                },
                'unit': {
                    'type': 'string'
                },
            },
            'required': ['city']
        }
    }
}]

telechat3_tool_call = '{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}'
telechat3_tool_call2 = '{"name": "get_weather", "arguments": {"city": "Shanghai", "unit": "fahrenheit"}}'


def _assert_template_backend_equal(template, data):
    template.template_backend = 'swift'
    template.set_mode('train')
    encoded = template.encode(data)
    template.template_backend = 'jinja'
    encoded2 = template.encode(data)
    assert encoded['input_ids'] == encoded2['input_ids']
    return encoded


def _assert_generation_backend_equal(template, data):
    template.set_mode('transformers')
    template.template_backend = 'swift'
    encoded = template.encode(data)
    template.template_backend = 'jinja'
    encoded2 = template.encode(data)
    assert encoded['input_ids'] == encoded2['input_ids']
    return template.safe_decode(encoded['input_ids'])


def _assert_telechat3_agent_template(model_id: str, template_type: str):
    from swift import get_processor, get_template
    from swift.model import get_matched_model_meta

    model_meta = get_matched_model_meta(model_id)
    assert model_meta.model_type == 'telechat3'
    assert model_meta.template == template_type
    tokenizer = get_processor(model_id)
    template = get_template(tokenizer)
    assert template.template_meta.template_type == template_type
    data = {
        'tools':
        telechat3_tools,
        'messages': [{
            'role': 'system',
            'content': 'You are a helpful assistant.'
        }, {
            'role': 'user',
            'content': 'weather?'
        }, {
            'role': 'tool_call',
            'content': telechat3_tool_call
        }, {
            'role': 'tool_response',
            'content': '{"temperature": 22}'
        }, {
            'role': 'assistant',
            'content': 'sunny'
        }]
    }
    _assert_template_backend_equal(template, data)

    data['messages'].insert(2, {'role': 'assistant', 'content': 'I will check.'})
    _assert_template_backend_equal(template, data)

    data = {
        'tools':
        telechat3_tools,
        'messages': [{
            'role': 'system',
            'content': 'You are a helpful assistant.'
        }, {
            'role': 'user',
            'content': 'weather in two cities?'
        }, {
            'role': 'tool_call',
            'content': telechat3_tool_call
        }, {
            'role': 'tool_call',
            'content': telechat3_tool_call2
        }, {
            'role': 'tool_response',
            'content': '{"temperature": 22}'
        }, {
            'role': 'tool_response',
            'content': '{"temperature": 28}'
        }, {
            'role': 'assistant',
            'content': 'Beijing is 22 and Shanghai is 28.'
        }]
    }
    _assert_template_backend_equal(template, data)

    thinking_arg = 'before</think>after'
    data = {
        'tools':
        telechat3_tools,
        'messages': [{
            'role': 'user',
            'content': 'echo the text'
        }, {
            'role': 'tool_call',
            'content': json.dumps({
                'name': 'get_weather',
                'arguments': {
                    'city': thinking_arg
                }
            })
        }]
    }
    encoded = _assert_template_backend_equal(template, data)
    assert thinking_arg in template.safe_decode(encoded['input_ids'])

    structured_tool_call = {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'arguments': {
                'city': thinking_arg
            }
        }
    }
    data = {
        'tools':
        telechat3_tools,
        'messages': [{
            'role': 'user',
            'content': 'echo the text'
        }, {
            'role': 'assistant',
            'content': '',
            'tool_calls': [structured_tool_call]
        }]
    }
    encoded = _assert_template_backend_equal(template, data)
    assert thinking_arg in template.safe_decode(encoded['input_ids'])

    structured_tool_call['function']['arguments'] = json.dumps({'city': thinking_arg}, separators=(',', ':'))
    encoded = _assert_template_backend_equal(template, data)
    assert thinking_arg in template.safe_decode(encoded['input_ids'])

    for key, pre_value, tool_value in [('loss', False, True), ('loss_scale', 0.2, 0.8)]:
        data = {
            'tools':
            telechat3_tools,
            'messages': [{
                'role': 'user',
                'content': 'weather?'
            }, {
                'role': 'assistant',
                'content': 'I will check.',
                key: pre_value
            }, {
                'role': 'tool_call',
                'content': telechat3_tool_call,
                key: tool_value
            }, {
                'role': 'tool_response',
                'content': '{"temperature": 22}'
            }, {
                'role': 'assistant',
                'content': 'sunny'
            }]
        }
        template.template_backend = 'swift'
        template.set_mode('train')
        is_binary_loss_scale = template.is_binary_loss_scale
        if key == 'loss_scale':
            template.is_binary_loss_scale = False
        encoded = template.encode(data)
        template.is_binary_loss_scale = is_binary_loss_scale
        assert len(encoded['input_ids']) == len(encoded['labels'])
        if key == 'loss_scale':
            assert len(encoded['input_ids']) == len(encoded['loss_scale'])
            assert pre_value in encoded['loss_scale']
            assert tool_value in encoded['loss_scale']
        else:
            labels = template.safe_decode(encoded['labels'])
            assert 'I will check.' not in labels
            assert 'Beijing' in labels
    return template


def test_telechat3():
    from swift import agent_template_map
    from swift.model import get_matched_model_meta

    template = _assert_telechat3_agent_template('TeleAI/TeleChat3-36B-Thinking', 'telechat3')
    model_meta = get_matched_model_meta('TeleAI/TeleChat3-105B-A4.7B-Thinking')
    assert model_meta.model_type == 'deepseek_v3'
    assert model_meta.template == 'telechat3'

    data = {
        'messages': [{
            'role': 'system',
            'content': ''
        }, {
            'role': 'user',
            'content': 'hi'
        }, {
            'role': 'assistant',
            'content': 'answer'
        }]
    }
    encoded = _assert_template_backend_equal(template, data)
    assert template.safe_decode(encoded['input_ids']) == '<_system>\n<_user>hi<_bot>answer<_end>\n'

    data = {
        'messages': [{
            'role': 'user',
            'content': 'question'
        }, {
            'role': 'assistant',
            'content': '<think>\nplan\n</think>\nanswer'
        }]
    }
    encoded = _assert_template_backend_equal(template, data)
    assert template.safe_decode(encoded['input_ids']) == '<_system><_user>question<_bot>answer<_end>\n'
    data['chat_template_kwargs'] = {'preserve_thinking': True}
    encoded = _assert_template_backend_equal(template, data)
    assert template.safe_decode(encoded['input_ids']) == '<_system><_user>question<_bot>answer<_end>\n'

    data = {'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': ' answer\n'}]}
    encoded = _assert_template_backend_equal(template, data)
    assert template.safe_decode(encoded['input_ids']) == '<_system><_user>q<_bot> answer\n<_end>\n'

    data = {'chat_template_kwargs': {'enable_thinking': False}, 'messages': [{'role': 'user', 'content': 'q'}]}
    assert _assert_generation_backend_equal(template, data) == '<_system><_user>q<_bot><think>\n'

    agent_template = agent_template_map['telechat3']()
    functions = agent_template.get_toolcall(f'<tool_call>\n{telechat3_tool_call}\n</tool_call>\n'
                                            f'<tool_call>\n{telechat3_tool_call2}\n</tool_call>')
    assert len(functions) == 2
    assert functions[0].arguments == '{"city": "Beijing", "unit": "celsius"}'
    assert functions[1].arguments == '{"city": "Shanghai", "unit": "fahrenheit"}'


def test_telechat3_infer():
    from swift import TransformersEngine

    engine = TransformersEngine('TeleAI/TeleChat3-36B-Thinking')
    engine.template.template_backend = 'jinja'
    messages = _infer(
        engine,
        num_tools=2,
        agent_tools=telechat3_tools,
        query='Use the get_weather tool to get the weather in Beijing. Return only a tool call.')
    assert messages[-1]['content']


def test_telechat3_coder():
    from swift import agent_template_map

    template = _assert_telechat3_agent_template('TeleAI/TeleChat3-Coder-36B-Thinking', 'telechat3_coder')

    data = {'messages': [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'answer'}]}
    encoded = _assert_template_backend_equal(template, data)
    assert template.safe_decode(encoded['input_ids']) == '<_system><_user>hi<_bot></think>answer<_end>'

    data = {
        'messages': [{
            'role': 'user',
            'content': 'hi'
        }, {
            'role': 'assistant',
            'content': '<think>plan</think>answer'
        }, {
            'role': 'user',
            'content': 'again'
        }, {
            'role': 'assistant',
            'content': 'done'
        }]
    }
    encoded = _assert_template_backend_equal(template, data)
    expected_input = '<_system><_user>hi<_bot></think>answer<_end><_user>again<_bot></think>done<_end>'
    assert template.safe_decode(encoded['input_ids']) == expected_input
    data['chat_template_kwargs'] = {'preserve_thinking': True}
    encoded = _assert_template_backend_equal(template, data)
    assert template.safe_decode(encoded['input_ids']) == expected_input

    for key, first_value, second_value in [('loss', False, True), ('loss_scale', 0.2, 0.8)]:
        data = {
            'messages': [{
                'role': 'user',
                'content': 'q'
            }, {
                'role': 'assistant',
                'content': 'left',
                key: first_value
            }, {
                'role': 'assistant',
                'content': 'right',
                key: second_value
            }]
        }
        with pytest.raises(ValueError, match=rf'different `{key}` values'):
            template.encode(data)

    data = {
        'messages': [{
            'role': 'user',
            'content': 'question'
        }, {
            'role': 'assistant',
            'content': '<think>\nplan\n</think>\nanswer'
        }]
    }
    encoded = _assert_template_backend_equal(template, data)
    expected_input = '<_system><_user>question<_bot><think>\nplan\n</think>answer<_end>'
    assert template.safe_decode(encoded['input_ids']) == expected_input

    cases = [
        ({
            'messages': [{
                'role': 'user',
                'content': 'q'
            }, {
                'role': 'assistant',
                'content': 'answer',
                'reasoning_content': 'plan'
            }]
        }, '<_system><_user>q<_bot><think>\nplan\n</think>answer<_end>'),
        ({
            'messages': [{
                'role': 'user',
                'content': 'q'
            }, {
                'role': 'assistant',
                'content': '<think>\n\n</think>answer'
            }]
        }, '<_system><_user>q<_bot></think>answer<_end>'),
        ({
            'messages': [{
                'role': 'user',
                'content': 'q'
            }, {
                'role': 'assistant',
                'content': 'prefix'
            }, {
                'role': 'assistant',
                'content': '<think>x</think>answer'
            }]
        }, '<_system><_user>q<_bot><think>\nx\n</think>answer<_end>'),
    ]
    for data, expected_input in cases:
        encoded = _assert_template_backend_equal(template, data)
        assert template.safe_decode(encoded['input_ids']) == expected_input

    data = {
        'chat_template_kwargs': {
            'clear_thinking': False
        },
        'messages': [{
            'role': 'user',
            'content': 'q1'
        }, {
            'role': 'assistant',
            'content': '<think>p1</think>a1'
        }, {
            'role': 'user',
            'content': 'q2'
        }, {
            'role': 'assistant',
            'content': '<think>p2</think>a2'
        }]
    }
    encoded = _assert_template_backend_equal(template, data)
    expected_input = ('<_system><_user>q1<_bot><think>\np1\n</think>a1<_end>'
                      '<_user>q2<_bot><think>\np2\n</think>a2<_end>')
    assert template.safe_decode(encoded['input_ids']) == expected_input

    data = {'chat_template_kwargs': {'enable_thinking': False}, 'messages': [{'role': 'user', 'content': 'q'}]}
    assert _assert_generation_backend_equal(template, data) == '<_system><_user>q<_bot><think>'

    data = {
        'tools':
        telechat3_tools,
        'messages': [{
            'role': 'user',
            'content': 'q1'
        }, {
            'role': 'tool_call',
            'content': telechat3_tool_call
        }, {
            'role': 'tool_response',
            'content': '{"temperature": 22}'
        }, {
            'role': 'assistant',
            'content': 'done'
        }, {
            'role': 'user',
            'content': 'q2'
        }]
    }
    rendered = _assert_generation_backend_equal(template, data)
    assert '<_user>q1<_bot></think><tool_call>' in rendered

    agent_template = agent_template_map['telechat3_coder']()
    functions = agent_template.get_toolcall('<tool_call>get_weather'
                                            '<param_key>city</param_key><param_value>Beijing</param_value>'
                                            '</tool_call><tool_call>get_weather'
                                            '<param_key>city</param_key><param_value>Shanghai</param_value>'
                                            '</tool_call>')
    assert len(functions) == 2
    assert functions[0].arguments == '{"city": "Beijing"}'
    assert functions[1].arguments == '{"city": "Shanghai"}'
    response = ('<tool_call>get_weather'
                '<param_key> city </param_key>\n<param_value> 22 </param_value>'
                '<param_key>unit</param_key> <param_value>null</param_value>'
                '</tool_call>')
    functions = agent_template.get_toolcall(response)
    assert len(functions) == 1
    assert functions[0].arguments == '{"city": 22, "unit": null}'
    functions = agent_template.get_toolcall_with_tools(response, telechat3_tools)
    assert len(functions) == 1
    assert functions[0].arguments == '{"city": "22", "unit": "null"}'

    from swift.infer_engine.infer_engine import InferEngine
    engine = SimpleNamespace(template=SimpleNamespace(agent_template=agent_template))
    tool_calls = InferEngine._get_toolcall(engine, response, telechat3_tools)
    assert len(tool_calls) == 1
    assert tool_calls[0].function.arguments == '{"city": "22", "unit": "null"}'


def test_telechat3_coder_infer():
    from swift import TransformersEngine

    engine = TransformersEngine('TeleAI/TeleChat3-Coder-36B-Thinking', template_type='telechat3_coder')
    engine.template.template_backend = 'jinja'
    messages = _infer(
        engine,
        num_tools=2,
        agent_tools=telechat3_tools,
        query='Use the get_weather tool to get the weather in Beijing. Return only a tool call.')
    assert messages[-1]['content']


def test_seed_oss():
    engine = TransformersEngine('ByteDance-Seed/Seed-OSS-36B-Instruct', load_model=False)

    template = engine.template
    dataset = load_dataset('AI-ModelScope/function-calling-chatml')[0]
    data = dataset[6]
    # To test multiple tool calls and responses, we duplicate some messages.
    data['messages'].insert(1, data['messages'][1])
    data['messages'].insert(3, data['messages'][3])

    # Incomplete tool function will cause seed template to throw an error.
    data['tools'] = [('{\n'
                      '    "name": "convert_temperature",\n'
                      '    "description": "Convert temperature from one unit to another",\n'
                      '    "parameters": {\n'
                      '        "type": "object",\n'
                      '        "properties": {\n'
                      '            "temperature": {\n'
                      '                "type": "number",\n'
                      '                "description": "The temperature value"\n'
                      '            },\n'
                      '            "from_unit": {\n'
                      '                "type": "string",\n'
                      '                "description": "The unit to convert from"\n'
                      '            },\n'
                      '            "to_unit": {\n'
                      '                "type": "string",\n'
                      '                "description": "The unit to convert to"\n'
                      '            }\n'
                      '        },\n'
                      '        "required": [\n'
                      '            "temperature",\n'
                      '            "from_unit",\n'
                      '            "to_unit"\n'
                      '        ]\n'
                      '    }\n'
                      '}'),
                     ('{\n'
                      '    "name": "get_current_date",\n'
                      '    "description": "Get the current date",\n'
                      '    "parameters":  {\n'
                      '        "type": "object",\n'
                      '        "properties": {\n'
                      '         "date": {\n'
                      '                "type": "number",\n'
                      '                "description": "The date value"}}}\n'
                      '}')]

    data['thinking_budget'] = 0

    template.template_backend = 'swift'
    template.set_mode('train')
    encoded = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded["labels"])}')
    import re
    expected_input_ids = re.sub(
        r'<seed:think>.*?</seed:think>', '', template.safe_decode(encoded['input_ids']), flags=re.DOTALL)
    template.template_backend = 'jinja'
    encoded2 = template.encode(data)
    print(f'input_ids: {template.safe_decode(encoded2["input_ids"])}')
    print(f'labels: {template.safe_decode(encoded2["labels"])}')
    assert template.safe_decode(encoded2['input_ids']) == expected_input_ids


if __name__ == '__main__':
    from swift import InferRequest, RequestConfig, TransformersEngine, agent_template_map, load_dataset

    # test_react_en()
    # test_react_zh()
    # test_qwen_en()
    # test_qwen_zh()
    # test_qwen_en_parallel()
    # test_qwen_zh_parallel()
    # test_hermes()
    # test_toolbench()
    # test_chatglm4()
    # test_glm4()
    # test_llama3()
    # test_llama4()
    # test_hunyuan()
    # test_glm4_5()
    # test_glm4_7()
    # test_qwen3_coder()
    # test_qwen3_5()
    # test_deepseek_v3_1()
    test_deepseek_v4()
    # test_seed_oss()
    # test_youtu()
