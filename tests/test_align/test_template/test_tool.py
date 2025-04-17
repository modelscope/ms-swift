import json

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
}]


def _test_tool(pt_engine, system=None):
    if system is None:
        system = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + json.dumps(tools[0]) + """
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""
    messages = [
        {
            'role': 'system',
            'content': system
        },
        {
            'role': 'user',
            'content': "How's the weather in Beijing today?"
        },
        {
            'role':
            'assistant',
            'content': ('<tool_call>\n{"name": "get_current_weather", "arguments": '
                        '{"location": "Beijing, China", "unit": "celsius"}}\n</tool_call>')
        },
        {
            'role': 'tool',
            'content': "{'temp': 25, 'description': 'Partly cloudy', 'status': 'success'}"
        },
    ]
    request_config = RequestConfig(max_tokens=512, temperature=0)
    response = pt_engine.infer([InferRequest(messages=messages)], request_config=request_config)
    return response[0].choices[0].message.content


def test_qwen2_5():
    pt_engine = PtEngine('Qwen/Qwen2.5-7B-Instruct')
    response = _test_tool(pt_engine)
    assert response == ('Today in Beijing, the temperature is 25 degrees Celsius with partly cloudy skies.')


def test_qwq():
    pt_engine = PtEngine('Qwen/QwQ-32B')
    response = _test_tool(pt_engine)
    assert response[-100:] == ('>\n\nThe current temperature in Beijing is **25°C** with **partly '
                               'cloudy** skies. Have a great day! 🌤️')


def test_deepseek_r1_distill():
    pt_engine = PtEngine('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    response = _test_tool(pt_engine, system='')
    assert response == ('Today in Beijing, the temperature is 25 degrees Celsius with partly cloudy skies.')


if __name__ == '__main__':
    from swift.llm import PtEngine, RequestConfig, get_template, get_model_tokenizer, InferRequest
    from swift.utils import get_logger, seed_everything
    logger = get_logger()
    # test_qwen2_5()
    test_qwq()
    # test_deepseek_r1_distill()
