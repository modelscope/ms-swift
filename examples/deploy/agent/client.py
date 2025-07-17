# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from openai import OpenAI

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_infer_request():
    messages = [{'role': 'user', 'content': "How's the weather in Beijing today?"}]
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
    return messages, tools


def infer(client, model: str, messages, tools):
    messages = messages.copy()
    query = messages[0]['content']
    resp = client.chat.completions.create(model=model, messages=messages, tools=tools, max_tokens=512, temperature=0)
    response = resp.choices[0].message.content
    print(f'query: {query}')
    print(f'response: {response}')
    print(f'tool_calls: {resp.choices[0].message.tool_calls}')

    tool = '{"temperature": 32, "condition": "Sunny", "humidity": 50}'
    print(f'tool_response: {tool}')
    messages += [{'role': 'assistant', 'content': response}, {'role': 'tool', 'content': tool}]
    resp = client.chat.completions.create(model=model, messages=messages, tools=tools, max_tokens=512, temperature=0)
    response2 = resp.choices[0].message.content
    print(f'response2: {response2}')


# streaming
def infer_stream(client, model: str, messages, tools):
    messages = messages.copy()
    query = messages[0]['content']
    gen = client.chat.completions.create(
        model=model, messages=messages, tools=tools, max_tokens=512, temperature=0, stream=True)
    response = ''
    print(f'query: {query}\nresponse: ', end='')
    for chunk in gen:
        if chunk is None:
            continue
        delta = chunk.choices[0].delta.content
        response += delta
        print(delta, end='', flush=True)
    print()
    print(f'tool_calls: {chunk.choices[0].delta.tool_calls}')

    tool = '{"temperature": 32, "condition": "Sunny", "humidity": 50}'
    print(f'tool_response: {tool}')
    messages += [{'role': 'assistant', 'content': response}, {'role': 'tool', 'content': tool}]
    gen = client.chat.completions.create(
        model=model, messages=messages, tools=tools, max_tokens=512, temperature=0, stream=True)
    print(f'query: {query}\nresponse2: ', end='')
    for chunk in gen:
        if chunk is None:
            continue
        print(chunk.choices[0].delta.content, end='', flush=True)
    print()


if __name__ == '__main__':
    host: str = '127.0.0.1'
    port: int = 8000
    client = OpenAI(
        api_key='EMPTY',
        base_url=f'http://{host}:{port}/v1',
    )
    model = client.models.list().data[0].id
    print(f'model: {model}')

    messages, tools = get_infer_request()
    infer(client, model, messages, tools)
    infer_stream(client, model, messages, tools)
