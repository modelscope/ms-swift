# Copyright (c) Alibaba, Inc. and its affiliates.
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['SWIFT_DEBUG'] = '1'


def infer(engine: 'InferEngine', infer_request: 'InferRequest'):
    stop = [engine.default_template.agent_template.keyword.observation]  # compat react_en
    request_config = RequestConfig(max_tokens=512, temperature=0, stop=stop)
    resp_list = engine.infer([infer_request], request_config)
    query = infer_request.messages[0]['content']
    response = resp_list[0].choices[0].message.content
    print(f'query: {query}')
    print(f'response: {response}')
    print(f'tool_calls: {resp_list[0].choices[0].message.tool_calls}')

    tool = '{"temperature": 32, "condition": "Sunny", "humidity": 50}'
    print(f'tool_response: {tool}')
    infer_request.messages += [{'role': 'assistant', 'content': response}, {'role': 'tool', 'content': tool}]
    resp_list = engine.infer([infer_request], request_config)
    response2 = resp_list[0].choices[0].message.content
    print(f'response2: {response2}')


def infer_stream(engine: 'InferEngine', infer_request: 'InferRequest'):
    stop = [engine.default_template.agent_template.keyword.observation]
    request_config = RequestConfig(max_tokens=512, temperature=0, stream=True, stop=stop)
    gen_list = engine.infer([infer_request], request_config)
    query = infer_request.messages[0]['content']
    response = ''
    print(f'query: {query}\nresponse: ', end='')
    for resp in gen_list[0]:
        if resp is None:
            continue
        delta = resp.choices[0].delta.content
        response += delta
        print(delta, end='', flush=True)
    print()
    print(f'tool_calls: {resp.choices[0].delta.tool_calls}')

    tool = '{"temperature": 32, "condition": "Sunny", "humidity": 50}'
    print(f'tool_response: {tool}\nresponse2: ', end='')
    infer_request.messages += [{'role': 'assistant', 'content': response}, {'role': 'tool', 'content': tool}]
    gen_list = engine.infer([infer_request], request_config)
    for resp in gen_list[0]:
        if resp is None:
            continue
        print(resp.choices[0].delta.content, end='', flush=True)
    print()


def get_infer_request():
    return InferRequest(
        messages=[{
            'role': 'user',
            'content': "How's the weather in Beijing today?"
        }],
        tools=[{
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
        }])


if __name__ == '__main__':
    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig
    from swift.plugin import agent_templates
    model = 'Qwen/Qwen2.5-3B'
    adapters = ['output/vx-xxx/checkpoint-xxx']
    engine = PtEngine(model, adapters=adapters, max_batch_size=8)

    # agent_template = agent_templates['hermes']()  # react_en/qwen_en/qwen_en_parallel
    # engine.default_template.agent_template = agent_template

    infer(engine, get_infer_request())
    infer_stream(engine, get_infer_request())
