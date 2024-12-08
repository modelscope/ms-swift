# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from openai import OpenAI

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer(client, model: str, messages):
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
    query = messages[0]['content']
    response = resp.choices[0].message.content
    print(f'query: {query}')
    print(f'response: {response}')
    return response


# streaming
def infer_stream(client, model: str, messages):
    gen = client.chat.completions.create(model=model, messages=messages, stream=True, temperature=0)
    print(f'messages: {messages}\nresponse: ', end='')
    for chunk in gen:
        print(chunk.choices[0].delta.content, end='', flush=True)
    print()


def run_client(host: str = '127.0.0.1', port: int = 8000):
    client = OpenAI(
        api_key='EMPTY',
        base_url=f'http://{host}:{port}/v1',
    )
    model = client.models.list().data[0].id
    print(f'model: {model}')

    query = 'Where is the capital of Zhejiang?'
    messages = [{'role': 'user', 'content': query}]
    response = infer(client, model, messages)
    messages.append({'role': 'assistant', 'content': response})
    messages.append({'role': 'user', 'content': 'What delicious food is there?'})
    infer_stream(client, model, messages)


if __name__ == '__main__':
    from swift.llm import run_deploy, DeployArguments
    with run_deploy(DeployArguments(model='Qwen/Qwen2.5-1.5B-Instruct', verbose=False, log_interval=-1)) as port:
        run_client(port=port)
