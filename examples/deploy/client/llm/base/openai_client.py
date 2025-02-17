# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from openai import OpenAI

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer(client, model: str, messages):
    query = messages[0]['content']
    print(f'query: {query}')
    resp = client.completions.create(model=model, prompt=query, max_tokens=64, temperature=0)
    response = resp.choices[0].text
    print(f'response: {response}')
    # or (The two calling methods are equivalent.)
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=64, temperature=0)
    response = resp.choices[0].message.content
    print(f'response: {response}')
    return response


def run_client(host: str = '127.0.0.1', port: int = 8000):
    client = OpenAI(
        api_key='EMPTY',
        base_url=f'http://{host}:{port}/v1',
    )
    model = client.models.list().data[0].id
    print(f'model: {model}')

    messages = [{'role': 'user', 'content': '浙江 -> 杭州\n安徽 -> 合肥\n四川 ->'}]
    infer(client, model, messages)


if __name__ == '__main__':
    from swift.llm import run_deploy, DeployArguments
    # NOTE: In a real deployment scenario, please comment out the context of run_deploy.
    with run_deploy(
            DeployArguments(
                model='Qwen/Qwen2.5-1.5B', verbose=False, log_interval=-1, infer_backend='pt',
                use_chat_template=False)) as port:
        run_client(port=port)
