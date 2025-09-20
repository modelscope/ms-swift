# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from openai import OpenAI

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer(client, model: str, messages):
    resp = client.chat.completions.create(model=model, messages=messages)
    classify = resp.choices[0].message.content
    print(f'messages: {messages}')
    print(f'classify: {classify}')
    return classify


def run_client(host: str = '127.0.0.1', port: int = 8000):
    client = OpenAI(
        api_key='EMPTY',
        base_url=f'http://{host}:{port}/v1',
    )
    model = client.models.list().data[0].id
    print(f'model: {model}')

    messages = [{
        'role': 'user',
        'content': 'What is the capital of China?',
    }, {
        'role': 'assistant',
        'content': 'Beijing',
    }]
    infer(client, model, messages)


if __name__ == '__main__':
    from swift.llm import run_deploy, DeployArguments
    with run_deploy(
            DeployArguments(
                model='/your/seq_cls/checkpoint-xxx',
                task_type='seq_cls',
                infer_backend='vllm',
                num_labels=2,
                verbose=False,
                log_interval=-1)) as port:
        run_client(port=port)
