# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from openai import OpenAI

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer(client, model: str, input):
    resp = client.embeddings.create(model=model, input=input)
    emb = resp.data[0].embedding
    shape = len(emb)
    sample = str(emb)
    if len(emb) > 6:
        sample = str(emb[:3])[:-1] + ', ..., ' + str(emb[-3:])[1:]
    print(f'query: {input}')
    print(f'Embedding(shape: [1, {shape}]): {sample}')
    return emb


def run_client(host: str = '127.0.0.1', port: int = 8000):
    client = OpenAI(
        api_key='EMPTY',
        base_url=f'http://{host}:{port}/v1',
    )
    model = client.models.list().data[0].id
    print(f'model: {model}')

    query = 'What is the capital of China?'
    response = infer(client, model, query)


if __name__ == '__main__':
    from swift.llm import run_deploy, DeployArguments
    with run_deploy(DeployArguments(model='Qwen/Qwen3-Embedding-0.6B', task_type='embedding', verbose=False, log_interval=-1)) as port:
        run_client(port=port)
