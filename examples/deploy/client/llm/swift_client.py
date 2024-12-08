# Copyright (c) Alibaba, Inc. and its affiliates.
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer_batch(engine: 'InferEngine', dataset: str):
    request_config = RequestConfig(max_tokens=512, temperature=0)
    dataset = load_dataset([dataset], strict=False, seed=42)[0]
    print(f'dataset: {dataset}')
    metric = InferStats()
    resp_list = engine.infer([InferRequest(**data) for data in dataset], request_config, metrics=[metric])
    query0 = dataset[0]['messages'][0]['content']
    print(f'query0: {query0}')
    print(f'response0: {resp_list[0].choices[0].message.content}')
    print(f'metric: {metric.compute()}')


def infer_stream(engine: 'InferEngine', infer_request: 'InferRequest'):
    request_config = RequestConfig(max_tokens=512, temperature=0, stream=True)
    metric = InferStats()
    gen = engine.infer([infer_request], request_config, metrics=[metric])
    query = infer_request.messages[0]['content']
    print(f'query: {query}\nresponse: ', end='')
    for resp_list in gen:
        print(resp_list[0].choices[0].delta.content, end='', flush=True)
    print()
    print(f'metric: {metric.compute()}')


def run_client(host: str = '127.0.0.1', port: int = 8000):
    engine = InferClient(host=host, port=port)
    print(f'models: {engine.models}')
    infer_batch(engine, 'AI-ModelScope/alpaca-gpt4-data-zh#1000')
    messages = [{'role': 'user', 'content': 'who are you?'}]
    infer_stream(engine, InferRequest(messages=messages))


if __name__ == '__main__':
    from swift.llm import (InferEngine, InferRequest, InferClient, RequestConfig, load_dataset, run_deploy,
                           DeployArguments)
    from swift.plugin import InferStats
    # TODO: The current 'pt' deployment does not support automatic batch.
    with run_deploy(
            DeployArguments(model='Qwen/Qwen2.5-1.5B-Instruct', verbose=False, log_interval=-1,
                            infer_backend='vllm')) as port:
        run_client(port=port)
