# Copyright (c) Alibaba, Inc. and its affiliates.
import asyncio
import os
from typing import List

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer_batch(engine: 'InferEngine', infer_requests: List['InferRequest']):
    request_config = RequestConfig(max_tokens=512, temperature=0)
    metric = InferStats()

    resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
    # # The asynchronous interface below is equivalent to the synchronous interface above.
    # async def _run():
    #     tasks = [engine.infer_async(infer_request, request_config) for infer_request in infer_requests]
    #     return await asyncio.gather(*tasks)
    # resp_list = asyncio.run(_run())

    query0 = infer_requests[0].messages[0]['content']
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
    # Here, `load_dataset` is used for convenience; `infer_batch` does not require creating a dataset.
    dataset = load_dataset(['AI-ModelScope/alpaca-gpt4-data-zh#1000'], seed=42)[0]
    print(f'dataset: {dataset}')
    infer_requests = [InferRequest(**data) for data in dataset]
    infer_batch(engine, infer_requests)

    messages = [{'role': 'user', 'content': 'who are you?'}]
    infer_stream(engine, InferRequest(messages=messages))


if __name__ == '__main__':
    from swift.llm import (InferEngine, InferRequest, InferClient, RequestConfig, load_dataset, run_deploy,
                           DeployArguments)
    from swift.plugin import InferStats
    # NOTE: In a real deployment scenario, please comment out the context of run_deploy.
    with run_deploy(
            DeployArguments(model='Qwen/Qwen2.5-1.5B-Instruct', verbose=False, log_interval=-1,
                            infer_backend='vllm')) as port:
        run_client(port=port)
