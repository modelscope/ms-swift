# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer_batch(engine: 'InferEngine', infer_requests: List['InferRequest']):
    request_config = RequestConfig(max_tokens=64, temperature=0)

    resp_list = engine.infer(infer_requests, request_config)

    query0 = infer_requests[0].messages[0]['content']
    print(f'query0: {query0}')
    print(f'response0: {resp_list[0].choices[0].message.content}')


def run_client(host: str = '127.0.0.1', port: int = 8000):
    engine = InferClient(host=host, port=port)
    print(f'models: {engine.models}')

    infer_requests = [InferRequest(messages=[{'role': 'user', 'content': '浙江 -> 杭州\n安徽 -> 合肥\n四川 ->'}])]
    infer_batch(engine, infer_requests)


if __name__ == '__main__':
    from swift.llm import InferEngine, InferRequest, InferClient, RequestConfig, run_deploy, DeployArguments
    # NOTE: In a real deployment scenario, please comment out the context of run_deploy.
    with run_deploy(
            DeployArguments(
                model='Qwen/Qwen2.5-1.5B', verbose=False, log_interval=-1, infer_backend='pt',
                use_chat_template=False)) as port:
        run_client(port=port)
