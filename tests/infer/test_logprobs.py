import os
from typing import Literal

import torch

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _prepare(infer_backend: Literal['vllm', 'pt', 'lmdeploy']):
    from swift.infer_engine import InferRequest

    if infer_backend == 'lmdeploy':
        from swift.infer_engine import LmdeployEngine
        engine = LmdeployEngine('Qwen/Qwen2-7B-Instruct', torch_dtype=torch.float32)
    elif infer_backend == 'pt':
        from swift.infer_engine import TransformersEngine
        engine = TransformersEngine('Qwen/Qwen2-7B-Instruct')
    elif infer_backend == 'vllm':
        from swift.infer_engine import VllmEngine
        engine = VllmEngine('Qwen/Qwen2-7B-Instruct')
    infer_requests = [
        InferRequest([{
            'role': 'user',
            'content': '晚上睡不着觉怎么办'
        }]),
        InferRequest([{
            'role': 'user',
            'content': 'hello! who are you'
        }])
    ]
    return engine, infer_requests


def test_infer(engine, infer_requests):
    from swift.infer_engine import RequestConfig
    from swift.metrics import InferStats

    request_config = RequestConfig(temperature=0, logprobs=True, top_logprobs=2)
    infer_stats = InferStats()

    response_list = engine.infer(infer_requests, request_config=request_config, metrics=[infer_stats])

    for response in response_list[:2]:
        print(response.choices[0].message.content)
    print(infer_stats.compute())


def test_stream(engine, infer_requests):
    from swift.infer_engine import RequestConfig
    from swift.metrics import InferStats

    infer_stats = InferStats()
    request_config = RequestConfig(temperature=0, stream=True, logprobs=True, top_logprobs=2)

    gen_list = engine.infer(infer_requests, request_config=request_config, metrics=[infer_stats])

    for response in gen_list[0]:
        if response is None:
            continue
        print(response.choices[0].delta.content, end='', flush=True)

    print(infer_stats.compute())


if __name__ == '__main__':
    engine, infer_requests = _prepare(infer_backend='pt')
    test_infer(engine, infer_requests)
    test_stream(engine, infer_requests)
