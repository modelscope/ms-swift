import os
from typing import Literal

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _prepare(infer_backend: Literal['vllm', 'pt', 'lmdeploy']):
    from swift.infer_engine import InferRequest
    if infer_backend == 'lmdeploy':
        from swift.infer_engine import LmdeployEngine
        engine = LmdeployEngine('OpenGVLab/InternVL2_5-2B', torch_dtype=torch.float32)
    elif infer_backend == 'pt':
        from swift.infer_engine import TransformersEngine
        engine = TransformersEngine('Qwen/Qwen2-7B-Instruct', max_batch_size=16)
    elif infer_backend == 'vllm':
        from swift.infer_engine import VllmEngine
        engine = VllmEngine('Qwen/Qwen2-7B-Instruct')
    infer_requests = [
        # InferRequest([{'role': 'user', 'content': '晚上睡不着觉怎么办'}]) for i in range(100)
        InferRequest([{
            'role': 'user',
            'content': 'hello! who are you'
        }]) for i in range(100)
    ]
    return engine, infer_requests


def test_infer(infer_backend):
    from swift.infer_engine import RequestConfig
    from swift.metrics import InferStats
    engine, infer_requests = _prepare(infer_backend=infer_backend)
    request_config = RequestConfig(temperature=0)
    infer_stats = InferStats()

    response_list = engine.infer(infer_requests, request_config=request_config, metrics=[infer_stats])

    for response in response_list[:2]:
        print(response.choices[0].message.content)
    print(infer_stats.compute())


def test_stream(infer_backend):
    from swift.infer_engine import RequestConfig
    from swift.metrics import InferStats
    engine, infer_requests = _prepare(infer_backend=infer_backend)
    infer_stats = InferStats()
    request_config = RequestConfig(temperature=0, stream=True, logprobs=True)

    gen_list = engine.infer(infer_requests, request_config=request_config, metrics=[infer_stats])

    for response in gen_list[0]:
        if response is None:
            continue
        print(response.choices[0].delta.content, end='', flush=True)
    print()
    print(infer_stats.compute())

    gen_list = engine.infer(infer_requests, request_config=request_config, use_tqdm=True, metrics=[infer_stats])

    for response in gen_list[0]:
        pass

    print(infer_stats.compute())


if __name__ == '__main__':
    test_infer('pt')
    # test_stream('pt')
