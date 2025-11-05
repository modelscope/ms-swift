import os
from typing import Literal

import torch

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _prepare(infer_backend: Literal['vllm', 'pt', 'lmdeploy']):
    from swift.llm import InferRequest, get_template

    if infer_backend == 'lmdeploy':
        from swift.llm import LmdeployEngine
        engine = LmdeployEngine('Qwen/Qwen2-7B-Instruct', torch.float32)
    elif infer_backend == 'pt':
        from swift.llm import PtEngine
        engine = PtEngine('Qwen/Qwen2-7B-Instruct')
    elif infer_backend == 'vllm':
        from swift.llm import VllmEngine
        engine = VllmEngine('Qwen/Qwen2-7B-Instruct')
    template = get_template(engine.model_meta.template, engine.processor)
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
    return engine, template, infer_requests


def test_infer(engine, template, infer_requests):
    from swift.llm import RequestConfig
    from swift.plugin import InferStats

    request_config = RequestConfig(temperature=0, logprobs=True, top_logprobs=2)
    infer_stats = InferStats()

    response_list = engine.infer(
        infer_requests, template=template, request_config=request_config, metrics=[infer_stats])

    for response in response_list[:2]:
        print(response.choices[0].message.content)
    print(infer_stats.compute())


def test_stream(engine, template, infer_requests):
    from swift.llm import RequestConfig
    from swift.plugin import InferStats

    infer_stats = InferStats()
    request_config = RequestConfig(temperature=0, stream=True, logprobs=True, top_logprobs=2)

    gen_list = engine.infer(infer_requests, template=template, request_config=request_config, metrics=[infer_stats])

    for response in gen_list[0]:
        if response is None:
            continue
        print(response.choices[0].delta.content, end='', flush=True)

    print(infer_stats.compute())


if __name__ == '__main__':
    engine, template, infer_requests = _prepare(infer_backend='pt')
    test_infer(engine, template, infer_requests)
    test_stream(engine, template, infer_requests)
