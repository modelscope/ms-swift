import os
from typing import Literal

import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _prepare(infer_backend: Literal['vllm', 'pt', 'lmdeploy']):
    from swift.llm import InferRequest, get_template
    if infer_backend == 'lmdeploy':
        from swift.llm import LmdeployEngine
        engine = LmdeployEngine('OpenGVLab/InternVL2_5-2B', torch.float32)
    elif infer_backend == 'pt':
        from swift.llm import PtEngine
        engine = PtEngine('Qwen/Qwen2-7B-Instruct', max_batch_size=16)
    elif infer_backend == 'vllm':
        from swift.llm import VllmEngine
        engine = VllmEngine('Qwen/Qwen2-7B-Instruct')
    template = get_template(engine.model_meta.template, engine.tokenizer)
    infer_requests = [
        # InferRequest([{'role': 'user', 'content': '晚上睡不着觉怎么办'}]) for i in range(100)
        InferRequest([{
            'role': 'user',
            'content': 'hello! who are you'
        }]) for i in range(100)
    ]
    return engine, template, infer_requests


def test_infer(infer_backend):
    from swift.llm import RequestConfig
    from swift.plugin import InferStats
    engine, template, infer_requests = _prepare(infer_backend=infer_backend)
    request_config = RequestConfig(temperature=0)
    infer_stats = InferStats()

    response_list = engine.infer(
        infer_requests, template=template, request_config=request_config, metrics=[infer_stats])

    for response in response_list[:2]:
        print(response.choices[0].message.content)
    print(infer_stats.compute())


def test_stream(infer_backend):
    from swift.llm import RequestConfig
    from swift.plugin import InferStats
    engine, template, infer_requests = _prepare(infer_backend=infer_backend)
    infer_stats = InferStats()
    request_config = RequestConfig(temperature=0, stream=True, logprobs=True)

    gen = engine.infer(infer_requests, template=template, request_config=request_config, metrics=[infer_stats])

    for response_list in gen:
        response = response_list[0]
        if response is None:
            continue
        print(response.choices[0].delta.content, end='', flush=True)
    print()
    print(infer_stats.compute())

    gen = engine.infer(
        infer_requests, template=template, request_config=request_config, use_tqdm=True, metrics=[infer_stats])

    for response_list in gen:
        pass

    print(infer_stats.compute())


if __name__ == '__main__':
    test_infer('pt')
    # test_stream('pt')
