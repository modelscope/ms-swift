import os
from typing import Dict, List

from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset


def batch_infer(engine: InferEngine, dataset: str):
    request_config = RequestConfig(max_tokens=512)
    dataset = load_dataset([dataset])[0]
    print(f'dataset: {dataset}')
    response = engine.infer(list(dataset), request_config)
    print(f'dataset[0]: {dataset[0]}')
    print(f'response[0]: {response[0].choices[0].message.content}')


def stream_infer(engine: InferEngine, messages: List[Dict[str, str]]):
    request_config = RequestConfig(max_tokens=512, stream=True)
    gen = engine.infer([InferRequest(messages)], request_config)
    print(f'messages: {messages}\nresponse: ', end='')
    for response in gen:
        print(response[0].choices[0].delta.content, end='', flush=True)
    print()


if __name__ == '__main__':
    model = 'Qwen/Qwen2.5-1.5B-Instruct'
    infer_backend = 'pt'

    if infer_backend == 'pt':
        engine = PtEngine(model, max_batch_size=64)
    elif infer_backend == 'vllm':
        from swift.llm import VllmEngine
        engine = VllmEngine(model)
    else:
        from swift.llm import LmdeployEngine
        engine = LmdeployEngine(model)
    batch_infer(engine, 'lvjianjin/AdvertiseGen#1000')
    messages = [{'role': 'user', 'content': '你是谁'}]
    stream_infer(engine, messages)
