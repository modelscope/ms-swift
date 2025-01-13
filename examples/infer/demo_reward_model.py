# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer_batch(engine: 'InferEngine', infer_requests: List['InferRequest']):
    resp_list = engine.infer(infer_requests)
    print(f'messages0: {infer_requests[0].messages}')
    print(f'response0: {resp_list[0].choices[0].message.content}')


if __name__ == '__main__':
    from swift.llm import InferEngine, InferRequest, PtEngine, load_dataset
    model = 'Shanghai_AI_Laboratory/internlm2-1_8b-reward'
    engine = PtEngine(model, max_batch_size=64)
    # Here, `load_dataset` is used for convenience; `infer_batch` does not require creating a dataset.
    dataset = load_dataset(['AI-ModelScope/alpaca-gpt4-data-zh#1000'], seed=42)[0]
    print(f'dataset: {dataset}')
    infer_requests = [InferRequest(**data) for data in dataset]
    infer_batch(engine, infer_requests)

    messages = [{
        'role': 'user',
        'content': "Hello! What's your name?"
    }, {
        'role': 'assistant',
        'content': 'My name is InternLM2! A helpful AI assistant. What can I do for you?'
    }]
    infer_batch(engine, [InferRequest(messages=messages)])
