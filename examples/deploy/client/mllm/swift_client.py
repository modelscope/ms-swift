# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Literal

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer_batch(engine: 'InferEngine', infer_requests: List['InferRequest']):
    request_config = RequestConfig(max_tokens=512, temperature=0)
    metric = InferStats()
    resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
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


def get_message(mm_type: Literal['text', 'image', 'video', 'audio']):
    if mm_type == 'text':
        message = {'role': 'user', 'content': 'who are you?'}
    elif mm_type == 'image':
        message = {
            'role':
            'user',
            'content': [
                {
                    'type': 'image',
                    # url or local_path or PIL.Image or base64
                    'image': 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png'
                },
                {
                    'type': 'text',
                    'text': 'How many sheep are there in the picture?'
                }
            ]
        }

    elif mm_type == 'video':
        # # use base64
        # import base64
        # with open('baby.mp4', 'rb') as f:
        #     vid_base64 = base64.b64encode(f.read()).decode('utf-8')
        # video = f'data:video/mp4;base64,{vid_base64}'

        # use url
        video = 'https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'
        message = {
            'role': 'user',
            'content': [{
                'type': 'video',
                'video': video
            }, {
                'type': 'text',
                'text': 'Describe this video.'
            }]
        }
    elif mm_type == 'audio':
        message = {
            'role':
            'user',
            'content': [{
                'type': 'audio',
                'audio': 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav'
            }, {
                'type': 'text',
                'text': 'What does this audio say?'
            }]
        }
    return message


def get_data(mm_type: Literal['text', 'image', 'video', 'audio']):
    data = {}
    if mm_type == 'text':
        messages = [{'role': 'user', 'content': 'who are you?'}]
    elif mm_type == 'image':
        # The number of <image> tags must be the same as len(images).
        messages = [{'role': 'user', 'content': '<image>How many sheep are there in the picture?'}]
        # Support URL/Path/base64/PIL.Image
        data['images'] = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']
    elif mm_type == 'video':
        messages = [{'role': 'user', 'content': '<video>Describe this video.'}]
        data['videos'] = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
    elif mm_type == 'audio':
        messages = [{'role': 'user', 'content': '<audio>What does this audio say?'}]
        data['audios'] = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav']
    data['messages'] = messages
    return data


def run_client(host: str = '127.0.0.1', port: int = 8000):
    engine = InferClient(host=host, port=port)
    print(f'models: {engine.models}')
    # Here, `load_dataset` is used for convenience; `infer_batch` does not require creating a dataset.
    dataset = load_dataset(['AI-ModelScope/LaTeX_OCR:small#1000'], seed=42)[0]
    print(f'dataset: {dataset}')
    infer_requests = [InferRequest(**data) for data in dataset]
    infer_batch(engine, infer_requests)

    infer_stream(engine, InferRequest(messages=[get_message(mm_type='video')]))
    # This writing is equivalent to the above writing.
    infer_stream(engine, InferRequest(**get_data(mm_type='video')))


if __name__ == '__main__':
    from swift.llm import (InferEngine, InferRequest, InferClient, RequestConfig, load_dataset, run_deploy,
                           DeployArguments)
    from swift.plugin import InferStats
    # NOTE: In a real deployment scenario, please comment out the context of run_deploy.
    with run_deploy(
            DeployArguments(model='Qwen/Qwen2-VL-2B-Instruct', verbose=False, log_interval=-1,
                            infer_backend='vllm')) as port:
        run_client(port=port)
