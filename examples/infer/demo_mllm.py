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
    # metric.reset()  # reuse


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
        message = {
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': 'https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'
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


if __name__ == '__main__':
    # The inference of the trained model can be referred to as:
    # https://github.com/modelscope/ms-swift/tree/main/examples/notebook
    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset
    from swift.plugin import InferStats
    infer_backend = 'pt'

    if infer_backend == 'pt':
        model = 'Qwen/Qwen2-Audio-7B-Instruct'
        mm_type = 'audio'
        dataset = 'speech_asr/speech_asr_aishell1_trainsets:validation#1000'
        engine = PtEngine(model, max_batch_size=64)
    elif infer_backend == 'vllm':
        from swift.llm import VllmEngine
        model = 'Qwen/Qwen2-VL-2B-Instruct'
        mm_type = 'video'
        dataset = 'AI-ModelScope/LaTeX_OCR:small#1000'
        engine = VllmEngine(model, max_model_len=32768, limit_mm_per_prompt={'image': 5, 'video': 2})
    elif infer_backend == 'lmdeploy':
        from swift.llm import LmdeployEngine
        model = 'OpenGVLab/InternVL2_5-1B'
        mm_type = 'video'
        dataset = 'AI-ModelScope/LaTeX_OCR:small#1000'
        engine = LmdeployEngine(model, vision_batch_size=8)

    # Here, `load_dataset` is used for convenience; `infer_batch` does not require creating a dataset.
    dataset = load_dataset([dataset], seed=42)[0]
    print(f'dataset: {dataset}')
    infer_requests = [InferRequest(**data) for data in dataset]
    infer_batch(engine, infer_requests)

    infer_stream(engine, InferRequest(messages=[get_message(mm_type)]))
    # This writing is equivalent to the above writing.
    infer_stream(engine, InferRequest(**get_data(mm_type)))
