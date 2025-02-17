# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Literal

from openai import OpenAI

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer(client, model: str, messages):
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=512, temperature=0)
    query = messages[0]['content']
    response = resp.choices[0].message.content
    print(f'query: {query}')
    print(f'response: {response}')
    return response


# streaming
def infer_stream(client, model: str, messages):
    gen = client.chat.completions.create(model=model, messages=messages, stream=True, temperature=0)
    print(f'messages: {messages}\nresponse: ', end='')
    for chunk in gen:
        print(chunk.choices[0].delta.content, end='', flush=True)
    print()


def get_message(mm_type: Literal['text', 'image', 'video', 'audio']):
    if mm_type == 'text':
        message = {'role': 'user', 'content': 'who are you?'}
    elif mm_type == 'image':
        message = {
            'role':
            'user',
            'content': [{
                'type': 'image',
                'image': 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png'
            }, {
                'type': 'text',
                'text': 'How many sheep are there in the picture?'
            }]
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


def run_client(host: str = '127.0.0.1', port: int = 8000):
    client = OpenAI(
        api_key='EMPTY',
        base_url=f'http://{host}:{port}/v1',
    )
    model = client.models.list().data[0].id
    print(f'model: {model}')

    query = 'who are you?'
    messages = [{'role': 'user', 'content': query}]
    response = infer(client, model, messages)
    messages.append({'role': 'assistant', 'content': response})
    messages.append(get_message(mm_type='video'))
    infer_stream(client, model, messages)


if __name__ == '__main__':
    from swift.llm import run_deploy, DeployArguments
    with run_deploy(DeployArguments(model='Qwen/Qwen2-VL-2B-Instruct', verbose=False, log_interval=-1)) as port:
        run_client(port=port)
