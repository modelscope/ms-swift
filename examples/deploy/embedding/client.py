# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from openai import OpenAI

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer(client, model: str, messages):
    resp = client.chat.completions.create(model=model, messages=messages)
    emb = resp.data[0]['embedding']
    shape = len(emb)
    sample = str(emb)
    if len(emb) > 6:
        sample = str(emb[:3])[:-1] + ', ..., ' + str(emb[-3:])[1:]
    print(f'query: {input}')
    print(f'Embedding(shape: [1, {shape}]): {sample}')
    return emb


def run_client(host: str = '127.0.0.1', port: int = 8000):
    client = OpenAI(
        api_key='EMPTY',
        base_url=f'http://{host}:{port}/v1',
    )
    model = client.models.list().data[0].id
    print(f'model: {model}')

    messages = [{
            'role': 'user',
            'content': [{
                'type': 'image',
                'image': 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png'
            }, {
                'type': 'text',
                'text': 'What is the capital of China?'
            }]
        }]
    response = infer(client, model, messages)


if __name__ == '__main__':
    from swift.llm import run_deploy, DeployArguments
    from modelscope import snapshot_download
    model_dir = snapshot_download('iic/gme-Qwen2-VL-2B-Instruct')
    # model_dir = '/mnt/nas3/yzhao/tastelikefeet/swift/output/gte_Qwen2-1.5B-instruct/v2-20250711-133234/checkpoint-1'
    with run_deploy(DeployArguments(model=model_dir, task_type='embedding', infer_backend='vllm', verbose=False, log_interval=-1)) as port:
        run_client(port=port)
