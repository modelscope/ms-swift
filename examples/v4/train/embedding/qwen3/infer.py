# For full-parameter training, please refer to:
# https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_embedding.py

import torch

from swift.infer_engine import InferRequest, TransformersEngine


def run_qwen3_emb():
    engine = TransformersEngine(
        'Qwen/Qwen3-Embedding-4B',
        task_type='embedding',
        attn_impl='flash_attention_2',
        adapters=['output/vx-xxx/checkpoint-xxx'])

    infer_requests = [
        InferRequest(messages=[
            {
                'role': 'user',
                'content': 'A dog sleeping under a table.'
            },
        ]),
        InferRequest(messages=[
            {
                'role': 'user',
                'content': 'a dog napping under a small table.'
            },
        ]),
        InferRequest(messages=[
            {
                'role': 'user',
                'content': 'a cat napping under a small tree.'
            },
        ])
    ]
    resp_list = engine.infer(infer_requests)
    embedding0 = torch.tensor(resp_list[0].data[0].embedding)
    embedding1 = torch.tensor(resp_list[1].data[0].embedding)
    embedding2 = torch.tensor(resp_list[2].data[0].embedding)
    embedding = torch.stack([embedding0, embedding1, embedding2])
    print(f'scores: {embedding @ embedding.T}')


if __name__ == '__main__':
    run_qwen3_emb()
