import torch

from swift.infer_engine import InferRequest, TransformersEngine


def run_qwen3_emb():
    engine = TransformersEngine(
        'Qwen/Qwen3-Embedding-4B', task_type='embedding', torch_dtype=torch.float16, attn_impl='flash_attention_2')

    infer_requests = [
        InferRequest(messages=[
            {
                'role':
                'user',
                'content':
                'Instruct: Given a web search query, retrieve relevant passages that answer the query\n'
                'Query:What is the capital of China?'
            },
        ]),
        InferRequest(messages=[
            {
                'role': 'user',
                'content': 'The capital of China is Beijing.'
            },
        ])
    ]
    resp_list = engine.infer(infer_requests)
    embedding0 = torch.tensor(resp_list[0].data[0].embedding)
    embedding1 = torch.tensor(resp_list[1].data[0].embedding)
    print(f'scores: {(embedding0 * embedding1).sum()}')


def run_qwen3_vl_emb():
    engine = TransformersEngine(
        'Qwen/Qwen3-VL-Embedding-2B', task_type='embedding', max_batch_size=2, attn_impl='flash_attention_2')

    infer_requests = [
        InferRequest(messages=[
            {
                'role': 'user',
                'content': 'A woman playing with her dog on a beach at sunset.'
            },
        ]),
        InferRequest(
            messages=[
                {
                    'role':
                    'user',
                    'content':
                    '<image>A woman shares a joyful moment with her golden retriever on a sun-drenched beach at '
                    'sunset, as the dog offers its paw in a heartwarming display of companionship and trust.'
                },
            ],
            images=['https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'])
    ]
    resp_list = engine.infer(infer_requests)
    embedding0 = torch.tensor(resp_list[0].data[0].embedding)
    embedding1 = torch.tensor(resp_list[1].data[0].embedding)
    print(f'scores: {(embedding0 * embedding1).sum()}')


if __name__ == '__main__':
    # run_qwen3_emb()
    run_qwen3_vl_emb()
