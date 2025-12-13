import torch

from swift.llm import InferRequest, PtEngine

if __name__ == '__main__':
    engine = PtEngine(
        'Qwen/Qwen3-Embedding-4B',
        task_type='embedding',
        torch_dtype=torch.float16,
        attn_implementation='flash_attention_2')

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
                'role': 'assistant',
                'content': 'The capital of China is Beijing.'
            },
        ])
    ]
    resp_list = engine.infer(infer_requests)
    embedding0 = torch.tensor(resp_list[0].data[0].embedding)
    embedding1 = torch.tensor(resp_list[1].data[0].embedding)
    print(f'scores: {(embedding0 * embedding1).sum()}')
