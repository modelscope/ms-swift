import torch

from swift.llm import InferRequest, PtEngine

if __name__ == '__main__':
    engine = PtEngine(
        'Qwen/Qwen3-Reranker-4B',
        task_type='generative_reranker',
        torch_dtype=torch.float16,
        attn_implementation='flash_attention_2')

    infer_request = InferRequest(
        messages=[{
            'role': 'system',
            'content': 'Given a web search query, retrieve relevant passages that answer the query'
        }, {
            'role': 'user',
            'content': 'What is the capital of China?'
        }, {
            'role': 'assistant',
            'content': 'The capital of China is Beijing.'
        }])

    response = engine.infer([infer_request])[0]
    print(f'scores: {response.choices[0].message.content}')
