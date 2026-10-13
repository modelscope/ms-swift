import torch

from swift.infer_engine import InferRequest, TransformersEngine


def run_qwen3_reranker():
    engine = TransformersEngine(
        'Qwen/Qwen3-Reranker-4B',
        task_type='generative_reranker',
        torch_dtype=torch.float16,
        attn_impl='flash_attention_2')

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


def run_jina_reranker():
    # jina-reranker-v3.5 uses the same integration; replace the model id to use it.
    engine = TransformersEngine(
        'jinaai/jina-reranker-v3',
        use_hf=True,
        task_type='reranker',
        torch_dtype=torch.bfloat16,
        reranker_use_activation=False)

    query = 'What is retrieval augmented generation?'
    documents = [
        'RAG grounds a language model with retrieved external documents.',
        'The Moon is Earth\'s natural satellite.',
    ]
    infer_requests = [
        InferRequest(messages=[
            {
                'role': 'user',
                'content': query
            },
            {
                'role': 'assistant',
                'content': document
            },
        ]) for document in documents
    ]
    responses = engine.infer(infer_requests)
    scores = [response.choices[0].message.content[0] for response in responses]
    print(sorted(zip(documents, scores), key=lambda item: item[1], reverse=True))


def run_qwen3_vl_reranker():
    engine = TransformersEngine(
        'Qwen/Qwen3-VL-Reranker-2B', task_type='generative_reranker', attn_impl='flash_attention_2')

    infer_request = InferRequest(
        messages=[{
            'role': 'system',
            'content': "Retrieval relevant image or text with user's query"
        }, {
            'role': 'user',
            'content': 'A woman playing with her dog on a beach at sunset.'
        }, {
            'role':
            'assistant',
            'content':
            '<image>A woman shares a joyful moment with her golden retriever on a sun-drenched beach '
            'at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.'
        }],
        images=['https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'])

    response = engine.infer([infer_request])[0]
    print(f'scores: {response.choices[0].message.content}')


if __name__ == '__main__':
    # run_qwen3_reranker()
    run_qwen3_vl_reranker()
