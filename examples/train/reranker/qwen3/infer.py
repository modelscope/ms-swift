# For full-parameter training, please refer to:
# https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_reranker.py

from swift.llm import InferRequest, PtEngine


def run_qwen3_reranker():
    engine = PtEngine(
        'Qwen/Qwen3-Reranker-4B',
        task_type='generative_reranker',
        attn_impl='flash_attention_2',
        adapters=['output/vx-xxx/checkpoint-xxx'])

    infer_requests = [
        InferRequest(messages=[{
            'role': 'user',
            'content': 'Mindful emotion regulation: An integrative review.'
        }, {
            'role':
            'assistant',
            'content':
            'Differential effects of mindful breathing, progressive muscle relaxation, and loving-kindness '
            'meditation on decentering and negative reactions to repetitive thoughts.'
        }]),
        InferRequest(messages=[{
            'role': 'user',
            'content': 'Mindful emotion regulation: An integrative review.'
        }, {
            'role': 'assistant',
            'content': 'Exploiting vulnerability to secure user privacy on a social networking site'
        }])
    ]

    responses = engine.infer(infer_requests)
    scores = [response.choices[0].message.content for response in responses]
    print(f'scores: {scores}')


if __name__ == '__main__':
    run_qwen3_reranker()
