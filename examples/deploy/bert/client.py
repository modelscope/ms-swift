from typing import List

from swift.llm import InferClient, InferRequest


def infer_batch(engine: InferClient, infer_requests: List[InferRequest]):
    resp_list = engine.infer(infer_requests)
    query0 = infer_requests[0].messages[0]['content']
    query1 = infer_requests[1].messages[0]['content']
    print(f'query0: {query0}')
    print(f'response0: {resp_list[0].choices[0].message.content}')
    print(f'query1: {query1}')
    print(f'response1: {resp_list[1].choices[0].message.content}')


if __name__ == '__main__':
    engine = InferClient(host='127.0.0.1', port=8000)
    models = engine.models
    print(f'models: {models}')
    infer_batch(engine, [
        InferRequest(messages=[{
            'role': 'user',
            'content': '今天天气真好呀'
        }]),
        InferRequest(messages=[{
            'role': 'user',
            'content': '真倒霉'
        }])
    ])
