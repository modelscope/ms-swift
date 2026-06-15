import os

from swift import TransformersEngine
from swift.infer_engine import InferRequest, RequestConfig
from swift.metrics import InferStats

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
engine = TransformersEngine('Qwen/Qwen2-0.5B', max_batch_size=4)


def test_batch_infer():
    infer_requests = [InferRequest([{'role': 'user', 'content': 'hello, who are you?'}]) for _ in range(4)]
    request_config = RequestConfig(temperature=0, max_tokens=32)
    infer_stats = InferStats()

    response_list = engine.infer(infer_requests, request_config=request_config, metrics=[infer_stats])

    assert len(response_list) == len(infer_requests)
    for response in response_list:
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None

    stats = infer_stats.compute()
    assert stats['num_samples'] > 0
    assert stats['num_generated_tokens'] > 0


def test_stream_infer():
    infer_requests = [InferRequest([{'role': 'user', 'content': 'What is 1+1? Answer briefly.'}])]
    request_config = RequestConfig(temperature=0, max_tokens=32, stream=True)
    infer_stats = InferStats()

    gen_list = engine.infer(infer_requests, request_config=request_config, metrics=[infer_stats])

    full_content = ''
    for chunk in gen_list[0]:
        if chunk is None:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            full_content += delta

    assert len(full_content) > 0, 'Stream infer produced no content'

    stats = infer_stats.compute()
    assert stats['num_samples'] > 0
    assert stats['num_generated_tokens'] > 0


def test_single_infer_with_system():
    infer_requests = [
        InferRequest([{
            'role': 'system',
            'content': 'You are a helpful assistant.'
        }, {
            'role': 'user',
            'content': 'Say hello in one word.'
        }])
    ]
    request_config = RequestConfig(temperature=0, max_tokens=16)

    response_list = engine.infer(infer_requests, request_config=request_config)

    assert len(response_list) == 1
    assert len(response_list[0].choices) > 0
    assert response_list[0].choices[0].message.content is not None


if __name__ == '__main__':
    test_batch_infer()
    test_stream_infer()
    test_single_infer_with_system()
