import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_engine():
    from swift.llm import SglangEngine, load_dataset, RequestConfig
    dataset = load_dataset('AI-ModelScope/alpaca-gpt4-data-zh#20')[0]
    engine = SglangEngine('Qwen/Qwen2.5-0.5B-Instruct', disable_radix_cache=True)
    request_config = RequestConfig(max_tokens=1024)
    resp_list = engine.infer(list(dataset), request_config=request_config)
    for resp in resp_list[:5]:
        print(resp)
    resp_list = engine.infer(list(dataset), request_config=request_config)
    for resp in resp_list[:5]:
        print(resp)


def test_engine_stream():
    from swift.llm import SglangEngine, load_dataset, RequestConfig
    dataset = load_dataset('AI-ModelScope/alpaca-gpt4-data-zh#1')[0]
    engine = SglangEngine('Qwen/Qwen2.5-0.5B-Instruct')
    request_config = RequestConfig(max_tokens=1024)
    resp_list = engine.infer(list(dataset), request_config=request_config, stream=True)
    resp_list = engine.infer(list(dataset), request_config=request_config, stream=True)
    for resp in resp_list[:5]:
        print(resp)


def test_infer():
    pass


def test_deploy():
    pass


if __name__ == '__main__':
    test_engine()
    # test_engine_stream()
