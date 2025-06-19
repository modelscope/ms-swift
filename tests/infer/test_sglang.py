import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_engine():
    from swift.llm import SglangEngine, load_dataset, RequestConfig
    dataset = load_dataset('AI-ModelScope/alpaca-gpt4-data-zh#20')[0]
    engine = SglangEngine('Qwen/Qwen2.5-0.5B-Instruct')
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
    request_config = RequestConfig(max_tokens=1024, stream=True)
    gen_list = engine.infer(list(dataset), request_config=request_config)
    for resp in gen_list[0]:
        print(resp.choices[0].delta.content, flush=True, end='')


def test_infer():
    from swift.llm import infer_main, InferArguments
    infer_main(
        InferArguments(model='Qwen/Qwen2.5-0.5B-Instruct', stream=True, infer_backend='sglang', max_new_tokens=2048))


def test_eval():
    from swift.llm import EvalArguments, eval_main
    eval_main(
        EvalArguments(
            model='Qwen/Qwen2-7B-Instruct',
            eval_dataset='arc_c',
            infer_backend='sglang',
            eval_backend='OpenCompass',
        ))


if __name__ == '__main__':
    test_engine()
    # test_engine_stream()
    # test_infer()
    # test_eval()
