def _test_client(print_logprobs: bool = False):
    import requests
    import time
    import aiohttp
    from pprint import pprint
    from swift.llm import InferClient, InferRequest, RequestConfig
    query = '123*234=?'

    while True:
        try:
            infer_client = InferClient()
        except aiohttp.ClientConnectorError:
            time.sleep(5)
            continue
        break
    models = infer_client.models
    print(f'models: {models}')

    infer_request = InferRequest(messages=[{'role': 'user', 'content': '你是谁'}])
    request_config = RequestConfig(seed=42, max_tokens=256, temperature=0.8, logprobs=True, top_logprobs=5)

    resp = infer_client.infer([infer_request], request_config=request_config)[0]
    response = resp.choices[0].message.content
    print(f'query: {query}')
    print(f'response: {response}')
    if print_logprobs:
        pprint(resp.choices[0].logprobs)

    request_config = RequestConfig(
        stream=True, seed=42, max_tokens=256, temperature=0.8, top_k=20, top_p=0.8, logprobs=True, top_logprobs=5)
    stream_resp = infer_client.infer([infer_request], request_config=request_config)
    print(f'query: {query}')
    print('response: ', end='')
    for chunk in stream_resp:
        chunk = chunk[0]
        print(chunk.choices[0].delta.content, end='', flush=True)
        if print_logprobs and chunk.choices[0].logprobs is not None:
            pprint(chunk.choices[0].logprobs)
    print()


def _test(infer_backend):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from swift.llm import DeployArguments
    from swift.llm import deploy_main
    import multiprocessing
    mp = multiprocessing.get_context('spawn')
    process = mp.Process(
        target=deploy_main,
        args=(DeployArguments(model='qwen/Qwen2-7B-Instruct', infer_backend=infer_backend, verbose=False), ))
    process.start()
    _test_client(True)
    process.terminate()


def test_vllm():
    _test('vllm')


def test_lmdeploy():
    _test('lmdeploy')


def test_pt():
    _test('pt')


def test_vllm_orgin():
    import os
    import subprocess
    import sys
    from modelscope import snapshot_download
    model_dir = snapshot_download('qwen/Qwen2-7B-Instruct')
    args = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', model_dir]
    process = subprocess.Popen(args)
    _test_client()
    process.terminate()


if __name__ == '__main__':
    # test_vllm_orgin()
    test_vllm()
    # test_lmdeploy()
    # test_pt()
