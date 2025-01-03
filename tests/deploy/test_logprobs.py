def _test_client(port: int, print_logprobs: bool = False, test_vlm: bool = False):
    import requests
    import time
    import aiohttp
    from pprint import pprint
    from swift.llm import InferClient, InferRequest, RequestConfig

    infer_client = InferClient(port=port)

    while True:
        try:
            models = infer_client.models
            print(f'models: {models}')
        except aiohttp.ClientConnectorError:
            time.sleep(5)
            continue
        break

    if test_vlm:
        query = '这是什么'
        # http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
        messages = [{
            'role':
            'user',
            'content': [
                {
                    'type': 'text',
                    'text': '这是什么'
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': 'cat.png'
                    }
                },
            ]
        }]
    else:
        query = '123*234=?'
        messages = [{'role': 'user', 'content': query}]

    infer_request = InferRequest(messages=messages)
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


def _test(infer_backend, test_vlm: bool = False):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from swift.llm import DeployArguments
    from swift.llm import deploy_main
    import multiprocessing
    mp = multiprocessing.get_context('spawn')
    model = 'Qwen/Qwen2-VL-7B-Instruct' if test_vlm else 'Qwen/Qwen2-7B-Instruct'
    args = DeployArguments(model=model, infer_backend=infer_backend, verbose=False)
    process = mp.Process(target=deploy_main, args=(args, ))
    process.start()
    _test_client(args.port, True, test_vlm)
    process.terminate()


def test_vllm_vlm():
    _test('vllm', test_vlm=True)


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
    model_dir = snapshot_download('Qwen/Qwen2-7B-Instruct')
    args = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', model_dir]
    process = subprocess.Popen(args)
    _test_client(8000)
    process.terminate()


if __name__ == '__main__':
    # test_vllm_orgin()
    # test_vllm()
    test_vllm_vlm()
    # test_lmdeploy()
    # test_pt()
