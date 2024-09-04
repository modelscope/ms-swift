def _test_client(print_logprobs: bool = False):
    import requests
    import time
    from pprint import pprint
    from swift.llm import get_model_list_client, XRequestConfig, inference_client
    query = '123*234=?'

    while True:
        try:
            model_list = get_model_list_client()
        except requests.exceptions.ConnectionError:
            time.sleep(5)
            continue
        break
    model_type = model_list.data[0].id
    is_chat = model_list.data[0].is_chat
    is_multimodal = model_list.data[0].is_multimodal
    print(f'model_type: {model_type}')

    request_config = XRequestConfig(seed=42, max_tokens=256, temperature=0.8, logprobs=True, top_logprobs=5)
    resp = inference_client(
        model_type, query, request_config=request_config, is_chat=is_chat, is_multimodal=is_multimodal)
    response = resp.choices[0].message.content
    print(f'query: {query}')
    print(f'response: {response}')
    if print_logprobs:
        pprint(resp.choices[0].logprobs)

    request_config = XRequestConfig(
        stream=True, seed=42, max_tokens=256, temperature=0.8, top_k=20, top_p=0.8, logprobs=True, top_logprobs=5)
    stream_resp = inference_client(
        model_type, query, request_config=request_config, is_chat=is_chat, is_multimodal=is_multimodal)
    print(f'query: {query}')
    print('response: ', end='')
    for chunk in stream_resp:
        print(chunk.choices[0].delta.content, end='', flush=True)
        if print_logprobs and chunk.choices[0].logprobs is not None:
            pprint(chunk.choices[0].logprobs)
    print()


def _test(infer_backend):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TIMEOUT'] = '-1'

    from swift.llm import DeployArguments
    from swift.llm.deploy import llm_deploy
    import multiprocessing
    mp = multiprocessing.get_context('spawn')
    process = mp.Process(
        target=llm_deploy,
        args=(DeployArguments(model_type='qwen2-7b-instruct', infer_backend=infer_backend, verbose=False), ))
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
    # test_vllm()
    # test_lmdeploy()
    test_pt()
