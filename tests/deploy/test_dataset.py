def _test_client():
    import time
    import aiohttp
    from swift.llm import InferClient, InferRequest, RequestConfig, load_dataset
    dataset = load_dataset(['alpaca_zh#100'], num_proc=4)

    while True:
        try:
            infer_client = InferClient()
        except aiohttp.ClientConnectorError:
            time.sleep(5)
            continue
        break
    models = infer_client.models
    print(f'models: {models}')
    infer_requests = []
    for data in dataset[0]:
        infer_requests.append(InferRequest(**data))
    request_config = RequestConfig(seed=42, max_tokens=256, temperature=0.8)

    resp = infer_client.infer(infer_requests, request_config=request_config, use_tqdm=False)
    print(len(resp))


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
    _test_client()
    process.terminate()


def test_vllm():
    _test('vllm')


def test_lmdeploy():
    _test('lmdeploy')


def test_pt():
    _test('pt')


def test_vllm_orgin():
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
