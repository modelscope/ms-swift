def _test_client(port=8000):
    import time
    import aiohttp
    from swift.llm import InferClient, InferRequest, RequestConfig, load_dataset, run_deploy
    dataset = load_dataset(['AI-ModelScope/alpaca-gpt4-data-zh#1000'], num_proc=4)
    infer_client = InferClient(port=port)
    while True:
        try:
            infer_client.models
            break
        except Exception:
            time.sleep(1)
            pass
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
    from swift.llm import run_deploy
    args = DeployArguments(model='Qwen/Qwen2-7B-Instruct', infer_backend=infer_backend, verbose=False)
    with run_deploy(args) as port:
        _test_client(port)


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
    model_dir = snapshot_download('Qwen/Qwen2-7B-Instruct')
    args = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', model_dir]
    process = subprocess.Popen(args)
    _test_client()
    process.terminate()


if __name__ == '__main__':
    # test_vllm_orgin()
    # test_vllm()
    test_lmdeploy()
    # test_pt()
