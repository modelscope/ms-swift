def test_benchmark(infer_backend: str) -> None:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TIMEOUT'] = '-1'
    import requests
    from swift.llm import DeployArguments, get_dataset, get_model_list_client, XRequestConfig, inference_client_async
    from swift.llm.deploy import llm_deploy
    import multiprocessing
    import time
    import asyncio
    from swift.utils import get_logger

    logger = get_logger()

    mp = multiprocessing.get_context('spawn')
    process = mp.Process(
        target=llm_deploy,
        args=(DeployArguments(model_type='qwen2-7b-instruct', infer_backend=infer_backend, verbose=False), ))
    process.start()

    dataset = get_dataset(['alpaca-zh#1000', 'alpaca-en#1000'])[0]
    query_list = dataset['query']
    request_config = XRequestConfig(seed=42, max_tokens=8192)

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

    tasks = []
    for query in query_list:
        tasks.append(
            inference_client_async(
                model_type, query, request_config=request_config, is_chat=is_chat, is_multimodal=is_multimodal))

    async def _batch_run(tasks):
        return await asyncio.gather(*tasks)

    resp_list = asyncio.run(_batch_run(tasks))
    logger.info(f'len(resp_list): {len(resp_list)}')
    logger.info(f'resp_list[0]: {resp_list[0]}')
    process.terminate()


def test_vllm_benchmark():
    test_benchmark('vllm')


def test_lmdeploy_benchmark():
    test_benchmark('lmdeploy')


if __name__ == '__main__':
    # test_vllm_benchmark()
    test_lmdeploy_benchmark()
