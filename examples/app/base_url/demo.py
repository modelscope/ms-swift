# Copyright (c) Alibaba, Inc. and its affiliates.
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    from swift.llm import AppArguments, app_main, DeployArguments, run_deploy
    # Here's a runnable demo provided.
    # In a real scenario, you can simply remove the deployed context.
    with run_deploy(
            DeployArguments(model='Qwen/Qwen2.5-1.5B-Instruct', verbose=False, log_interval=-1, infer_backend='vllm'),
            return_url=True) as url:
        app_main(AppArguments(model='Qwen2.5-1.5B-Instruct', base_url=url, stream=True, max_new_tokens=2048))
