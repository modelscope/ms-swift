# Copyright (c) Alibaba, Inc. and its affiliates.
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    from swift.llm import EvalArguments, eval_main, run_deploy, DeployArguments
    # Here's a runnable demo provided. Use the eval_url method for evaluation.
    # In a real scenario, you can simply remove the deployed context.
    print(EvalArguments.list_eval_dataset())
    with run_deploy(
            DeployArguments(model='Qwen/Qwen2.5-1.5B-Instruct', verbose=False, log_interval=-1, infer_backend='vllm'),
            return_url=True) as url:
        eval_main(EvalArguments(model='Qwen2.5-1.5B-Instruct', eval_url=url, eval_dataset=['ARC_c']))
