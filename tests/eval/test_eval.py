import os

from swift.llm import SwiftEval

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_eval_llm():
    from swift.llm import EvalArguments, eval_main
    eval_main(EvalArguments(model='qwen/Qwen2-7B-Instruct', eval_dataset='arc_c', infer_backend='vllm'))


def test_eval_mllm():
    from swift.llm import EvalArguments, eval_main
    eval_main(
        EvalArguments(model='qwen/Qwen2-VL-7B-Instruct', eval_dataset=['realWorldQA', 'arc_c'], infer_backend='vllm'))


def test_eval_url():
    from swift.llm import EvalArguments, eval_main, DeployArguments
    deploy_args = DeployArguments(model='qwen/Qwen2-VL-7B-Instruct')

    with SwiftEval.run_deploy(deploy_args) as url:
        eval_main(EvalArguments(eval_url=url, eval_dataset=['realWorldQA', 'arc_c'], infer_backend='vllm'))


if __name__ == '__main__':
    # test_eval_llm()
    # test_eval_mllm()
    test_eval_url()
