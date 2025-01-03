import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_eval_llm():
    from swift.llm import EvalArguments, eval_main
    eval_main(EvalArguments(model='Qwen/Qwen2-7B-Instruct', eval_dataset='arc_c', infer_backend='vllm'))


def test_eval_mllm():
    from swift.llm import EvalArguments, eval_main
    eval_main(
        EvalArguments(model='Qwen/Qwen2-VL-7B-Instruct', eval_dataset=['realWorldQA', 'arc_c'], infer_backend='vllm'))


def test_eval_url():
    from swift.llm import EvalArguments, eval_main, DeployArguments, run_deploy
    deploy_args = DeployArguments(model='Qwen/Qwen2-VL-7B-Instruct', infer_backend='vllm', verbose=False)

    with run_deploy(deploy_args, return_url=True) as url:
        eval_main(EvalArguments(model='Qwen2-VL-7B-Instruct', eval_url=url, eval_dataset=['arc_c']))


if __name__ == '__main__':
    # test_eval_llm()
    # test_eval_mllm()
    test_eval_url()
