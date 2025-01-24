import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

infer_backend = 'vllm'


def test_eval_native():
    from swift.llm import EvalArguments, eval_main
    eval_main(
        EvalArguments(
            model='Qwen/Qwen2.5-0.5B-Instruct',
            eval_dataset='arc',
            infer_backend=infer_backend,
            eval_backend='Native',
            eval_limit=10))


def test_eval_llm():
    from swift.llm import EvalArguments, eval_main
    eval_main(
        EvalArguments(
            model='Qwen/Qwen2-7B-Instruct',
            eval_dataset='arc_c',
            infer_backend=infer_backend,
            eval_backend='OpenCompass',
            eval_limit=10))


def test_eval_mllm():
    from swift.llm import EvalArguments, eval_main
    eval_main(
        EvalArguments(
            model='Qwen/Qwen2-VL-7B-Instruct',
            eval_dataset=['realWorldQA'],
            infer_backend=infer_backend,
            eval_backend='VLMEvalKit',
            eval_limit=10))


def test_eval_url():
    from swift.llm import EvalArguments, eval_main, DeployArguments, run_deploy
    deploy_args = DeployArguments(model='Qwen/Qwen2-VL-7B-Instruct', infer_backend=infer_backend, verbose=False)

    with run_deploy(deploy_args, return_url=True) as url:
        eval_main(EvalArguments(model='Qwen2-VL-7B-Instruct', eval_url=url, eval_dataset=['arc_c']))


if __name__ == '__main__':
    # test_eval_llm()
    # test_eval_mllm()
    # test_eval_url()
    test_eval_native()
