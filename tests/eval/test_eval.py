import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_eval_llm():
    from swift.llm import EvalArguments, eval_main
    eval_main(EvalArguments(model='qwen/Qwen2-7B-Instruct', eval_dataset='arc_e', infer_backend='vllm'))


def test_eval_mllm():
    from swift.llm import EvalArguments, eval_main
    eval_main(EvalArguments(model='qwen/Qwen2-VL-7B-Instruct', eval_dataset='realWorldQA', infer_backend='vllm'))


if __name__ == '__main__':
    # test_eval_llm()
    test_eval_mllm()
