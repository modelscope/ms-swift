import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_eval_llm():
    from swift.llm import EvalArguments, eval_main
    eval_main(EvalArguments(model='qwen/Qwen2-7B-Instruct', eval_dataset='arc_e'))


def test_eval_mllm():
    pass


if __name__ == '__main__':
    test_eval_llm()
