import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_eval_llm():
    from swift.llm import EvalArguments, eval_main


def test_eval_mllm():
    pass


if __name__ == '__main__':
    pass
