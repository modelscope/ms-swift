import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_cogvlm():
    from swift.llm import infer_main, InferArguments
    infer_main(
        InferArguments(model='ZhipuAI/cogvlm2-video-llama3-chat'))


if __name__ == '__main__':
    test_cogvlm()
