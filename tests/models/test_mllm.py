import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_cogvlm():
    from swift.llm import infer_main, InferArguments, sft_main, TrainArguments
    # infer_main(InferArguments(model='ZhipuAI/cogvlm2-video-llama3-chat'))
    sft_main(
        TrainArguments(
            model='ZhipuAI/cogvlm2-video-llama3-chat',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#200', 'swift/VideoChatGPT:Generic#200']))


if __name__ == '__main__':
    test_cogvlm()
