import os

from tests._test_utils import setup_device_env

setup_device_env('0')


def test_cogvlm():
    from swift import InferArguments, SftArguments, infer_main, sft_main

    # infer_main(InferArguments(model='ZhipuAI/cogvlm2-video-llama3-chat'))
    sft_main(
        SftArguments(
            model='ZhipuAI/cogvlm2-video-llama3-chat',
            dataset=['AI-ModelScope/alpaca-gpt4-data-zh#200', 'swift/VideoChatGPT:Generic#200'],
            split_dataset_ratio=0.01))


if __name__ == '__main__':
    test_cogvlm()
