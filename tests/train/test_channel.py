import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_channel():
    from swift.llm import sft_main, TrainArguments
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2.5-VL-7B-Instruct',
            dataset=['channel.jsonl#1000'],
            channels=['aaa', 'abc'],
            loss_type='channel_loss'))


if __name__ == '__main__':
    test_channel()
