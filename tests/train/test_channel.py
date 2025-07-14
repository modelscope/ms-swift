import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_channel():
    from swift.llm import sft_main, TrainArguments
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2.5-VL-7B-Instruct',
            dataset=['channel.jsonl#1000'],
            split_dataset_ratio=0.01,
            packing=True,
            max_length=128,
            channels=['aaa', 'abc'],
            attn_impl='flash_attn',
            loss_type='channel_loss',
            eval_steps=10))


if __name__ == '__main__':
    test_channel()
