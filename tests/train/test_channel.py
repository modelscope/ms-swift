import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test_channel():
    from swift.llm import sft_main, TrainArguments
    sft_main(
        TrainArguments(
            model='Qwen/Qwen2.5-7B-Instruct',
            dataset=['channel.jsonl#1000'],
            split_dataset_ratio=0.01,
            enable_channel_loss=True,
            packing=True,
            max_length=128,
            attn_impl='flash_attn',
            load_from_cache_file=False,
            deepspeed='zero2',
            eval_steps=5))


if __name__ == '__main__':
    test_channel()
