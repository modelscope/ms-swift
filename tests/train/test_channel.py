import os

from tests._test_utils import setup_device_env

setup_device_env('0')


def test_channel():
    from swift import SftArguments, sft_main
    sft_main(
        SftArguments(
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
