# Copyright (c) Alibaba, Inc. and its affiliates.


def try_init_unsloth():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tuner_backend', type=str, default='peft')
    args, _ = parser.parse_known_args()
    if args.tuner_backend == 'unsloth':
        import unsloth


if __name__ == '__main__':
    from swift.cli.utils import try_use_single_device_mode
    try_use_single_device_mode()
    try_init_unsloth()
    from swift.ray import try_init_ray
    try_init_ray()
    from swift.llm import sft_main
    sft_main()
