# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import sft_main

if __name__ == '__main__':
    ckpt_dir = sft_main()
    print(f'best_ckpt: {ckpt_dir}')
