# Copyright (c) Alibaba, Inc. and its affiliates.
import custom

from swift.llm.run import sft_main

if __name__ == '__main__':
    best_ckpt_dir = sft_main()
    print(f'best_ckpt_dir: {best_ckpt_dir}')
