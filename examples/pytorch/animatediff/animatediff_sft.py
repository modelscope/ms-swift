# Copyright (c) Alibaba, Inc. and its affiliates.

from swift.llm.run import animatediff_main

if __name__ == '__main__':
    best_ckpt_dir = animatediff_main()
    print(f'best_ckpt_dir: {best_ckpt_dir}')
