# Copyright (c) Alibaba, Inc. and its affiliates.
import os

if int(os.environ.get('UNSLOTH_PATCH_TRL', '0')) != 0:
    import unsloth

from swift.llm import sft_main

if __name__ == '__main__':
    sft_main()
