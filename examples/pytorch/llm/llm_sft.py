# Copyright (c) Alibaba, Inc. and its affiliates.
import custom

from swift.llm.run import sft_main

if __name__ == '__main__':
    output = sft_main()
    print(f'sft_main output: {output}')
