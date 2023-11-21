# Copyright (c) Alibaba, Inc. and its affiliates.
import custom

from swift.llm.run import infer_main

if __name__ == '__main__':
    result = infer_main()
    print(f'infer_main result: {result}')
