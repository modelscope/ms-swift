# Copyright (c) Alibaba, Inc. and its affiliates.
import custom

from swift.llm import sft_main

if __name__ == '__main__':
    # seed_value = 42
    # import random
    # import numpy as np 
    # import torch 
    # random.seed(seed_value)
    # np.random.seed(seed_value)
    # torch.manual_seed(seed_value)
    # torch.cuda.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    output = sft_main()
