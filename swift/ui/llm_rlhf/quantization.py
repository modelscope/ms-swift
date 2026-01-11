# Copyright (c) Alibaba, Inc. and its affiliates.
from ..llm_train import Quantization


class RLHFQuantization(Quantization):

    group = 'llm_rlhf'
