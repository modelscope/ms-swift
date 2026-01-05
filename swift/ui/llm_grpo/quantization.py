# Copyright (c) Alibaba, Inc. and its affiliates.
from ..llm_train import Quantization


class GRPOQuantization(Quantization):

    group = 'llm_grpo'
