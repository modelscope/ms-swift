# Copyright (c) ModelScope Contributors. All rights reserved.
from ..llm_train import Quantization


class GRPOQuantization(Quantization):

    group = 'llm_grpo'
