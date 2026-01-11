# Copyright (c) Alibaba, Inc. and its affiliates.
from ..llm_train import LoRA


class GRPOLoRA(LoRA):

    group = 'llm_grpo'
