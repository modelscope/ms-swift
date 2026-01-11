# Copyright (c) Alibaba, Inc. and its affiliates.
from ..llm_train import LoRA


class RLHFLoRA(LoRA):

    group = 'llm_rlhf'
