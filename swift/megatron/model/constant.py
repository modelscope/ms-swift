# Copyright (c) Alibaba, Inc. and its affiliates.
class LLMMegatronModelType:
    gpt = 'gpt'
    qwen3_next = 'qwen3_next'


class MLLMMegatronModelType:
    qwen2_vl = 'qwen2_vl'
    qwen2_5_vl = 'qwen2_5_vl'
    qwen3_vl = 'qwen3_vl'
    qwen2_5_omni = 'qwen2_5_omni'
    ovis2_5 = 'ovis2_5'

    internvl3 = 'internvl3'
    glm4_5v = 'glm4_5v'


class MegatronModelType(LLMMegatronModelType, MLLMMegatronModelType):
    pass
