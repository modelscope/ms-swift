# Copyright (c) Alibaba, Inc. and its affiliates.
class LLMMegatronModelType:
    gpt = 'gpt'
    qwen3_next = 'qwen3_next'


class MLLMMegatronModelType:
    qwen2_vl = 'qwen2_vl'
    qwen2_5_vl = 'qwen2_5_vl'
    qwen3_vl = 'qwen3_vl'
    qwen2_5_omni = 'qwen2_5_omni'
    qwen3_omni = 'qwen3_omni'
    ovis2_5 = 'ovis2_5'

    internvl3 = 'internvl3'
    internvl_hf = 'internvl_hf'
    glm4_5v = 'glm4_5v'
    kimi_vl = 'kimi_vl'


class MegatronModelType(LLMMegatronModelType, MLLMMegatronModelType):
    pass
