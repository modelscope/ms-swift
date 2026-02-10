# Copyright (c) ModelScope Contributors. All rights reserved.
class LLMMegatronModelType:
    gpt = 'gpt'
    qwen3_next = 'qwen3_next'
    olmoe = 'olmoe'
    glm4 = 'glm4'
    minimax_m2 = 'minimax_m2'

    qwen3_emb = 'qwen3_emb'


class MLLMMegatronModelType:
    qwen2_vl = 'qwen2_vl'
    qwen2_5_vl = 'qwen2_5_vl'
    qwen3_vl = 'qwen3_vl'
    qwen2_5_omni = 'qwen2_5_omni'
    qwen3_omni = 'qwen3_omni'
    qwen3_5 = 'qwen3_5'
    ovis2_5 = 'ovis2_5'

    internvl3 = 'internvl3'
    internvl_hf = 'internvl_hf'
    glm4v = 'glm4v'
    glm4v_moe = 'glm4v_moe'
    kimi_vl = 'kimi_vl'
    llama4 = 'llama4'


class MegatronModelType(LLMMegatronModelType, MLLMMegatronModelType):
    pass
