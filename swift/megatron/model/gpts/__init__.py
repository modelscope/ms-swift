# Copyright (c) ModelScope Contributors. All rights reserved.
from swift.model import ModelType
from ..constant import MegatronModelType
from ..register import MegatronModelMeta, register_megatron_model
from . import glm4, minimax_m2, olmoe, qwen3_emb, qwen3_next

register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.gpt,
        [
            ModelType.qwen2,
            ModelType.llama,
            ModelType.codefuse_codellama,
            ModelType.yi,
            ModelType.openbuddy_llama,
            ModelType.qwen3,
            ModelType.qwen3_reranker,
            ModelType.qwen2_moe,
            ModelType.qwen3_moe,
            ModelType.internlm3,
            ModelType.mimo,
            ModelType.deepseek,
            ModelType.deepseek_v2,
            ModelType.deepseek_v3,
            ModelType.dots1,
            ModelType.ernie4_5,
            ModelType.ernie4_5_moe,
            ModelType.glm4_moe,
            ModelType.glm4_moe_lite,
            ModelType.gpt_oss,
        ],
    ))
