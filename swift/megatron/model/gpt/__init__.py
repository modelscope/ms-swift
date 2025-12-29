# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import ModelType
from ..constant import MegatronModelType
from ..register import MegatronModelMeta, register_megatron_model
from . import glm4, qwen3_next

register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.gpt,
        [
            ModelType.qwen2,
            ModelType.qwen2_5,
            ModelType.llama,
            ModelType.codefuse_codellama,
            ModelType.marco_o1,
            ModelType.deepseek,
            ModelType.deepseek_r1_distill,
            ModelType.yi,
            ModelType.skywork_o1,
            ModelType.openbuddy_llama,
            ModelType.openbuddy_llama3,
            ModelType.megrez,
            ModelType.reflection,
            ModelType.numina,
            ModelType.ziya,
            ModelType.mengzi3,
            ModelType.qwen3,
            ModelType.qwen3_thinking,
            ModelType.qwen3_nothinking,
            ModelType.qwen2_moe,
            ModelType.qwen3_moe,
            ModelType.qwen3_moe_thinking,
            ModelType.qwen3_coder,
            ModelType.internlm3,
            ModelType.mimo,
            ModelType.mimo_rl,
            ModelType.moonlight,
            ModelType.kimi_k2,
            ModelType.deepseek_moe,
            ModelType.deepseek_v2,
            ModelType.deepseek_v3,
            ModelType.dots1,
            ModelType.ernie,
            ModelType.glm4_moe,
            ModelType.ernie_thinking,
            ModelType.gpt_oss,
        ],
    ))
