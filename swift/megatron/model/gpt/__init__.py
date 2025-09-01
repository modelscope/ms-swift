# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass
from typing import Any, Callable, Dict

from torch import nn
from transformers import PretrainedConfig

from swift.llm import ModelType
from ..constant import MegatronModelType
from ..register import MegatronModelMeta, register_megatron_model
from .config import convert_gpt_hf_config
from .hf2mcore import convert_hf2mcore as convert_hf2mcore_gpt
from .mcore2hf import convert_mcore2hf as convert_mcore2hf_gpt
from .model import model_provider as gpt_model_provider


@dataclass
class GPTMegatronModelMeta(MegatronModelMeta):
    model_provider: Callable[[], nn.Module] = gpt_model_provider
    convert_hf_config: Callable[[PretrainedConfig], Dict[str, Any]] = convert_gpt_hf_config
    convert_mcore2hf: Callable[[nn.Module, nn.Module], None] = convert_mcore2hf_gpt
    convert_hf2mcore: Callable[[nn.Module, nn.Module], None] = convert_hf2mcore_gpt


register_megatron_model(
    GPTMegatronModelMeta(MegatronModelType.gpt, [
        ModelType.qwen2,
        ModelType.qwen2_5,
        ModelType.qwq,
        ModelType.qwq_preview,
        ModelType.qwen2_5_math,
        ModelType.llama,
        ModelType.llama3,
        ModelType.llama3_1,
        ModelType.llama3_2,
        ModelType.longwriter_llama3_1,
        ModelType.codefuse_codellama,
        ModelType.marco_o1,
        ModelType.deepseek,
        ModelType.deepseek_r1_distill,
        ModelType.yi,
        ModelType.yi_coder,
        ModelType.sus,
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
        ModelType.internlm3,
        ModelType.mimo,
        ModelType.mimo_rl,
        ModelType.moonlight,
        ModelType.deepseek_moe,
        ModelType.deepseek_v2,
        ModelType.deepseek_v2_5,
        ModelType.deepseek_r1,
        ModelType.dots1,
        ModelType.ernie,
        ModelType.glm4_5,
        ModelType.deepseek_v3_1,
    ]))
