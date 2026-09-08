# Copyright (c) ModelScope Contributors. All rights reserved.
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.utils import WeightsMapper


class UEmbedForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    # Accept both the original AutoModel checkpoint and exported CausalLM weights.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        'language_model.': 'model.language_model.',
        'visual.': 'model.visual.',
    }) | Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper
