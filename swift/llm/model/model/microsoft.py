# Copyright (c) Alibaba, Inc. and its affiliates.
from types import MethodType
from typing import Any, Dict

from transformers import PretrainedConfig

from swift.llm import TemplateType
from swift.utils import get_env_args
from ..constant import LLMModelType, MLLMModelType
from ..patcher import patch_output_clone
from ..register import (Model, ModelGroup, ModelMeta, get_model_tokenizer_from_local,
                        get_model_tokenizer_with_flash_attn, register_model)
from ..utils import ignore_check_imports, use_submodel_func


def get_model_tokenizer_phi3_vision(model_dir: str,
                                    model_config: PretrainedConfig,
                                    model_kwargs: Dict[str, Any],
                                    load_model: bool = True,
                                    **kwargs):
    processor_kwargs = {}
    if 'Phi-3.5-vision-instruct' in model_dir:
        kwargs['num_crops'] = kwargs.get('num_crops') or 4
    if 'num_crops' in kwargs:
        processor_kwargs['num_crops'] = get_env_args('num_crops', int, kwargs['num_crops'])
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, **processor_kwargs)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_config, model_kwargs, load_model, **kwargs)
    tokenizer.processor = processor

    if load_model:
        patch_output_clone(model.model.vision_embed_tokens.wte)

    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.phi3_vl,
        [
            # llama2
            ModelGroup([
                Model('LLM-Research/Phi-3-vision-128k-instruct', 'microsoft/Phi-3-vision-128k-instruct'),
                Model('LLM-Research/Phi-3.5-vision-instruct', 'microsoft/Phi-3.5-vision-instruct'),
            ],
                       requires=['transformers>=4.36'],
                       tags=['multi-modal', 'vision'],
                       ignore_file_pattern=[r'.+\.bin$']),
        ],
        TemplateType.phi3_vl,
        get_model_tokenizer_phi3_vision,
        architectures=['MambaForCausalLM'],
        support_flash_attn=True,
        support_vllm=True,
    ))


def get_model_tokenizer_florence(model_dir: str,
                                 model_config: PretrainedConfig,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    with ignore_check_imports():
        model, tokenizer = get_model_tokenizer_with_flash_attn(
            model_dir, model_config, model_kwargs, load_model, tokenizer=processor.tokenizer, **kwargs)

    tokenizer.processor = processor
    # model.vision_tower.enable_checkpoint = True
    use_submodel_func(model, 'language_model', ['generate', 'forward'])
    return model, tokenizer


register_model(
    ModelMeta(
        MLLMModelType.florence,
        [
            # llama2
            ModelGroup([
                Model('AI-ModelScope/Florence-2-base', 'microsoft/Florence-2-base'),
                Model('AI-ModelScope/Florence-2-base-ft', 'microsoft/Florence-2-base-ft'),
                Model('AI-ModelScope/Florence-2-large', 'microsoft/Florence-2-large'),
                Model('AI-ModelScope/Florence-2-large-ft', 'microsoft/Florence-2-large-ft'),
            ],
                       requires=['transformers>=4.36'],
                       tags=['multi-modal', 'vision'],
                       ignore_file_pattern=[r'.+\.bin$']),
        ],
        TemplateType.phi3_vl,
        get_model_tokenizer_phi3_vision,
        architectures=['MambaForCausalLM'],
        support_flash_attn=True,
    ))


def get_model_tokenizer_phi3_small(model_dir: str,
                                   model_config: PretrainedConfig,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   **kwargs):
    attn_type = AttentionType(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', 'sdpa'))
    attn_type.update_config(model_config)
    model, tokenizer = get_model_tokenizer_from_repo(
        model_dir, torch_dtype, model_kwargs, load_model, model_config=model_config, **kwargs)

    def rotary_emb(self, query_states, key_states, **kwargs):
        q_type = query_states.dtype
        k_type = key_states.dtype
        query_states, key_states = self.rotory_emb_origin(query_states, key_states, **kwargs)
        query_states = query_states.to(q_type)
        key_states = key_states.to(k_type)
        return query_states, key_states

    for i in range(32):
        re = model.model.layers[i].self_attn.rotary_emb
        re.rotory_emb_origin = re.forward
        re.forward = MethodType(rotary_emb, re)
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.phi3_small,
        [
            ModelGroup([
                Model('LLM-Research/Phi-3-small-8k-instruct', 'microsoft/Phi-3-small-8k-instruct'),
                Model('LLM-Research/Phi-3-small-128k-instruct', 'microsoft/Phi-3-small-128k-instruct'),
            ],
                       requires=['transformers>=4.36'],
                       tags=['multi-modal', 'vision'],
                       ignore_file_pattern=[r'.+\.bin$']),
        ],
        TemplateType.phi3,
        get_model_tokenizer_phi3_small,
        architectures=['MambaForCausalLM'],
        support_flash_attn=True,
        support_gradient_checkpointing=False,
        support_vllm=True,
    ))


def get_model_tokenizer_phi(model_dir: str,
                            config: PretrainedConfig,
                            model_kwargs: Dict[str, Any],
                            load_model: bool = True,
                            **kwargs):
    attn_type = AttentionImpl(kwargs.pop('use_flash_attn', None), kwargs.pop('attn_type', None))
    config.flash_attn = attn_type.to_bool()
    return get_model_tokenizer_from_local(model_dir, config, model_kwargs, load_model, **kwargs)


register_model(
    ModelMeta(
        LLMModelType.phi2,
        [
            ModelGroup([
                Model('AI-ModelScope/phi-2', 'microsoft/phi-2'),
            ],
                       requires=['transformers>=4.36'],
                       tags=['coding'],
                       ignore_file_pattern=[r'.+\.bin$']),
        ],
        TemplateType.default,
        get_model_tokenizer_phi3_small,
        architectures=['MambaForCausalLM'],
        support_flash_attn=True,
        support_vllm=True,
        support_gradient_checkpointing=False,
    ))

register_model(
    ModelMeta(
        LLMModelType.phi3,
        [
            ModelGroup([
                Model('LLM-Research/Phi-3-mini-128k-instruct', 'microsoft/Phi-3-mini-128k-instruct'),
                Model('LLM-Research/Phi-3-medium-4k-instruct', 'microsoft/Phi-3-medium-4k-instruct'),
                Model('LLM-Research/Phi-3-medium-128k-instruct', 'microsoft/Phi-3-medium-128k-instruct'),
                Model('LLM-Research/Phi-3-mini-4k-instruct', 'microsoft/Phi-3-mini-4k-instruct'),
                Model('LLM-Research/Phi-3.5-mini-instruct', 'microsoft/Phi-3.5-mini-instruct'),
            ],
                       requires=['transformers>=4.36']),
            ModelGroup([
                Model('LLM-Research/Phi-3.5-MoE-instruct', 'microsoft/Phi-3.5-MoE-instruct'),
            ],
                       requires=['transformers>=4.36'],
                       tags=['moe']),
        ],
        TemplateType.phi3,
        get_model_tokenizer_with_flash_attn,
        architectures=['MambaForCausalLM'],
        support_flash_attn=True,
        support_vllm=True,
    ))
