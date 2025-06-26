# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
from typing import Any, Dict

from swift.llm import TemplateType
from ..constant import LLMModelType, MLLMModelType
from ..model_arch import ModelArch
from ..patcher import patch_output_clone, patch_output_to_input_device
from ..register import Model, ModelGroup, ModelMeta, get_model_tokenizer_with_flash_attn, register_model
from ..utils import ModelInfo, git_clone_github, use_submodel_func

register_model(
    ModelMeta(
        LLMModelType.deepseek, [
            ModelGroup([
                Model('deepseek-ai/deepseek-llm-7b-base', 'deepseek-ai/deepseek-llm-7b-base'),
                Model('deepseek-ai/deepseek-llm-7b-chat', 'deepseek-ai/deepseek-llm-7b-chat'),
                Model('deepseek-ai/deepseek-llm-67b-base', 'deepseek-ai/deepseek-llm-67b-base'),
                Model('deepseek-ai/deepseek-llm-67b-chat', 'deepseek-ai/deepseek-llm-67b-chat'),
            ]),
            ModelGroup(
                [
                    Model('deepseek-ai/deepseek-math-7b-base', 'deepseek-ai/deepseek-math-7b-base'),
                    Model('deepseek-ai/deepseek-math-7b-instruct', 'deepseek-ai/deepseek-math-7b-instruct'),
                    Model('deepseek-ai/deepseek-math-7b-rl', 'deepseek-ai/deepseek-math-7b-rl'),
                ],
                tags=['math'],
            ),
            ModelGroup(
                [
                    Model('deepseek-ai/deepseek-coder-1.3b-base', 'deepseek-ai/deepseek-coder-1.3b-base'),
                    Model('deepseek-ai/deepseek-coder-1.3b-instruct', 'deepseek-ai/deepseek-coder-1.3b-instruct'),
                    Model('deepseek-ai/deepseek-coder-6.7b-base', 'deepseek-ai/deepseek-coder-6.7b-base'),
                    Model('deepseek-ai/deepseek-coder-6.7b-instruct', 'deepseek-ai/deepseek-coder-6.7b-instruct'),
                    Model('deepseek-ai/deepseek-coder-33b-base', 'deepseek-ai/deepseek-coder-33b-base'),
                    Model('deepseek-ai/deepseek-coder-33b-instruct', 'deepseek-ai/deepseek-coder-33b-instruct'),
                ],
                tags=['coding'],
            ),
        ],
        TemplateType.deepseek,
        get_model_tokenizer_with_flash_attn,
        architectures=['LlamaForCausalLM'],
        model_arch=ModelArch.llama))


def get_model_tokenizer_deepseek_moe(model_dir: str,
                                     model_info: ModelInfo,
                                     model_kwargs: Dict[str, Any],
                                     load_model: bool = True,
                                     **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)
    if model is not None:
        # fix dtype bug
        mlp_cls = model.model.layers[1].mlp.__class__

        for module in model.modules():
            if isinstance(module, mlp_cls):
                patch_output_to_input_device(module)
    return model, tokenizer


register_model(
    ModelMeta(
        LLMModelType.deepseek_moe,
        [
            ModelGroup([
                Model('deepseek-ai/deepseek-moe-16b-chat', 'deepseek-ai/deepseek-moe-16b-chat'),
                Model('deepseek-ai/deepseek-moe-16b-base', 'deepseek-ai/deepseek-moe-16b-base'),
            ], ),
        ],
        TemplateType.deepseek,
        get_model_tokenizer_deepseek_moe,
        architectures=['DeepseekForCausalLM'],
    ))

register_model(
    ModelMeta(
        LLMModelType.deepseek_v2,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-Coder-V2-Instruct', 'deepseek-ai/DeepSeek-Coder-V2-Instruct'),
                Model('deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', 'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct'),
                Model('deepseek-ai/DeepSeek-Coder-V2-Base', 'deepseek-ai/DeepSeek-Coder-V2-Base'),
                Model('deepseek-ai/DeepSeek-Coder-V2-Lite-Base', 'deepseek-ai/DeepSeek-Coder-V2-Lite-Base'),
                Model('deepseek-ai/DeepSeek-V2-Lite', 'deepseek-ai/DeepSeek-V2-Lite'),
                Model('deepseek-ai/DeepSeek-V2-Lite-Chat', 'deepseek-ai/DeepSeek-V2-Lite-Chat'),
                Model('deepseek-ai/DeepSeek-V2', 'deepseek-ai/DeepSeek-V2'),
                Model('deepseek-ai/DeepSeek-V2-Chat', 'deepseek-ai/DeepSeek-V2-Chat'),
            ]),
        ],
        TemplateType.deepseek,
        get_model_tokenizer_deepseek_moe,
        architectures=['DeepseekV2ForCausalLM'],
        model_arch=ModelArch.deepseek_v2,
        requires=['transformers>=4.39.3'],
    ))

register_model(
    ModelMeta(
        LLMModelType.deepseek_v2_5,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-V2.5', 'deepseek-ai/DeepSeek-V2.5'),
                Model('deepseek-ai/DeepSeek-V2.5-1210', 'deepseek-ai/DeepSeek-V2.5-1210'),
                Model('deepseek-ai/DeepSeek-V3-Base', 'deepseek-ai/DeepSeek-V3-Base'),
                Model('deepseek-ai/DeepSeek-V3', 'deepseek-ai/DeepSeek-V3'),
                Model('deepseek-ai/DeepSeek-V3-0324', 'deepseek-ai/DeepSeek-V3-0324'),
            ]),
            ModelGroup([
                Model('cognitivecomputations/DeepSeek-V3-awq', 'cognitivecomputations/DeepSeek-V3-AWQ'),
                Model('cognitivecomputations/DeepSeek-V3-0324-AWQ', 'cognitivecomputations/DeepSeek-V3-0324-AWQ')
            ]),
            ModelGroup([
                Model('deepseek-ai/DeepSeek-Prover-V2-7B', 'deepseek-ai/DeepSeek-Prover-V2-7B'),
                Model('deepseek-ai/DeepSeek-Prover-V2-671B', 'deepseek-ai/DeepSeek-Prover-V2-671B'),
            ]),
            ModelGroup([
                Model('unsloth/DeepSeek-V3-bf16', 'unsloth/DeepSeek-V3-bf16'),
                Model('unsloth/DeepSeek-V3-0324-BF16', 'unsloth/DeepSeek-V3-0324-BF16'),
                Model('unsloth/DeepSeek-Prover-V2-671B-BF16', 'unsloth/DeepSeek-Prover-V2-671B-BF16'),
            ])
        ],
        TemplateType.deepseek_v2_5,
        get_model_tokenizer_deepseek_moe,
        architectures=['DeepseekV2ForCausalLM', 'DeepseekV3ForCausalLM'],
        model_arch=ModelArch.deepseek_v2,
        requires=['transformers>=4.39.3'],
    ))


def _get_deepseek_vl(processor, llm_prefix, model_dir, *args, **kwargs):
    kwargs['tokenizer'] = processor.tokenizer
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    if model:
        llm = getattr(model, llm_prefix)
        patch_output_clone(llm.model.embed_tokens)
        patch_output_to_input_device(llm.model.embed_tokens)
        use_submodel_func(model, llm_prefix)
        model.generation_config = llm.generation_config
    return model, processor


def get_model_tokenizer_deepseek_vl(model_dir: str, *args, **kwargs):
    # compat with python==3.10
    if sys.version_info.minor >= 10:
        import collections
        import collections.abc
        for type_name in collections.abc.__all__:
            setattr(collections, type_name, getattr(collections.abc, type_name))
    local_repo_path = kwargs.get('local_repo_path')
    if not local_repo_path:
        local_repo_path = git_clone_github('https://github.com/deepseek-ai/DeepSeek-VL')
    sys.path.append(local_repo_path)
    from deepseek_vl.models import VLChatProcessor
    processor = VLChatProcessor.from_pretrained(model_dir)
    return _get_deepseek_vl(processor, 'language_model', model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.deepseek_vl,
        [
            ModelGroup([
                Model('deepseek-ai/deepseek-vl-1.3b-chat', 'deepseek-ai/deepseek-vl-1.3b-chat'),
                Model('deepseek-ai/deepseek-vl-7b-chat', 'deepseek-ai/deepseek-vl-7b-chat'),
            ], ),
        ],
        TemplateType.deepseek_vl,
        get_model_tokenizer_deepseek_vl,
        architectures=['MultiModalityCausalLM'],
        model_arch=ModelArch.deepseek_vl,
        tags=['vision'],
    ))


def get_model_tokenizer_deepseek_janus(model_dir: str, *args, **kwargs):
    local_repo_path = kwargs.get('local_repo_path')
    if not local_repo_path:
        local_repo_path = git_clone_github('https://github.com/deepseek-ai/Janus')
    sys.path.append(local_repo_path)
    from janus.models import VLChatProcessor

    processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_dir)
    return _get_deepseek_vl(processor, 'language_model', model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.deepseek_janus,
        [
            ModelGroup([
                Model('deepseek-ai/Janus-1.3B', 'deepseek-ai/Janus-1.3B'),
            ]),
        ],
        TemplateType.deepseek_janus,
        get_model_tokenizer_deepseek_janus,
        model_arch=ModelArch.deepseek_janus,
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        MLLMModelType.deepseek_janus_pro,
        [
            ModelGroup([
                Model('deepseek-ai/Janus-Pro-1B', 'deepseek-ai/Janus-Pro-1B'),
                Model('deepseek-ai/Janus-Pro-7B', 'deepseek-ai/Janus-Pro-7B'),
            ]),
        ],
        TemplateType.deepseek_janus_pro,
        get_model_tokenizer_deepseek_janus,
        model_arch=ModelArch.deepseek_janus,
        tags=['vision'],
    ))


def get_model_tokenizer_deepseek_vl2(model_dir: str, *args, **kwargs):
    local_repo_path = kwargs.get('local_repo_path')
    if not local_repo_path:
        local_repo_path = git_clone_github('https://github.com/deepseek-ai/DeepSeek-VL2')
    sys.path.append(local_repo_path)
    try:
        from deepseek_vl2.models import DeepseekVLV2Processor
    except ImportError:
        # compat transformers>=4.42
        import transformers
        transformers.models.llama.modeling_llama.LlamaFlashAttention2 = None
        from deepseek_vl2.models import DeepseekVLV2Processor
    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_dir)
    return _get_deepseek_vl(processor, 'language', model_dir, *args, **kwargs)


register_model(
    ModelMeta(
        MLLMModelType.deepseek_vl2,
        [
            ModelGroup([
                Model('deepseek-ai/deepseek-vl2-tiny', 'deepseek-ai/deepseek-vl2-tiny'),
                Model('deepseek-ai/deepseek-vl2-small', 'deepseek-ai/deepseek-vl2-small'),
                Model('deepseek-ai/deepseek-vl2', 'deepseek-ai/deepseek-vl2'),
            ]),
        ],
        TemplateType.deepseek_vl2,
        get_model_tokenizer_deepseek_vl2,
        model_arch=ModelArch.deepseek_vl2,
        requires=['transformers<4.42'],
        architectures=['DeepseekV2ForCausalLM'],
        tags=['vision'],
    ))

register_model(
    ModelMeta(
        LLMModelType.deepseek_r1,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-R1', 'deepseek-ai/DeepSeek-R1'),
                Model('deepseek-ai/DeepSeek-R1-Zero', 'deepseek-ai/DeepSeek-R1-Zero'),
                Model('deepseek-ai/DeepSeek-R1-0528', 'deepseek-ai/DeepSeek-R1-0528'),
            ]),
            ModelGroup([
                Model('cognitivecomputations/DeepSeek-R1-awq', 'cognitivecomputations/DeepSeek-R1-AWQ'),
                Model('cognitivecomputations/DeepSeek-R1-0528-AWQ', 'cognitivecomputations/DeepSeek-R1-0528-AWQ'),
            ]),
            ModelGroup([
                Model('unsloth/DeepSeek-R1-BF16', 'unsloth/DeepSeek-R1-BF16'),
                Model('unsloth/DeepSeek-R1-Zero-BF16', 'unsloth/DeepSeek-R1-Zero-BF16'),
                Model('unsloth/DeepSeek-R1-0528-BF16', 'unsloth/DeepSeek-R1-0528-BF16'),
            ]),
        ],
        TemplateType.deepseek_r1,
        get_model_tokenizer_deepseek_moe,
        architectures=['DeepseekV3ForCausalLM'],
        model_arch=ModelArch.deepseek_v2,
        requires=['transformers>=4.39.3'],
    ))

register_model(
    ModelMeta(
        LLMModelType.deepseek_r1_distill,
        [
            ModelGroup([
                Model('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'),
                Model('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'),
                Model('deepseek-ai/DeepSeek-R1-Distill-Qwen-14B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'),
                Model('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'),
                Model('iic/QwenLong-L1-32B', 'Tongyi-Zhiwen/QwenLong-L1-32B'),
            ],
                       requires=['transformers>=4.37']),
            ModelGroup([
                Model('deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'),
                Model('deepseek-ai/DeepSeek-R1-Distill-Llama-70B', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'),
            ]),
            ModelGroup([
                Model('deepseek-ai/DeepSeek-R1-0528-Qwen3-8B', 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'),
            ]),
        ],
        TemplateType.deepseek_r1,
        get_model_tokenizer_with_flash_attn,
        architectures=['Qwen2ForCausalLM', 'LlamaForCausalLM', 'Qwen3ForCausalLM'],
        model_arch=ModelArch.llama,
    ))
