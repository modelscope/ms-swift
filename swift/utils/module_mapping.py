import dataclasses
from collections import OrderedDict
from typing import Optional, Union


@dataclasses.dataclass
class MultiModelKeys:

    language_model: str = None

    projector: Optional[str] = None

    vision_tower: str = None

    vision_resampler: str = None


@dataclasses.dataclass
class ModelKeys:

    model_type: str = None

    module_list: str = None

    embedding: str = None

    mlp: str = None

    down_proj: str = None

    attention: str = None

    o_proj: str = None

    q_proj: str = None

    k_proj: str = None

    v_proj: str = None

    qkv_proj: str = None

    qk_proj: str = None

    qa_proj: str = None

    qb_proj: str = None

    kva_proj: str = None

    kvb_proj: str = None

    output: str = None


LLAMA_KEYS = ModelKeys(
    module_list='model.layers',
    mlp='model.layers.{}.mlp',
    down_proj='model.layers.{}.mlp.down_proj',
    attention='model.layers.{}.self_attn',
    o_proj='model.layers.{}.self_attn.o_proj',
    q_proj='model.layers.{}.self_attn.q_proj',
    k_proj='model.layers.{}.self_attn.k_proj',
    v_proj='model.layers.{}.self_attn.v_proj',
    embedding='model.embed_tokens',
    output='lm_head',
)

INTERNLM2_KEYS = ModelKeys(
    module_list='model.layers',
    mlp='model.layers.{}.feed_forward',
    down_proj='model.layers.{}.feed_forward.w2',
    attention='model.layers.{}.attention',
    o_proj='model.layers.{}.attention.wo',
    qkv_proj='model.layers.{}.attention.wqkv',
    embedding='model.tok_embeddings',
    output='output',
)

CHATGLM_KEYS = ModelKeys(
    module_list='transformer.encoder.layers',
    mlp='transformer.encoder.layers.{}.mlp',
    down_proj='transformer.encoder.layers.{}.mlp.dense_4h_to_h',
    attention='transformer.encoder.layers.{}.self_attention',
    o_proj='transformer.encoder.layers.{}.self_attention.dense',
    qkv_proj='transformer.encoder.layers.{}.self_attention.query_key_value',
    embedding='transformer.embedding',
    output='transformer.output_layer',
)

BAICHUAN_KEYS = ModelKeys(
    module_list='model.layers',
    mlp='model.layers.{}.mlp',
    down_proj='model.layers.{}.mlp.down_proj',
    attention='model.layers.{}.self_attn',
    qkv_proj='model.layers.{}.self_attn.W_pack',
    embedding='model.embed_tokens',
    output='lm_head',
)

YUAN_KEYS = ModelKeys(
    module_list='model.layers',
    mlp='model.layers.{}.mlp',
    down_proj='model.layers.{}.mlp.down_proj',
    attention='model.layers.{}.self_attn',
    qk_proj='model.layers.{}.self_attn.qk_proj',
    o_proj='model.layers.{}.self_attn.o_proj',
    q_proj='model.layers.{}.self_attn.q_proj',
    k_proj='model.layers.{}.self_attn.k_proj',
    v_proj='model.layers.{}.self_attn.v_proj',
    embedding='model.embed_tokens',
    output='lm_head',
)

CODEFUSE_KEYS = ModelKeys(
    module_list='gpt_neox.layers',
    mlp='gpt_neox.layers.{}.mlp',
    down_proj='gpt_neox.layers.{}.mlp.dense_4h_to_h',
    attention='gpt_neox.layers.{}.attention',
    o_proj='gpt_neox.layers.{}.attention.dense',
    qkv_proj='gpt_neox.layers.{}.attention.query_key_value',
    embedding='gpt_neox.embed_in',
    output='gpt_neox.embed_out',
)

PHI2_KEYS = ModelKeys(
    module_list='transformer.h',
    mlp='transformer.h.{}.mlp',
    down_proj='transformer.h.{}.mlp.c_proj',
    attention='transformer.h.{}.mixer',
    o_proj='transformer.h.{}.mixer.out_proj',
    qkv_proj='transformer.h.{}.mixer.Wqkv',
    embedding='transformer.embd',
    output='lm_head',
)

QWEN_KEYS = ModelKeys(
    module_list='transformer.h',
    mlp='transformer.h.{}.mlp',
    down_proj='transformer.h.{}.mlp.c_proj',
    attention='transformer.h.{}.attn',
    o_proj='transformer.h.{}.attn.c_proj',
    qkv_proj='transformer.h.{}.attn.c_attn',
    embedding='transformer.wte',
    output='lm_head',
)

PHI3_KEYS = ModelKeys(
    module_list='model.layers',
    mlp='model.layers.{}.mlp',
    down_proj='model.layers.{}.mlp.down_proj',
    attention='model.layers.{}.self_attn',
    o_proj='model.layers.{}.self_attn.o_proj',
    qkv_proj='model.layers.{}.self_attn.qkv_proj',
    embedding='model.embed_tokens',
    output='lm_head',
)

PHI3_SMALL_KEYS = ModelKeys(
    module_list='model.layers',
    mlp='model.layers.{}.mlp',
    down_proj='model.layers.{}.mlp.down_proj',
    attention='model.layers.{}.self_attn',
    o_proj='model.layers.{}.self_attn.dense',
    qkv_proj='model.layers.{}.self_attn.query_key_value',
    embedding='model.embed_tokens',
    output='lm_head',
)

DEEPSEEK_V2_KEYS = ModelKeys(
    module_list='model.layers',
    mlp='model.layers.{}.mlp',
    down_proj='model.layers.{}.mlp.down_proj',
    attention='model.layers.{}.self_attn',
    o_proj='model.layers.{}.self_attn.o_proj',
    qa_proj='model.layers.{}.self_attn.q_a_proj',
    qb_proj='model.layers.{}.self_attn.q_b_proj',
    kva_proj='model.layers.{}.self_attn.kv_a_proj_with_mqa',
    kvb_proj='model.layers.{}.self_attn.kv_b_proj',
    embedding='model.embed_tokens',
    output='lm_head',
)

LLAVA_KEYS = MultiModelKeys(
    language_model='language_model',
    projector='multi_modal_projector',
    vision_tower='vision_tower',
)

LLAVA_NEXT_VIDEO_KEYS = MultiModelKeys(
    language_model='language_model',
    projector='multi_modal_projector',
    vision_tower='vision_tower',
    vision_resampler='vision_resampler',
)

LLAVA_LLAMA_KEYS = MultiModelKeys(
    language_model='model.layers',
    projector='model.mm_projector',
    vision_tower='model.vision_tower',
)

INTERNLM_XCOMPOSER_KEYS = MultiModelKeys(
    language_model='model',
    projector='vision_proj',
    vision_tower='vit',
)

INTERNVL_KEYS = MultiModelKeys(
    language_model='language_model',
    projector='mlp1',
    vision_tower='vision_model',
)

DEEPSEEK_VL_KEYS = MultiModelKeys(
    language_model='language_model',
    projector='aligner',
    vision_tower='vision_model',
)

MINICPM_V_KEYS = MultiModelKeys(
    language_model='llm',
    projector='resampler',
    vision_tower='vpm',
)

PHI3V_KEYS = MultiModelKeys(
    language_model='model.layers',
    projector='model.vision_embed_tokens.img_projection',
    vision_tower='model.vision_embed_tokens.img_processor',
)

COGVLM_KEYS = MultiModelKeys(
    language_model='model.layers',
    projector=None,
    vision_tower='model.vision',
)

FLORENCE_KEYS = MultiModelKeys(
    language_model='language_model',
    projector='image_projection',
    vision_tower='vision_tower',
)

QWEN_VL_KEYS = MultiModelKeys(
    language_model='transformer.h',
    projector=None,
    vision_tower='transformer.visual',
)

QWEN_AUDIO_KEYS = MultiModelKeys(
    language_model='transformer.h',
    projector=None,
    vision_tower='transformer.audio',
)

QWEN2_AUDIO_KEYS = MultiModelKeys(
    language_model='language_model',
    projector='multi_modal_projector',
    vision_tower='audio_tower',
)

GLM4V_KEYS = MultiModelKeys(
    language_model='transformer.encoder',
    projector=None,
    vision_tower='transformer.vision',
)

MODEL_KEYS_MAPPING = OrderedDict([
    # MLLM here
    ('qwen_audio', QWEN_AUDIO_KEYS),
    ('qwen_vl', QWEN_VL_KEYS),
    ('qwen2_audio', QWEN2_AUDIO_KEYS),
    ('glm4v', GLM4V_KEYS),
    ('llava_next_video', LLAVA_NEXT_VIDEO_KEYS),
    ('llava_next', LLAVA_KEYS),
    ('llava_llama', LLAVA_LLAMA_KEYS),
    ('llava', LLAVA_KEYS),
    ('yi_vl', LLAVA_LLAMA_KEYS),
    ('internlm_xcomposer', INTERNLM_XCOMPOSER_KEYS),
    ('internvl', INTERNVL_KEYS),
    ('deepseek_vl', DEEPSEEK_VL_KEYS),
    ('paligemma', LLAVA_KEYS),
    ('minicpm_v', MINICPM_V_KEYS),
    ('phi3v', PHI3V_KEYS),
    ('cogvlm2', COGVLM_KEYS),
    ('cogvlm', COGVLM_KEYS),
    ('cogagent', COGVLM_KEYS),
    ('florence', FLORENCE_KEYS),
    # LLM begins here
    ('llama', LLAMA_KEYS),
    ('mistral', LLAMA_KEYS),
    ('qwen1half', LLAMA_KEYS),
    ('qwen2', LLAMA_KEYS),
    ('yi', LLAMA_KEYS),
    ('gemma', LLAMA_KEYS),
    ('internlm2', LLAMA_KEYS),
    ('internlm', LLAMA_KEYS),
    ('deepseek-v2', LLAMA_KEYS),
    ('deepseek', LLAMA_KEYS),
    ('openbuddy', LLAMA_KEYS),
    ('xverse', LLAMA_KEYS),
    ('orion', LLAMA_KEYS),
    ('bluelm', LLAMA_KEYS),
    ('ziya', LLAMA_KEYS),
    ('skywork', LLAMA_KEYS),
    ('chatglm', LLAMA_KEYS),
    ('glm4', LLAMA_KEYS),
    ('baichuan', LLAMA_KEYS),
    ('yuan', LLAMA_KEYS),
    ('codefuse', LLAMA_KEYS),
    ('phi2', LLAMA_KEYS),
    ('qwen', LLAMA_KEYS),
    ('phi3-small', LLAMA_KEYS),
    ('phi3', LLAMA_KEYS),
    ('minicpm', LLAMA_KEYS),
])


def get_regex_for_mm_default_lora(model_type: str):
    if not model_type:
        return None
    if model_type not in MODEL_KEYS_MAPPING:
        return None

    mapping: Union[MultiModelKeys, ModelKeys] = MODEL_KEYS_MAPPING[model_type]
    if not isinstance(mapping, MultiModelKeys):
        return None
    llm = mapping.language_model
    projector = mapping.projector
    _regex = f'^({llm}'
    if projector:
        _regex += f'|{projector}'
    _regex += ')(?!.*(lm_head|output|emb|wte|shared)).*'
    return _regex
