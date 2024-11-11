from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Union


@dataclass
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


@dataclass
class MultiModelKeys(ModelKeys):
    language_model: Union[List[str], str] = field(default_factory=list)
    connector: Union[List[str], str] = field(default_factory=list)
    vision_tower: Union[List[str], str] = field(default_factory=list)
    generator: Union[List[str], str] = field(default_factory=list)

    def __post_init__(self):
        # compat
        for key in ['language_model', 'connector', 'vision_tower', 'generator']:
            v = getattr(self, key)
            if isinstance(v, str):
                setattr(self, key, [v])
            if v is None:
                setattr(self, key, [])


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
    connector='multi_modal_projector',
    vision_tower='vision_tower',
)

LLAVA_NEXT_VIDEO_KEYS = MultiModelKeys(
    language_model='language_model',
    connector=['multi_modal_projector', 'vision_resampler'],
    vision_tower='vision_tower',
)

LLAVA_LLAMA_KEYS = MultiModelKeys(
    language_model='model.layers',
    connector='model.mm_projector',
    vision_tower='model.vision_tower',
)

INTERNLM_XCOMPOSER_KEYS = MultiModelKeys(
    language_model='model',
    connector='vision_proj',
    vision_tower='vit',
)

INTERNVL_KEYS = MultiModelKeys(
    language_model='language_model',
    connector='mlp1',
    vision_tower='vision_model',
)

MPLUG_OWL3_KEYS = MultiModelKeys(
    language_model='language_model',
    connector='vision2text_model',
    vision_tower='vision_model',
)

DEEPSEEK_VL_KEYS = MultiModelKeys(
    language_model='language_model',
    connector='aligner',
    vision_tower='vision_model',
)

MINICPM_V_KEYS = MultiModelKeys(
    language_model='llm',
    connector='resampler',
    vision_tower='vpm',
)

PHI3V_KEYS = MultiModelKeys(
    language_model='model.layers',
    connector='model.vision_embed_tokens.img_projection',
    vision_tower='model.vision_embed_tokens.img_processor',
)

COGVLM_KEYS = MultiModelKeys(
    language_model='model.layers',
    vision_tower='model.vision',
)

FLORENCE_KEYS = MultiModelKeys(
    language_model='language_model',
    connector='image_projection',
    vision_tower='vision_tower',
)

QWEN_VL_KEYS = MultiModelKeys(
    language_model='transformer.h',
    vision_tower='transformer.visual',
)

QWEN_AUDIO_KEYS = MultiModelKeys(
    language_model='transformer.h',
    vision_tower='transformer.audio',
)

QWEN2_AUDIO_KEYS = MultiModelKeys(
    language_model='language_model',
    connector='multi_modal_projector',
    vision_tower='audio_tower',
)

QWEN2_VL_KEYS = MultiModelKeys(
    language_model='model',
    vision_tower='visual',
)

GLM4V_KEYS = MultiModelKeys(
    language_model='transformer.encoder',
    vision_tower='transformer.vision',
)

IDEFICS3_KEYS = MultiModelKeys(
    language_model='model.text_model',
    connector='model.connector',
    vision_tower='model.vision_model',
)

LLAMA3_1_OMNI = MultiModelKeys(
    language_model='model.layers',
    connector='model.speech_projector',
    vision_tower='model.speech_encoder',
    generator='speech_generator',
)

GOT_OCR2 = MultiModelKeys(
    language_model='model.layers',
    connector='model.mm_projector_vary',
    vision_tower='model.vision_tower_high',
)

LLAMA3_2_VISION = MultiModelKeys(
    language_model='language_model',
    connector='multi_modal_projector',
    vision_tower='vision_model',
)

OVIS1_6 = MultiModelKeys(
    language_model='llm',
    vision_tower='visual_tokenizer',
)

MOLMO_KEYS = MultiModelKeys(
    language_model='model.transformer',
    vision_tower='model.vision_backbone',
)
DEEPSPEED_JANUS = MultiModelKeys(
    language_model='language_model',
    vision_tower='vision_model',
    connector='aligner',
    generator=['gen_vision_model', 'gen_aligner', 'gen_head', 'gen_embed'])

EMU3_CHAT_KEYS = MultiModelKeys(language_model='model', )

MODEL_KEYS_MAPPING = OrderedDict([
    # MLLM here
    ('qwen_audio', QWEN_AUDIO_KEYS),
    ('qwen_vl', QWEN_VL_KEYS),
    ('qwen2_audio', QWEN2_AUDIO_KEYS),
    ('qwen2_vl', QWEN2_VL_KEYS),
    ('glm4v', GLM4V_KEYS),
    ('llava_next_video', LLAVA_NEXT_VIDEO_KEYS),
    ('llava_llama', LLAVA_LLAMA_KEYS),
    ('llava', LLAVA_KEYS),
    ('internlm_xcomposer', INTERNLM_XCOMPOSER_KEYS),
    ('internvl', INTERNVL_KEYS),
    ('deepseek_vl', DEEPSEEK_VL_KEYS),
    ('minicpm_v', MINICPM_V_KEYS),
    ('phi3v', PHI3V_KEYS),
    ('cogvlm', COGVLM_KEYS),
    ('florence', FLORENCE_KEYS),
    ('idefics3', IDEFICS3_KEYS),
    ('mplug_owl3', MPLUG_OWL3_KEYS),
    ('llama3_1_omni', LLAMA3_1_OMNI),
    ('got_ocr2', GOT_OCR2),
    ('llama3_2_vision', LLAMA3_2_VISION),
    ('ovis1_6', OVIS1_6),
    ('molmo', MOLMO_KEYS),
    ('deepseek_janus', DEEPSPEED_JANUS),
    ('emu3_chat', EMU3_CHAT_KEYS),
    # LLM begins here
    ('llama', LLAMA_KEYS),
    ('mistral', LLAMA_KEYS),
    ('qwen1half', LLAMA_KEYS),
    ('qwen2', LLAMA_KEYS),
    ('yi', LLAMA_KEYS),
    ('gemma', LLAMA_KEYS),
    ('internlm2', INTERNLM2_KEYS),
    ('internlm', LLAMA_KEYS),
    ('deepseek-v2', DEEPSEEK_V2_KEYS),
    ('deepseek', LLAMA_KEYS),
    ('openbuddy', LLAMA_KEYS),
    ('xverse', LLAMA_KEYS),
    ('orion', LLAMA_KEYS),
    ('bluelm', LLAMA_KEYS),
    ('ziya', LLAMA_KEYS),
    ('skywork', LLAMA_KEYS),
    ('chatglm', CHATGLM_KEYS),
    ('glm4', CHATGLM_KEYS),
    ('baichuan', BAICHUAN_KEYS),
    ('yuan', YUAN_KEYS),
    ('codefuse', CODEFUSE_KEYS),
    ('phi2', PHI2_KEYS),
    ('qwen', QWEN_KEYS),
    ('phi3-small', PHI3_SMALL_KEYS),
    ('phi3', PHI3_KEYS),
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
    connector = mapping.connector
    assert isinstance(llm, (list, tuple)) and isinstance(connector,
                                                         (list, tuple)), f'llm: {llm}, connector: {connector}'
    _regex = []
    for module in llm + connector:
        _regex.append(f'{module}')
    regex = '|'.join(_regex)
    regex = f'^({regex})(?!.*(lm_head|output|emb|wte|shared)).*'
    return regex
