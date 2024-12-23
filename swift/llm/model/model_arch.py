# Copyright (c) Alibaba, Inc. and its affiliates.
from dataclasses import dataclass, field
from typing import List, Optional, Union


class LLMModelArch:
    qwen = 'qwen'
    llama = 'llama'
    internlm2 = 'internlm2'
    chatglm = 'chatglm'
    deepseek_v2 = 'deepseek_v2'
    baichuan = 'baichuan'

    yuan = 'yuan'
    codefuse = 'codefuse'
    phi2 = 'phi2'
    phi3 = 'phi3'
    phi3_small = 'phi3_small'
    telechat = 'telechat'
    dbrx = 'dbrx'


class MLLMModelArch:
    qwen_vl = 'qwen_vl'
    qwen_audio = 'qwen_audio'
    qwen2_vl = 'qwen2_vl'
    qwen2_audio = 'qwen2_audio'

    cogvlm = 'cogvlm'
    glm4v = 'glm4v'
    glm_edge_v = 'glm_edge_v'

    llama3_1_omni = 'llama3_1_omni'
    llama3_2_vision = 'llama3_2_vision'

    llava_hf = 'llava_hf'
    llava_next_video_hf = 'llava_next_video_hf'

    llava_llama = 'llava_llama'
    llava_mistral = 'llava_mistral'

    xcomposer = 'xcomposer'
    internvl = 'internvl'
    minicpmv = 'minicpmv'
    deepseek_vl = 'deepseek_vl'
    deepseek_vl2 = 'deepseek_vl2'
    deepseek_janus = 'deepseek_janus'

    mplug_owl2 = 'mplug_owl2'
    mplug_owl2_1 = 'mplug_owl2_1'
    mplug_owl3 = 'mplug_owl3'
    doc_owl2 = 'doc_owl2'

    phi3v = 'phi3v'
    florence = 'florence'
    idefics3 = 'idefics3'

    got_ocr2 = 'got_ocr2'
    ovis1_6 = 'ovis1_6'
    molmo = 'molmo'
    emu3_chat = 'emu3_chat'
    megrez_omni = 'megrez_omni'


class ModelArch(LLMModelArch, MLLMModelArch):
    pass


@dataclass
class ModelKeys:

    arch_name: str = None

    embedding: str = None
    module_list: str = None
    lm_head: str = None

    q_proj: str = None
    k_proj: str = None
    v_proj: str = None
    o_proj: str = None
    attention: str = None

    mlp: str = None
    down_proj: str = None

    qkv_proj: str = None
    qk_proj: str = None
    qa_proj: str = None
    qb_proj: str = None
    kv_proj: str = None
    kva_proj: str = None
    kvb_proj: str = None


@dataclass
class MultiModelKeys(ModelKeys):
    language_model: Union[str, List[str]] = field(default_factory=list)
    aligner: Union[str, List[str]] = field(default_factory=list)
    vision_tower: Union[str, List[str]] = field(default_factory=list)
    generator: Union[str, List[str]] = field(default_factory=list)

    def __post_init__(self):
        for key in ['language_model', 'aligner', 'vision_tower', 'generator']:
            v = getattr(self, key)
            if isinstance(v, str):
                setattr(self, key, [v])
            if v is None:
                setattr(self, key, [])


MODEL_ARCH_MAPPING = {}


def register_model_arch(model_arch: ModelKeys, *, exist_ok: bool = False) -> None:
    """
    model_type: The unique ID for the model type. Models with the same model_type share
        the same architectures, template, get_function, etc.
    """
    arch_name = model_arch.arch_name
    if not exist_ok and arch_name in MODEL_ARCH_MAPPING:
        raise ValueError(f'The `{arch_name}` has already been registered in the MODEL_ARCH_MAPPING.')

    MODEL_ARCH_MAPPING[arch_name] = model_arch


register_model_arch(
    ModelKeys(
        LLMModelArch.llama,
        module_list='model.layers',
        mlp='model.layers.{}.mlp',
        down_proj='model.layers.{}.mlp.down_proj',
        attention='model.layers.{}.self_attn',
        o_proj='model.layers.{}.self_attn.o_proj',
        q_proj='model.layers.{}.self_attn.q_proj',
        k_proj='model.layers.{}.self_attn.k_proj',
        v_proj='model.layers.{}.self_attn.v_proj',
        embedding='model.embed_tokens',
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.internlm2,
        module_list='model.layers',
        mlp='model.layers.{}.feed_forward',
        down_proj='model.layers.{}.feed_forward.w2',
        attention='model.layers.{}.attention',
        o_proj='model.layers.{}.attention.wo',
        qkv_proj='model.layers.{}.attention.wqkv',
        embedding='model.tok_embeddings',
        lm_head='output',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.chatglm,
        module_list='transformer.encoder.layers',
        mlp='transformer.encoder.layers.{}.mlp',
        down_proj='transformer.encoder.layers.{}.mlp.dense_4h_to_h',
        attention='transformer.encoder.layers.{}.self_attention',
        o_proj='transformer.encoder.layers.{}.self_attention.dense',
        qkv_proj='transformer.encoder.layers.{}.self_attention.query_key_value',
        embedding='transformer.embedding',
        lm_head='transformer.output_layer'))

register_model_arch(
    ModelKeys(
        LLMModelArch.telechat,
        module_list='transformer.h',
        mlp='transformer.h.{}.mlp',
        down_proj='transformer.h.{}.mlp.down_proj',
        attention='transformer.h.{}.self_attention',
        o_proj='transformer.h.{}.self_attention.dense',
        q_proj='transformer.h.{}.self_attention.query',
        kv_proj='transformer.h.{}.self_attention.key_value',
        embedding='transformer.word_embeddings',
        lm_head='lm_head'))

register_model_arch(
    ModelKeys(
        LLMModelArch.baichuan,
        module_list='model.layers',
        mlp='model.layers.{}.mlp',
        down_proj='model.layers.{}.mlp.down_proj',
        attention='model.layers.{}.self_attn',
        qkv_proj='model.layers.{}.self_attn.W_pack',
        embedding='model.embed_tokens',
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.yuan,
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
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.codefuse,
        module_list='gpt_neox.layers',
        mlp='gpt_neox.layers.{}.mlp',
        down_proj='gpt_neox.layers.{}.mlp.dense_4h_to_h',
        attention='gpt_neox.layers.{}.attention',
        o_proj='gpt_neox.layers.{}.attention.dense',
        qkv_proj='gpt_neox.layers.{}.attention.query_key_value',
        embedding='gpt_neox.embed_in',
        lm_head='gpt_neox.embed_out',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.phi2,
        module_list='model.layers',
        mlp='model.layers.{}.mlp',
        down_proj='model.layers.{}.mlp.fc2',
        attention='model.layers.{}.self_attn',
        o_proj='model.layers.{}.self_attn.dense',
        q_proj='model.layers.{}.self_attn.q_proj',
        k_proj='model.layers.{}.self_attn.k_proj',
        v_proj='model.layers.{}.self_attn.v_proj',
        embedding='model.embed_tokens',
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.qwen,
        module_list='transformer.h',
        mlp='transformer.h.{}.mlp',
        down_proj='transformer.h.{}.mlp.c_proj',
        attention='transformer.h.{}.attn',
        o_proj='transformer.h.{}.attn.c_proj',
        qkv_proj='transformer.h.{}.attn.c_attn',
        embedding='transformer.wte',
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.dbrx,
        module_list='transformer.blocks',
        mlp='transformer.blocks.{}.ffn',
        attention='transformer.blocks.{}.norm_attn_norm.attn',
        o_proj='transformer.blocks.{}.norm_attn_norm.attn.out_proj',
        qkv_proj='transformer.blocks.{}.norm_attn_norm.attn.Wqkv',
        embedding='transformer.wte',
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.phi3,
        module_list='model.layers',
        mlp='model.layers.{}.mlp',
        down_proj='model.layers.{}.mlp.down_proj',
        attention='model.layers.{}.self_attn',
        o_proj='model.layers.{}.self_attn.o_proj',
        qkv_proj='model.layers.{}.self_attn.qkv_proj',
        embedding='model.embed_tokens',
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.phi3_small,
        module_list='model.layers',
        mlp='model.layers.{}.mlp',
        down_proj='model.layers.{}.mlp.down_proj',
        attention='model.layers.{}.self_attn',
        o_proj='model.layers.{}.self_attn.dense',
        qkv_proj='model.layers.{}.self_attn.query_key_value',
        embedding='model.embed_tokens',
        lm_head='lm_head',
    ))

register_model_arch(
    ModelKeys(
        LLMModelArch.deepseek_v2,
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
        lm_head='lm_head',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.llava_hf,
        language_model='language_model',
        aligner='multi_modal_projector',
        vision_tower='vision_tower',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.llava_mistral,
        language_model='model.layers',
        aligner='model.mm_projector',
        vision_tower='model.vision_tower',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.llava_next_video_hf,
        language_model='language_model',
        aligner=['multi_modal_projector'],
        vision_tower='vision_tower'))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.llava_llama,
        language_model='model.layers',
        aligner='model.mm_projector',
        vision_tower='model.vision_tower',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.xcomposer,
        language_model='model',
        aligner='vision_proj',
        vision_tower='vit',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.internvl,
        language_model='language_model',
        aligner='mlp1',
        vision_tower='vision_model',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.mplug_owl3,
        language_model='language_model',
        aligner='vision2text_model',
        vision_tower='vision_model',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.doc_owl2,
        language_model='model.layers',
        aligner=['model.vision2text', 'model.hr_compressor'],
        vision_tower='model.vision_model',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.deepseek_vl,
        language_model='language_model',
        aligner='aligner',
        vision_tower='vision_model',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.deepseek_janus,
        language_model='language_model',
        vision_tower='vision_model',
        aligner='aligner',
        generator=['gen_vision_model', 'gen_aligner', 'gen_head', 'gen_embed']))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.deepseek_vl2,
        language_model='language',
        vision_tower='vision',
        aligner='projector',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.minicpmv,
        language_model='llm',
        aligner='resampler',
        vision_tower='vpm',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.phi3v,
        language_model='model.layers',
        aligner='model.vision_embed_tokens.img_projection',
        vision_tower='model.vision_embed_tokens.img_processor',
    ))

register_model_arch(MultiModelKeys(
    MLLMModelArch.cogvlm,
    language_model='model.layers',
    vision_tower='model.vision',
))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.florence,
        language_model='language_model',
        vision_tower='vision_tower',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.qwen_vl,
        language_model='transformer.h',
        vision_tower='transformer.visual',
    ))
# TODO: check lm_head, ALL
register_model_arch(
    MultiModelKeys(
        MLLMModelArch.qwen_audio,
        language_model='transformer.h',
        vision_tower='transformer.audio',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.qwen2_audio,
        language_model='language_model',
        aligner='multi_modal_projector',
        vision_tower='audio_tower',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.qwen2_vl,
        language_model='model',
        aligner='visual.merger',
        vision_tower='visual',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.glm4v,
        language_model='transformer.encoder',
        vision_tower='transformer.vision',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.idefics3,
        language_model='model.text_model',
        aligner='model.connector',
        vision_tower='model.vision_model',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.llama3_1_omni,
        language_model='model.layers',
        aligner='model.speech_projector',
        vision_tower='model.speech_encoder',
        generator='speech_generator',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.got_ocr2,
        language_model='model.layers',
        aligner='model.mm_projector_vary',
        vision_tower='model.vision_tower_high',
    ))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.llama3_2_vision,
        language_model='language_model',
        aligner='multi_modal_projector',
        vision_tower='vision_model',
    ))

register_model_arch(MultiModelKeys(
    MLLMModelArch.ovis1_6,
    language_model='llm',
    vision_tower='visual_tokenizer',
))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.molmo,
        language_model='model.transformer',
        vision_tower='model.vision_backbone',
        aligner='model.vision_backbone.image_projector'))

register_model_arch(
    MultiModelKeys(
        MLLMModelArch.megrez_omni,
        language_model='llm',
        vision_tower=['vision', 'audio'],
    ))

register_model_arch(MultiModelKeys(MLLMModelArch.emu3_chat, language_model='model'))

register_model_arch(
    MultiModelKeys(MLLMModelArch.glm_edge_v, language_model='model.layers', vision_tower='model.vision'))


def get_model_arch(arch_name: Optional[str]) -> Optional[ModelKeys]:
    return MODEL_ARCH_MAPPING.get(arch_name)
