# Copyright (c) ModelScope Contributors. All rights reserved.
from itertools import chain
from typing import List


class LLMModelType:
    qwen = 'qwen'
    qwen2 = 'qwen2'
    qwen2_moe = 'qwen2_moe'
    qwen3 = 'qwen3'
    qwen3_moe = 'qwen3_moe'
    qwen3_next = 'qwen3_next'
    qwen3_emb = 'qwen3_emb'
    qwen3_reranker = 'qwen3_reranker'

    qwen2_gte = 'qwen2_gte'

    codefuse_qwen = 'codefuse_qwen'
    modelscope_agent = 'modelscope_agent'

    llama = 'llama'
    yi = 'yi'
    gpt_oss = 'gpt_oss'
    seed_oss = 'seed_oss'

    codefuse_codellama = 'codefuse_codellama'

    chatglm2 = 'chatglm2'
    chatglm3 = 'chatglm3'
    chatglm4 = 'chatglm4'
    glm4 = 'glm4'
    glm4_moe = 'glm4_moe'
    glm4_moe_lite = 'glm4_moe_lite'

    glm_edge = 'glm_edge'
    codefuse_codegeex2 = 'codefuse_codegeex2'
    codegeex4 = 'codegeex4'

    internlm = 'internlm'
    internlm2 = 'internlm2'
    internlm3 = 'internlm3'

    deepseek = 'deepseek'
    deepseek_v2 = 'deepseek_v2'
    deepseek_v3 = 'deepseek_v3'
    deepseek_v3_2 = 'deepseek_v3_2'

    openbuddy_llama = 'openbuddy_llama'
    openbuddy_mistral = 'openbuddy_mistral'
    openbuddy_mixtral = 'openbuddy_mixtral'

    baichuan = 'baichuan'
    baichuan2 = 'baichuan2'
    baichuan_m1 = 'baichuan_m1'

    minicpm = 'minicpm'
    minicpm_chatml = 'minicpm_chatml'
    minicpm3 = 'minicpm3'
    minicpm_moe = 'minicpm_moe'

    telechat = 'telechat'
    telechat2 = 'telechat2'

    mistral = 'mistral'
    devstral = 'devstral'
    zephyr = 'zephyr'
    mixtral = 'mixtral'
    mistral_nemo = 'mistral_nemo'
    mistral_2501 = 'mistral_2501'
    wizardlm2 = 'wizardlm2'
    wizardlm2_moe = 'wizardlm2_moe'

    phi2 = 'phi2'
    phi3_small = 'phi3_small'
    phi3 = 'phi3'
    phi3_moe = 'phi3_moe'
    phi4 = 'phi4'

    minimax = 'minimax'
    minimax_m1 = 'minimax_m1'
    minimax_m2 = 'minimax_m2'

    gemma = 'gemma'
    gemma2 = 'gemma2'
    gemma3_text = 'gemma3_text'

    skywork = 'skywork'

    ling = 'ling'
    ling2 = 'ling2'
    ring2 = 'ring2'
    yuan2 = 'yuan2'
    orion = 'orion'
    xverse = 'xverse'
    xverse_moe = 'xverse_moe'
    seggpt = 'seggpt'
    bluelm = 'bluelm'
    c4ai = 'c4ai'
    dbrx = 'dbrx'
    grok = 'grok'
    mamba = 'mamba'
    polylm = 'polylm'
    aya = 'aya'
    mimo = 'mimo'
    dots1 = 'dots1'
    hunyuan = 'hunyuan'
    hunyuan_v1_dense = 'hunyuan_v1_dense'
    ernie4_5 = 'ernie4_5'
    ernie4_5_moe = 'ernie4_5_moe'
    gemma_emb = 'gemma_emb'
    longchat = 'longchat'
    iquestcoder = 'iquestcoder'
    youtu_llm = 'youtu_llm'

    modern_bert_gte_reranker = 'modern_bert_gte_reranker'
    bge_reranker = 'bge_reranker'

    olmoe = 'olmoe'


class BertModelType:
    modern_bert = 'modern_bert'
    modern_bert_gte = 'modern_bert_gte'
    bert = 'bert'


class RMModelType:
    internlm2_reward = 'internlm2_reward'
    qwen2_reward = 'qwen2_reward'
    qwen2_5_prm = 'qwen2_5_prm'
    llama3_2_reward = 'llama3_2_reward'
    gemma_reward = 'gemma_reward'


class MLLMModelType:
    qwen_vl = 'qwen_vl'
    qwen_audio = 'qwen_audio'
    qwen2_vl = 'qwen2_vl'
    qwen2_5_vl = 'qwen2_5_vl'
    qwen2_5_omni = 'qwen2_5_omni'
    qwen3_omni_moe = 'qwen3_omni_moe'
    qwen2_audio = 'qwen2_audio'
    qwen3_vl = 'qwen3_vl'
    qwen3_vl_moe = 'qwen3_vl_moe'
    qwen3_vl_emb = 'qwen3_vl_emb'
    qwen3_vl_reranker = 'qwen3_vl_reranker'
    qwen2_gme = 'qwen2_gme'
    ovis1_6 = 'ovis1_6'
    ovis2 = 'ovis2'
    ovis2_5 = 'ovis2_5'
    midashenglm = 'midashenglm'

    chatglm4v = 'chatglm4v'
    glm4v = 'glm4v'
    glm4v_moe = 'glm4v_moe'
    glm_edge_v = 'glm_edge_v'
    cogvlm = 'cogvlm'
    cogagent_vqa = 'cogagent_vqa'
    cogagent_chat = 'cogagent_chat'
    cogvlm2 = 'cogvlm2'
    cogvlm2_video = 'cogvlm2_video'

    internvl = 'internvl'
    internvl_phi3 = 'internvl_phi3'
    internvl2 = 'internvl2'
    internvl2_phi3 = 'internvl2_phi3'
    internvl2_5 = 'internvl2_5'
    internvl3 = 'internvl3'
    internvl_hf = 'internvl_hf'
    internvl3_5 = 'internvl3_5'
    internvl3_5_gpt = 'internvl3_5_gpt'
    internvl_gpt_hf = 'internvl_gpt_hf'
    interns1 = 'interns1'
    xcomposer2 = 'xcomposer2'
    xcomposer2_4khd = 'xcomposer2_4khd'
    xcomposer2_5 = 'xcomposer2_5'
    xcomposer2_5_ol_audio = 'xcomposer2_5_ol_audio'

    llama3_2_vision = 'llama3_2_vision'
    llama4 = 'llama4'
    llama3_1_omni = 'llama3_1_omni'

    llava1_5_hf = 'llava1_5_hf'
    llava1_6_mistral_hf = 'llava1_6_mistral_hf'
    llava1_6_vicuna_hf = 'llava1_6_vicuna_hf'
    llava1_6_yi_hf = 'llava1_6_yi_hf'
    llama3_llava_next_hf = 'llama3_llava_next_hf'
    llava_next_qwen_hf = 'llava_next_qwen_hf'
    llava_next_video_hf = 'llava_next_video_hf'
    llava_next_video_yi_hf = 'llava_next_video_yi_hf'
    llava_onevision_hf = 'llava_onevision_hf'
    yi_vl = 'yi_vl'
    ernie_vl = 'ernie_vl'

    llava_llama3_1_hf = 'llava_llama3_1_hf'  # DaozeZhang
    llava_llama3_hf = 'llava_llama3_hf'  # xtuner

    llava1_6_mistral = 'llava1_6_mistral'
    llava1_6_yi = 'llava1_6_yi'
    llava_next_qwen = 'llava_next_qwen'
    llama3_llava_next = 'llama3_llava_next'
    llava_onevision1_5 = 'llava_onevision1_5'

    deepseek_vl = 'deepseek_vl'
    deepseek_vl2 = 'deepseek_vl2'
    deepseek_janus = 'deepseek_janus'
    deepseek_janus_pro = 'deepseek_janus_pro'
    deepseek_ocr = 'deepseek_ocr'
    deepseek_ocr2 = 'deepseek_ocr2'

    minicpmv = 'minicpmv'
    minicpmv2_5 = 'minicpmv2_5'
    minicpmv2_6 = 'minicpmv2_6'
    minicpmo2_6 = 'minicpmo2_6'
    minicpmv4 = 'minicpmv4'
    minicpmv4_5 = 'minicpmv4_5'

    minimax_vl = 'minimax_vl'

    mplug_owl2 = 'mplug_owl2'
    mplug_owl2_1 = 'mplug_owl2_1'
    mplug_owl3 = 'mplug_owl3'
    mplug_owl3_241101 = 'mplug_owl3_241101'
    doc_owl2 = 'doc_owl2'

    emu3_gen = 'emu3_gen'
    emu3_chat = 'emu3_chat'
    got_ocr2 = 'got_ocr2'
    got_ocr2_hf = 'got_ocr2_hf'
    step_audio = 'step_audio'
    step_audio2_mini = 'step_audio2_mini'
    kimi_vl = 'kimi_vl'
    keye_vl = 'keye_vl'
    keye_vl_1_5 = 'keye_vl_1_5'
    dots_ocr = 'dots_ocr'
    sail_vl2 = 'sail_vl2'

    phi3_vision = 'phi3_vision'
    phi4_multimodal = 'phi4_multimodal'
    florence = 'florence'
    idefics3 = 'idefics3'
    paligemma = 'paligemma'
    molmo = 'molmo'
    molmoe = 'molmoe'
    pixtral = 'pixtral'
    megrez_omni = 'megrez_omni'
    valley = 'valley'
    gemma3_vision = 'gemma3_vision'
    gemma3n = 'gemma3n'
    mistral3 = 'mistral3'
    mistral3_2506 = 'mistral3_2506'
    paddle_ocr = 'paddle_ocr'
    hunyuan_ocr = 'hunyuan_ocr'
    step3_vl = 'step3_vl'

    jina_reranker_m0 = 'jina_reranker_m0'


class ModelType(LLMModelType, MLLMModelType, BertModelType, RMModelType):

    @classmethod
    def get_model_name_list(cls) -> List[str]:

        def _get_model_name_list(cls):
            res = []
            for k in cls.__dict__:
                if k.startswith('__'):
                    continue
                value = getattr(cls, k)
                if isinstance(value, str):
                    res.append(value)
            return res

        return list(
            chain.from_iterable(
                _get_model_name_list(model_type_cls)
                for model_type_cls in [LLMModelType, MLLMModelType, BertModelType, RMModelType]))
