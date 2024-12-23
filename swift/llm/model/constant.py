# Copyright (c) Alibaba, Inc. and its affiliates.
# Classification criteria for model_type: same model architecture, tokenizer (get function), template.
from typing import List


class LLMModelType:
    qwen = 'qwen'
    qwen2 = 'qwen2'
    qwen2_5 = 'qwen2_5'
    qwen2_moe = 'qwen2_moe'
    qwq = 'qwq'

    codefuse_qwen = 'codefuse_qwen'
    modelscope_agent = 'modelscope_agent'
    marco_o1 = 'marco_o1'

    llama = 'llama'
    llama3 = 'llama3'
    llama3_1 = 'llama3_1'
    llama3_2 = 'llama3_2'
    reflection = 'reflection'
    megrez = 'megrez'
    yi = 'yi'
    yi_coder = 'yi_coder'
    sus = 'sus'

    codefuse_codellama = 'codefuse_codellama'
    mengzi3 = 'mengzi3'
    ziya = 'ziya'
    numina = 'numina'
    atom = 'atom'

    chatglm2 = 'chatglm2'
    chatglm3 = 'chatglm3'
    glm4 = 'glm4'
    glm_edge = 'glm_edge'
    codefuse_codegeex2 = 'codefuse_codegeex2'
    codegeex4 = 'codegeex4'
    longwriter_llama3_1 = 'longwriter_llama3_1'

    internlm = 'internlm'
    internlm2 = 'internlm2'

    deepseek = 'deepseek'
    deepseek_moe = 'deepseek_moe'
    deepseek_v2 = 'deepseek_v2'
    deepseek_v2_5 = 'deepseek_v2_5'

    openbuddy_llama = 'openbuddy_llama'
    openbuddy_llama3 = 'openbuddy_llama3'
    openbuddy_mistral = 'openbuddy_mistral'
    openbuddy_mixtral = 'openbuddy_mixtral'

    baichuan = 'baichuan'
    baichuan2 = 'baichuan2'

    minicpm = 'minicpm'
    minicpm_chatml = 'minicpm_chatml'
    minicpm3 = 'minicpm3'
    minicpm_moe = 'minicpm_moe'

    telechat = 'telechat'
    telechat2 = 'telechat2'
    telechat2_115b = 'telechat2_115b'

    mistral = 'mistral'
    zephyr = 'zephyr'
    mixtral = 'mixtral'
    mistral_nemo = 'mistral_nemo'
    wizardlm2 = 'wizardlm2'
    wizardlm2_moe = 'wizardlm2_moe'

    phi2 = 'phi2'
    phi3_small = 'phi3_small'
    phi3 = 'phi3'
    phi3_moe = 'phi3_moe'

    gemma = 'gemma'
    gemma2 = 'gemma2'

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
    skywork = 'skywork'
    aya = 'aya'


class MLLMModelType:
    qwen_vl = 'qwen_vl'
    qwen_audio = 'qwen_audio'
    qwen2_vl = 'qwen2_vl'
    qwen2_audio = 'qwen2_audio'
    qvq = 'qvq'
    ovis1_6 = 'ovis1_6'

    glm4v = 'glm4v'
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
    xcomposer2 = 'xcomposer2'
    xcomposer2_4khd = 'xcomposer2_4khd'
    xcomposer2_5 = 'xcomposer2_5'
    xcomposer2_5_ol_audio = 'xcomposer2_5_ol_audio'

    llama3_2_vision = 'llama3_2_vision'
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

    llava_llama3_1_hf = 'llava_llama3_1_hf'  # DaozeZhang
    llava_llama3_hf = 'llava_llama3_hf'  # xtuner

    llava1_6_mistral = 'llava1_6_mistral'
    llava1_6_yi = 'llava1_6_yi'
    llava_next_qwen = 'llava_next_qwen'
    llama3_llava_next = 'llama3_llava_next'

    deepseek_vl = 'deepseek_vl'
    deepseek_vl2 = 'deepseek_vl2'
    deepseek_janus = 'deepseek_janus'

    minicpmv = 'minicpmv'
    minicpmv2_6 = 'minicpmv2_6'
    minicpmv2_5 = 'minicpmv2_5'

    mplug_owl2 = 'mplug_owl2'
    mplug_owl2_1 = 'mplug_owl2_1'
    mplug_owl3 = 'mplug_owl3'
    mplug_owl3_241101 = 'mplug_owl3_241101'
    doc_owl2 = 'doc_owl2'

    emu3_gen = 'emu3_gen'
    emu3_chat = 'emu3_chat'
    got_ocr2 = 'got_ocr2'

    phi3_vision = 'phi3_vision'
    florence = 'florence'
    idefics3 = 'idefics3'
    paligemma = 'paligemma'
    molmo = 'molmo'
    molmoe = 'molmoe'
    pixtral = 'pixtral'
    megrez_omni = 'megrez_omni'


class ModelType(LLMModelType, MLLMModelType):

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

        return _get_model_name_list(LLMModelType) + _get_model_name_list(MLLMModelType)
