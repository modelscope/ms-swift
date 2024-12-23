# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import List


class LLMTemplateType:
    chatml = 'chatml'
    default = 'default'
    dummy = 'dummy'

    qwen = 'qwen'
    qwen2_5 = 'qwen2_5'
    qwq = 'qwq'
    marco_o1 = 'marco_o1'
    modelscope_agent = 'modelscope_agent'

    llama = 'llama'  # llama2
    llama3 = 'llama3'
    llama3_2 = 'llama3_2'
    reflection = 'reflection'
    megrez = 'megrez'
    yi_coder = 'yi_coder'
    sus = 'sus'

    numina = 'numina'
    ziya = 'ziya'
    atom = 'atom'
    mengzi = 'mengzi'

    chatglm2 = 'chatglm2'
    glm4 = 'glm4'
    codegeex4 = 'codegeex4'
    longwriter_llama = 'longwriter_llama'

    internlm = 'internlm'
    internlm2 = 'internlm2'

    deepseek = 'deepseek'
    deepseek_coder = 'deepseek_coder'
    deepseek_v2_5 = 'deepseek_v2_5'

    openbuddy = 'openbuddy'
    openbuddy2 = 'openbuddy2'
    baichuan = 'baichuan'
    minicpm = 'minicpm'
    telechat = 'telechat'
    telechat2 = 'telechat2'
    telechat2_115b = 'telechat2_115b'

    codefuse = 'codefuse'
    codefuse_codellama = 'codefuse_codellama'

    mistral_nemo = 'mistral_nemo'
    zephyr = 'zephyr'
    wizardlm2 = 'wizardlm2'
    wizardlm2_moe = 'wizardlm2_moe'
    gemma = 'gemma'
    phi3 = 'phi3'

    yuan = 'yuan'
    xverse = 'xverse'
    skywork = 'skywork'
    bluelm = 'bluelm'
    orion = 'orion'

    aya = 'aya'
    c4ai = 'c4ai'
    dbrx = 'dbrx'


class MLLMTemplateType:
    qwen_vl = 'qwen_vl'
    qwen_audio = 'qwen_audio'
    qwen2_vl = 'qwen2_vl'
    qwen2_audio = 'qwen2_audio'
    qvq = 'qvq'
    ovis1_6 = 'ovis1_6'

    llama3_1_omni = 'llama3_1_omni'
    llama3_2_vision = 'llama3_2_vision'

    llava1_5_hf = 'llava1_5_hf'
    llava1_6_mistral_hf = 'llava1_6_mistral_hf'
    llava1_6_vicuna_hf = 'llava1_6_vicuna_hf'
    llava1_6_yi_hf = 'llava1_6_yi_hf'
    llama3_llava_next_hf = 'llama3_llava_next_hf'
    llava_next_qwen_hf = 'llava_next_qwen_hf'
    llava_onevision_hf = 'llava_onevision_hf'
    llava_next_video_hf = 'llava_next_video_hf'

    llava_llama3_1_hf = 'llava_llama3_1_hf'  # DaozeZhang
    llava_llama3_hf = 'llava_llama3_hf'  # xtuner
    # lmms-lab
    llava1_6_mistral = 'llava1_6_mistral'
    llava1_6_yi = 'llava1_6_yi'
    llava_next_qwen = 'llava_next_qwen'
    llama3_llava_next = 'llama3_llava_next'

    yi_vl = 'yi_vl'

    internvl = 'internvl'
    internvl_phi3 = 'internvl_phi3'
    internvl2 = 'internvl2'
    internvl2_phi3 = 'internvl2_phi3'
    internvl2_5 = 'internvl2_5'

    xcomposer2 = 'ixcomposer2'
    xcomposer2_4khd = 'xcomposer2_4khd'
    xcomposer2_5 = 'xcomposer2_5'

    cogagent_chat = 'cogagent_chat'
    cogagent_vqa = 'cogagent_vqa'
    cogvlm = 'cogvlm'
    cogvlm2 = 'cogvlm2'
    cogvlm2_video = 'cogvlm2_video'
    glm4v = 'glm4v'
    glm_edge_v = 'glm_edge_v'

    minicpmv = 'minicpmv'
    minicpmv2_5 = 'minicpmv2_5'
    minicpmv2_6 = 'minicpmv2_6'

    deepseek_vl = 'deepseek_vl'
    deepseek_vl2 = 'deepseek_vl2'
    deepseek_janus = 'deepseek_janus'

    mplug_owl2 = 'mplug_owl2'
    mplug_owl3 = 'mplug_owl3'
    mplug_owl3_241101 = 'mplug_owl3_241101'
    doc_owl2 = 'doc_owl2'

    emu3_chat = 'emu3_chat'
    emu3_gen = 'emu3_gen'

    got_ocr2 = 'got_ocr2'
    idefics3 = 'idefics3'
    pixtral = 'pixtral'
    paligemma = 'paligemma'
    phi3_vision = 'phi3_vision'
    florence = 'florence'
    molmo = 'molmo'
    megrez_omni = 'megrez_omni'


class TemplateType(LLMTemplateType, MLLMTemplateType):

    @classmethod
    def get_template_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__'):
                continue
            value = cls.__dict__[k]
            if isinstance(value, str):
                res.append(value)
        return res
