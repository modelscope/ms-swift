# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import List


class LLMTemplateType:
    default = 'default'
    qwen = 'qwen'
    qwen2_5 = 'qwen2_5'
    qwq = 'qwq'
    chatml = 'chatml'
    marco_o1 = 'marco_o1'

    llama = 'llama'  # llama2
    llama3 = 'llama3'
    llama3_2 = 'llama3_2'
    longwriter_llama3 = 'longwriter_llama3'
    reflection = 'reflection'

    chatglm2 = 'chatglm2'
    chatglm3 = 'chatglm3'
    chatglm4 = 'chatglm4'
    codegeex4 = 'codegeex4'

    internlm = 'internlm'
    internlm2 = 'internlm2'

    deepseek = 'deepseek'
    deepseek_coder = 'deepseek_coder'
    deepseek2_5 = 'deepseek2_5'

    baichuan = 'baichuan'
    openbuddy = 'openbuddy'
    openbuddy2 = 'openbuddy2'
    minicpm = 'minicpm'
    telechat = 'telechat'
    telechat2 = 'telechat2'

    yi_coder = 'yi_coder'
    codefuse = 'codefuse'
    codefuse_codellama = 'codefuse_codellama'

    numina_math = 'numina_math'
    mistral_nemo = 'mistral_nemo'
    gemma = 'gemma'
    wizardlm2_awq = 'wizardlm2_awq'
    wizardlm2 = 'wizardlm2'
    atom = 'atom'
    phi3 = 'phi3'
    c4ai = 'c4ai'
    dbrx = 'dbrx'

    yuan = 'yuan'
    xverse = 'xverse'
    ziya = 'ziya'
    skywork = 'skywork'
    bluelm = 'bluelm'
    zephyr = 'zephyr'
    sus = 'sus'
    orion = 'orion'
    modelscope_agent = 'modelscope_agent'
    mengzi = 'mengzi'
    aya = 'aya'


class MLLMTemplateType:
    qwen_vl = 'qwen_vl'
    qwen_audio = 'qwen_audio'
    qwen2_vl = 'qwen2_vl'
    qwen2_audio = 'qwen2_audio'
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

    xcomposer2 = 'ixcomposer2'
    xcomposer2_4khd = 'xcomposer2_4khd'
    xcomposer2_5 = 'xcomposer2_5'

    cogagent_chat = 'cogagent_chat'
    cogagent_vqa = 'cogagent_vqa'
    cogvlm = 'cogvlm'
    cogvlm2_video = 'cogvlm2_video'
    glm4v = 'glm4v'

    minicpmv = 'minicpmv'
    minicpmv2_5 = 'minicpmv2_5'
    minicpmv2_6 = 'minicpmv2_6'

    deepseek_vl = 'deepseek_vl'
    deepseek_janus = 'deepseek_janus'
    mplug_owl2 = 'mplug_owl2'
    mplug_owl3 = 'mplug_owl3'
    mplug_owl3v = 'mplug_owl3v'
    got_ocr2 = 'got_ocr2'

    florence = 'florence'
    idefics3 = 'idefics3'
    pixtral = 'pixtral'
    paligemma = 'paligemma'
    phi3_vl = 'phi3_vl'

    emu3_chat = 'emu3_chat'
    emu3_gen = 'emu3_gen'
    janus = 'janus'
    molmo = 'molmo'


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
