# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import List


class LLMTemplateType:
    default = 'default'
    qwen = 'qwen'
    qwen2_5 = 'qwen2_5'
    chatml = 'chatml'

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
    deepseek2 = 'deepseek2'
    deepseek2_5 = 'deepseek2_5'

    baichuan = 'baichuan'
    openbuddy = 'openbuddy'
    openbuddy2 = 'openbuddy2'
    minicpm = 'minicpm'
    telechat = 'telechat'
    telechat_v2 = 'telechat_v2'

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


class MLLMTemplateType:
    qwen_vl = 'qwen_vl'
    qwen_audio = 'qwen_audio'
    qwen2_vl = 'qwen2_vl'
    qwen2_audio = 'qwen2_audio'

    llama3_1_omni = 'llama3_1_omni'
    llama3_2_vision = 'llama3_2_vision'

    llava1_5 = 'llava1_5'
    llava_mistral = 'llava_mistral'
    llava_vicuna = 'llava_vicuna'
    llava_yi = 'llava_yi'
    llama3_llava_next_hf = 'llama3_llava_next_hf'
    llava_qwen_hf = 'llama_qwen_hf'
    llava_onevision_qwen = 'llava_onevision_qwen'
    llava_next_video = 'llava_next_video'
    llava_next_video_yi = 'llava_next_video_yi'

    llava_next_llama3 = 'llava_next_llama3'  # DaozeZhang
    llava_llama_instruct = 'llava_llama_instruct'  # xtuner
    llama3_llava_next = 'llama3_llava_next'  # lmms-lab
    llava_qwen = 'llava_qwen'  # lmms-lab
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
    mplug_owl2 = 'mplug_owl2'
    mplug_owl3 = 'mplug_owl3'
    got_ocr2 = 'got_ocr2'

    florence = 'florence'
    idefics3 = 'idefics3'
    pixtral = 'pixtral'
    paligemma = 'paligemma'
    phi3_vl = 'phi3_vl'

    emu3_chat = 'emu3_chat'
    emu3_gen = 'emu3_gen'
    janus = 'janus'


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
