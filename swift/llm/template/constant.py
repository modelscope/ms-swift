# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import List


class LLMTemplateType:
    # base
    default_generation = 'default-generation'
    chatglm_generation = 'chatglm-generation'

    # chat
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
    deepseek_coder = 'deepseek-coder'
    deepseek2 = 'deepseek2'
    deepseek2_5 = 'deepseek2_5'

    baichuan = 'baichuan'
    openbuddy = 'openbuddy'
    openbuddy2 = 'openbuddy2'
    minicpm = 'minicpm'
    telechat = 'telechat'
    telechat_v2 = 'telechat-v2'

    yi_coder = 'yi-coder'
    codefuse = 'codefuse'
    codefuse_codellama = 'codefuse-codellama'

    numina_math = 'numina-math'
    mistral_nemo = 'mistral-nemo'
    gemma = 'gemma'
    wizardlm2_awq = 'wizardlm2-awq'
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
    # base
    qwen_vl_generation = 'qwen_vl_generation'
    qwen_audio_generation = 'qwen_audio_generation'
    qwen2_vl_generation = 'qwen2_vl_generation'
    qwen2_audio_generation = 'qwen2_audio_generation'

    llama3_2_vision_generation = 'llama3_2_vision_generation'

    # chat
    qwen_vl = 'qwen_vl'
    qwen_audio = 'qwen_audio'
    qwen2_vl = 'qwen2_vl'
    qwen2_audio = 'qwen2_audio'

    llama3_1_omni = 'llama3_1-omni'
    llama3_2_vision = 'llama3_2-vision'

    llava1_5 = 'llava1_5'
    llava_mistral = 'llava_mistral'
    llava_vicuna = 'llava_vicuna'
    llava_yi = 'llava_yi'
    llama3_llava_next_hf = 'llama3_llava_next_hf'
    llava_qwen_hf = 'llama-qwen-hf'
    llava_onevision_qwen = 'llava-onevision-qwen'
    llava_next_video = 'llava-next-video'
    llava_next_video_yi = 'llava-next-video-yi'

    llava_next_llama3 = 'llava-next-llama3'  # DaozeZhang
    llava_llama_instruct = 'llava_llama_instruct'  # xtuner
    llama3_llava_next = 'llama3-llava-next'  # lmms-lab
    llava_qwen = 'llava-qwen'  # lmms-lab
    yi_vl = 'yi-vl'

    internvl = 'internvl'
    internvl2 = 'internvl2'
    internvl_phi3 = 'internvl-phi3'
    internvl2_phi3 = 'internvl2-phi3'

    internlm_xcomposer2 = 'internlm-xcomposer2'
    internlm_xcomposer2_4khd = 'internlm-xcomposer2-4khd'
    internlm_xcomposer2_5 = 'internlm-xcomposer2_5'

    cogagent_chat = 'cogagent-chat'
    cogagent_instruct = 'cogagent-instruct'
    cogvlm = 'cogvlm'
    cogvlm2_video = 'cogvlm2-video'
    glm4v = 'glm4v'

    minicpm_v = 'minicpm-v'
    minicpm_v_v2_5 = 'minicpm-v-v2_5'
    minicpm_v_v2_6 = 'minicpm-v-v2_6'

    deepseek_vl = 'deepseek-vl'
    mplug_owl2 = 'mplug-owl2'
    mplug_owl3 = 'mplug_owl3'
    got_ocr2 = 'got_ocr2'

    florence = 'florence'
    idefics3 = 'idefics3'
    pixtral = 'pixtral'
    paligemma = 'paligemma'
    phi3_vl = 'phi3-vl'


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
