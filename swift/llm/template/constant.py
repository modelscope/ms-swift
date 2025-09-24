# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import List


class LLMTemplateType:
    chatml = 'chatml'
    default = 'default'
    dummy = 'dummy'

    qwen = 'qwen'
    qwen2_5 = 'qwen2_5'
    qwen2_5_math = 'qwen2_5_math'
    qwen2_5_math_prm = 'qwen2_5_math_prm'
    qwen3 = 'qwen3'
    qwen3_thinking = 'qwen3_thinking'
    qwen3_nothinking = 'qwen3_nothinking'
    qwen3_coder = 'qwen3_coder'
    qwen3_emb = 'qwen3_emb'
    qwen3_reranker = 'qwen3_reranker'
    qwq_preview = 'qwq_preview'
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
    gpt_oss = 'gpt_oss'
    seed_oss = 'seed_oss'

    minimax = 'minimax'
    minimax_m1 = 'minimax_m1'
    minimax_vl = 'minimax_vl'

    numina = 'numina'
    ziya = 'ziya'
    atom = 'atom'
    mengzi = 'mengzi'
    bge_reranker = 'bge_reranker'

    chatglm2 = 'chatglm2'
    glm4 = 'glm4'
    glm4_0414 = 'glm4_0414'
    glm4_z1_rumination = 'glm4_z1_rumination'
    glm4_5 = 'glm4_5'
    codegeex4 = 'codegeex4'
    longwriter_llama = 'longwriter_llama'

    internlm = 'internlm'
    internlm2 = 'internlm2'
    internlm3 = 'internlm3'

    deepseek = 'deepseek'
    deepseek_coder = 'deepseek_coder'
    deepseek_v2_5 = 'deepseek_v2_5'
    deepseek_r1 = 'deepseek_r1'
    deepseek_v3_1 = 'deepseek_v3_1'

    openbuddy = 'openbuddy'
    openbuddy2 = 'openbuddy2'
    baichuan = 'baichuan'
    baichuan_m1 = 'baichuan_m1'
    minicpm = 'minicpm'
    telechat = 'telechat'
    telechat2 = 'telechat2'

    codefuse = 'codefuse'
    codefuse_codellama = 'codefuse_codellama'

    skywork = 'skywork'
    skywork_o1 = 'skywork_o1'

    mistral_nemo = 'mistral_nemo'
    mistral_2501 = 'mistral_2501'
    devstral = 'devstral'
    zephyr = 'zephyr'
    wizardlm2 = 'wizardlm2'
    wizardlm2_moe = 'wizardlm2_moe'
    gemma = 'gemma'
    gemma3_text = 'gemma3_text'
    phi3 = 'phi3'
    phi4 = 'phi4'

    ling = 'ling'
    ling2 = 'ling2'
    ring2 = 'ring2'
    yuan = 'yuan'
    xverse = 'xverse'
    bluelm = 'bluelm'
    orion = 'orion'
    moonlight = 'moonlight'
    mimo_rl = 'mimo_rl'
    dots1 = 'dots1'
    hunyuan_moe = 'hunyuan_moe'
    hunyuan = 'hunyuan'
    ernie = 'ernie'
    ernie_thinking = 'ernie_thinking'
    longchat = 'longchat'

    aya = 'aya'
    c4ai = 'c4ai'
    dbrx = 'dbrx'

    bert = 'bert'


class RMTemplateType:
    internlm2_reward = 'internlm2_reward'


class MLLMTemplateType:
    qwen_vl = 'qwen_vl'
    qwen_audio = 'qwen_audio'
    qwen2_vl = 'qwen2_vl'
    qwen2_5_vl = 'qwen2_5_vl'
    qwen2_5_omni = 'qwen2_5_omni'
    qwen3_omni = 'qwen3_omni'
    qwen2_audio = 'qwen2_audio'
    qwen3_vl = 'qwen3_vl'
    qwen2_gme = 'qwen2_gme'
    qvq = 'qvq'
    ovis1_6 = 'ovis1_6'
    ovis1_6_llama3 = 'ovis1_6_llama3'
    ovis2 = 'ovis2'
    ovis2_5 = 'ovis2_5'
    mimo_vl = 'mimo_vl'
    midashenglm = 'midashenglm'

    llama3_1_omni = 'llama3_1_omni'
    llama3_2_vision = 'llama3_2_vision'
    llama4 = 'llama4'

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
    internvl3_5 = 'internvl3_5'
    internvl3_5_gpt = 'internvl3_5_gpt'
    interns1 = 'interns1'
    internvl_hf = 'internvl_hf'

    xcomposer2 = 'ixcomposer2'
    xcomposer2_4khd = 'xcomposer2_4khd'
    xcomposer2_5 = 'xcomposer2_5'

    cogagent_chat = 'cogagent_chat'
    cogagent_vqa = 'cogagent_vqa'
    cogvlm = 'cogvlm'
    cogvlm2 = 'cogvlm2'
    cogvlm2_video = 'cogvlm2_video'
    glm4v = 'glm4v'
    glm4_1v = 'glm4_1v'
    glm_edge_v = 'glm_edge_v'
    glm4_5v = 'glm4_5v'

    minicpmv = 'minicpmv'
    minicpmv2_5 = 'minicpmv2_5'
    minicpmv2_6 = 'minicpmv2_6'
    minicpmo2_6 = 'minicpmo2_6'
    minicpmv4 = 'minicpmv4'
    minicpmv4_5 = 'minicpmv4_5'

    deepseek_vl = 'deepseek_vl'
    deepseek_vl2 = 'deepseek_vl2'
    deepseek_janus = 'deepseek_janus'
    deepseek_janus_pro = 'deepseek_janus_pro'

    mplug_owl2 = 'mplug_owl2'
    mplug_owl3 = 'mplug_owl3'
    mplug_owl3_241101 = 'mplug_owl3_241101'
    doc_owl2 = 'doc_owl2'

    emu3_chat = 'emu3_chat'
    emu3_gen = 'emu3_gen'

    got_ocr2 = 'got_ocr2'
    got_ocr2_hf = 'got_ocr2_hf'
    step_audio = 'step_audio'
    step_audio2_mini = 'step_audio2_mini'
    kimi_vl = 'kimi_vl'
    keye_vl = 'keye_vl'
    keye_vl_1_5 = 'keye_vl_1_5'
    dots_ocr = 'dots_ocr'
    sail_vl2 = 'sail_vl2'

    idefics3 = 'idefics3'
    pixtral = 'pixtral'
    paligemma = 'paligemma'
    phi3_vision = 'phi3_vision'
    phi4_multimodal = 'phi4_multimodal'
    florence = 'florence'
    molmo = 'molmo'
    megrez_omni = 'megrez_omni'
    valley = 'valley'
    gemma3_vision = 'gemma3_vision'
    gemma3n = 'gemma3n'
    mistral_2503 = 'mistral_2503'


class TemplateType(LLMTemplateType, MLLMTemplateType, RMTemplateType):

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
