from typing import List


class LLMDatasetName:
    # general
    ms_bench = 'ms_bench'
    alpaca_en = 'alpaca_en'
    alpaca_zh = 'alpaca_zh'
    multi_alpaca = 'multi_alpaca'
    instinwild = 'instinwild'
    cot_en = 'cot_en'
    cot_zh = 'cot_zh'
    instruct_en = 'instruct_en'
    firefly_zh = 'firefly_zh'
    gpt4all_en = 'gpt4all_en'
    sharegpt = 'sharegpt'
    tulu_v2_sft_mixture = 'tulu_v2_sft_mixture'
    wikipedia_zh = 'wikipedia_zh'
    open_orca = 'open_orca'
    sharegpt_gpt4 = 'sharegpt_gpt4'
    deepctrl_sft = 'deepctrl_sft'
    coig_cqia = 'coig_cqia'
    ruozhiba = 'ruozhiba'
    long_alpaca_12k = 'long_alpaca_12k'
    lmsys_chat_1m = 'lmsys_chat_1m'
    guanaco = 'guanaco'

    # agent
    ms_agent = 'ms_agent'
    ms_agent_for_agentfabric = 'ms_agent_for_agentfabric'
    ms_agent_multirole = 'ms_agent_multirole'
    toolbench_for_alpha_umi = 'toolbench_for_alpha_umi'
    damo_agent_zh = 'damo_agent_zh'
    damo_agent_zh_mini = 'damo_agent_zh_mini'
    agent_instruct_all_en = 'agent_instruct_all_en'
    msagent_pro = 'msagent_pro'
    toolbench = 'toolbench'

    # coding
    code_alpaca_en = 'code_alpaca_en'
    leetcode_python_en = 'leetcode_python_en'
    codefuse_python_en = 'codefuse_python_en'
    codefuse_evol_instruction_zh = 'codefuse_evol_instruction_zh'
    # medical
    medical_en = 'medical_en'
    medical_zh = 'medical_zh'
    disc_med_sft_zh = 'disc_med_sft_zh'
    # law
    lawyer_llama_zh = 'lawyer_llama_zh'
    tigerbot_law_zh = 'tigerbot_law_zh'
    disc_law_sft_zh = 'disc_law_sft_zh'
    # math
    blossom_math_zh = 'blossom_math_zh'
    school_math_zh = 'school_math_zh'
    open_platypus_en = 'open_platypus_en'
    # sql
    text2sql_en = 'text2sql_en'
    sql_create_context_en = 'sql_create_context_en'
    synthetic_text_to_sql = 'synthetic_text_to_sql'
    # text_generation
    advertise_gen_zh = 'advertise_gen_zh'
    dureader_robust_zh = 'dureader_robust_zh'

    # classification
    cmnli_zh = 'cmnli_zh'
    jd_sentiment_zh = 'jd_sentiment_zh'
    hc3_zh = 'hc3_zh'
    hc3_en = 'hc3_en'
    dolly_15k = 'dolly_15k'
    zhihu_kol = 'zhihu_kol'
    zhihu_kol_filtered = 'zhihu_kol_filtered'
    # other
    finance_en = 'finance_en'
    poetry_zh = 'poetry_zh'
    webnovel_zh = 'webnovel_zh'
    generated_chat_zh = 'generated_chat_zh'
    self_cognition = 'self_cognition'
    swift_mix = 'swift_mix'

    cls_fudan_news_zh = 'cls_fudan_news_zh'
    ner_java_zh = 'ner_jave_zh'

    # rlhf
    hh_rlhf = 'hh_rlhf'
    hh_rlhf_cn = 'hh_rlhf_cn'
    orpo_dpo_mix_40k = 'orpo_dpo_mix_40k'
    stack_exchange_paired = 'stack_exchange_paired'
    shareai_llama3_dpo_emoji = 'shareai_llama3_dpo_emoji'
    ultrafeedback_kto = 'ultrafeedback_kto'

    # for awq
    pileval = 'pileval'


class MLLMDatasetName:
    # <img></img>
    coco_en = 'coco_en'
    coco_en_mini = 'coco_en_mini'
    # images
    coco_en_2 = 'coco_en_2'
    coco_en_2_mini = 'coco_en_2_mini'
    capcha_images = 'capcha_images'
    latex_ocr_print = 'latex_ocr_print'
    latex_ocr_handwrite = 'latex_ocr_handwrite'
    # for qwen_audio
    aishell1_zh = 'aishell1_zh'
    aishell1_zh_mini = 'aishell1_zh_mini'
    # for video
    video_chatgpt = 'video_chatgpt'

    # visual rlhf
    rlaif_v = 'rlaif_v'

    mantis_instruct = 'mantis_instruct'
    llava_data_instruct = 'llava_data_instruct'
    midefics = 'midefics'
    gqa = 'gqa'
    text_caps = 'text_caps'
    refcoco_unofficial_caption = 'refcoco_unofficial_caption'
    refcoco_unofficial_grounding = 'refcoco_unofficial_grounding'
    refcocog_unofficial_caption = 'refcocog_unofficial_caption'
    refcocog_unofficial_grounding = 'refcocog_unofficial_grounding'
    a_okvqa = 'a_okvqa'
    okvqa = 'okvqa'
    ocr_vqa = 'ocr_vqa'
    grit = 'grit'
    llava_instruct_mix = 'llava_instruct_mix'
    gpt4v_dataset = 'gpt4v_dataset'
    lnqa = 'lnqa'
    science_qa = 'science_qa'
    mind2web = 'mind2web'
    sharegpt_4o_image = 'sharegpt_4o_image'
    pixelprose = 'pixelprose'

    m3it = 'm3it'
    # additional images
    sharegpt4v = 'sharegpt4v'

    llava_instruct_150k = 'llava_instruct_150k'
    llava_pretrain = 'llava_pretrain'

    sa1b_dense_caption = 'sa1b_dense_caption'
    sa1b_paired_caption = 'sa1b_paired_caption'


class DatasetName(LLMDatasetName, MLLMDatasetName):

    @classmethod
    def get_dataset_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__'):
                continue
            value = cls.__dict__[k]
            if isinstance(value, str):
                res.append(value)
        return res
