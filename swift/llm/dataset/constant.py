from typing import List


class LLMDatasetName:
    # general
    ms_bench = 'ms-bench'
    alpaca_en = 'alpaca-en'
    alpaca_zh = 'alpaca-zh'
    multi_alpaca = 'multi-alpaca'
    instinwild = 'instinwild'
    cot_en = 'cot-en'
    cot_zh = 'cot-zh'
    instruct_en = 'instruct-en'
    firefly_zh = 'firefly-zh'
    gpt4all_en = 'gpt4all-en'
    sharegpt = 'sharegpt'
    tulu_v2_sft_mixture = 'tulu-v2-sft-mixture'
    wikipedia_zh = 'wikipedia-zh'
    open_orca = 'open-orca'
    sharegpt_gpt4 = 'sharegpt-gpt4'
    deepctrl_sft = 'deepctrl-sft'
    coig_cqia = 'coig-cqia'
    ruozhiba = 'ruozhiba'
    long_alpaca_12k = 'long-alpaca-12k'
    lmsys_chat_1m = 'lmsys-chat-1m'
    guanaco = 'guanaco'

    # agent
    ms_agent = 'ms-agent'
    ms_agent_for_agentfabric = 'ms-agent-for-agentfabric'
    ms_agent_multirole = 'ms-agent-multirole'
    toolbench_for_alpha_umi = 'toolbench-for-alpha-umi'
    damo_agent_zh = 'damo-agent-zh'
    damo_agent_zh_mini = 'damo-agent-zh-mini'
    agent_instruct_all_en = 'agent-instruct-all-en'
    msagent_pro = 'msagent-pro'
    toolbench = 'toolbench'

    # coding
    code_alpaca_en = 'code-alpaca-en'
    leetcode_python_en = 'leetcode-python-en'
    codefuse_python_en = 'codefuse-python-en'
    codefuse_evol_instruction_zh = 'codefuse-evol-instruction-zh'
    # medical
    medical_en = 'medical-en'
    medical_zh = 'medical-zh'
    disc_med_sft_zh = 'disc-med-sft-zh'
    # law
    lawyer_llama_zh = 'lawyer-llama-zh'
    tigerbot_law_zh = 'tigerbot-law-zh'
    disc_law_sft_zh = 'disc-law-sft-zh'
    # math
    blossom_math_zh = 'blossom-math-zh'
    school_math_zh = 'school-math-zh'
    open_platypus_en = 'open-platypus-en'
    # sql
    text2sql_en = 'text2sql-en'
    sql_create_context_en = 'sql-create-context-en'
    synthetic_text_to_sql = 'synthetic-text-to-sql'
    # text-generation
    advertise_gen_zh = 'advertise-gen-zh'
    dureader_robust_zh = 'dureader-robust-zh'

    # classification
    cmnli_zh = 'cmnli-zh'
    jd_sentiment_zh = 'jd-sentiment-zh'
    hc3_zh = 'hc3-zh'
    hc3_en = 'hc3-en'
    dolly_15k = 'dolly-15k'
    zhihu_kol = 'zhihu-kol'
    zhihu_kol_filtered = 'zhihu-kol-filtered'
    # other
    finance_en = 'finance-en'
    poetry_zh = 'poetry-zh'
    webnovel_zh = 'webnovel-zh'
    generated_chat_zh = 'generated-chat-zh'
    self_cognition = 'self-cognition'
    swift_mix = 'swift-mix'

    cls_fudan_news_zh = 'cls-fudan-news-zh'
    ner_java_zh = 'ner-jave-zh'

    # rlhf
    hh_rlhf = 'hh-rlhf'
    hh_rlhf_cn = 'hh-rlhf-cn'
    orpo_dpo_mix_40k = 'orpo-dpo-mix-40k'
    stack_exchange_paired = 'stack-exchange-paired'
    shareai_llama3_dpo_zh_en_emoji = 'shareai-llama3-dpo-zh-en-emoji'
    ultrafeedback_kto = 'ultrafeedback-kto'

    # for awq
    pileval = 'pileval'


class MLLMDatasetName:
    # <img></img>
    coco_en = 'coco-en'
    coco_en_mini = 'coco-en-mini'
    # images
    coco_en_2 = 'coco-en-2'
    coco_en_2_mini = 'coco-en-2-mini'
    capcha_images = 'capcha-images'
    latex_ocr_print = 'latex-ocr-print'
    latex_ocr_handwrite = 'latex-ocr-handwrite'
    # for qwen-audio
    aishell1_zh = 'aishell1-zh'
    aishell1_zh_mini = 'aishell1-zh-mini'
    # for video
    video_chatgpt = 'video-chatgpt'

    # visual rlhf
    rlaif_v = 'rlaif-v'

    mantis_instruct = 'mantis-instruct'
    llava_data_instruct = 'llava-data-instruct'
    midefics = 'midefics'
    gqa = 'gqa'
    text_caps = 'text-caps'
    refcoco_unofficial_caption = 'refcoco-unofficial-caption'
    refcoco_unofficial_grounding = 'refcoco-unofficial-grounding'
    refcocog_unofficial_caption = 'refcocog-unofficial-caption'
    refcocog_unofficial_grounding = 'refcocog-unofficial-grounding'
    a_okvqa = 'a-okvqa'
    okvqa = 'okvqa'
    ocr_vqa = 'ocr-vqa'
    grit = 'grit'
    llava_instruct_mix = 'llava-instruct-mix'
    gpt4v_dataset = 'gpt4v-dataset'
    lnqa = 'lnqa'
    science_qa = 'science-qa'
    mind2web = 'mind2web'
    sharegpt_4o_image = 'sharegpt-4o-image'
    pixelprose = 'pixelprose'

    m3it = 'm3it'
    # additional images
    sharegpt4v = 'sharegpt4v'

    llava_instruct_150k = 'llava-instruct-150k'
    llava_pretrain = 'llava-pretrain'

    sa1b_dense_caption = 'sa1b-dense-caption'
    sa1b_paired_caption = 'sa1b-paired-caption'


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
