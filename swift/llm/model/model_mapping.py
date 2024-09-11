from typing import List


class MMModelType:

    # qwen-vl
    qwen_vl = 'qwen-vl'
    qwen_vl_chat = 'qwen-vl-chat'
    qwen_vl_chat_int4 = 'qwen-vl-chat-int4'

    #qwen2-vl
    qwen2_vl_2b_instruct = 'qwen2-vl-2b-instruct'
    qwen2_vl_2b_instruct_gptq_int4 = 'qwen2-vl-2b-instruct-gptq-int4'
    qwen2_vl_2b_instruct_gptq_int8 = 'qwen2-vl-2b-instruct-gptq-int8'
    qwen2_vl_2b_instruct_awq = 'qwen2-vl-2b-instruct-awq'
    qwen2_vl_7b_instruct = 'qwen2-vl-7b-instruct'
    qwen2_vl_7b_instruct_gptq_int4 = 'qwen2-vl-7b-instruct-gptq-int4'
    qwen2_vl_7b_instruct_gptq_int8 = 'qwen2-vl-7b-instruct-gptq-int8'
    qwen2_vl_7b_instruct_awq = 'qwen2-vl-7b-instruct-awq'

    # qwen-audio
    qwen_audio = 'qwen-audio'
    qwen_audio_chat = 'qwen-audio-chat'
    qwen2_audio_7b = 'qwen2-audio-7b'
    qwen2_audio_7b_instruct = 'qwen2-audio-7b-instruct'

    # llava-hf
    llava1_5_7b_instruct = 'llava1_5-7b-instruct'
    llava1_5_13b_instruct = 'llava1_5-13b-instruct'
    llava1_6_mistral_7b_instruct = 'llava1_6-mistral-7b-instruct'
    llava1_6_vicuna_7b_instruct = 'llava1_6-vicuna-7b-instruct'
    llava1_6_vicuna_13b_instruct = 'llava1_6-vicuna-13b-instruct'
    llava1_6_llama3_1_8b_instruct = 'llava1_6-llama3_1-8b-instruct'
    llava1_6_yi_34b_instruct = 'llava1_6-yi-34b-instruct'
    llama3_llava_next_8b_hf = 'llama3-llava-next-8b-hf'
    llava_next_72b_hf = 'llava-next-72b-hf'
    llava_next_110b_hf = 'llava-next-110b-hf'

    llava_onevision_qwen2_0_5b_ov = 'llava-onevision-qwen2-0_5b-ov'
    llava_onevision_qwen2_7b_ov = 'llava-onevision-qwen2-7b-ov'
    llava_onevision_qwen2_72b_ov = 'llava-onevision-qwen2-72b-ov'

    # llava
    llama3_llava_next_8b = 'llama3-llava-next-8b'
    llava_next_72b = 'llava-next-72b'
    llava_next_110b = 'llava-next-110b'

    # llava_next_video-hf
    llava_next_video_7b_instruct = 'llava-next-video-7b-instruct'
    llava_next_video_7b_32k_instruct = 'llava-next-video-7b-32k-instruct'
    llava_next_video_7b_dpo_instruct = 'llava-next-video-7b-dpo-instruct'
    llava_next_video_34b_instruct = 'llava-next-video-34b-instruct'

    # yi-vl
    yi_vl_6b_chat = 'yi-vl-6b-chat'
    yi_vl_34b_chat = 'yi-vl-34b-chat'

    # llava-llama (xtuner)
    llava_llama3_8b_v1_1 = 'llava-llama3-8b-v1_1'

    # glm4v
    glm4v_9b_chat = 'glm4v-9b-chat'

    # internvl
    internvl_chat_v1_5 = 'internvl-chat-v1_5'
    internvl_chat_v1_5_int8 = 'internvl-chat-v1_5-int8'
    mini_internvl_chat_2b_v1_5 = 'mini-internvl-chat-2b-v1_5'
    mini_internvl_chat_4b_v1_5 = 'mini-internvl-chat-4b-v1_5'

    # internvl2
    internvl2_1b = 'internvl2-1b'
    internvl2_2b = 'internvl2-2b'
    internvl2_4b = 'internvl2-4b'
    internvl2_8b = 'internvl2-8b'
    internvl2_26b = 'internvl2-26b'
    internvl2_40b = 'internvl2-40b'
    internvl2_llama3_76b = 'internvl2-llama3-76b'
    internvl2_2b_awq = 'internvl2-2b-awq'
    internvl2_8b_awq = 'internvl2-8b-awq'
    internvl2_26b_awq = 'internvl2-26b-awq'
    internvl2_40b_awq = 'internvl2-40b-awq'
    internvl2_llama3_76b_awq = 'internvl2-llama3-76b-awq'

    # deepseek-vl
    deepseek_vl_1_3b_chat = 'deepseek-vl-1_3b-chat'
    deepseek_vl_7b_chat = 'deepseek-vl-7b-chat'

    # deepseek-v2
    deepseek_v2 = 'deepseek-v2'
    deepseek_v2_chat = 'deepseek-v2-chat'
    deepseek_v2_lite = 'deepseek-v2-lite'
    deepseek_v2_lite_chat = 'deepseek-v2-lite-chat'

    # deepseek-v2.5
    deepseek_v2_5 = 'deepseek-v2_5'

    # paligemma
    paligemma_3b_pt_224 = 'paligemma-3b-pt-224'
    paligemma_3b_pt_448 = 'paligemma-3b-pt-448'
    paligemma_3b_pt_896 = 'paligemma-3b-pt-896'
    paligemma_3b_mix_224 = 'paligemma-3b-mix-224'
    paligemma_3b_mix_448 = 'paligemma-3b-mix-448'

    # minicpm-v
    minicpm_v_3b_chat = 'minicpm-v-3b-chat'
    minicpm_v_v2_chat = 'minicpm-v-v2-chat'
    minicpm_v_v2_5_chat = 'minicpm-v-v2_5-chat'
    minicpm_v_v2_6_chat = 'minicpm-v-v2_6-chat'

    # mplug-owl
    mplug_owl2_chat = 'mplug-owl2-chat'  # llama
    mplug_owl2_1_chat = 'mplug-owl2_1-chat'  # qwen
    mplug_owl3_7b_chat = 'mplug-owl3-7b-chat'

    # phi3-v
    phi3_vision_128k_instruct = 'phi3-vision-128k-instruct'
    phi3_5_vision_instruct = 'phi3_5-vision-instruct'

    # cogagent
    cogvlm_17b_chat = 'cogvlm-17b-chat'
    cogvlm2_19b_chat = 'cogvlm2-19b-chat'  # chinese
    cogvlm2_en_19b_chat = 'cogvlm2-en-19b-chat'
    cogvlm2_video_13b_chat = 'cogvlm2-video-13b-chat'
    cogagent_18b_chat = 'cogagent-18b-chat'
    cogagent_18b_instruct = 'cogagent-18b-instruct'

    # florence
    florence_2_base = 'florence-2-base'
    florence_2_base_ft = 'florence-2-base-ft'
    florence_2_large = 'florence-2-large'
    florence_2_large_ft = 'florence-2-large-ft'


class NLPModelType:
    # qwen
    qwen_1_8b = 'qwen-1_8b'
    qwen_1_8b_chat = 'qwen-1_8b-chat'
    qwen_1_8b_chat_int4 = 'qwen-1_8b-chat-int4'
    qwen_1_8b_chat_int8 = 'qwen-1_8b-chat-int8'
    qwen_7b = 'qwen-7b'
    qwen_7b_chat = 'qwen-7b-chat'
    qwen_7b_chat_int4 = 'qwen-7b-chat-int4'
    qwen_7b_chat_int8 = 'qwen-7b-chat-int8'
    qwen_14b = 'qwen-14b'
    qwen_14b_chat = 'qwen-14b-chat'
    qwen_14b_chat_int4 = 'qwen-14b-chat-int4'
    qwen_14b_chat_int8 = 'qwen-14b-chat-int8'
    qwen_72b = 'qwen-72b'
    qwen_72b_chat = 'qwen-72b-chat'
    qwen_72b_chat_int4 = 'qwen-72b-chat-int4'
    qwen_72b_chat_int8 = 'qwen-72b-chat-int8'

    # modelscope_agent
    modelscope_agent_7b = 'modelscope-agent-7b'
    modelscope_agent_14b = 'modelscope-agent-14b'

    # qwen1.5
    qwen1half_0_5b = 'qwen1half-0_5b'
    qwen1half_1_8b = 'qwen1half-1_8b'
    qwen1half_4b = 'qwen1half-4b'
    qwen1half_7b = 'qwen1half-7b'
    qwen1half_14b = 'qwen1half-14b'
    qwen1half_32b = 'qwen1half-32b'
    qwen1half_72b = 'qwen1half-72b'
    qwen1half_110b = 'qwen1half-110b'
    codeqwen1half_7b = 'codeqwen1half-7b'
    qwen1half_moe_a2_7b = 'qwen1half-moe-a2_7b'
    qwen1half_0_5b_chat = 'qwen1half-0_5b-chat'
    qwen1half_1_8b_chat = 'qwen1half-1_8b-chat'
    qwen1half_4b_chat = 'qwen1half-4b-chat'
    qwen1half_7b_chat = 'qwen1half-7b-chat'
    qwen1half_14b_chat = 'qwen1half-14b-chat'
    qwen1half_32b_chat = 'qwen1half-32b-chat'
    qwen1half_72b_chat = 'qwen1half-72b-chat'
    qwen1half_110b_chat = 'qwen1half-110b-chat'
    qwen1half_moe_a2_7b_chat = 'qwen1half-moe-a2_7b-chat'
    codeqwen1half_7b_chat = 'codeqwen1half-7b-chat'

    # qwen1.5 gptq
    qwen1half_0_5b_chat_int4 = 'qwen1half-0_5b-chat-int4'
    qwen1half_1_8b_chat_int4 = 'qwen1half-1_8b-chat-int4'
    qwen1half_4b_chat_int4 = 'qwen1half-4b-chat-int4'
    qwen1half_7b_chat_int4 = 'qwen1half-7b-chat-int4'
    qwen1half_14b_chat_int4 = 'qwen1half-14b-chat-int4'
    qwen1half_32b_chat_int4 = 'qwen1half-32b-chat-int4'
    qwen1half_72b_chat_int4 = 'qwen1half-72b-chat-int4'
    qwen1half_110b_chat_int4 = 'qwen1half-110b-chat-int4'
    qwen1half_0_5b_chat_int8 = 'qwen1half-0_5b-chat-int8'
    qwen1half_1_8b_chat_int8 = 'qwen1half-1_8b-chat-int8'
    qwen1half_4b_chat_int8 = 'qwen1half-4b-chat-int8'
    qwen1half_7b_chat_int8 = 'qwen1half-7b-chat-int8'
    qwen1half_14b_chat_int8 = 'qwen1half-14b-chat-int8'
    qwen1half_72b_chat_int8 = 'qwen1half-72b-chat-int8'
    qwen1half_moe_a2_7b_chat_int4 = 'qwen1half-moe-a2_7b-chat-int4'

    # qwen1.5 awq
    qwen1half_0_5b_chat_awq = 'qwen1half-0_5b-chat-awq'
    qwen1half_1_8b_chat_awq = 'qwen1half-1_8b-chat-awq'
    qwen1half_4b_chat_awq = 'qwen1half-4b-chat-awq'
    qwen1half_7b_chat_awq = 'qwen1half-7b-chat-awq'
    qwen1half_14b_chat_awq = 'qwen1half-14b-chat-awq'
    qwen1half_32b_chat_awq = 'qwen1half-32b-chat-awq'
    qwen1half_72b_chat_awq = 'qwen1half-72b-chat-awq'
    qwen1half_110b_chat_awq = 'qwen1half-110b-chat-awq'
    codeqwen1half_7b_chat_awq = 'codeqwen1half-7b-chat-awq'

    # qwen2
    qwen2_0_5b = 'qwen2-0_5b'
    qwen2_0_5b_instruct = 'qwen2-0_5b-instruct'
    qwen2_0_5b_instruct_int4 = 'qwen2-0_5b-instruct-int4'
    qwen2_0_5b_instruct_int8 = 'qwen2-0_5b-instruct-int8'
    qwen2_0_5b_instruct_awq = 'qwen2-0_5b-instruct-awq'
    qwen2_1_5b = 'qwen2-1_5b'
    qwen2_1_5b_instruct = 'qwen2-1_5b-instruct'
    qwen2_1_5b_instruct_int4 = 'qwen2-1_5b-instruct-int4'
    qwen2_1_5b_instruct_int8 = 'qwen2-1_5b-instruct-int8'
    qwen2_1_5b_instruct_awq = 'qwen2-1_5b-instruct-awq'
    qwen2_7b = 'qwen2-7b'
    qwen2_7b_instruct = 'qwen2-7b-instruct'
    qwen2_7b_instruct_int4 = 'qwen2-7b-instruct-int4'
    qwen2_7b_instruct_int8 = 'qwen2-7b-instruct-int8'
    qwen2_7b_instruct_awq = 'qwen2-7b-instruct-awq'
    qwen2_72b = 'qwen2-72b'
    qwen2_72b_instruct = 'qwen2-72b-instruct'
    qwen2_72b_instruct_int4 = 'qwen2-72b-instruct-int4'
    qwen2_72b_instruct_int8 = 'qwen2-72b-instruct-int8'
    qwen2_72b_instruct_awq = 'qwen2-72b-instruct-awq'
    qwen2_57b_a14b = 'qwen2-57b-a14b'
    qwen2_57b_a14b_instruct = 'qwen2-57b-a14b-instruct'
    qwen2_57b_a14b_instruct_int4 = 'qwen2-57b-a14b-instruct-int4'

    qwen2_math_1_5b = 'qwen2-math-1_5b'
    qwen2_math_1_5b_instruct = 'qwen2-math-1_5b-instruct'
    qwen2_math_7b = 'qwen2-math-7b'
    qwen2_math_7b_instruct = 'qwen2-math-7b-instruct'
    qwen2_math_72b = 'qwen2-math-72b'
    qwen2_math_72b_instruct = 'qwen2-math-72b-instruct'

    # chatglm
    chatglm2_6b = 'chatglm2-6b'
    chatglm2_6b_32k = 'chatglm2-6b-32k'
    chatglm3_6b_base = 'chatglm3-6b-base'
    chatglm3_6b = 'chatglm3-6b'
    chatglm3_6b_32k = 'chatglm3-6b-32k'
    chatglm3_6b_128k = 'chatglm3-6b-128k'
    codegeex2_6b = 'codegeex2-6b'
    glm4_9b = 'glm4-9b'
    glm4_9b_chat = 'glm4-9b-chat'
    glm4_9b_chat_1m = 'glm4-9b-chat-1m'
    codegeex4_9b_chat = 'codegeex4-9b-chat'

    # llama2
    llama2_7b = 'llama2-7b'
    llama2_7b_chat = 'llama2-7b-chat'
    llama2_13b = 'llama2-13b'
    llama2_13b_chat = 'llama2-13b-chat'
    llama2_70b = 'llama2-70b'
    llama2_70b_chat = 'llama2-70b-chat'
    llama2_7b_aqlm_2bit_1x16 = 'llama2-7b-aqlm-2bit-1x16'  # aqlm

    # llama3
    llama3_8b = 'llama3-8b'
    llama3_8b_instruct = 'llama3-8b-instruct'
    llama3_8b_instruct_int4 = 'llama3-8b-instruct-int4'
    llama3_8b_instruct_int8 = 'llama3-8b-instruct-int8'
    llama3_8b_instruct_awq = 'llama3-8b-instruct-awq'
    llama3_70b = 'llama3-70b'
    llama3_70b_instruct = 'llama3-70b-instruct'
    llama3_70b_instruct_int4 = 'llama3-70b-instruct-int4'
    llama3_70b_instruct_int8 = 'llama3-70b-instruct-int8'
    llama3_70b_instruct_awq = 'llama3-70b-instruct-awq'

    # llama3.1
    llama3_1_8b = 'llama3_1-8b'
    llama3_1_8b_instruct = 'llama3_1-8b-instruct'
    llama3_1_8b_instruct_awq = 'llama3_1-8b-instruct-awq'
    llama3_1_8b_instruct_gptq_int4 = 'llama3_1-8b-instruct-gptq-int4'
    llama3_1_8b_instruct_bnb = 'llama3_1-8b-instruct-bnb'
    llama3_1_70b = 'llama3_1-70b'
    llama3_1_70b_instruct = 'llama3_1-70b-instruct'
    llama3_1_70b_instruct_fp8 = 'llama3_1-70b-instruct-fp8'
    llama3_1_70b_instruct_awq = 'llama3_1-70b-instruct-awq'
    llama3_1_70b_instruct_gptq_int4 = 'llama3_1-70b-instruct-gptq-int4'
    llama3_1_70b_instruct_bnb = 'llama3_1-70b-instruct-bnb'
    llama3_1_405b = 'llama3_1-405b'
    llama3_1_405b_instruct = 'llama3_1-405b-instruct'
    llama3_1_405b_instruct_fp8 = 'llama3_1-405b-instruct-fp8'
    llama3_1_405b_instruct_awq = 'llama3_1-405b-instruct-awq'
    llama3_1_405b_instruct_gptq_int4 = 'llama3_1-405b-instruct-gptq-int4'
    llama3_1_405b_instruct_bnb = 'llama3_1-405b-instruct-bnb'

    # reflection
    reflection_llama_3_1_70b = 'reflection-llama_3_1-70b'

    # long writer
    longwriter_glm4_9b = 'longwriter-glm4-9b'
    longwriter_llama3_1_8b = 'longwriter-llama3_1-8b'

    # chinese-llama-alpaca
    chinese_llama_2_1_3b = 'chinese-llama-2-1_3b'
    chinese_llama_2_7b = 'chinese-llama-2-7b'
    chinese_llama_2_7b_16k = 'chinese-llama-2-7b-16k'
    chinese_llama_2_7b_64k = 'chinese-llama-2-7b-64k'
    chinese_llama_2_13b = 'chinese-llama-2-13b'
    chinese_llama_2_13b_16k = 'chinese-llama-2-13b-16k'
    chinese_alpaca_2_1_3b = 'chinese-alpaca-2-1_3b'
    chinese_alpaca_2_7b = 'chinese-alpaca-2-7b'
    chinese_alpaca_2_7b_16k = 'chinese-alpaca-2-7b-16k'
    chinese_alpaca_2_7b_64k = 'chinese-alpaca-2-7b-64k'
    chinese_alpaca_2_13b = 'chinese-alpaca-2-13b'
    chinese_alpaca_2_13b_16k = 'chinese-alpaca-2-13b-16k'
    llama_3_chinese_8b = 'llama-3-chinese-8b'
    llama_3_chinese_8b_instruct = 'llama-3-chinese-8b-instruct'

    # idefics
    idefics3_8b_llama3 = 'idefics3-8b-llama3'

    # atom
    atom_7b = 'atom-7b'
    atom_7b_chat = 'atom-7b-chat'

    # yi
    yi_6b = 'yi-6b'
    yi_6b_200k = 'yi-6b-200k'
    yi_6b_chat = 'yi-6b-chat'
    yi_6b_chat_awq = 'yi-6b-chat-awq'
    yi_6b_chat_int8 = 'yi-6b-chat-int8'
    yi_9b = 'yi-9b'
    yi_9b_200k = 'yi-9b-200k'
    yi_34b = 'yi-34b'
    yi_34b_200k = 'yi-34b-200k'
    yi_34b_chat = 'yi-34b-chat'
    yi_34b_chat_awq = 'yi-34b-chat-awq'
    yi_34b_chat_int8 = 'yi-34b-chat-int8'
    # yi1.5
    yi_1_5_6b = 'yi-1_5-6b'
    yi_1_5_6b_chat = 'yi-1_5-6b-chat'
    yi_1_5_9b = 'yi-1_5-9b'
    yi_1_5_9b_chat = 'yi-1_5-9b-chat'
    yi_1_5_9b_chat_16k = 'yi-1_5-9b-chat-16k'
    yi_1_5_34b = 'yi-1_5-34b'
    yi_1_5_34b_chat = 'yi-1_5-34b-chat'
    yi_1_5_34b_chat_16k = 'yi-1_5-34b-chat-16k'
    yi_1_5_6b_chat_awq_int4 = 'yi-1_5-6b-chat-awq-int4'
    yi_1_5_6b_chat_gptq_int4 = 'yi-1_5-6b-chat-gptq-int4'
    yi_1_5_9b_chat_awq_int4 = 'yi-1_5-9b-chat-awq-int4'
    yi_1_5_9b_chat_gptq_int4 = 'yi-1_5-9b-chat-gptq-int4'
    yi_1_5_34b_chat_awq_int4 = 'yi-1_5-34b-chat-awq-int4'
    yi_1_5_34b_chat_gptq_int4 = 'yi-1_5-34b-chat-gptq-int4'

    # yi-coder
    yi_coder_1_5b = 'yi-coder-1_5b'
    yi_coder_1_5b_chat = 'yi-coder-1_5b-chat'
    yi_coder_9b = 'yi-coder-9b'
    yi_coder_9b_chat = 'yi-coder-9b-chat'

    # internlm
    internlm_7b = 'internlm-7b'
    internlm_7b_chat = 'internlm-7b-chat'
    internlm_7b_chat_8k = 'internlm-7b-chat-8k'
    internlm_20b = 'internlm-20b'
    internlm_20b_chat = 'internlm-20b-chat'

    # internlm2
    internlm2_1_8b = 'internlm2-1_8b'
    internlm2_1_8b_sft_chat = 'internlm2-1_8b-sft-chat'
    internlm2_1_8b_chat = 'internlm2-1_8b-chat'
    internlm2_7b_base = 'internlm2-7b-base'
    internlm2_7b = 'internlm2-7b'
    internlm2_7b_sft_chat = 'internlm2-7b-sft-chat'
    internlm2_7b_chat = 'internlm2-7b-chat'
    internlm2_20b_base = 'internlm2-20b-base'
    internlm2_20b = 'internlm2-20b'
    internlm2_20b_sft_chat = 'internlm2-20b-sft-chat'
    internlm2_20b_chat = 'internlm2-20b-chat'

    # internlm2.5
    internlm2_5_1_8b = 'internlm2_5-1_8b'
    internlm2_5_1_8b_chat = 'internlm2_5-1_8b-chat'
    internlm2_5_7b = 'internlm2_5-7b'
    internlm2_5_7b_chat = 'internlm2_5-7b-chat'
    internlm2_5_7b_chat_1m = 'internlm2_5-7b-chat-1m'
    internlm2_5_20b = 'internlm2_5-20b'
    internlm2_5_20b_chat = 'internlm2_5-20b-chat'

    # internlm2-math
    internlm2_math_7b = 'internlm2-math-7b'
    internlm2_math_7b_chat = 'internlm2-math-7b-chat'
    internlm2_math_20b = 'internlm2-math-20b'
    internlm2_math_20b_chat = 'internlm2-math-20b-chat'

    # internlm-xcomposer2
    internlm_xcomposer2_7b_chat = 'internlm-xcomposer2-7b-chat'
    internlm_xcomposer2_4khd_7b_chat = 'internlm-xcomposer2-4khd-7b-chat'
    internlm_xcomposer2_5_7b_chat = 'internlm-xcomposer2_5-7b-chat'

    # deepseek
    deepseek_7b = 'deepseek-7b'
    deepseek_7b_chat = 'deepseek-7b-chat'
    deepseek_moe_16b = 'deepseek-moe-16b'
    deepseek_moe_16b_chat = 'deepseek-moe-16b-chat'
    deepseek_67b = 'deepseek-67b'
    deepseek_67b_chat = 'deepseek-67b-chat'

    # deepseek-coder
    deepseek_coder_1_3b = 'deepseek-coder-1_3b'
    deepseek_coder_1_3b_instruct = 'deepseek-coder-1_3b-instruct'
    deepseek_coder_6_7b = 'deepseek-coder-6_7b'
    deepseek_coder_6_7b_instruct = 'deepseek-coder-6_7b-instruct'
    deepseek_coder_33b = 'deepseek-coder-33b'
    deepseek_coder_33b_instruct = 'deepseek-coder-33b-instruct'

    # deepseek2-coder
    deepseek_coder_v2_instruct = 'deepseek-coder-v2-instruct'
    deepseek_coder_v2_lite_instruct = 'deepseek-coder-v2-lite-instruct'
    deepseek_coder_v2 = 'deepseek-coder-v2'
    deepseek_coder_v2_lite = 'deepseek-coder-v2-lite'

    # deepseek-math
    deepseek_math_7b = 'deepseek-math-7b'
    deepseek_math_7b_instruct = 'deepseek-math-7b-instruct'
    deepseek_math_7b_chat = 'deepseek-math-7b-chat'

    # numina-math
    numina_math_7b = 'numina-math-7b'

    # gemma
    gemma_2b = 'gemma-2b'
    gemma_7b = 'gemma-7b'
    gemma_2b_instruct = 'gemma-2b-instruct'
    gemma_7b_instruct = 'gemma-7b-instruct'
    gemma2_2b = 'gemma2-2b'
    gemma2_9b = 'gemma2-9b'
    gemma2_27b = 'gemma2-27b'
    gemma2_2b_instruct = 'gemma2-2b-instruct'
    gemma2_9b_instruct = 'gemma2-9b-instruct'
    gemma2_27b_instruct = 'gemma2-27b-instruct'

    # minicpm
    minicpm_1b_sft_chat = 'minicpm-1b-sft-chat'
    minicpm_2b_sft_chat = 'minicpm-2b-sft-chat'
    minicpm_2b_chat = 'minicpm-2b-chat'
    minicpm_2b_128k = 'minicpm-2b-128k'
    minicpm_moe_8x2b = 'minicpm-moe-8x2b'
    minicpm3_4b = 'minicpm3-4b'

    # openbuddy
    openbuddy_llama_65b_chat = 'openbuddy-llama-65b-chat'
    openbuddy_llama2_13b_chat = 'openbuddy-llama2-13b-chat'
    openbuddy_llama2_70b_chat = 'openbuddy-llama2-70b-chat'
    openbuddy_llama3_8b_chat = 'openbuddy-llama3-8b-chat'
    openbuddy_llama3_70b_chat = 'openbuddy-llama3-70b-chat'
    openbuddy_mistral_7b_chat = 'openbuddy-mistral-7b-chat'
    openbuddy_zephyr_7b_chat = 'openbuddy-zephyr-7b-chat'
    openbuddy_deepseek_67b_chat = 'openbuddy-deepseek-67b-chat'
    openbuddy_mixtral_moe_7b_chat = 'openbuddy-mixtral-moe-7b-chat'
    openbuddy_llama3_1_8b_chat = 'openbuddy-llama3_1-8b-chat'

    # mistral
    mistral_7b = 'mistral-7b'
    mistral_7b_v2 = 'mistral-7b-v2'
    mistral_7b_instruct = 'mistral-7b-instruct'
    mistral_7b_instruct_v2 = 'mistral-7b-instruct-v2'
    mistral_7b_instruct_v3 = 'mistral-7b-instruct-v3'
    mistral_nemo_base_2407 = 'mistral-nemo-base-2407'
    mistral_nemo_instruct_2407 = 'mistral-nemo-instruct-2407'
    mistral_large_instruct_2407 = 'mistral-large-instruct-2407'
    mixtral_moe_7b = 'mixtral-moe-7b'
    mixtral_moe_7b_instruct = 'mixtral-moe-7b-instruct'
    mixtral_moe_7b_aqlm_2bit_1x16 = 'mixtral-moe-7b-aqlm-2bit-1x16'  # aqlm
    mixtral_moe_8x22b_v1 = 'mixtral-moe-8x22b-v1'

    # wizardlm
    wizardlm2_7b_awq = 'wizardlm2-7b-awq'
    wizardlm2_8x22b = 'wizardlm2-8x22b'

    # baichuan
    baichuan_7b = 'baichuan-7b'
    baichuan_13b = 'baichuan-13b'
    baichuan_13b_chat = 'baichuan-13b-chat'

    # baichuan2
    baichuan2_7b = 'baichuan2-7b'
    baichuan2_7b_chat = 'baichuan2-7b-chat'
    baichuan2_7b_chat_int4 = 'baichuan2-7b-chat-int4'
    baichuan2_13b = 'baichuan2-13b'
    baichuan2_13b_chat = 'baichuan2-13b-chat'
    baichuan2_13b_chat_int4 = 'baichuan2-13b-chat-int4'

    # yuan
    yuan2_2b_instruct = 'yuan2-2b-instruct'
    yuan2_2b_janus_instruct = 'yuan2-2b-janus-instruct'
    yuan2_51b_instruct = 'yuan2-51b-instruct'
    yuan2_102b_instruct = 'yuan2-102b-instruct'
    yuan2_m32 = 'yuan2-m32'

    # xverse
    xverse_7b = 'xverse-7b'
    xverse_7b_chat = 'xverse-7b-chat'
    xverse_13b = 'xverse-13b'
    xverse_13b_chat = 'xverse-13b-chat'
    xverse_65b = 'xverse-65b'
    xverse_65b_v2 = 'xverse-65b-v2'
    xverse_65b_chat = 'xverse-65b-chat'
    xverse_13b_256k = 'xverse-13b-256k'
    xverse_moe_a4_2b = 'xverse-moe-a4_2b'

    # orion
    orion_14b = 'orion-14b'
    orion_14b_chat = 'orion-14b-chat'

    # vivo
    bluelm_7b = 'bluelm-7b'
    bluelm_7b_32k = 'bluelm-7b-32k'
    bluelm_7b_chat = 'bluelm-7b-chat'
    bluelm_7b_chat_32k = 'bluelm-7b-chat-32k'

    # ziya
    ziya2_13b = 'ziya2-13b'
    ziya2_13b_chat = 'ziya2-13b-chat'

    # skywork
    skywork_13b = 'skywork-13b'
    skywork_13b_chat = 'skywork-13b-chat'

    # zephyr
    zephyr_7b_beta_chat = 'zephyr-7b-beta-chat'

    # other
    polylm_13b = 'polylm-13b'
    seqgpt_560m = 'seqgpt-560m'
    sus_34b_chat = 'sus-34b-chat'

    # tongyi-finance
    tongyi_finance_14b = 'tongyi-finance-14b'
    tongyi_finance_14b_chat = 'tongyi-finance-14b-chat'
    tongyi_finance_14b_chat_int4 = 'tongyi-finance-14b-chat-int4'

    # codefuse
    codefuse_codellama_34b_chat = 'codefuse-codellama-34b-chat'
    codefuse_codegeex2_6b_chat = 'codefuse-codegeex2-6b-chat'
    codefuse_qwen_14b_chat = 'codefuse-qwen-14b-chat'

    # phi
    phi2_3b = 'phi2-3b'
    phi3_4b_4k_instruct = 'phi3-4b-4k-instruct'
    phi3_4b_128k_instruct = 'phi3-4b-128k-instruct'
    phi3_small_8k_instruct = 'phi3-small-8k-instruct'
    phi3_medium_4k_instruct = 'phi3-medium-4k-instruct'
    phi3_small_128k_instruct = 'phi3-small-128k-instruct'
    phi3_medium_128k_instruct = 'phi3-medium-128k-instruct'

    phi3_5_mini_instruct = 'phi3_5-mini-instruct'
    phi3_5_moe_instruct = 'phi3_5-moe-instruct'

    # mamba
    mamba_130m = 'mamba-130m'
    mamba_370m = 'mamba-370m'
    mamba_390m = 'mamba-390m'
    mamba_790m = 'mamba-790m'
    mamba_1_4b = 'mamba-1.4b'
    mamba_2_8b = 'mamba-2.8b'

    # teleAI
    telechat_7b = 'telechat-7b'
    telechat_12b = 'telechat-12b'
    telechat_12b_v2 = 'telechat-12b-v2'
    telechat_12b_v2_gptq_int4 = 'telechat-12b-v2-gptq-int4'

    # grok-1
    grok_1 = 'grok-1'

    # dbrx
    dbrx_instruct = 'dbrx-instruct'
    dbrx_base = 'dbrx-base'

    # mengzi
    mengzi3_13b_base = 'mengzi3-13b-base'

    # c4ai
    c4ai_command_r_v01 = 'c4ai-command-r-v01'
    c4ai_command_r_plus = 'c4ai-command-r-plus'

    # codestral
    codestral_22b = 'codestral-22b'


class ModelType(NLPModelType, MMModelType):

    @classmethod
    def get_model_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__') or k == 'get_model_name_list':
                continue
            res.append(cls.__dict__[k])
        return res
