# Supported models and datasets
## Table of Contents
- [Models](#Models)
  - [LLM](#LLM)
  - [MLLM](#MLLM)
- [Datasets](#Datasets)

## Models
The table below introcudes all models supported by SWIFT:
- Model List: The model_type information registered in SWIFT.
- Default Lora Target Modules: Default lora_target_modules used by the model.
- Default Template: Default template used by the model.
- Support Flash Attn: Whether the model supports [flash attention](https://github.com/Dao-AILab/flash-attention) to accelerate sft and infer.
- Support VLLM: Whether the model supports [vllm](https://github.com/vllm-project/vllm) to accelerate infer and deployment.
- Requires: The extra requirements used by the model.


### LLM
| Model ID | HF Model ID | Model ID | HF Model ID | Model Type | Architectures | Default Template(for sft) |  |Requires | Tags |
| -------- | ----------- | ----------- | ------------------------- | ------------------ | ------------ | ---------------- | ---------------- | -------- | ---- |


### MLLM
| Model ID | HF Model ID | Model ID | HF Model ID | Model Type | Architectures | Default Template(for sft) |  |Requires | Tags |
| -------- | ----------- | ----------- | ------------------------- | ------------------ | ------------ | ---------------- | ---------------- | -------- | ---- |


## Datasets
The table below introduces the datasets supported by SWIFT:
- Dataset Name: The dataset name registered in SWIFT.
- Dataset ID: The dataset id in [ModelScope](https://www.modelscope.cn/my/overview).
- Size: The data row count of the dataset.
- Statistic: Dataset statistics. We use the number of tokens for statistics, which helps adjust the max_length hyperparameter. We concatenate the training and validation sets of the dataset and then compute the statistics. We use qwen's tokenizer to tokenize the dataset. Different tokenizers produce different statistics. If you want to obtain token statistics for tokenizers of other models, you can use the script to get them yourself.

| MS Dataset ID | HF Dataset ID | Subset name | Real Subset  | Subset split | Dataset Size | Statistic (token) | Tags |
| ------------ | ------------- | ----------- |------------- | -------------| -------------| ----------------- | ---- |
|None|lmms-lab/GQA|default|default|train_all_instructions|-|Dataset is too huge, please click the original link to view the dataset stat.|multi-modal, en, vqa, quality|
|None|HuggingFaceTB/cosmopedia|auto_math_text,khanacademy,openstax,stanford,stories,web_samples_v1,web_samples_v2,wikihow|auto_math_text,khanacademy,openstax,stanford,stories,web_samples_v1,web_samples_v2,wikihow|train|-|Dataset is too huge, please click the original link to view the dataset stat.|multi-domain, en, qa|
|None|HuggingFaceFW/fineweb|default|default|train|-|Dataset is too huge, please click the original link to view the dataset stat.|pretrain, quality|
|None|allenai/c4|default|default|train|-|Dataset is too huge, please click the original link to view the dataset stat.|pretrain, quality|
|None|tiiuae/falcon-refinedweb|default|default|train|-|Dataset is too huge, please click the original link to view the dataset stat.|pretrain, quality|
|None|cerebras/SlimPajama-627B|default|default|train|-|Dataset is too huge, please click the original link to view the dataset stat.|pretrain, quality|
|AI-ModelScope/COIG-CQIA|None|chinese_traditional,coig_pc,exam,finance,douban,human_value,logi_qa,ruozhiba,segmentfault,wiki,wikihow,xhs,zhihu|chinese_traditional,coig_pc,exam,finance,douban,human_value,logi_qa,ruozhiba,segmentfault,wiki,wikihow,xhs,zhihu|train|44694|331.2±693.8, min=34, max=19288|general, 🔥|
|AI-ModelScope/CodeAlpaca-20k|HuggingFaceH4/CodeAlpaca_20K|default|default|train|20022|99.3±57.6, min=30, max=857|code, en|
|AI-ModelScope/DISC-Law-SFT|ShengbinYue/DISC-Law-SFT|default|default|train|166758|1799.0±474.9, min=769, max=3151|chat, law, 🔥|
|AI-ModelScope/DISC-Med-SFT|Flmc/DISC-Med-SFT|default|default|train|464885|426.5±178.7, min=110, max=1383|chat, medical, 🔥|
|AI-ModelScope/Duet-v0.5|G-reen/Duet-v0.5|default|default|train|5000|1157.4±189.3, min=657, max=2344|CoT, en|
|AI-ModelScope/GuanacoDataset|JosephusCheung/GuanacoDataset|default|default|train|31563|250.3±70.6, min=95, max=987|chat, zh|
|AI-ModelScope/LLaVA-Instruct-150K|None|default|default|train|623302|630.7±143.0, min=301, max=1166|chat, multi-modal, vision|
|AI-ModelScope/LLaVA-Pretrain|liuhaotian/LLaVA-Pretrain|default|default|train|-|Dataset is too huge, please click the original link to view the dataset stat.|chat, multi-modal, quality|
|AI-ModelScope/LaTeX_OCR|linxy/LaTeX_OCR|default,synthetic_handwrite|default,synthetic_handwrite|train|162149|117.6±44.9, min=41, max=312|chat, ocr, multi-modal, vision|
|AI-ModelScope/LongAlpaca-12k|Yukang/LongAlpaca-12k|default|default|train|11998|9941.8±3417.1, min=4695, max=25826|long-sequence, QA|
|AI-ModelScope/M3IT|None|coco,vqa-v2,shapes,shapes-rephrased,coco-goi-rephrased,snli-ve,snli-ve-rephrased,okvqa,a-okvqa,viquae,textcap,docvqa,science-qa,imagenet,imagenet-open-ended,imagenet-rephrased,coco-goi,clevr,clevr-rephrased,nlvr,coco-itm,coco-itm-rephrased,vsr,vsr-rephrased,mocheg,mocheg-rephrased,coco-text,fm-iqa,activitynet-qa,msrvtt,ss,coco-cn,refcoco,refcoco-rephrased,multi30k,image-paragraph-captioning,visual-dialog,visual-dialog-rephrased,iqa,vcr,visual-mrc,ivqa,msrvtt-qa,msvd-qa,gqa,text-vqa,ocr-vqa,st-vqa,flickr8k-cn|coco,vqa-v2,shapes,shapes-rephrased,coco-goi-rephrased,snli-ve,snli-ve-rephrased,okvqa,a-okvqa,viquae,textcap,docvqa,science-qa,imagenet,imagenet-open-ended,imagenet-rephrased,coco-goi,clevr,clevr-rephrased,nlvr,coco-itm,coco-itm-rephrased,vsr,vsr-rephrased,mocheg,mocheg-rephrased,coco-text,fm-iqa,activitynet-qa,msrvtt,ss,coco-cn,refcoco,refcoco-rephrased,multi30k,image-paragraph-captioning,visual-dialog,visual-dialog-rephrased,iqa,vcr,visual-mrc,ivqa,msrvtt-qa,msvd-qa,gqa,text-vqa,ocr-vqa,st-vqa,flickr8k-cn|train|-|Dataset is too huge, please click the original link to view the dataset stat.|chat, multi-modal, vision|
|AI-ModelScope/Magpie-Qwen2-Pro-200K-Chinese|Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese|default|default|train|200000|448.4±223.5, min=87, max=4098|chat, sft, 🔥, zh|
|AI-ModelScope/Magpie-Qwen2-Pro-200K-English|Magpie-Align/Magpie-Qwen2-Pro-200K-English|default|default|train|200000|609.9±277.1, min=257, max=4098|chat, sft, 🔥, en|
|AI-ModelScope/Magpie-Qwen2-Pro-300K-Filtered|Magpie-Align/Magpie-Qwen2-Pro-300K-Filtered|default|default|train|300000|556.6±288.6, min=175, max=4098|chat, sft, 🔥|
|AI-ModelScope/MathInstruct|TIGER-Lab/MathInstruct|default|default|train|262040|253.3±177.4, min=42, max=2193|math, cot, en, quality|
|AI-ModelScope/Open-Platypus|garage-bAInd/Open-Platypus|default|default|train|24926|389.0±256.4, min=55, max=3153|chat, math, quality|
|AI-ModelScope/OpenOrca|None|default,3_5M|default,3_5M|train|-|Dataset is too huge, please click the original link to view the dataset stat.|chat, multilingual, general|
|AI-ModelScope/OpenOrca-Chinese|yys/OpenOrca-Chinese|default|default|train|-|Dataset is too huge, please click the original link to view the dataset stat.|QA, zh, general, quality|
|AI-ModelScope/SFT-Nectar|AstraMindAI/SFT-Nectar|default|default|train|131201|441.9±307.0, min=45, max=3136|cot, en, quality|
|AI-ModelScope/ShareGPT-4o|OpenGVLab/ShareGPT-4o|image_caption|image_caption|images|57289|599.8±140.4, min=214, max=1932|vqa, multi-modal|
|AI-ModelScope/ShareGPT4V|None|default|default|train|-|Dataset is too huge, please click the original link to view the dataset stat.|chat, multi-modal, vision|
|AI-ModelScope/SkyPile-150B|Skywork/SkyPile-150B|default|default|train|-|Dataset is too huge, please click the original link to view the dataset stat.|pretrain, quality, zh|
|AI-ModelScope/WizardLM_evol_instruct_V2_196k|WizardLM/WizardLM_evol_instruct_V2_196k|default|default|train|109184|483.3±338.4, min=27, max=3735|chat, en|
|AI-ModelScope/alpaca-cleaned|yahma/alpaca-cleaned|default|default|train|51760|170.1±122.9, min=29, max=1028|chat, general, bench, quality|
|AI-ModelScope/alpaca-gpt4-data-en|vicgalle/alpaca-gpt4|default|default|train|52002|167.6±123.9, min=29, max=607|chat, general, 🔥|
|AI-ModelScope/alpaca-gpt4-data-zh|llm-wizard/alpaca-gpt4-data-zh|default|default|train|48818|157.2±93.2, min=27, max=544|chat, general, 🔥|
|AI-ModelScope/blossom-math-v2|Azure99/blossom-math-v2|default|default|train|10000|175.4±59.1, min=35, max=563|chat, math, 🔥|
|AI-ModelScope/captcha-images|None|default|default|train,validation|8000|47.0±0.0, min=47, max=47|chat, multi-modal, vision|
|AI-ModelScope/databricks-dolly-15k|databricks/databricks-dolly-15k|default|default|train|15011|199.0±268.8, min=26, max=5987|multi-task, en, quality|
|AI-ModelScope/deepctrl-sft-data|None|default,en|default,en|train|-|Dataset is too huge, please click the original link to view the dataset stat.|chat, general, sft, multi-round|
|AI-ModelScope/firefly-train-1.1M|YeungNLP/firefly-train-1.1M|default|default|train|1649399|204.3±365.3, min=28, max=9306|chat, general|
|AI-ModelScope/generated_chat_0.4M|BelleGroup/generated_chat_0.4M|default|default|train|396004|272.7±51.1, min=78, max=579|chat, character-dialogue|
|AI-ModelScope/guanaco_belle_merge_v1.0|Chinese-Vicuna/guanaco_belle_merge_v1.0|default|default|train|693987|133.8±93.5, min=30, max=1872|QA, zh|
|AI-ModelScope/hh_rlhf_cn|None|hh_rlhf,harmless_base_cn,harmless_base_en,helpful_base_cn,helpful_base_en|hh_rlhf,harmless_base_cn,harmless_base_en,helpful_base_cn,helpful_base_en|train,test|362909|142.3±107.5, min=25, max=1571|rlhf, dpo, pairwise, 🔥|
|AI-ModelScope/lawyer_llama_data|Skepsun/lawyer_llama_data|default|default|train|21476|224.4±83.9, min=69, max=832|chat, law|
|AI-ModelScope/leetcode-solutions-python|None|default|default|train|2359|723.8±233.5, min=259, max=2117|chat, coding, 🔥|
|AI-ModelScope/lmsys-chat-1m|lmsys/lmsys-chat-1m|default|default|train|166211|545.8±3272.8, min=22, max=219116|chat, em|
|AI-ModelScope/ms_agent_for_agentfabric|None|default|default|train|30000|615.7±198.7, min=251, max=2055|chat, agent, multi-round, 🔥|
