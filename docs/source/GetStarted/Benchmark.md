# Benchmark
本页面主要介绍了各个方法, 模型之间对比的Benchmark. 以及数据集的相关信息.

## Method



## Model
下表介绍了swift介入的模型的相关信息.
- Model List: 模型在swift中注册的model_name的列表.
- Default Lora Target Modules: 对应模型的默认lora_target_modules.
- Default Template: 对应模型的默认template.
- Requires: 对应模型所需的额外依赖要求.

| Model List | Default Lora Target Modules | Default Template | Requires |
| ---------  | --------------------------- | ---------------- | -------- |
|qwen-7b, qwen-14b, qwen-vl, polylm-13b|c_attn|default-generation|-|
|qwen-7b-chat, qwen-14b-chat, qwen-vl-chat|c_attn|chatml|-|
|qwen-7b-chat-int4, qwen-14b-chat-int4, qwen-7b-chat-int8, qwen-14b-chat-int8, qwen-vl-chat-int4|c_attn|chatml|auto_gptq>=0.4.2|
|baichuan-7b, baichuan-13b|W_pack|default-generation|transformers<4.34|
|baichuan-13b-chat|W_pack|baichuan|transformers<4.34|
|baichuan2-7b, baichuan2-13b|W_pack|default-generation|-|
|baichuan2-7b-chat, baichuan2-13b-chat, baichuan2-7b-chat-int4, baichuan2-13b-chat-int4|W_pack|baichuan|-|
|chatglm2-6b, chatglm2-6b-32k|query_key_value|chatglm2|-|
|chatglm3-6b-base|query_key_value|chatglm-generation|-|
|chatglm3-6b, chatglm3-6b-32k|query_key_value|chatglm3|-|
|llama2-7b, llama2-13b, llama2-70b, internlm-7b, internlm-20b, xverse-7b, xverse-13b, xverse-65b, bluelm-7b, bluelm-7b-32k, yi-6b, yi-34b, ziya2-13b, skywork-13b|q_proj, k_proj, v_proj|default-generation|-|
|llama2-7b-chat, llama2-13b-chat, llama2-70b-chat|q_proj, k_proj, v_proj|llama|-|
|openbuddy-llama2-13b-chat, openbuddy-llama-65b-chat, openbuddy-llama2-70b-chat|q_proj, k_proj, v_proj|openbuddy|-|
|openbuddy-mistral-7b-chat|q_proj, k_proj, v_proj|openbuddy|transformers>=4.34|
|internlm-7b-chat, internlm-7b-chat-8k, internlm-20b-chat|q_proj, k_proj, v_proj|internlm|-|
|xverse-7b-chat, xverse-13b-chat|q_proj, k_proj, v_proj|xverse|-|
|bluelm-7b-chat, bluelm-7b-chat-32k|q_proj, k_proj, v_proj|bluelm|-|
|mistral-7b|q_proj, k_proj, v_proj|default-generation|transformers>=4.34|
|mistral-7b-chat|q_proj, k_proj, v_proj|llama|transformers>=4.34|
|ziya2-13b-chat|q_proj, k_proj, v_proj|ziya|-|
|skywork-13b-chat|q_proj, k_proj, v_proj|skywork|-|
|seqgpt-560m|query_key_value|default-generation|-|

## Dataset
下表介绍了swift接入的数据集的相关信息.
- Dataset Name: 数据集在swift中注册的dataset_name.
- Dataset ID: 数据集在[ModelScope](https://www.modelscope.cn/my/overview)上的dataset_id.
- Size: 数据集中的数据样本数量.
- Statistic: 数据集的统计量. 我们使用token数进行统计, 而不是字符数, 这对于调整`max_length`超参数有帮助. 我们将数据集的训练集和验证集进行拼接, 然后进行统计. 我们使用qwen的tokenizer对数据集进行分词. 不同的tokenizer的统计量不同, 如果你要获取其他的模型的tokenizer的token统计量, 可以通过[脚本](https://github.com/modelscope/swift/tree/main/benchmark/run_dataset.py)自行获取.

| Dataset Name | Dataset ID | Size | Statistic (token) |
| ------------ | ---------- | ---- | ----------------- |
|alpaca-en|[AI-ModelScope/alpaca-gpt4-data-en](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-en/summary)|52002|178.2±125.8, min=28, max=742|
|alpaca-zh|[AI-ModelScope/alpaca-gpt4-data-zh](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-zh/summary)|48818|164.1±93.9, min=28, max=858|
|multi-alpaca-all|[damo/nlp_polylm_multialpaca_sft](https://modelscope.cn/datasets/damo/nlp_polylm_multialpaca_sft/summary)|131867|114.9±50.6, min=28, max=1228|
|instinwild-en|[wyj123456/instinwild](https://modelscope.cn/datasets/wyj123456/instinwild/summary)|52191|162.2±69.7, min=35, max=765|
|instinwild-zh|[wyj123456/instinwild](https://modelscope.cn/datasets/wyj123456/instinwild/summary)|51504|132.3±45.1, min=30, max=1436|
|cot-en|[YorickHe/CoT](https://modelscope.cn/datasets/YorickHe/CoT/summary)|74771|124.7±64.8, min=53, max=8322|
|cot-zh|[YorickHe/CoT_zh](https://modelscope.cn/datasets/YorickHe/CoT_zh/summary)|74771|119.5±70.8, min=45, max=9638|
|firefly-all-zh|[wyj123456/firefly](https://modelscope.cn/datasets/wyj123456/firefly/summary)|1649399|180.1±260.4, min=28, max=12518|
|instruct-en|[wyj123456/instruct](https://modelscope.cn/datasets/wyj123456/instruct/summary)|888970|270.9±331.2, min=28, max=7254|
|gpt4all-en|[wyj123456/GPT4all](https://modelscope.cn/datasets/wyj123456/GPT4all/summary)|806199|304.5±384.1, min=29, max=7393|
|sharegpt-en|[huangjintao/sharegpt](https://modelscope.cn/datasets/huangjintao/sharegpt/summary)|99799|1047.7±431.9, min=24, max=7909|
|sharegpt_zh|[huangjintao/sharegpt](https://modelscope.cn/datasets/huangjintao/sharegpt/summary)|135399|808.3±771.7, min=23, max=65320|
|damo-agent-zh|[damo/MSAgent-Bench](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary)|422276|967.7±440.9, min=323, max=31537|
|damo-agent-mini-zh|[damo/MSAgent-Bench](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary)|40116|1232.9±350.1, min=560, max=4984|
|agent-instruct-all-en|[huangjintao/AgentInstruct_copy](https://modelscope.cn/datasets/huangjintao/AgentInstruct_copy/summary)|1866|1146.3±635.5, min=208, max=6414|
|code-alpaca-en|[wyj123456/code_alpaca_en](https://modelscope.cn/datasets/wyj123456/code_alpaca_en/summary)|20016|102.1±60.1, min=31, max=1778|
|code-python-zh|[codefuse-ai/CodeExercise-Python-27k](https://modelscope.cn/datasets/codefuse-ai/CodeExercise-Python-27k/summary)|27224|485.6±193.9, min=47, max=3084|
|leetcode-python-en|[AI-ModelScope/leetcode-solutions-python](https://modelscope.cn/datasets/AI-ModelScope/leetcode-solutions-python/summary)|2359|725.8±233.5, min=261, max=2119|
|medical-en|[huangjintao/medical_zh](https://modelscope.cn/datasets/huangjintao/medical_zh/summary)|117617|259.4±89.1, min=38, max=2566|
|medical-zh|[huangjintao/medical_zh](https://modelscope.cn/datasets/huangjintao/medical_zh/summary)|1950972|169.2±219.7, min=28, max=27353|
|medical-mini-zh|[huangjintao/medical_zh](https://modelscope.cn/datasets/huangjintao/medical_zh/summary)|100500|169.8±219.0, min=28, max=15868|
|lawyer-llama-zh|[AI-ModelScope/lawyer_llama_data](https://modelscope.cn/datasets/AI-ModelScope/lawyer_llama_data/summary)|21476|196.4±91.7, min=29, max=926|
|tigerbot-law-zh|[AI-ModelScope/tigerbot-law-plugin](https://modelscope.cn/datasets/AI-ModelScope/tigerbot-law-plugin/summary)|55895|111.9±126.4, min=39, max=18880|
|blossom-math-zh|[AI-ModelScope/blossom-math-v2](https://modelscope.cn/datasets/AI-ModelScope/blossom-math-v2/summary)|10000|171.3±58.7, min=37, max=565|
|school-math-zh|[AI-ModelScope/school_math_0.25M](https://modelscope.cn/datasets/AI-ModelScope/school_math_0.25M/summary)|248480|159.6±72.1, min=35, max=3452|
|text2sql-en|[AI-ModelScope/texttosqlv2_25000_v2](https://modelscope.cn/datasets/AI-ModelScope/texttosqlv2_25000_v2/summary)|25000|276.6±326.4, min=40, max=1977|
|sql-create-context-en|[AI-ModelScope/sql-create-context](https://modelscope.cn/datasets/AI-ModelScope/sql-create-context/summary)|78577|82.2±17.8, min=38, max=458|
|advertise-gen-zh|[lvjianjin/AdvertiseGen](https://modelscope.cn/datasets/lvjianjin/AdvertiseGen/summary)|98399|133.6±21.7, min=54, max=244|
|dureader-robust-zh|[modelscope/DuReader_robust-QG](https://modelscope.cn/datasets/modelscope/DuReader_robust-QG/summary)|17899|244.1±137.4, min=63, max=1419|
|cmnli-zh|[clue](https://modelscope.cn/datasets/clue/summary)|417904|85.6±16.6, min=54, max=202|
|jd-sentiment-zh|[DAMO_NLP/jd](https://modelscope.cn/datasets/DAMO_NLP/jd/summary)|50000|69.0±83.2, min=42, max=4042|
|finance-en|[wyj123456/finance_en](https://modelscope.cn/datasets/wyj123456/finance_en/summary)|68911|137.6±134.3, min=28, max=3527|
|poetry-zh|[modelscope/chinese-poetry-collection](https://modelscope.cn/datasets/modelscope/chinese-poetry-collection/summary)|390309|57.2±9.4, min=25, max=85|
|cls-fudan-news-zh|[damo/zh_cls_fudan-news](https://modelscope.cn/datasets/damo/zh_cls_fudan-news/summary)|4959|3236.4±2547.5, min=93, max=19550|
|ner-jave-zh|[damo/zh_ner-JAVE](https://modelscope.cn/datasets/damo/zh_ner-JAVE/summary)|1266|120.3±45.5, min=46, max=225|
|coco-en|[modelscope/coco_2014_caption](https://modelscope.cn/datasets/modelscope/coco_2014_caption/summary)|454617|94.9±2.8, min=90, max=147|
