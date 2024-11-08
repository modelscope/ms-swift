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
|None|cerebras/SlimPajama-627B|default|default|train|-|Dataset is too huge, please click the original link to view the dataset stat.|pretrain, quality|
|None|HuggingFaceFW/fineweb|default|default|train|-|Dataset is too huge, please click the original link to view the dataset stat.|pretrain, quality|
|None|HuggingFaceTB/cosmopedia|auto_math_text,khanacademy,openstax,stanford,stories,web_samples_v1,web_samples_v2,wikihow|auto_math_text,khanacademy,openstax,stanford,stories,web_samples_v1,web_samples_v2,wikihow|train|-|Dataset is too huge, please click the original link to view the dataset stat.|multi-domain, en, qa|
|None|allenai/c4|default|default|train|-|Dataset is too huge, please click the original link to view the dataset stat.|pretrain, quality|
|None|tiiuae/falcon-refinedweb|default|default|train|-|Dataset is too huge, please click the original link to view the dataset stat.|pretrain, quality|
|AI-ModelScope/COIG-CQIA|None|chinese_traditional,coig_pc,exam,finance,douban,human_value,logi_qa,ruozhiba,segmentfault,wiki,wikihow,xhs,zhihu|chinese_traditional,coig_pc,exam,finance,douban,human_value,logi_qa,ruozhiba,segmentfault,wiki,wikihow,xhs,zhihu|train|44694|331.2±693.8, min=34, max=19288|general, 🔥|
|AI-ModelScope/CodeAlpaca-20k|HuggingFaceH4/CodeAlpaca_20K|default|default|train|20022|99.3±57.6, min=30, max=857|code, en|
|AI-ModelScope/DISC-Law-SFT|ShengbinYue/DISC-Law-SFT|default|default|train|166758|1799.0±474.9, min=769, max=3151|chat, law, 🔥|
|AI-ModelScope/DISC-Med-SFT|Flmc/DISC-Med-SFT|default|default|train|464885|426.5±178.7, min=110, max=1383|chat, medical, 🔥|
|AI-ModelScope/Duet-v0.5|G-reen/Duet-v0.5|default|default|train|5000|1157.4±189.3, min=657, max=2344|CoT, en|
|AI-ModelScope/GuanacoDataset|JosephusCheung/GuanacoDataset|default|default|train|31563|250.3±70.6, min=95, max=987|chat, zh|
