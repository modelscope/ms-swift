<h1 align="center">LLM SFT Example</h1>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.9.3-5D91D4.svg"></a>
<a href="https://github.com/modelscope/swift/"><img src="https://img.shields.io/badge/ms--swift-Build from source-6FEBB9.svg"></a>
</p>


<p align="center">
<a href="https://modelscope.cn/home">ModelScope Hub</a>
<br>
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish
</p>


## 🎉 News
- 🔥 2023.10.15: Support for **bluelm** series models: bluelm-7b, bluelm-7b-chat, bluelm-7b-32k, bluelm-7b-chat-32k. The corresponding shell script can be found in `scripts/bluelm_7b_chat`.
- 2023.10.31: Support Web UI. Run command: python app.py.
- 2023.10.30: Support for **skywork-13b** series models: skywork-13b, skywork-13b-chat. The corresponding shell script can be found in `scripts/skywork_13b`.
- 🔥 2023.10.27: Support for **chatglm3** series models: chatglm3-6b-base, chatglm3-6b, chatglm3-6b-32k. The corresponding shell script can be found in `scripts/chatglm3_6b`.
- 🔥 2023.10.24: Use the **registration mechanism** to add models, **datasets**, and chat templates. To customize models, datasets, and chat templates, refer to the "User Guide" section. The corresponding Python file can be found in `custom.py`, and the corresponding shell script can be found in `scripts/custom`.
- 🔥 2023.10.17: Supported **int4, int8** models: qwen-7b-chat-int4, qwen-14b-chat-int4, qwen-vl-chat-int4, baichuan2-7b-chat-int4, baichuan2-13b-chat-int4, qwen-7b-chat-int8, qwen-14b-chat-int8. The corresponding shell script can be found at `scripts/qwen_7b_chat_int4`, `scripts/qwen_14b_chat_int4`, `scripts/qwen_vl_chat_int4`, `scripts/qwen_7b_chat_int8`, `scripts/qwen_14b_chat_int8`.
- 2023.10.15: Supported **ziya2-13b** model series: ziya2-13b, ziya2-13b-chat. The corresponding shell script can be found at `scripts/ziya2_13b_chat`.
- 2023.10.12: Supported **mistral-7b** model series: openbuddy-mistral-7b-chat, mistral-7b, mistral-7b-chat. The corresponding shell script can be found at `scripts/openbuddy_mistral_7b_chat`, `scripts/mistral_7b_chat`.
- 🔥 2023.10.7: Supported **DeepSpeed ZeRO-2**, enabling LoRA (not just QLoRA) to run DDP on 2*A10. The corresponding shell script can be found at `scripts/qwen_7b_chat/lora_ddp_ds/sft.sh`.
- 2023.10.4: Supported datasets in the fields of mathematics, law, SQL, and coding: blossom-math-zh, school-math-zh, text2sql-en, sql-create-context-en, lawyer-llama-zh, tigerbot-law-zh, leetcode-python-en.
- 🔥 2023.9.25: Supported **qwen-14b** model series: qwen-14b, qwen-14b-chat. The corresponding shell script can be found at `scripts/qwen_14b`, `scripts/qwen_14b_chat`.
- 2023.9.18: Supported **internlm-20b** model series: internlm-20b, internlm-20b-chat. The corresponding shell script can be found at `scripts/internlm_20b`, `scripts/internlm_20b_chat`.
- 2023.9.12: Supported training with **MP+DDP** to accelerate full-parameter fine-tuning speed. The corresponding shell script can be found at `scripts/qwen_7b_chat/full_mp_ddp/sft.sh`.
- 2023.9.5: Supported training that only saves model weights without saving intermediate states such as optimizer weights required for checkpoint resumption, avoiding long checkpoint-saving times and large storage space in full-parameter fine-tuning. You can check the command-line parameter `--only_save_model` in the `sft.sh` script.
- 2023.9.5: Supported **openbuddy-llama2-70b-chat** model. The corresponding shell script can be found at `scripts/openbuddy_llama2_70b_chat`.
- 2023.9.3: Supported **baichuan2** model series: baichuan2-7b, baichuan2-7b-chat, baichuan2-13b, baichuan2-13b-chat. The corresponding shell script can be found at `scripts/baichuan2_7b`, `scripts/baichuan2_7b_chat`, `scripts/baichuan2_13b_chat`.


## ✨ Features
- Supported SFT Methods: [lora](https://arxiv.org/abs/2106.09685), [qlora](https://arxiv.org/abs/2305.14314), full(full parameter fine-tuning)
- Supported Features: quantization, DDP, model parallelism, gradient checkpointing, pushing to modelscope hub, custom datasets, multimodal and agent SFT, mutli-round chat, ...
- Supported Models:
  - qwen series: [qwen-7b](https://modelscope.cn/models/qwen/Qwen-7B/summary), [qwen-7b-chat](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary), [qwen-14b](https://modelscope.cn/models/qwen/Qwen-14B/summary), [qwen-14b-chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary), [qwen-7b-chat-int4](https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int4/summary), [qwen-14b-chat-int4](https://modelscope.cn/models/qwen/Qwen-14B-Chat-Int4/summary), [qwen-7b-chat-int8](https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int8/summary), [qwen-14b-chat-int8](https://modelscope.cn/models/qwen/Qwen-14B-Chat-Int8/summary)
  - qwen-vl series: [qwen-vl](https://modelscope.cn/models/qwen/Qwen-VL/summary), [qwen-vl-chat](https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary), [qwen-vl-chat-int4](https://modelscope.cn/models/qwen/Qwen-VL-Chat-Int4/summary)
  - baichuan series: [baichuan-7b](https://modelscope.cn/models/baichuan-inc/baichuan-7B/summary), [baichuan-13b](https://modelscope.cn/models/baichuan-inc/Baichuan-13B-Base/summary), [baichuan-13b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan-13B-Chat/summary), [baichuan2-7b](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Base/summary), [baichuan2-7b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat/summary), [baichuan2-13b](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Base/summary), [baichuan2-13b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat/summary), [baichuan2-7b-chat-int4](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat-4bits/summary), [baichuan2-13b-chat-int4](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat-4bits/summary)
  - chatglm series: [chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary), [chatglm2-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm2-6b-32k/summary), [chatglm3-6b-base](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-base/summary), [chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary), [chatglm3-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-32k/summary)
  - llama series: [llama2-7b](https://modelscope.cn/models/modelscope/Llama-2-7b-ms/summary), [llama2-7b-chat](https://modelscope.cn/models/modelscope/Llama-2-7b-chat-ms/summary), [llama2-13b](https://modelscope.cn/models/modelscope/Llama-2-13b-ms/summary), [llama2-13b-chat](https://modelscope.cn/models/modelscope/Llama-2-13b-chat-ms/summary), [llama2-70b](https://modelscope.cn/models/modelscope/Llama-2-70b-ms/summary), [llama2-70b-chat](https://modelscope.cn/models/modelscope/Llama-2-70b-chat-ms/summary)
  - openbuddy series: [openbuddy-llama2-13b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama2-13b-v8.1-fp16/summary), [openbuddy-llama-65b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama-65b-v8-bf16/summary), [openbuddy-llama2-70b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama2-70b-v10.1-bf16/summary), [openbuddy-mistral-7b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-mistral-7b-v13.1/summary)
  - internlm series: [internlm-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-7b/summary), [internlm-7b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-v1_1/summary), [internlm-7b-chat-8k](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-8k/summary), [internlm-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-20b/summary), [internlm-20b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-20b/summary)
  - xverse series: [xverse-7b](https://modelscope.cn/models/xverse/XVERSE-7B/summary), [xverse-7b-chat](https://modelscope.cn/models/xverse/XVERSE-7B-Chat/summary), [xverse-13b](https://modelscope.cn/models/xverse/XVERSE-13B/summary), [xverse-13b-chat](https://modelscope.cn/models/xverse/XVERSE-13B-Chat/summary)
  - bluelm series: [bluelm-7b](https://modelscope.cn/models/vivo-ai/BlueLM-7B-Base/summary), [bluelm-7b-chat](https://modelscope.cn/models/vivo-ai/BlueLM-7B-Chat/summary), [bluelm-7b-32k](https://modelscope.cn/models/vivo-ai/BlueLM-7B-Base-32K/summary), [bluelm-7b-chat-32k](https://modelscope.cn/models/vivo-ai/BlueLM-7B-Chat-32K/summary)
  - mistral series: [mistral-7b](https://modelscope.cn/models/AI-ModelScope/Mistral-7B-v0.1/summary), [mistral-7b-chat](https://modelscope.cn/models/AI-ModelScope/Mistral-7B-Instruct-v0.1/summary)
  - ziya series: [ziya2-13b](https://modelscope.cn/models/Fengshenbang/Ziya2-13B-Base/summary), [ziya2-13b-chat](https://modelscope.cn/models/Fengshenbang/Ziya2-13B-Chat/summary)
  - skywork series: [skywork-13b](https://modelscope.cn/models/skywork/Skywork-13B-base/summary), [skywork-13b-chat](https://modelscope.cn/models/skywork/Skywork-13B-chat/summary)
  - other: [polylm-13b](https://modelscope.cn/models/damo/nlp_polylm_13b_text_generation/summary), [seqgpt-560m](https://modelscope.cn/models/damo/nlp_seqgpt-560m/summary)
- Supported Datasets:
  - NLP:
    - General: 🔥[alpaca-en](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-en/summary)(gpt4), 🔥[alpaca-zh](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-zh/summary)(gpt4), [multi-alpaca-all](https://www.modelscope.cn/datasets/damo/nlp_polylm_multialpaca_sft/summary), [instinwild-en](https://www.modelscope.cn/datasets/wyj123456/instinwild/summary), [instinwild-zh](https://www.modelscope.cn/datasets/wyj123456/instinwild/summary), [cot-en](https://www.modelscope.cn/datasets/YorickHe/CoT/summary), [cot-zh](https://www.modelscope.cn/datasets/YorickHe/CoT/summary), [firefly-all-zh](https://www.modelscope.cn/datasets/wyj123456/firefly/summary), [instruct-en](https://www.modelscope.cn/datasets/wyj123456/instruct/summary), [gpt4all-en](https://www.modelscope.cn/datasets/wyj123456/GPT4all/summary), [sharegpt-en](https://www.modelscope.cn/datasets/huangjintao/sharegpt/summary), [sharegpt-zh](https://www.modelscope.cn/datasets/huangjintao/sharegpt/summary)
    - Agent: [damo-agent-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary), 🔥[damo-agent-mini-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary), 🔥[agent-instruct-all-en](https://modelscope.cn/datasets/ZhipuAI/AgentInstruct/summary)
    - Coding: [code-alpaca-en](https://www.modelscope.cn/datasets/wyj123456/code_alpaca_en/summary), [code-python-zh](https://modelscope.cn/datasets/codefuse-ai/CodeExercise-Python-27k/summary), 🔥[leetcode-python-en](https://modelscope.cn/datasets/AI-ModelScope/leetcode-solutions-python/summary)
    - Medical: [medical-en](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary), [medical-zh](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary), [medical-mini-zh](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary)
    - Law: 🔥[lawyer-llama-zh](https://modelscope.cn/datasets/AI-ModelScope/lawyer_llama_data/summary), [tigerbot-law-zh](https://modelscope.cn/datasets/AI-ModelScope/tigerbot-law-plugin/summary)
    - Math: 🔥[blossom-math-zh](https://modelscope.cn/datasets/AI-ModelScope/blossom-math-v2/summary), [school-math-zh](https://modelscope.cn/datasets/AI-ModelScope/school_math_0.25M/summary)
    - SQL: [text2sql-en](https://modelscope.cn/datasets/AI-ModelScope/texttosqlv2_25000_v2/summary), 🔥[sql-create-context-en](https://modelscope.cn/datasets/AI-ModelScope/sql-create-context/summary)
    - Text Generation: 🔥[advertise-gen-zh](https://modelscope.cn/datasets/lvjianjin/AdvertiseGen/summary), 🔥[dureader-robust-zh](https://modelscope.cn/datasets/modelscope/DuReader_robust-QG/summary)
    - Classification: [cmnli-zh](https://www.modelscope.cn/datasets/modelscope/clue/summary), 🔥[jd-sentiment-zh](https://modelscope.cn/datasets/DAMO_NLP/jd/summary)
    - Other: [finance-en](https://www.modelscope.cn/datasets/wyj123456/finance_en/summary), [poetry-zh](https://www.modelscope.cn/datasets/modelscope/chinese-poetry-collection/summary), [cls-fudan-news-zh](https://modelscope.cn/datasets/damo/zh_cls_fudan-news/summary), [ner-jave-zh](https://modelscope.cn/datasets/damo/zh_ner-JAVE/summary)
  - Multi-Modal: 🔥[coco-en](https://modelscope.cn/datasets/modelscope/coco_2014_caption/summary)
  - Custom Dataset
- Supported Templates:
  - Text Generation: default-generation, chatglm-generation
  - Chat: default, chatml(qwen), baichuan, chatglm2, chatglm3, llama, openbuddy, internlm, xverse, ziya, skywork, bluelm


## 🛠️ Preparing the Experimental Environment
Experimental environment: A10, 3090, V100, A100, ...
```bash
# Setting up a global mirror for pip and installing related Python packages
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
git clone https://github.com/modelscope/swift.git
cd swift
pip install .[llm]
# The following script needs to be executed in this directory.
cd examples/pytorch/llm

# If you want to use DeepSpeed:
pip install deepspeed -U

# If you want to use qlora training based on auto_gptq (recommended, better performance than bnb):
# auto_gptq has version mapping with cuda versions，please refer to https://github.com/PanQiWei/AutoGPTQ#quick-installation
pip install auto_gptq
pip install optimum -U

# If you want to use qlora training based on bnb:
pip install bitsandbytes -U
```


## 🚀 Basic Usage
Quickly fine-tune, infer with LLM, and build a Web-UI. Please make sure you have read the "Preparing the Experimental Environment" section.

### Run using Python
```python
# Experimental environment: A10, 3090, A100, ...
# 16GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments
)
from swift.llm.run import infer_main, sft_main, web_ui_main

model_type = ModelType.qwen_7b_chat_int4
sft_args = SftArguments(
    model_type=model_type,
    eval_steps=50,
    train_dataset_sample=2000,
    dataset=[DatasetName.leetcode_python_en],
    output_dir='output',
    gradient_checkpointing=True)
best_ckpt_dir = sft_main(sft_args)
print(f'best_ckpt_dir: {best_ckpt_dir}')
torch.cuda.empty_cache()
infer_args = InferArguments(
    ckpt_dir=best_ckpt_dir,
    load_args_from_ckpt_dir=True,
    stream=True,
    show_dataset_sample=5)
infer_main(infer_args)
torch.cuda.empty_cache()
web_ui_main(infer_args)
```

### Run using Swift CLI
**SFT**:
```bash
# Experimental environment: A10, 3090, A100, ...
# 10GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft --model_id_or_path qwen/Qwen-7B-Chat-Int4 --dataset blossom-math-zh

# Using DDP
# Experimental environment: 2 * 3090
# 2 * 10GB GPU memory
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat-Int4 \
    --dataset blossom-math-zh \

# Using custom dataset
CUDA_VISIBLE_DEVICES=0 swift sft --model_id_or_path qwen/Qwen-7B-Chat-Int4 --custom_train_dataset_path chatml.jsonl
```

**Inference**:
```bash
CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx'
```

**Web-UI**:
```bash
CUDA_VISIBLE_DEVICES=0 swift web-ui --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx'
```


## 🌟 Run SFT and Inference
Performace: full(nice) > lora > qlora(auto_gptq) > qlora(bnb)

Training GPU memory: qlora(low,3090) > lora > full(2*A100)

**Tips**:
- You can set `--gradient_checkpointing true` during training to **save GPU memory**, but this will slightly decrease the training speed. This is useful if you need to train LLM on **consumer-grade GPU**, e.g. 3090.
- If you want to use quantization based on **auto_gptq**, you need to install auto_gptq first: `pip install auto_gptq -U`.
  The models available with auto_gptq are: `qwen-7b-chat-int4`, `qwen-14b-chat-int4`, `qwen-7b-chat-int8`, `qwen-14b-chat-int8`.
  If the script provides multiple versions of qlora SFT, including both non-quantized models and int4/int8 models, it is **recommended to use the script for the int4/int8 model versions**.
- If you want to use the quantization parameter `quantization_bit`, you need to install `bitsandbytes` first: `pip install bitsandbytes -U`.
- If you want to use deepspeed, you need to `pip install deepspeed -U`. Using deepspeed can **save GPU memory**, but this may slightly decrease the training speed.
- If you are using older GPUs like **V100**, you need to set `--dtype fp16`, because they do not support bf16.
- qwen recommends installing [**flash-attn**](https://github.com/Dao-AILab/flash-attention), which will accelerate the training and inference speed and reduce GPU memory usage (A10, 3090, V100 machines do not support flash-attn).
- If you want to perform **second pre-training** instead of SFT, you can refer to the `DatasetName.tigerbot_law_zh` dataset and its corresponding sh file: `scripts/qwen_7b/qlora_ddp`.
- If you want to push weights to the ModelScope Hub during training, you need to set `--push_to_hub true`.
- If you want to merge LoRA weights and save them during inference, you need to set `--merge_lora_and_save true`. It is **not recommended to merge quantized models**, as this can result in performance degradation, specifically in the case of qlora.
- Below is a shell script for running `qwen_7b_chat` directly (you just need to specify `ckpt_dir` during inference to execute it smoothly). For more model scripts, you can check the `scripts` folder. If you want to **customize a shell script**, it is recommended to refer to the script in `scripts/qwen_7b_chat`.
```bash
# sft(qlora) and infer qwen-7b-chat-int8, Requires 16GB GPU memory.
# Recommended experimental environment: V100, A10, 3090
bash scripts/qwen_7b_chat_int8/qlora/sft.sh
bash scripts/qwen_7b_chat_int8/qlora/infer.sh

# sft(qlora+ddp+deepspeed) and infer qwen-7b-chat-int8, Requires 2*19GB GPU memory.
# Recommended experimental environment: V100, A10, 3090
bash scripts/qwen_7b_chat_int8/qlora_ddp_ds/sft.sh
bash scripts/qwen_7b_chat_int8/qlora_ddp_ds/infer.sh

# sft(qlora) and infer qwen-7b-chat-int4, Requires 13GB GPU memory.
# Recommended experimental environment: V100, A10, 3090
bash scripts/qwen_7b_chat_int4/qlora/sft.sh
bash scripts/qwen_7b_chat_int4/qlora/infer.sh

# sft(qlora+ddp+deepspeed) and infer qwen-7b-chat-int4, Requires 2*16GB GPU memory.
# Recommended experimental environment: V100, A10, 3090
bash scripts/qwen_7b_chat_int4/qlora_ddp_ds/sft.sh
bash scripts/qwen_7b_chat_int4/qlora_ddp_ds/infer.sh

# sft(lora) and infer qwen-7b-chat, Requires 60GB GPU memory.
# Recommended experimental environment: A100
bash scripts/qwen_7b_chat/lora/sft.sh
bash scripts/qwen_7b_chat/lora/infer.sh

# sft(lora+ddp) and infer qwen-7b-chat, Requires 2*60GB GPU memory.
# Recommended experimental environment: A100
bash scripts/qwen_7b_chat/lora_ddp/sft.sh
bash scripts/qwen_7b_chat/lora_ddp/infer.sh

# sft(lora+ddp+deepspeed) and infer qwen-7b-chat, Requires 2*18GB GPU memory.
# Recommended experimental environment: A10, 3090
bash scripts/qwen_7b_chat/lora_ddp_ds/sft.sh
bash scripts/qwen_7b_chat/lora_ddp_ds/infer.sh

# sft(lora+mp+ddp) and infer qwen-7b-chat, Requires 4*15GB GPU memory.
# Recommended experimental environment: A10, 3090
bash scripts/qwen_7b_chat/lora_mp_ddp/sft.sh
bash scripts/qwen_7b_chat/lora_mp_ddp/infer.sh

# sft(full+mp) and infer qwen-7b-chat, Requires 2*75GB GPU memory.
# Recommended experimental environment: A100
bash scripts/qwen_7b_chat/full_mp/sft.sh
bash scripts/qwen_7b_chat/full_mp/infer.sh

# sft(full+mp+ddp) and infer qwen-7b-chat, Requires 4*75GB GPU memory.
# Recommended experimental environment: A100
bash scripts/qwen_7b_chat/full_mp_ddp/sft.sh
bash scripts/qwen_7b_chat/full_mp_ddp/infer.sh

# The qlora script based on bnb below is no longer recommended for use. Please prioritize using the qlora script based on auto_gptq.
# sft(qlora) and infer qwen-7b-chat, Requires 13GB GPU memory.
# Recommended experimental environment: A10, 3090
bash scripts/qwen_7b_chat/qlora/sft.sh
bash scripts/qwen_7b_chat/qlora/infer.sh

# sft(qlora+ddp) and infer qwen-7b-chat, Requires 2*14GB GPU memory.
# Recommended experimental environment: A10, 3090
bash scripts/qwen_7b_chat/qlora_ddp/sft.sh
bash scripts/qwen_7b_chat/qlora_ddp/infer.sh

# sft(qlora+ddp+deepspeed) and infer qwen-7b-chat, Requires 2*16GB GPU memory.
# Recommended experimental environment: A10, 3090
bash scripts/qwen_7b_chat/qlora_ddp_ds/sft.sh
bash scripts/qwen_7b_chat/qlora_ddp_ds/infer.sh
```


## 📝 User Guide

### Custom Dataset
We support two methods for **customizing datasets**.
1. [Recommended] **Command line arguments**: It is **more convenient for supporting local custom datasets**.
2. **Registering datasets**: It is more flexible and allows for **further extension and development of swift**, but it requires some programming skills. Method 1 relies on Method 2 for implementation.

#### 📌 [Recommended] Command Line Arguments
Explanation of command line arguments:
1. `--custom_train_dataset_path`: The default value is `None`, which means no custom dataset is used. You can specify it in the following format: `--custom_train_dataset_path alpaca.csv` or specify multiple training datasets like `--custom_train_dataset_path alpaca.csv chatml.jsonl swift.jsonl`. The script will automatically preprocess and concatenate them.

   > You can also combine public datasets with custom datasets for training: `--dataset blossom-math-zh --custom_train_dataset_path custom_math.jsonl`.

2. `--custom_val_dataset_path`: The default value is `None`, which means no custom validation dataset is used. If you specify `custom_train_dataset_path`, the custom dataset's validation set will be split according to the command line argument `dataset_test_ratio`. The format of the command line input can be referred to the `--custom_train_dataset_path` format.

The script supports `csv` and `jsonl` file formats. The files you pass in need to conform to the following dataset formats. The csv format file only supports instruction fine-tuning, which means there is no history. The jsonl format file supports system and history.

Format 1:
```csv
instruction,input,output
11111,22222,33333
aaaaa,bbbbb,ccccc
AAAAA,BBBBB,CCCCC
```

```jsonl
{"instruction": "11111", "input": "aaaaa", "output": "AAAAA"}
{"instruction": "22222", "input": "bbbbb", "output": "BBBBB"}
{"instruction": "33333", "input": "ccccc", "output": "CCCCC"}
```

Format 2:
```jsonl
{"query": "55555", "response": "66666", "history": [["11111", "22222"], ["33333", "44444"]]}
{"query": "eeeee", "response": "fffff", "history": [["aaaaa", "bbbbb"], ["ccccc", "ddddd"]]}
{"query": "EEEEE", "response": "FFFFF", "history": [["AAAAA", "BBBBB"], ["CCCCC", "DDDDD"]]}
```

Format 3:
```jsonl
{"conversations": [{"from": "user", "value": "11111"}, {"from": "assistant", "value": "22222"}, {"from": "user", "value": "33333"}, {"from": "assistant", "value": "44444"}]}
{"conversations": [{"from": "user", "value": "aaaaa"}, {"from": "assistant", "value": "bbbbb"}, {"from": "user", "value": "ccccc"}, {"from": "assistant", "value": "ddddd"}]}
{"conversations": [{"from": "user", "value": "AAAAA"}, {"from": "assistant", "value": "BBBBB"}, {"from": "user", "value": "CCCCC"}, {"from": "assistant", "value": "DDDDD"}]}
```

Format 4:
```jsonl
{"messages": [{"role": "user", "content": "11111"}, {"role": "assistant", "content": "22222"}, {"role": "user", "content": "33333"}, {"role": "assistant", "content": "44444"}]}
{"messages": [{"role": "user", "content": "aaaaa"}, {"role": "assistant", "content": "bbbbb"}, {"role": "user", "content": "ccccc"}, {"role": "assistant", "content": "ddddd"}]}
{"messages": [{"role": "user", "content": "AAAAA"}, {"role": "assistant", "content": "BBBBB"}, {"role": "user", "content": "CCCCC"}, {"role": "assistant", "content": "DDDDD"}]}
```


#### Registering Datasets
Here is an example of a **registering a dataset**. Running the shell script for this custom dataset can be found in `scripts/custom`.

```python
import ast
from swift.llm import (
    register_dataset, get_dataset, ConversationsPreprocessor, get_dataset_from_repo
)

class CustomDatasetName:
    agent_instruct_all_en = 'agent-instruct-all-en'

_agent_instruct_subset_list = [
    'alfworld', 'db', 'kg', 'mind2web', 'os', 'webshop'
]

def repair_conversations_agent_instruct(s: str) -> str:
    s = s.replace('}\n {', '},\n {')
    return ast.literal_eval(s)


register_dataset(
    CustomDatasetName.agent_instruct_all_en,
    'huangjintao/AgentInstruct_copy',
    [(subset, 'train') for subset in _agent_instruct_subset_list],
    None,
    ConversationsPreprocessor(
        'human',
        'gpt',
        repair_conversations=repair_conversations_agent_instruct),
    get_dataset_from_repo,
    task='chat')

if __name__ == '__main__':
    train_dataset, _ = get_dataset([CustomDatasetName.agent_instruct_all_en],
                                   0., check_dataset_strategy='warning')
    print(train_dataset)
    print(train_dataset[0].keys())
```
The `register_dataset` function registers the dataset in the `DATASET_MAPPING`. The function parameters are as follows:
- `dataset_name`: Required parameter that represents the name of the dataset, which is also the unique ID of the dataset.
- `dataset_id_or_path`: Required. Represents the `dataset_id` on the ModelScope Hub or the local `dataset_dir` where the dataset is located.
- `train_subset_split_list`: Default value is `None`. If you are using the function `get_dataset_from_repo` to fetch the dataset, this parameter is a `List[Union[str, Tuple[str, str], List[str]]]`. It is a list of (subset_name, split) pairs, where we concatenate these subsets to form the complete training dataset. If the elements in the list are strings, the default `subset_name` is 'default'. If you are using a different `get_function`, the meaning of this parameter can be customized, for example, if `dataset_id_or_path` represents `model_dir`, this parameter can represent the filename of the training set.
- `train_subset_split_list`: Default value is `None`. The meaning of this parameter is similar to `train_subset_split_list`.
- `preprocess_func`: Default value is `None`. Represents the method for preprocessing the function.
- `get_function`: Default value is `None`. The function used to retrieve the dataset. If None is passed, the decorator scheme is used for dataset registration, where the `register_dataset` function returns `Callable[[GetDatasetFunction], GetDatasetFunction]`. This scheme requires users with some python knowledge. If a function is passed, the normal registration scheme is used. If importing datasets from the ModelScope Hub, the `get_dataset_from_repo` function is commonly used.
  The `get_function` function has no restrictions, you just need to return either `HfDataset` or `Tuple[HfDataset, Optional[HfDataset]]`. In the case where only the `train_dataset` is returned, the data processing function will split a portion of the dataset as the validation dataset (based on the command line hyperparameter `dataset_test_ratio`). If two datasets are returned, they will be used as the training and validation datasets respectively. We support fine-tuning with multiple datasets. The training and validation portions of each sub-dataset will be concatenated separately, and the final merged training and validation datasets will be returned.
  The returned `HfDataset` needs to adhere to certain specifications. If it is for instruction fine-tuning (single-turn dialogue), it should include the `query` and `response` fields, representing the user's query for instruction fine-tuning and the AI assistant's response, respectively. You can refer to the `alpaca-zh` dataset for more details. If it is for multi-turn dialogue, it needs to include an additional `history` field, representing the history of the conversation. You can refer to the `damo-agent-mini-zh` dataset for more details. If each example in the dataset has a different `system`, an additional `system` field is required. You can also refer to the `damo-agent-mini-zh` dataset for more details. We only calculate and optimize the loss for the `response` part.
- `task`: The task for which the dataset is intended. This parameter is generally not required to be set.
- `function_kwargs`: Default is `{}`, used to pass arguments to `get_function` to support the `partial` functionality in the decorator scheme. This parameter is generally not required to be set.
- `**kwargs`: Other parameters used for annotating the dataset. This parameter is generally not required to be set.

### Custom Model

Here is an example of a **custom model**. Running the shell script for this custom model can be found in `scripts/custom`.

```python
from swift.llm import (
    register_model, LoRATM, get_model_tokenizer_from_repo, get_model_tokenizer
)
import torch
from torch import dtype as Dtype
from typing import Dict, Any

class CustomModelType:
    tigerbot_13b_chat = 'tigerbot-13b-chat'

class CustomTemplateType:
    tigerbot = 'tigerbot'

@register_model(CustomModelType.tigerbot_13b_chat,
                'TigerResearch/tigerbot-13b-chat-v4', LoRATM.llama2,
                CustomTemplateType.tigerbot)
def get_tigerbot_model_tokenizer(model_dir: str,
                                 torch_dtype: Dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if use_flash_attn:
        require_version('transformers>=4.34')
        logger.info('Setting use_flash_attention_2: True')
        model_kwargs['use_flash_attention_2'] = True
    return get_model_tokenizer_from_repo(model_dir, torch_dtype, model_kwargs,
                                         load_model, **kwargs)

# Usage without decorators:
# register_model(CustomModelType.tigerbot_13b_chat,
#                'TigerResearch/tigerbot-13b-chat-v4', LoRATM.llama2,
#                CustomTemplateType.tigerbot, get_tigerbot_model_tokenizer)

if __name__ == '__main__':
    model_kwargs = {'device_map': 'auto'}
    model, tokenizer = get_model_tokenizer(CustomModelType.tigerbot_13b_chat, torch.bfloat16, use_flash_attn=False)
    print(model, tokenizer)
```

The `register_model` function registers the model in `MODEL_MAPPING`, and its parameters are as follows:

- `model_type`: Required. It represents the name of the model, which is also the unique ID.
- `model_id_or_path`: Required. It represents the `model_id` of the model in the ModelScope Hub or the local model directory `model_dir`.
- `lora_target_modules`: Default is `None`. It represents the `lora_target_modules` used when specified as `--lora_target_modules DEFAULT` in the shell script or when not specified.
- `template`: Default is `TemplateType.default`. It represents the default chat template used when not specified as `--template` in the shell script.
- `get_function`: Default is `None`. It is a function used to retrieve the model and tokenizer. If `None` is passed, the decorator approach is used for model registration, and the `register_model` function will return `Callable[[GetModelTokenizerFunction], GetModelTokenizerFunction]`. This approach is intended for users with some Python knowledge. If a function is passed, the regular approach is used for registration. Typically, `get_model_tokenizer_from_repo` is used as a parameter, which returns the model and tokenizer. If there is a need for patching the model code or other customization requirements, it can be achieved by customizing this function.
- `requires`: Default is `[]`. It represents the dependencies specific to the model, different from other models. This parameter is generally not required.
- `torch_dtype`: Default is `None`. It represents the recommended `torch_dtype` used by the model. This parameter is generally not required.
- `automodel_class`: Default is `AutoModelForCausalLM`. It represents the class called by `from_pretrained`. If you are using models like `roberta-base`, this parameter needs to be modified. This parameter is generally not required.
- `revision`: Default is `'master'`. It is used to specify the version number of the model. This parameter is not effective if `model_id_or_path` is a local model directory. This parameter is generally not required.
- `ignore_file_pattern`: Default is `None`. It represents the regular expression pattern of the file names to be ignored during downloading, which is passed to `snapshot_download`. For example, `r'.+\.bin$'`, `r'.+\.savetensors$'`, etc. This parameter is generally not required.
- `max_length`: Default is `None`. It is used to annotate the maximum length of the model. This parameter is generally not required.
- `function_kwargs`: Default is `{}`. It is used to pass arguments to `get_function` to support the `partial` functionality in the decorator approach. This parameter is generally not required.
- `**kwargs`: Other parameters used to annotate model capabilities. This parameter is generally not required.

### Custom Chat Template

Here is an example of a **custom template**. Running the shell script for this custom template can be found in `scripts/custom`.

```python
from swift.llm import (
    register_template, Template, get_template, get_model_tokenizer, ModelType, inference
)
class CustomTemplateType:
    tigerbot = 'tigerbot'

# Ref: https://github.com/TigerResearch/TigerBot/blob/main/infer.py
register_template(
    CustomTemplateType.tigerbot,
    Template([], ['\n\n### Instruction:\n{{QUERY}}\n\n### Response:\n'], [],
             [['eos_token_id']]))

if __name__ == '__main__':
    # only for test
    _, tokenizer = get_model_tokenizer(ModelType.qwen_7b_chat, load_model=False)
    template = get_template(CustomTemplateType.tigerbot, tokenizer)
    inputs = {'query': '浙江的省会在哪里?', 'response': '杭州',
              'system': 'you are a helpful assistant!',
              'history': [('你好!', '你好! 我是AI智能助手. '),
                          ('1+1=?', '2')]}
    print(tokenizer.decode(template.encode(inputs)['input_ids']))
```
The `register_template` function registers the conversation template in the `TEMPLATE_MAPPING`. The function takes the following arguments:
- `template_type`: Required. It represents the name of the conversation template and serves as the unique ID for the template.
- `template`: Required. It takes a `Template` object as input. Initializing the `Template` requires four parameters: `prefix`, `prompt`, `chat_sep`, and `suffix`.

The template initialization function retrieves the complete chat template based on these four components, enabling support for pre-training, text generation-style SFT, and various chat-based SFT. The meanings of these four configuration components are as follows:
- `prefix`: Represents the prefix part of the chat template, usually including the system part and relevant formats, prefix tokens, BOS token, etc. We use `{{SYSTEM}}` as a placeholder for the system part.
- `prompt`: Represents a round of dialogue in the chat template. We use `{{QUERY}}` as a placeholder for the human inquiry part in each round of dialogue, `{{ROUND0}}` represents the placeholder for the current round of dialogue, counting from 0, and `{{ROUND1}}` counting from 1. The assistant's reply is concatenated after the `prompt`, so we did not design a placeholder for it.
- `chat_sep`: If multiple rounds of dialogue are needed, `chat_sep` serves as the separator between each round of dialogue, such as a newline, etc. If set to None, the Template does not support multi-turn conversations.
- `suffix`: Serves as the suffix part of the chat template, usually the EOS token. It is appended after the last round of dialogue. Only the response part of the last round will calculate the loss and be optimized, while the other parts will not calculate the loss.



### sft.sh Command Line Arguments
- `--model_type`: Represents the selected model type. The default value is `None`, which means if `model_id_or_path` is not specified, `'qwen-7b-chat'` will be chosen. If `model_id_or_path` is specified, the `model_type` will be inferred based on `model_id_or_path` and `MODEL_MAPPING`. These two parameters cannot be specified simultaneously. The available `model_type` options can be found in `MODEL_MAPPING.keys()`.
- `--model_id_or_path`: Represents the `model_id` of the model in the ModelScope Hub or the local model directory `model_dir`. It is case-insensitive and the default value is `None`. If `--model_id_or_path` is not registered, an exception will be raised. You can specify the model type using `model_type` or `model_id_or_path`.
- `--model_revision`: Represents the version number of the `model_id` in the ModelScope Hub. The default value is `None`. If `model_id_or_path` is a local model directory, this parameter is ignored. If `model_revision` is set to `None`, the revision registered in `MODEL_MAPPING` will be used. Otherwise, the `model_revision` will be forced to be used.
- `--model_cache_dir`: Default is `None`. If the model has already been cached locally and the cache path is not the default cache path of ModelScope, you can specify this parameter to import the model and tokenizer from the cache directory.
- `--sft_type`: Represents the fine-tuning method, default is `'lora'`. The possible values are: 'lora', 'full'. If you want to use lora or qlora, you need to select `--sft_type lora`. For qlora, an additional setting `--quantization_bit 4` is required. If you want to use full-parameter fine-tuning, you need to select `--sft_type full`.
- `--tuner_backend`: Represents the backend support for lora and qlora, default is `'swift'`. The possible values are: 'swift', 'peft'.
- `--template_type`: Represents the type of dialogue template used, default is `None`, which means it retrieves the template based on `model_type` from `MODEL_MAPPING`. Available `template_type` can be checked using `TEMPLATE_MAPPING.keys()`.
- `--output_dir`: Represents the directory for storing checkpoints, default is `'output'`. We will concatenate `model_type` and fine-tuning version number to this directory. This allows users to perform multiple comparative experiments on different models without changing the `output_dir` command-line argument.
- `--add_output_dir_suffix`: Default is `True`, which means that `model_type` and the fine-tuning version number suffix will be appended to the `output_dir` directory. If you want to avoid this behavior, you can set it to `False`.
- `--ddp_backend`: Represents the backend support for distributed training, default is `'nccl'`. The possible values are: 'nccl', 'gloo', 'mpi', 'ccl'.
- `--seed`: Global seed value, default is 42. In distributed training, to avoid each process using the same dropout, etc., we set `seed=seed+rank`.
- `--resume_from_checkpoint`: Used for resuming training from a checkpoint, default is `None`. You can set it to the path of the checkpoint, for example: `'output/qwen-7b-chat/vx_xxx/checkpoint-xxx'`, to resume training from that checkpoint.
- `--dtype`: The torch_dtype used when loading the base model, default is `None`, which means automatic selection of the dtype: if the machine does not support bf16, fp16 will be used instead. If the `MODEL_MAPPING` specifies a torch_dtype for the corresponding model, it will be used; otherwise, bf16 will be used. The available values are: 'bf16', 'fp16', 'fp32'.
- `--dataset`: Used to select the training dataset, default is `'blossom-math-zh'`. Available datasets can be checked using `DATASET_MAPPING.keys()`. If you want to use multiple datasets for training, you can separate them using ',' or ' ', for example: `alpaca-en,alpaca-zh` or `alpaca-en alpaca-zh`.
- `--dataset_seed`: Used to specify the seed for dataset processing. The default value is `42`. It is present in the form of `random_state` and does not affect the global seed.
- `--dataset_test_ratio`: Specifies the ratio for splitting the sub-dataset into training and validation sets, default is `0.01`. This parameter is ignored if the sub-dataset has already been split into training and validation sets. When multiple sub-datasets are specified in `dataset` and the function for retrieving the sub-dataset does not perform the split (i.e., returns `HfDataset` or `Tuple[HfDataset, None]` instead of `Tuple[HfDataset, HfDataset]`), we need to split the sub-dataset. Finally, we concatenate the training and validation parts of these sub-datasets to generate the training and validation sets for the complete fine-tuning dataset.
- `--train_dataset_sample`: Samples from the complete training dataset, default is `20000`, to speed up training. This parameter is used to avoid the issue of training time being too long for a single epoch when the dataset is large. LoRA convergence is usually fast and does not require a large number of data samples for fine-tuning. If you specify `-1`, the full training dataset will be used for training, which is typically used in the setting of full-parameter fine-tuning.
- `--system`: The system used in the dialogue template, default is `'you are a helpful assistant!'`.
- `--max_length`: Maximum token length, default is `2048`. This helps to avoid out-of-memory (OOM) issues caused by individual samples that are too long. If a data sample exceeds the `max_length`, the frontmost tokens will be truncated: `input_ids[-max_length:]`. If set to -1, there is no restriction.
- `--check_dataset_strategy`: The default value is `'none'`, which means no checking will be done. If you are training an LLM model, it is recommended to use `'warning'` as the data checking strategy. If your training objective is sentence classification or Masked LM tasks, it is suggested to set it as `'none'`.
- `custom_train_dataset_path`: The default value is `None`. Please refer to the `Custom Dataset` module in the README.md for specific meanings.
- `custom_val_dataset_path`: The default value is `None`. Please refer to the `Custom Dataset` module in the README.md for specific meanings.
- `--quantization_bit`: Specifies whether to perform quantization and the number of quantization bits, default is `0`, which means no quantization. Quantization is only supported for the lora fine-tuning method and not for full-parameter fine-tuning.
- `--bnb_4bit_comp_dtype`: When performing 4-bit quantization, we need to dequantize it during the model's forward and backward passes. This parameter specifies the torch_dtype after dequantization. Default is `None`, which means it remains consistent with `dtype`. The possible values are: 'fp16', 'bf16', 'fp32'. This parameter is ignored when `quantization_bit` is 0.
- `--bnb_4bit_quant_type`: The quantization type for 4-bit quantization, default is `'nf4'`. The possible values are: 'nf4', 'fp4'. This parameter is ignored when `quantization_bit` is 0.
- `--bnb_4bit_use_double_quant`: Whether to enable double quantization during 4-bit quantization, default is `True`. This parameter is ignored when `quantization_bit` is 0.
- `--lora_target_modules`: Specifies the LoRA module, default is `None`. If `lora_target_modules` is `None` or set to `DEFAULT`, it will look for `lora_target_modules` in `MODEL_MAPPING` based on `model_type` (default is set to qkv). If set to `ALL`, all Linear layers (excluding the head) will be specified as LoRA modules. This parameter only takes effect when `sft_type` is set to 'lora'.
- `--lora_rank`: Default is `8`. This parameter only takes effect when `sft_type` is set to `'lora'`.
- `--lora_alpha`: Default is `32`. This parameter only takes effect when `sft_type` is set to `'lora'`.
- `--lora_dropout_p`: Default is `0.05`. This parameter only takes effect when `sft_type` is set to `'lora'`.
- `--gradient_checkpointing`: Whether to enable gradient checkpointing, default is `False`. This parameter can be used to save GPU memory, although it slightly slows down the training speed. This parameter is particularly effective when `max_length` and `batch_size` are large.
- `--deepspeed_config_path`: Used to specify the path to the DeepSpeed configuration file. Default is `None`, which means DeepSpeed is not enabled. DeepSpeed can help save GPU memory. We have provided a default configuration file for ZeRO-2: `ds_config/zero2.json`.
- `--batch_size`: Batch size during training, default is `1`. Increasing the batch size can improve GPU utilization but may not necessarily speed up training because within a batch, padding is applied to shorter sentences based on the length of the longest sentence in the batch, introducing unnecessary computations.
- `--eval_batch_size`: Batch size during evaluation, default is `None`. If `predict_with_generate` is set to `True`, it is set to `1`; if `predict_with_generate` is `False`, it is set to `batch_size`.
- `--num_train_epochs`: Number of training epochs, default is `1`. If `max_steps >= 0`, it overrides `num_train_epochs`.
- `--max_steps`: Maximum number of training steps, default is `-1`. If `max_steps >= 0`, it overrides `num_train_epochs`.
- `--optim`: Default is `'adamw_torch'`.
- `--learning_rate`: Default is `None`. If `sft_type` is `'lora'`, it is set to `1e-4`; if `sft_type` is `'full'`, it is set to `2e-5`.
- `--weight_decay`: Default is `0.01`.
- `--gradient_accumulation_steps`: Gradient accumulation, default is `16`. `total_batch_size = batch_size * gradient_accumulation_steps * world_size`.
- `--max_grad_norm`: Gradient clipping, default is `1`.
- `--predict_with_generate`: Whether to use a generative approach during evaluation, default is `False`. If set to `False`, it uses `loss` for evaluation. If set to `True`, it uses metrics such as `ROUGE-L` for evaluation. Note that using the generative approach for evaluation is time-consuming, so use it with caution.
- `--lr_scheduler_type`: Default is `'cosine'`.
- `--warmup_ratio`: Ratio of warmup steps to the total training steps, default is `0.05`.
- `--eval_steps`: Perform evaluation every specified number of steps, default is `50`.
- `--save_steps`: Save the model every specified number of steps, default is `None`, which sets it to `eval_steps`.
- `--only_save_model`: Whether to only save the model parameters without storing the intermediate states required for resuming training. The default value is `None`. If `sft_type` is 'lora' and DeepSpeed is not used (deepspeed_config_path is None), it is set to False; otherwise, it is set to True (e.g., when using full parameter fine-tuning or DeepSpeed).
- `--save_total_limit`: The number of checkpoints to save. The default value is `2`, which saves the best and last checkpoints. If set to -1, it saves all checkpoints.
- `--logging_steps`: Number of training steps to print training information (e.g., loss, learning_rate, etc.). Default is `5`.
- `--dataloader_num_workers`: The number of worker processes to use for data loading. The default value is `1`.
- `--push_to_hub`: Whether to synchronize the training checkpoints to the ModelScope Hub. The default value is `False`.
- `--hub_model_id`: The model id of the ModelScope Hub to push to. The default value is `None`, which is set to `f'{model_type}-{sft_type}'`. You can set it to a specific model id or repository name. The user name will be inferred from the `hub_token`. If the remote repository does not exist, a new repository will be created. If it exists, the previous repository will be reused. This parameter only takes effect when `push_to_hub` is set to True.
- `--hub_private_repo`: Whether to set the permission of the model repository in the ModelScope Hub to private. The default value is `True`. This parameter only takes effect when `push_to_hub` is set to True.
- `--push_hub_strategy`: The push strategy. The default value is `'push_best'`. Available options are: 'end', 'push_best', 'push_last', 'checkpoint', 'all_checkpoints'. 'push_best' means that the best model will be pushed and overwrite the previous weights every time the weights are saved. 'push_last' means that the last weights will be pushed and overwrite the previous weights every time the weights are saved. This parameter only takes effect when `push_to_hub` is set to True.
- `--hub_token`: The SDK token required for pushing to the ModelScope Hub. You can obtain it from https://modelscope.cn/my/myaccesstoken. The default value is `None`, which retrieves the token from the environment variable `MODELSCOPE_API_TOKEN`. This parameter only takes effect when `push_to_hub` is set to True.
- `--test_oom_error`: Used to check if training will encounter an out-of-memory (OOM) error. The default value is `False`. If set to True, the training set will be sorted in reverse order of `max_length` to facilitate OOM testing. This parameter is generally used for testing, so please use it with caution.
- `--use_flash_attn`: Whether to use flash attention. The default value is `None`. For installation steps of flash attention, please refer to https://github.com/Dao-AILab/flash-attention.
- `--ignore_args_error`: Whether to ignore errors raised by command-line argument mismatch, default is `False`. If you need to copy the code to a notebook for execution, you should set it to True.
- `--logging_dir`: Default is `None`. If not specified, it is set to `f'{self.output_dir}/runs'`, which represents the directory where TensorBoard files are stored.
- `--max_new_tokens`: The maximum number of new tokens to generate. The default value is `2048`. This parameter only takes effect when `predict_with_generate` is set to True.
- `--do_sample`: Whether to use sampling during generation. The default value is `True`. This parameter only takes effect when `predict_with_generate` is set to True.
- `--temperature`: The temperature value for sampling during generation. The default value is `0.9`. This parameter only takes effect when `predict_with_generate` is set to True.
- `--top_k`: The value of k for top-k sampling during generation. The default value is `20`. This parameter only takes effect when `predict_with_generate` is set to True.
- `--top_p`: The cumulative probability threshold for top-p sampling during generation. The default value is `0.9`. This parameter only takes effect when `predict_with_generate` is set to True.
- `--repetition_penalty`: The repetition penalty applied during generation. The default value is `1.05`. This parameter only takes effect when `predict_with_generate` is set to True.


### infer.sh Command Line Arguments
- `--model_type`: Default value is `None`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--model_id_or_path`: Default value is `None`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`. It is recommended to use the `model_type` approach for specification.
- `--model_revision`: Default value is `None`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`. This parameter is not effective if `model_id_or_path` is `None` or if it refers to a local model directory.
- `--sft_type`: Default value is `'lora'`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--template_type`: Default value is `None`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--ckpt_dir`: Required field, value is the checkpoint path saved during the SFT phase, e.g., `'/path/to/your/vx_xxx/checkpoint-xxx'`.
- `--load_args_from_ckpt_dir`: Whether to load configuration information from the `sft_args.json` file in `ckpt_dir`. The default value is `True`. The imported keys include: `model_id_or_path`, `model_revision`, `sft_type`, `template_type`, `dtype`, `system`, `quantization_bit`, `bnb_4bit_comp_dtype`, `bnb_4bit_quant_type`, `bnb_4bit_use_double_quant`. If `eval_human` is set to False, the following keys will also be imported: `dataset`, `dataset_seed`, `dataset_test_ratio`, `check_dataset_strategy`, `custom_train_dataset_path`, `custom_val_dataset_path`.
- `--eval_human`: Whether to evaluate using the validation set from the dataset or manually evaluate the model. Default value is `False`. This allows us to get an intuitive understanding of the model's performance after fine-tuning.
- `--seed`: Default value is `42`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--dtype`: Default value is `None`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--dataset`: Default value is `'blossom-math-zh'`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`. This parameter only takes effect when `eval_human` is set to False.
- `--dataset_seed`: Default value is `42`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`. This parameter only takes effect when `eval_human` is set to False.
- `--dataset_test_ratio`: Default value is `0.01`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`. This parameter only takes effect when `eval_human` is set to False.
- `--show_dataset_sample`: Indicates the number of samples from the validation set to evaluate and display. Default value is `10`. This parameter only takes effect when `eval_human` is set to False.
- `--system`: Default value is `'you are a helpful assistant!'`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--max_length`: Default value is `2048`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--check_dataset_strategy`: The default value is `'none'`, For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--custom_train_dataset_path`: 默认值为`None`. 具体的含义参考README.md中的`自定义数据集`模块.
- `--custom_val_dataset_path`: 默认值为`None`. 具体的含义参考README.md中的`自定义数据集`模块.
- `--quantization_bit`: Default value is 0. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--bnb_4bit_comp_dtype`: Default value is `None`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`. This parameter is not effective if `quantization_bit` is set to 0.
- `--bnb_4bit_quant_type`: Default value is `'nf4'`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`. This parameter is not effective if `quantization_bit` is set to 0.
- `--bnb_4bit_use_double_quant`: Default value is `True`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`. This parameter is not effective if `quantization_bit` is set to 0.
- `--max_new_tokens`: Maximum number of new tokens to generate. Default value is `2048`.
- `--do_sample`: Whether to use greedy decoding or sampling for generation. Default value is `True`.
- `--temperature`: Default value is `0.9`. This parameter only takes effect when `do_sample` is set to True.
- `--top_k`: Default value is `20`. This parameter only takes effect when `do_sample` is set to True.
- `--top_p`: Default value is `0.9`. This parameter only takes effect when `do_sample` is set to True.
- `--repetition_penalty`: Default value is `1.05`.
- `--use_flash_attn`: Default value is `None`, which means 'auto'. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--ignore_args_error`: Default value is `False`. For specific parameter details, please refer to the `sft.sh Command Line Arguments`.
- `--stream`: Whether to use streaming output. Default value is `True`.
- `--merge_lora_and_save`: Whether to merge the lora weights into the base model and save the complete weights. Default value is `False`. The weights will be saved in a directory named `checkpoint-xxx-merged` at the same level as `ckpt_dir`, e.g., `'/path/to/your/vx_xxx/checkpoint-xxx-merged'`.
- `--overwrite_generation_config`: Whether to save the generation_config used for evaluation as a `generation_config.json` file. Default value is `False`. The generate_config file saved during training will be overwritten.
