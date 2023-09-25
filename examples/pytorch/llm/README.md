
<h1 align="center">LLM SFT Example</h1>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.8.4-5D91D4.svg"></a>
<a href="https://github.com/modelscope/swift/"><img src="https://img.shields.io/badge/ms--swift-%E2%89%A51.1.0-6FEBB9.svg"></a>
</p>

<p align="center">
<a href="https://modelscope.cn/home">Modelscope Hub</a>
<br>
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish
</p>

## Features
1. supported SFT methods: [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314), full(full parameter fine-tuning)
2. supported models:
   1. qwen series: qwen-7b, [qwen-7b-chat](https://github.com/QwenLM/Qwen-7B)
   2. qwen-vl series: qwen-vl, [qwen-vl-chat](https://github.com/QwenLM/Qwen-VL)
   3. baichuan series: baichuan-7b, baichuan-13b, baichuan-13b-chat, baichuan2-7b, [baichuan2-7b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat/summary), baichuan2-13b, baichuan2-13b-chat
   4. chatglm2 series: [chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary), chatglm2-6b-32k
   5. llama series: llama2-7b, llama2-7b-chat, llama2-13b, llama2-13b-chat, llama2-70b, [llama2-70b-chat](https://modelscope.cn/models/modelscope/Llama-2-70b-chat-ms/summary)
   6. openbuddy-llama series: openbuddy-llama2-13b, openbuddy-llama-65b, [openbuddy-llama2-70b](https://modelscope.cn/models/OpenBuddy/openbuddy-llama2-70b-v10.1-bf16/summary)
   7. internlm series: internlm-7b, internlm-7b-chat, internlm-7b-chat-8k, [internlm-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-20b/summary), [internlm-20b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-20b/summary)
   8. other: [polylm-13b](https://modelscope.cn/models/damo/nlp_polylm_13b_text_generation/summary), [seqgpt-560m](https://modelscope.cn/models/damo/nlp_seqgpt-560m/summary)
3. supported features: quantization, DDP, model parallelism(device map), gradient checkpointing, gradient accumulation, pushing to modelscope hub, custom datasets, multimodal and agent SFT, mutli-round chat, ...
4. supported datasets:
   1. NLP: [alpaca-en](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-en/summary)(gpt4), [alpaca-zh](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-zh/summary)(gpt4), finance-en, multi-alpaca-all, code-en, instinwild-en, instinwild-zh, cot-en, cot-zh, firefly-all-zh, poetry-zh, instruct-en, gpt4all-en, cmnli-zh, [jd-zh](https://modelscope.cn/datasets/DAMO_NLP/jd/summary), [dureader-robust-zh](https://modelscope.cn/datasets/modelscope/DuReader_robust-QG/summary), medical-en, medical-zh, medical-mini-zh, sharegpt-en, sharegpt-zh, [code-python-zh](https://modelscope.cn/datasets/codefuse-ai/CodeExercise-Python-27k/summary), [advertise-gen](https://modelscope.cn/datasets/lvjianjin/AdvertiseGen/summary)
   2. Agent: [damo-agent-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary), damo-agent-mini-zh
   3. Multi-Modal: [coco-en](https://modelscope.cn/datasets/modelscope/coco_2014_caption/summary)
   4. Other: [cls-fudan-news-zh](https://modelscope.cn/datasets/damo/zh_cls_fudan-news/files), [ner-jave-zh](https://modelscope.cn/datasets/damo/zh_ner-JAVE/summary)
5. supported templates: chatml(qwen), baichuan, chatglm2, llama, openbuddy-llama, default, default-generation

## Prepare the Environment
Experimental environment: V100, A10, 3090, A100, ...
```bash
# Installing miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

# Setting up a conda virtual environment
conda create --name ms-sft python=3.10
conda activate ms-sft

# Setting up a global mirror for pip and installing related Python packages
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt -U
```

## Run SFT and Inference
Performace: full(nice) > lora > qlora

Training GPU memory: qlora(low,3090) > lora > full(2*A100)
```bash
# Clone the repository, install ms-swift from source code and enter the code directory.
git clone https://github.com/modelscope/swift.git
cd swift
pip install .
cd examples/pytorch/llm

# sft lora and infer qwen-7b-chat, Requires 38GB GPU memory.
# You can save GPU memory by setting `--gradient_checkpointing true`, but this will slightly decrease the training speed.
# If you want to push weights into modelscope hub during training, you need to set '--push_to_hub true'.
# Recommended experimental environment: A100
bash scripts/qwen_7b_chat/lora/sft.sh
# If you want to merge LoRA weight and save it, you need to set `--merge_lora_and_save true`.
bash scripts/qwen_7b_chat/lora/infer.sh

# sft(lora+ddp) and infer qwen-7b-chat, Requires 2*38GB GPU memory.
# Recommended experimental environment: A100
bash scripts/qwen_7b_chat/lora_ddp/sft.sh
bash scripts/qwen_7b_chat/lora_ddp/infer.sh

# sft(lora+mp+ddp) and infer qwen-7b-chat, Requires 4*15GB GPU memory.
# Recommended experimental environment: V100, A10, 3090
bash scripts/qwen_7b_chat/lora_mp_ddp/sft.sh
bash scripts/qwen_7b_chat/lora_mp_ddp/infer.sh

# sft(qlora) and infer qwen-7b-chat, Requires 10GB GPU memory.
# If you want to use quantification, you need to `pip install bitsandbytes -U`
# Recommended experimental environment: V100, A10, 3090
bash scripts/qwen_7b_chat/qlora/sft.sh
bash scripts/qwen_7b_chat/qlora/infer.sh

# sft(qlora+ddp) and infer qwen-7b-chat, Requires 2*14GB GPU memory.
# Recommended experimental environment: V100, A10, 3090
bash scripts/qwen_7b_chat/qlora_ddp/sft.sh
bash scripts/qwen_7b_chat/qlora_ddp/infer.sh

# sft(full+mp) and infer qwen-7b-chat, Requires 2*75GB GPU memory.
# Recommended experimental environment: A100
bash scripts/qwen_7b_chat/full_mp/sft.sh
bash scripts/qwen_7b_chat/full_mp/infer.sh

# sft(full+mp+ddp) and infer qwen-7b-chat, Requires 4*75GB GPU memory.
# Recommended experimental environment: A100
bash scripts/qwen_7b_chat/full_mp_ddp/sft.sh
bash scripts/qwen_7b_chat/full_mp_ddp/infer.sh

# For more scripts, please see `scripts/` folder.
```

## Extend Datasets
1. If you want to expand the dataset, you can modify the `DATASET_MAPPING` in the `utils/dataset.py` file by adding a set of mappings. The key represents the name of the dataset, and the value represents the function to retrieve the dataset, which should return an `HfDataset`. For fine-tuning with single-turn conversations, the dataset should include the `query` and `response` fields, representing the user's query and the assistant's response for fine-tuning, respectively. You can refer to the `alpaca-zh` dataset for more details. If you're dealing with multi-turn conversations, you need to include an additional `history` field, which can be referenced from the `damo-agent-mini-zh` dataset. If each example in the dataset has a different `system` field, you'll need to include the `system` field as well.
2. If you want to expand the model, you can modify the `MODEL_MAPPING` in the `utils/model.py` file. The `model_id` can be specified as a local path, in which case the `revision` parameter is not used.
3. If you want to expand the templates, you can modify the `TEMPLATE_MAPPING` in the `utils/preprocess.py` file.
