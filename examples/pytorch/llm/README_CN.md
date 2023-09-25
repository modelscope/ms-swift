
<h1 align="center">大模型微调的例子</h1>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.8.4-5D91D4.svg"></a>
<a href="https://github.com/modelscope/swift/"><img src="https://img.shields.io/badge/ms--swift-%E2%89%A51.0.0-6FEBB9.svg">
</p>

<p align="center">
<a href="https://modelscope.cn/home">魔搭社区</a>
<br>
        中文&nbsp ｜ &nbsp<a href="README.md">English</a>
</p>


## 特性
1. 支持的SFT方法: [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314), 全参数微调
2. 支持的模型:
   1. qwen 系列: qwen-7b, [qwen-7b-chat](https://github.com/QwenLM/Qwen-7B)
   2. qwen-vl 系列: qwen-vl, [qwen-vl-chat](https://github.com/QwenLM/Qwen-VL)
   3. baichuan 系列: baichuan-7b, baichuan-13b, baichuan-13b-chat, baichuan2-7b, [baichuan2-7b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat/summary), baichuan2-13b, baichuan2-13b-chat
   4. chatglm2 系列: [chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary), chatglm2-6b-32k
   5. llama 系列: llama2-7b, llama2-7b-chat, llama2-13b, llama2-13b-chat, llama2-70b, [llama2-70b-chat](https://modelscope.cn/models/modelscope/Llama-2-70b-chat-ms/summary)
   6. openbuddy-llama 系列: openbuddy-llama2-13b, openbuddy-llama-65b, [openbuddy-llama2-70b](https://modelscope.cn/models/OpenBuddy/openbuddy-llama2-70b-v10.1-bf16/summary)
   7. internlm 系列: internlm-7b, internlm-7b-chat, internlm-7b-chat-8k, [internlm-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-20b/summary), [internlm-20b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-20b/summary)
   8. other: [polylm-13b](https://modelscope.cn/models/damo/nlp_polylm_13b_text_generation/summary), [seqgpt-560m](https://modelscope.cn/models/damo/nlp_seqgpt-560m/summary)
3. 支持的特性: 模型量化, DDP, 模型并行(device_map), gradient checkpointing, 梯度累加, 支持推送ModelScope Hub, 自定义数据集, 多模态和Agent SFT, 多轮对话, ...
4. 支持的数据集:
   1. NLP: [alpaca-en](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-en/summary)(gpt4), [alpaca-zh](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-zh/summary)(gpt4), finance-en, multi-alpaca-all, code-en, instinwild-en, instinwild-zh, cot-en, cot-zh, firefly-all-zh, poetry-zh, instruct-en, gpt4all-en, cmnli-zh, [jd-zh](https://modelscope.cn/datasets/DAMO_NLP/jd/summary), [dureader-robust-zh](https://modelscope.cn/datasets/modelscope/DuReader_robust-QG/summary), medical-en, medical-zh, medical-mini-zh, sharegpt-en, sharegpt-zh, [code-python-zh](https://modelscope.cn/datasets/codefuse-ai/CodeExercise-Python-27k/summary), [advertise-gen](https://modelscope.cn/datasets/lvjianjin/AdvertiseGen/summary)
   2. Agent: [damo-agent-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary), damo-agent-mini-zh
   3. 多模态: [coco-en](https://modelscope.cn/datasets/modelscope/coco_2014_caption/summary)
   4. 其他: [cls-fudan-news-zh](https://modelscope.cn/datasets/damo/zh_cls_fudan-news/files), [ner-jave-zh](https://modelscope.cn/datasets/damo/zh_ner-JAVE/summary)
5. 支持的对话模板: chatml(qwen), baichuan, chatglm2, llama, openbuddy-llama, default, default-generation

## 准备实验环境
实验环境: V100, A10, 3090, A100均可.
```bash
# 安装miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# 一直[ENTER], 最后一个选项yes即可
sh Miniconda3-latest-Linux-x86_64.sh

# conda虚拟环境搭建
conda create --name ms-sft python=3.10
conda activate ms-sft

# pip设置全局镜像与相关python包安装
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt -U
```

## 微调和推理
性能: full(优) > lora > qlora

训练显存: qlora(低,3090) > lora > full(2*A100)
```bash
# clone仓库, 安装ms-swift, 然后进入代码目录
git clone https://github.com/modelscope/swift.git
cd swift
pip install .
cd examples/pytorch/llm

# 微调(lora)+推理 qwen-7b-chat, 需要38GB显存.
# 你可以通过设置`--gradient_checkpointing true`来节约显存, 但这会略微降低训练速度.
# 如果你想在训练时, 将权重push到modelscope hub中, 你需要设置`--push_to_hub true`.
# 推荐的实验环境: A100
bash scripts/qwen_7b_chat/lora/sft.sh
# 如何你想要合并LoRA权重并保存，你需要设置`--merge_lora_and_save true`
bash scripts/qwen_7b_chat/lora/infer.sh

# 微调(lora+ddp)+推理 qwen-7b-chat, 需要2卡*38GB显存.
# 推荐的实验环境: A100
bash scripts/qwen_7b_chat/lora_ddp/sft.sh
bash scripts/qwen_7b_chat/lora_ddp/infer.sh

# 微调(lora+mp+ddp)+推理 qwen-7b-chat, 需要4卡*15GB显存.
# 推荐的实验环境: V100, 3090, A10
bash scripts/qwen_7b_chat/lora_mp_ddp/sft.sh
bash scripts/qwen_7b_chat/lora_mp_ddp/infer.sh

# 微调(qlora)+推理 qwen-7b-chat, 需要10GB显存.
# 如果你想要使用量化, 你需要`pip install bitsandbytes -U`
# 推荐的实验环境: V100, 3090, A10
bash scripts/qwen_7b_chat/qlora/sft.sh
bash scripts/qwen_7b_chat/qlora/infer.sh

# 微调(qlora+ddp)+推理 qwen-7b-chat, 需要2卡*14GB显存.
# 推荐的实验环境: V100, 3090, A10
bash scripts/qwen_7b_chat/qlora_ddp/sft.sh
bash scripts/qwen_7b_chat/qlora_ddp/infer.sh

# 微调(full+mp)+推理 qwen-7b-chat, 需要2卡*75G显存.
# 推荐的实验环境: A100
bash scripts/qwen_7b_chat/full_mp/sft.sh
bash scripts/qwen_7b_chat/full_mp/infer.sh

# 微调(full+mp+ddp)+推理 qwen-7b-chat, 需要4卡*75G显存.
# 推荐的实验环境: A100
bash scripts/qwen_7b_chat/full_mp_ddp/sft.sh
bash scripts/qwen_7b_chat/full_mp_ddp/infer.sh

# 更多的scripts脚本, 可以看`scripts`文件夹.
```

## 拓展数据集
1. 如果你想要拓展数据集, 你可以修改`utils/dataset.py`文件中的`DATASET_MAPPING`加入一组映射, key为数据集的名称, value为获取数据集的函数, 该函数需要返回一个`HfDataset`. 其中指令微调(单轮对话)需包含`query`, `response`字段, 分别代表指令微调的用户询问和AI助手的回答, 具体可以参考`alpaca-zh`数据集. 如果是多轮对话, 则需要额外加上`history`字段, 具体可以参考`damo-agent-mini-zh`数据集. 如果每个数据集样例的具有不同的`system`, 则需要额外加上system字段.
2. 如果你想要拓展模型, 你可以修改`utils/model.py`文件中的`MODEL_MAPPING`. `model_id`可以指定为本地路径, 这种情况下, `revision`参数不起作用.
3. 如果你想要拓展template, 你可以修改`utils/preprocess.py`文件中的`TEMPLATE_MAPPING`.
