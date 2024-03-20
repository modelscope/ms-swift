# SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning)

<p align="center">
    <br>
    <img src="resources/banner.png"/>
    <br>
<p>

<p align="center">
<a href="https://modelscope.cn/home">魔搭社区</a>
<br>
        中文&nbsp ｜ &nbsp<a href="README.md">English</a>&nbsp ｜ &nbsp<a href="https://github.com/modelscope/swift/blob/main/docs/source/GetStarted/%E5%BF%AB%E9%80%9F%E4%BD%BF%E7%94%A8.md">文档</a>
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.9.5-5D91D4.svg"></a>
<a href="https://pypi.org/project/ms-swift/"><img src="https://badge.fury.io/py/ms-swift.svg"></a>
<a href="https://github.com/modelscope/swift/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/swift"></a>
<a href="https://pepy.tech/project/ms-swift"><img src="https://pepy.tech/badge/ms-swift"></a>
<a href="https://github.com/modelscope/swift/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
</p>

##  📖 目录
- [简介](#-简介)
- 🔥[新闻](#-新闻)
- [安装](#-安装)
- 🔥[快速开始](#-快速开始)
- [文档](#-文档)
- [License](#license)
- [引用](#-引用)
- [联系我们](#-联系我们)

## 📝 简介
LLM-Adapter支持近200种LLM和MLLM（多模态大模型）的训练推理评测和部署。开发者可以直接将我们的框架应用到自己的Research和生产环境中，实现模型训练评测到应用的完整链路。此外，我们除支持了[PEFT](https://github.com/huggingface/peft)提供的Adapters外，也提供了一个完整的适配器库以支持最新的训练技术，如NEFTune、LoRA+、LLaMA-PRO等，这个适配器库可以脱离训练脚本直接使用在自己的自定脚本中。

为方便不熟悉深度学习的用户使用，我们提供了一个Gradio的web-ui用于控制训练和推理，并提供了配套的深度学习课程和最佳实践供新手入门。

此外，我们也在拓展其他模态的能力，目前我们支持了AnimateDiff的全参数训练和LoRA训练。

## 🎉 新闻
- 🔥2024.03.12: 支持**deepseek-vl**系列推理和微调, 最佳实践可以查看[这里](https://github.com/modelscope/swift/tree/main/docs/source/Multi-Modal/deepseek-vl最佳实践.md).
- 🔥2024.03.11: 支持[GaLore](https://arxiv.org/abs/2403.03507), 用于在全参数训练中有效减小显存占用至原来的1/2.
- 🔥2024.03.10: Qwen1.5-7B-Chat与Qwen1.5-72B-Chat从微调到部署[全流程最佳实践](https://github.com/modelscope/swift/blob/main/docs/source/LLM/Qwen1.5%E5%85%A8%E6%B5%81%E7%A8%8B%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md).
- 🔥2024.03.09: 支持MAMBA模型的训练和推理, 使用[这个脚本](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/mamba-1.4b/lora/sft.sh)来开始训练！.
- 2024.03.09: 支持AQLM量化模型的训练和推理, 使用[这个脚本](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/llama2_7b_aqlm_2bit_1x16/lora/sft.sh)开始训练！
- 2024.03.06: 支持AWQ量化模型的训练和推理, 使用[这个Qwen1.5-AWQ模型脚本](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/qwen1half_7b_chat_awq/lora/sft.sh)开始训练, 并支持[yi-9b](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/yi_9b/lora_zero3)的训练和推理.
- 🔥2024.02.29: 支持[LLaMA PRO](https://arxiv.org/pdf/2401.02415.pdf), 使用[这个脚本](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/yi_6b_chat/llamapro/sft.sh)即可开始训练.
- 🔥2024.02.29: 支持[LoRA+](https://arxiv.org/pdf/2402.12354.pdf), 使用[这个脚本](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/yi_6b_chat/lorap/sft.sh)即可开始训练.
- 2024.02.25: 支持`swift export`, 对模型进行**AWQ/GPTQ**量化导出, 以及推送ModelScope Hub. 具体可以查看文档: [LLM量化文档](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E9%87%8F%E5%8C%96%E6%96%87%E6%A1%A3.md).
<details><summary>更多</summary>

- 2024.02.22: 支持gemma系列: gemma-2b, [gemma-2b-instruct](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/gemma_2b_instruct), gemma-7b, gemma-7b-instruct.
- 2024.02.16: 支持deepseek-math系列: deepseek-math-7b, deepseek-math-7b-instruct, deepseek-math-7b-chat.
- 🔥2024.02.05: 支持**Qwen1.5**系列模型, 支持的所有Qwen1.5系列模型请查看[模型列表](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.md#%E6%A8%A1%E5%9E%8B). 提供了[qwen1half-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen1half_7b_chat), [qwen1half-7b-chat-int8](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen1half_7b_chat_int8)微调的脚本.
- 2024.02.05: 支持扩散模型如**SDXL**, **SD**, **ControlNet**的训练, 同时也支持**DreamBooth**的训练, 详情可以查看对应的[训练脚本](https://github.com/modelscope/swift/tree/main/examples/pytorch/sdxl/scripts).
- 2024.02.01: 支持minicpm系列: [minicpm-2b-sft-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/minicpm_2b_sft_chat), minicpm-2b-chat.
- 🔥2024.02.01: 支持数据集打混来减少 **灾难性遗忘问题**. 使用`--train_dataset_mix_ratio 2.0`开启训练！同时我们也开源了通用知识数据集 [ms-bench](https://www.modelscope.cn/datasets/iic/ms_bench/summary).
- 🔥2024.02.01: 支持Agent训练！Agent训练算法源自这篇[论文](https://arxiv.org/pdf/2309.00986.pdf). 我们也增加了[ms-agent](https://www.modelscope.cn/datasets/iic/ms_agent/summary)这个优质的agent数据集. 使用[这个脚本](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/qwen_7b_chat/lora/sft.sh)开启Agent训练!
- 🔥2024.02.01: 支持在DPO训练中增加SFT loss来减少KL散度loss造成的生成重复问题.
- 2024.02.01: 支持在训练中使用AdaLoRA和IA3两个adapter.
- 2024.02.01: 支持在AnimateDiff训练中使用`--merge_lora`参数.
- 2024.01.30: 支持[internlm-xcomposer2-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/internlm_xcomposer2_7b_chat).
- 🔥2024.01.30: 支持[ZeRO-3](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat/full_ddp_zero3/), 只需要指定`--deepspeed default-zero3`即可.
- 2024.01.29: 支持internlm2-math系列: internlm2-math-7b, internlm2-math-7b-chat, internlm2-math-20b, internlm2-math-20b-chat.
- 🔥2024.01.26: 支持[yi-vl-6b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yi_vl_6b_chat), yi-vl-34b-chat.
- 2024.01.24: 支持codefuse-codegeex2-6b-chat, codefuse-qwen-14b-chat.
- 2024.01.23: 支持orion系列: orion-14b, [orion-14b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/orion_14b_chat).
- 2024.01.20: 支持[xverse-13b-256k](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/xverse_13b_256k), xverse-65b-v2, xverse-65b-chat.
- 🔥2024.01.17: 支持internlm2系列: internlm2-7b-base, internlm2-7b, [internlm2-7b-sft-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/internlm2_7b_sft_chat), internlm2-7b-chat, internlm2-20b-base, internlm2-20b, internlm2-20b-sft-chat, internlm2-20b-chat.
- 2024.01.15: 支持yuan系列: yuan2-2b-instruct, [yuan2-2b-janus-instruct](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yuan2_2b_janus_instruct), yuan2-51b-instruct, yuan2-102b-instruct.
- 🔥2024.01.12: 支持**deepseek-moe**系列: deepseek-moe-16b, [deepseek-moe-16b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/deepseek_moe_16b_chat).
- 🔥2024.01.04: 支持**VLLM部署**, 兼容**OpenAI API**样式, 具体可以查看[VLLM推理加速与部署](https://github.com/modelscope/swift/blob/main/docs/source/LLM/VLLM推理加速与部署.md#部署).
- 2024.01.04: 更新[Benchmark](https://github.com/modelscope/swift/blob/main/docs/source/LLM/Benchmark.md), 方便查看不同模型训练的速度和所需显存.
- 🔥 2023.12.29: 支持web-ui进行sft训练和推理，安装ms-swift后使用`swift web-ui`开启
- 🔥 2023.12.29: 支持 DPO RLHF(Reinforcement Learning from Human Feedback) 和三个用于此任务的数据集: AI-ModelScope/stack-exchange-paired 以及 AI-ModelScope/hh-rlhf 以及 AI-ModelScope/hh_rlhf_cn. 查看[文档](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E4%BA%BA%E7%B1%BB%E5%AF%B9%E9%BD%90%E8%AE%AD%E7%BB%83%E6%96%87%E6%A1%A3.md)开启训练！
- 🔥 2023.12.28: 支持SCEdit! 该tuner可显著降低U-Net中的显存占用，并支持低显存可控图像生成（取代ControlNet），阅读下面的章节来了解详细信息
- 2023.12.23: 支持[codegeex2-6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/codegeex2_6b).
- 2023.12.19: 支持[phi2-3b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/phi2_3b).
- 2023.12.18: 支持VLLM进行推理加速.
- 2023.12.15: 支持deepseek, deepseek-coder系列: deepseek-7b, deepseek-7b-chat, deepseek-67b, deepseek-67b-chat, openbuddy-deepseek-67b-chat, deepseek-coder-1_3b, deepseek-coder-1_3b-instruct, deepseek-coder-6_7b, deepseek-coder-6_7b-instruct, deepseek-coder-33b, deepseek-coder-33b-instruct.
- 2023.12.13: 支持mistral-7b-instruct-v2, [mixtral-moe-7b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/mixtral_7b_moe), [mixtral-moe-7b-instruct](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/mixtral_7b_moe_instruct).
- 2023.12.09: 支持`freeze_parameters`参数, 作为lora和全参数训练的折中方案. 对应的sh可以查看[full_freeze_ddp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/full_freeze_ddp). 支持`disable_tqdm`, `lazy_tokenize`, `preprocess_num_proc`参数, 具体可以查看[命令行参数](https://github.com/modelscope/swift/blob/main/docs/source/LLM/命令行参数.md).
- 2023.12.08: 支持[sus-34b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/sus_34b_chat), 支持yi-6b-200k, yi-34b-200k.
- 2023.12.07: 支持[Multi-Node DDP训练](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E5%BE%AE%E8%B0%83%E6%96%87%E6%A1%A3.md#%E4%BD%BF%E7%94%A8cli).
- 2023.12.05: 支持模型: zephyr-7b-beta-chat, openbuddy-zephyr-7b-chat. 支持数据集: hc3-zh, hc3-en.
- 🔥 2023.12.02: [自我认知微调最佳实践](https://github.com/modelscope/swift/blob/main/docs/source/LLM/自我认知微调最佳实践.md), **10分钟对大模型进行自我认知微调**, 创建专属于自己的大模型.
- 🔥 2023.11.30: 支持**qwen-1_8b**, **qwen-72b**, **qwen-audio**系列模型的训练的推理. 对应的sh脚本可以查看[qwen_1_8b_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_1_8b_chat), [qwen_72b_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_72b_chat), [qwen_audio_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_audio_chat)
- 🔥 2023.11.29: 支持**AnimateDiff**的训练和推理
- 🔥 2023.11.24: 支持**yi-34b-chat**, **codefuse-codellama-34b-chat**模型. 对应的sh脚本可以查看[yi_34b_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yi_34b_chat), [codefuse_codellama_34b_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/codefuse_codellama_34b_chat).
- 🔥 2023.11.18: 支持**tongyi-finance-14b**系列模型: tongyi-finance-14b, tongyi-finance-14b-chat, tongyi-finance-14b-chat-int4. 对应的sh脚本可以查看[tongyi_finance_14b_chat_int4](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/tongyi_finance_14b_chat_int4).
- 2023.11.16: 支持更多模型的**flash attn**支持: qwen系列, qwen-vl系列, llama系列, openbuddy系列, mistral系列, yi系列, ziya系列. 请使用`use_flash_attn`参数.
- 🔥 2023.11.11: 支持**NEFTune**, 使用`Swift.prepare_model(model, NEFTuneConfig())`即可开启.
- 🔥 2023.11.11: 支持**命令行**训练推理和**Web-UI**推理, 详情可以查看下方的`使用Swift CLI运行`章节.
- 🔥 2023.11.11: 支持模型训练后的**部署**链路(vllm/chatglm.cpp/xinference)，详情可以查看[官方文档](./docs/source/GetStarted/部署指南.md).
- 🔥 2023.11.10: 支持**bluelm**系列模型: bluelm-7b, bluelm-7b-chat, bluelm-7b-32k, bluelm-7b-chat-32k. 对应的sh脚本可以查看[bluelm_7b_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/bluelm_7b_chat).
- 🔥 2023.11.08: 支持**xverse-65b**模型的训练和推理流程，脚本在[xverse_65b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/xverse_65b).
- 🔥 2023.11.07: 支持**yi-6b**, **yi-34b**模型的训练和推理流程，脚本在[yi_6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yi_6b), [yi_34b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yi_34b).
- 🔥 2023.10.30: 支持 **QA-LoRA** 和 **LongLoRA**两种新的tuners.
- 🔥 2023.10.30: 支持使用**ROME**(Rank One Model Editing)来编辑模型，在无需训练的情况下即可给模型灌注新知识！
- 2023.10.30: 支持**skywork-13b**系列模型: skywork-13b, skywork-13b-chat. 对应的sh脚本可以查看[skywork_13b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/skywork_13b).
- 🔥 2023.10.27: 支持**chatglm3**系列模型: chatglm3-6b-base, chatglm3-6b, chatglm3-6b-32k. 对应的sh脚本可以查看[chatglm3_6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/chatglm3_6b).
- 🔥 2023.10.17: 支持**int4**, **int8**模型的SFT: qwen-7b-chat-int4, qwen-14b-chat-int4, qwen-vl-chat-int4, baichuan2-7b-chat-int4, baichuan2-13b-chat-int4, qwen-7b-chat-int8, qwen-14b-chat-int8.
- 2023.10.15: 支持**ziya2-13b**系列模型: ziya2-13b, ziya2-13b-chat.
- 2023.10.12: 支持**mistral-7b**系列模型: openbuddy-mistral-7b-chat, mistral-7b, mistral-7b-instruct.
- 🔥 2023.10.07: 支持**DeepSpeed ZeRO-2**, 使得lora(不仅仅是qlora)可以在双卡A10上运行DDP.
- 2023.10.04: 支持更多数学, 法律, SQL, 代码领域的数据集: blossom-math-zh, school-math-zh, text2sql-en, sql-create-context-en, lawyer-llama-zh, tigerbot-law-zh, leetcode-python-en.
- 🔥 2023.09.25: 支持**qwen-14b**系列: qwen-14b, qwen-14b-chat.
- 2023.09.18: 支持**internlm-20b**系列: internlm-20b, internlm-20b-chat.
- 2023.09.12: 支持**MP+DDP**对全参数训练进行加速.
- 2023.09.05: 支持**openbuddy-llama2-70b-chat**.
- 2023.09.03: 支持**baichuan2**系列: baichuan2-7b, baichuan2-7b-chat, baichuan2-13b, baichuan2-13b-chat.
</details>

## 🛠️ 安装

SWIFT在Python环境中运行。请确保您的Python版本高于3.8。

- 方法1：使用pip命令安装SWIFT：

```shell
# 全量能力
pip install ms-swift[all] -U
# 仅使用LLM
pip install ms-swift[llm] -U
# 仅使用AIGC
pip install ms-swift[aigc] -U
# 仅使用Adapters
pip install ms-swift -U
```

- 方法2：通过源代码安装SWIFT（方便运行训练推理脚本），请运行以下命令：

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .[llm]
```

SWIFT依赖torch>=1.13，建议torch>=2.0.0。

- 方法3：在我们的Docker镜像中使用SWIFT

```shell
# China-Hangzhou image
docker pull registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.1.2-tf2.14.0-1.13.1
# US-west image 
docker pull registry.us-west-1.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.1.2-tf2.14.0-1.13.1
```

## 🚀 快速开始

本章节介绍基本使用，更丰富的使用方式请查看[文档部分](#-文档)。

### web-ui

```shell
swift web-ui
```

### 训练

#### 支持的训练过程

| 训练过程 | 训练方式                                                   |
| -------- | ---------------------------------------------------------- |
| 预训练   | 续写                                                       |
| 微调     | 单轮/多轮/Agent训练/自我认知/多模态图文问答/多模态语音问答 |
| 人类对齐 | DPO                                                        |
| 文生图   | DreamBooth                                                 |
| 文生视频 | -                                                          |


#### 单卡训练

通过如下命令启动单卡微调：

```shell
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset ms-bench-mini \
    --train_dataset_sample 1000 \
    --logging_steps 5 \
    --max_length 2048 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.4 \
    --output_dir output \
    --lora_target_modules ALL \
    --self_cognition_sample 500 \
    --model_name 小黄 'Xiao Huang' \
    --model_author 魔搭 ModelScope
```

#### 模型并行训练

模型并行训练在上述命令基础上修改`CUDA_VISIBLE_DEVICES`环境变量：

```shell
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset ms-bench-mini \
    --train_dataset_sample 1000 \
    --logging_steps 5 \
    --max_length 2048 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.4 \
    --output_dir output \
    --lora_target_modules ALL \
    --self_cognition_sample 500 \
    --model_name 小黄 'Xiao Huang' \
    --model_author 魔搭 ModelScope
```

#### 数据并行训练

数据并行训练在上述命令基础上修改`NPROC_PER_NODE`环境变量：

```shell
# 如果CUDA_VISIBLE_DEVICES数量是NPROC_PER_NODE的整数倍（大于1），则按照NPROC_PER_NODE启动数据并行，CUDA_VISIBLE_DEVICES数量/NPROC_PER_NODE启动模型并行
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset ms-bench-mini \
    --train_dataset_sample 1000 \
    --logging_steps 5 \
    --max_length 2048 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.4 \
    --output_dir output \
    --lora_target_modules ALL \
    --self_cognition_sample 500 \
    --model_name 小黄 'Xiao Huang' \
    --model_author 魔搭 ModelScope
```

#### Deepspeed训练

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset ms-bench-mini \
    --train_dataset_sample 1000 \
    --logging_steps 5 \
    --max_length 2048 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.4 \
    --output_dir output \
    --lora_target_modules ALL \
    --self_cognition_sample 500 \
    --model_name 小黄 'Xiao Huang' \
    --model_author 魔搭 ModelScope \
    --deepspeed default-zero3
```

### 推理

```shell
swift infer --model_type qwen1half-7b-chat --stream true
```

##### VLLM推理

```shell
swift infer --model_type qwen1half-7b-chat --infer_backend vllm --stream true
```

### 评测

```shell
swift eval --model_type qwen1half-7b-chat --eval_dataset mmlu ceval
```

### 导出

```shell
swift export --model_type qwen1half-7b-chat --quant_bits 4 --quant_method awq
```

### 部署

```shell
swift deploy --model_type qwen1half-7b-chat --infer_backend vllm --max_model_len 8192
```

### 支持的模型

#### LLM模型

| 模型类型                                            | 模型介绍                                                     | 语言      | 模型大小                  | 模型类型                                |
| --------------------------------------------------- | ------------------------------------------------------------ | --------- | ------------------------- | --------------------------------------- |
| Qwen/Qwen1.5                                        | [通义千问1.0和1.5系列模型](https://github.com/QwenLM)        | 中文/英文 | 1.8B-72B,包含量化版本     | base模型/chat模型                       |
| ChatGLM2/ChatGLM3/Codegeex2                         | [智谱ChatGLM系列模型](https://github.com/THUDM/)             | 中文/英文 | 6B                        | base模型/chat模型                       |
| Baichuan/Baichuan2                                  | [百川1和百川2](https://github.com/baichuan-inc)              | 中文/英文 | 7B-13B                    | base模型/chat模型                       |
| Yuan2                                               | [浪潮源系列模型](https://github.com/IEIT-Yuan)               | 中文/英文 | 2B-102B                   | chat模型                                |
| XVerse                                              | [元象系列模型](https://github.com/xverse-ai)                 | 中文/英文 | 7B-65B                    | base模型/chat模型                       |
| LLaMA2                                              | [LLaMA2系列模型](https://github.com/facebookresearch/llama)  | 英文      | 7B-70B，包含量化版本      | base模型/chat模型                       |
| Mistral/Mistral-MoE                                 | [Mistral系列模型](https://github.com/mistralai/mistral-src)  | 英文      | 7B，包含量化版本和MoE版本 | base模型/chat模型                       |
| YI                                                  | [01AI的YI系列模型](https://github.com/01-ai)                 | 中文/英文 | 6B-34B                    | base模型/chat模型                       |
| InternLM/InternLM2/InternLM2-Math                   | [浦江实验室书生浦语系列模型](https://github.com/InternLM/InternLM) | 中文/英文 | 1.8B-20B                  | base模型/chat模型/数学模型              |
| DeepSeek/DeepSeek-Coder/DeepSeek-Math               | [幻方系列模型](https://github.com/deepseek-ai)               | 中文/英文 | 1.3B-67B                  | base模型/chat模型/代码生成模型/数学模型 |
| MAMBA                                               | [MAMBA时序卷积模型](https://github.com/state-spaces/mamba)   | 英文      | 130M-2.8B                 | base模型                                |
| Gemma                                               | [Google Gemma系列模型](https://github.com/google/gemma_pytorch) | 英文      | 2B-7B                     | base模型/chat模型                       |
| MiniCPM                                             | [OpenBmB MiniCPM系列模型](https://github.com/OpenBMB/MiniCPM) | 中文/英文 | 2B-3B                     | chat模型                                |
| OpenBuddy                                           | [OpenBuddy系列模型](https://github.com/OpenBuddy/OpenBuddy)  | 中文/英文 | 7B-67B                    | base模型/chat模型                       |
| Orion                                               | [猎户星空系列模型](https://github.com/OrionStarAI)           | 中文/英文 | 14B                       | base模型/chat模型                       |
| BlueLM                                              | [VIVO蓝心大模型](https://github.com/vivo-ai-lab/BlueLM)      | 中文/英文 | 7B                        | base模型/chat模型                       |
| Ziya2                                               | [封神榜系列模型](https://github.com/IDEA-CCNL/Fengshenbang-LM) | 中文/英文 | 13B                       | base模型/chat模型                       |
| Skywork                                             | [昆仑天工系列模型](https://github.com/SkyworkAI/Skywork)     | 中文/英文 | 13B                       | base模型/chat模型                       |
| Zephyr                                              | 基于Mistral的zephyr系列模型                                  | 英文      | 7B                        | chat模型                                |
| PolyLM                                              | [通义实验室自研的PolyLM系列模型](https://github.com/DAMO-NLP-MT/PolyLM) | 多语种    | 13B                       | base模型                                |
| SeqGPT                                              | [通义实验室自研的文本理解模型，用于信息抽取和文本分类](https://github.com/Alibaba-NLP/SeqGPT) | 中文      | 560M                      | 语义理解模型                            |
| SUS                                                 | [南方科技大学基于YI Fine-Tune的模型](https://github.com/SUSTech-IDEA/SUS-Chat) | 中文/英文 | 34B                       | chat模型                                |
| Tongyi-Finance                                      | [通义金融系列模型](https://github.com/QwenLM/Qwen)           | 中文/英文 | 13B                       | 经济类目base模型/chat模型               |
| CodeFuse-CodeLLaMA/CodeFuse-Codegeex2/CodeFuse-Qwen | [蚂蚁CodeFuse系列模型](https://github.com/codefuse-ai)       | 中文/英文 | 6B-34B                    | 代码生成模型                            |
| phi2                                                | 微软PHI2模型                                                 | 英文      | 3B                        | 生成模型                                |

#### MLLM模型

| 模型类型        | 模型介绍                                                     | 语言      | 模型大小         | 模型类型          |
| --------------- | ------------------------------------------------------------ | --------- | ---------------- | ----------------- |
| Qwen-VL         | [通义千问视觉模型](https://github.com/QwenLM)                | 中文/英文 | 7B，包含量化版本 | base模型/chat模型 |
| Qwen-Audio      | [通义千问语音模型](https://github.com/QwenLM)                | 中文/英文 | 7B               | base模型/chat模型 |
| YI-VL           | [01AI的YI系列视觉模型](https://github.com/01-ai)             | 中文/英文 | 6B-34B           | chat模型          |
| xcomposer2      | [浦江实验室书生浦语视觉模型](https://github.com/InternLM/InternLM) | 中文/英文 | 7B               | chat模型          |
| DeepSeek-VL     | [幻方系列视觉模型](https://github.com/deepseek-ai)           | 中文/英文 | 1.3B-7B          | chat模型          |
| MiniCPM-VL      | [OpenBmB MiniCPM视觉模型](https://github.com/OpenBMB/MiniCPM) | 中文/英文 | 3B               | chat模型          |
| CogAgent/CogVLM | [智谱ChatGLM视觉问答和Agent模型](https://github.com/THUDM/)  | 中文/英文 | 17B-18B          | chat模型          |

#### 扩散模型

| 模型类型         | 模型介绍                                                     | 语言 | 模型类型 |
| ---------------- | ------------------------------------------------------------ | ---- | -------- |
| AnimateDiff      | [AnimateDiff动画模型](https://github.com/guoyww/AnimateDiff) | 英文 | 文生视频 |
| SD1.5/SD2.0/SDXL | [StabilityAI系列扩散模型](https://github.com/Stability-AI)   | 英文 | 文生图   |

### 支持的开源数据集

| 数据集类型 | 训练任务 | 文档                                                         |
| ---------- | :------- | ------------------------------------------------------------ |
| 通用       | 微调     | 🔥ms-bench, 🔥ms-bench-mini, 🔥alpaca-en(gpt4), 🔥alpaca-zh(gpt4), multi-alpaca-all, instinwild-en, instinwild-zh, cot-en, cot-zh, firefly-all-zh, instruct-en, gpt4all-en, sharegpt-en, sharegpt-zh, tulu-v2-sft-mixture, wikipedia-zh, open-orca, open-orca-gpt4, sharegpt-gpt4, 🔥sharegpt-gpt4-mini. |
| Agent      | 微调     | 🔥ms-agent, damo-mini-agent-zh, damo-agent-zh, agent-instruct-all-en. |
| 通用       | 人类对齐 | 🔥hh-rlhf-cn, stack-exchange-paired, hh-rlhf-harmless-base, hh-rlhf-helpful-base, hh-rlhf-helpful-online, hh-rlhf-helpful-rejection-sampled, hh-rlhf-red-team-attempts, hh-rlhf-cn-harmless-base-cn, hh-rlhf-cn-helpful-base-cn, hh-rlhf-cn-harmless-base-en, hh-rlhf-cn-helpful-base-en. |
| 代码       | 微调     | code-alpaca-en, 🔥leetcode-python-en, 🔥codefuse-python-en, 🔥codefuse-evol-instruction-zh. |
| 医疗       | 微调     | medical-en, medical-zh, medical-mini-zh, 🔥disc-med-sft-zh.   |
| 法律       | 微调     | lawyer-llama-zh, tigerbot-law-zh, 🔥disc-law-sft-zh.          |
| 数学       | 微调     | 🔥blossom-math-zh, school-math-zh, open-platypus-en.          |
| SQL        | 微调     | text2sql-en, 🔥sql-create-context-en.                         |
| 文本生成   | 微调     | 🔥advertise-gen-zh, 🔥dureader-robust-zh.                      |
| 分类       | 微调     | cmnli-zh, 🔥cmnli-mini-zh, 🔥jd-sentiment-zh, 🔥hc3-zh, 🔥hc3-en. |
| 量化辅助   | 量化     | pileval.                                                     |
| 其他       | 微调     | finance-en, poetry-zh, webnovel-zh, generated-chat-zh, cls-fudan-news-zh, ner-jave-zh. |
| 视觉       | 微调     | coco-en, 🔥coco-mini-en, coco-mini-en-2, capcha-images.       |
| 音频       | 微调     | aishell1-zh, 🔥aishell1-mini-zh.                              |

### 支持的技术

| 技术名称                                                     |
| ------------------------------------------------------------ |
| 🔥LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685) |
| 🔥LoRA+: [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/pdf/2402.12354.pdf) |
| 🔥LLaMA PRO: [LLAMA PRO: Progressive LLaMA with Block Expansion](https://arxiv.org/pdf/2401.02415.pdf) |
| 🔥SCEdit: [SCEdit: Efficient and Controllable Image Diffusion Generation via Skip Connection Editing](https://arxiv.org/abs/2312.11392)  < [arXiv](https://arxiv.org/abs/2312.11392)  \|  [Project Page](https://scedit.github.io/) > |
| 🔥NEFTune: [Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914) |
| QA-LoRA:[Quantization-Aware Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2309.14717) |
| LongLoRA: [Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307) |
| ROME: [Rank-One Editing of Encoder-Decoder Models](https://arxiv.org/abs/2211.13317) |
| Adapter: [Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751) |
| Prompt Tuning: [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119) |
| Side: [Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks](https://arxiv.org/abs/1912.13503) |
| Res-Tuning: [Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone](https://arxiv.org/abs/2310.19859)  < [arXiv](https://arxiv.org/abs/2310.19859)  \|  [Project Page](https://res-tuning.github.io/)  \|  [Usage](docs/source/GetStarted/ResTuning.md) > |
| [PEFT](https://github.com/huggingface/peft)提供的tuners, 如IA3, AdaLoRA等 |

### 支持的硬件

| 硬件环境                  | 备注                            |
| ------------------------- | ------------------------------- |
| CPU                       |                                 |
| RTX20系列/30系列/40系列等 | 30序列之后可使用BF16和FlashAttn |
| 计算卡系列 A10/A100等     |                                 |
| 华为昇腾NPU               |                                 |

### Benchmark

## 📃文档

### 深度学习教程

| 教程名称                                                     |
| ------------------------------------------------------------ |
| [深度学习入门](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/A.%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8%E4%BB%8B%E7%BB%8D.md) |
| [大模型基础知识](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/B.%E9%AD%94%E6%90%AD%E7%A4%BE%E5%8C%BA%E5%92%8CLLM%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86.md) |
| [提示词工程](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/C.%E6%8F%90%E7%A4%BA%E8%AF%8D%E5%B7%A5%E7%A8%8B-prompt%20engineering.md) |
| [Transformer结构介绍](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/D.Transformer%E7%BB%93%E6%9E%84.md) |
| [训练技术选型](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/E.%E6%8A%80%E6%9C%AF%E9%80%89%E5%9E%8B.md) |
| [数据预处理](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/F.%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86.md) |
| [量化](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/G.%E9%87%8F%E5%8C%96.md) |
| [训练](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/H.%E8%AE%AD%E7%BB%83.md) |
| [推理](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/I.LLM%E5%92%8C%E5%A4%9A%E6%A8%A1%E6%80%81%E6%A8%A1%E5%9E%8B%E9%AB%98%E6%95%88%E6%8E%A8%E7%90%86%E5%AE%9E%E8%B7%B5.md) |
| [部署](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/J.%E9%83%A8%E7%BD%B2.md) |
| [评估](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/K.%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%87%AA%E5%8A%A8%E8%AF%84%E4%BC%B0%E7%90%86%E8%AE%BA%E5%92%8C%E5%AE%9E%E6%88%98--LLM%20Automatic%20Evaluation.md) |

### 使用文档

| 文档名称                                                     |
| ------------------------------------------------------------ |
| [使用web-ui](https://github.com/modelscope/swift/blob/main/docs/source/GetStarted/%E7%95%8C%E9%9D%A2%E8%AE%AD%E7%BB%83%E6%8E%A8%E7%90%86.md) |
| [使用tuners](https://github.com/modelscope/swift/blob/main/docs/source/GetStarted/%E4%BD%BF%E7%94%A8tuners.md) |
| [LLM微调](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E5%BE%AE%E8%B0%83%E6%96%87%E6%A1%A3.md) |
| [LLM推理](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E6%8E%A8%E7%90%86%E6%96%87%E6%A1%A3.md) |
| [LLM量化](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E9%87%8F%E5%8C%96%E6%96%87%E6%A1%A3.md) |
| [LLM推理加速和部署](https://github.com/modelscope/swift/blob/main/docs/source/LLM/VLLM%E6%8E%A8%E7%90%86%E5%8A%A0%E9%80%9F%E4%B8%8E%E9%83%A8%E7%BD%B2.md) |
| [量化](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/G.%E9%87%8F%E5%8C%96.md) |
| [命令行参数](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.md) |
| [支持的模型和数据集](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.md) |
| [自定义新模型和数据集](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E8%87%AA%E5%AE%9A%E4%B9%89%E4%B8%8E%E6%8B%93%E5%B1%95.md) |
| [Agent微调最佳实践](https://github.com/modelscope/swift/blob/main/docs/source/LLM/Agent%E5%BE%AE%E8%B0%83%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md) [自我认知微调最佳实践](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E8%87%AA%E6%88%91%E8%AE%A4%E7%9F%A5%E5%BE%AE%E8%B0%83%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md) [Qwen1.5最佳实践](https://github.com/modelscope/swift/blob/main/docs/source/LLM/Qwen1.5%E5%85%A8%E6%B5%81%E7%A8%8B%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md) [多模态模型训练最佳实践](https://github.com/modelscope/swift/tree/main/docs/source/Multi-Modal) |
| [RLHF](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E4%BA%BA%E7%B1%BB%E5%AF%B9%E9%BD%90%E8%AE%AD%E7%BB%83%E6%96%87%E6%A1%A3.md) |
| [AnimateDiff训练](https://github.com/modelscope/swift/blob/main/docs/source/AIGC/AnimateDiff%E5%BE%AE%E8%B0%83%E6%8E%A8%E7%90%86%E6%96%87%E6%A1%A3.md) |


### 最佳实践

## 🔍 了解更多

- [ModelScope库](https://github.com/modelscope/modelscope/) ModelScope库是ModelScope项目的模型库，包含了各模态热门的深度学习模型。

- [将自己的模型贡献给ModelScope](https://modelscope.cn/docs/ModelScope%E6%A8%A1%E5%9E%8B%E6%8E%A5%E5%85%A5%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88)

## License

本框架使用[Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE)进行许可。模型和数据集请查看原资源页面并遵守对应License。

## 引用


## ☎ 联系我们

您可以通过加我们的微信群, 来和我们联系和交流:

<p align="left">
<img src="asset/wechat.png" width="250" style="display: inline-block;">
</p>


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/swift&type=Date)](https://star-history.com/#modelscope/swift&Date)
