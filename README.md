# SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning)

<p align="center">
    <br>
    <img src="resources/banner.png"/>
    <br>
<p>
<p align="center">
<a href="https://modelscope.cn/home">ModelScope Community Website</a>
<br>
        <a href="README_CN.md">中文</a> &nbsp ｜ &nbsp English &nbsp
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

## 📖 Table of Contents
- [Introduction](#-Introduction)
- [News](#-News) 
- [Installation](#-installation)
- [Getting Started](#-getting-started)
- [Documentation](#-documentation)
- [Learn More](#-learn-more)
- [License](#-License)
- [Citation](#-citation)
- [Contact Us](#-contact-us)

## 📝 Introduction
SWIFT supports training, inference, evaluation and deployment of nearly 200 LLMs and MLLMs (multimodal large models). Developers can directly apply our framework to their own research and production environments to realize the complete workflow from model training and evaluation to application. In addition to supporting the lightweight training solutions provided by [PEFT](https://github.com/huggingface/peft), we also provide a complete Adapters library to support the latest training techniques such as NEFTune, LoRA+, LLaMA-PRO, etc. This adapter library can be used directly in your own custom workflow without our training scripts.

To facilitate use by users unfamiliar with deep learning, we provide a Gradio web-ui for controlling training and inference, as well as accompanying deep learning courses and best practices for beginners.

Additionally, we are expanding capabilities for other modalities. Currently, we support full-parameter training and LoRA training for AnimateDiff.

## 🎉 News
- 🔥2024.03.12: Support inference and fine-tuning for **deepseek-vl** series. Best practices can be found [here](docs/source_en/Multi-Modal/deepseek-vl-best-practice.md).  
- 🔥2024.03.11: Support [GaLore](https://arxiv.org/abs/2403.03507) for effectively reducing memory usage to 1/2 of the original in full-parameter training.
- 🔥2024.03.10: [End-to-end best practices](docs/source_en/LLM/Qwen1.5-best-practice.md) from fine-tuning to deployment for Qwen1.5-7B-Chat and Qwen1.5-72B-Chat. 
- 🔥2024.03.09: Support training and inference of MAMBA model, use [this script](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/mamba-1.4b/lora/sft.sh) to start training!
- 2024.03.09: Support training and inference of AQLM quantized model, use [this script](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/llama2_7b_aqlm_2bit_1x16/lora/sft.sh) to start training!  
- 2024.03.06: Support training and inference of AWQ quantized model, use [this Qwen1.5-AWQ model script](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/qwen1half_7b_chat_awq/lora/sft.sh) to start training, and support training and inference of [yi-9b](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/yi_9b/lora_zero3).
- 🔥2024.02.29: Support [LLaMA PRO](https://arxiv.org/pdf/2401.02415.pdf), simply use [this script](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/yi_6b_chat/llamapro/sft.sh) to start training.  
- 🔥2024.02.29: Support [LoRA+](https://arxiv.org/pdf/2402.12354.pdf), simply use [this script](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/yi_6b_chat/lorap/sft.sh) to start training.
- 2024.02.25: Support `swift export` to quantize models using **AWQ/GPTQ** and push to ModelScope Hub. See documentation: [LLM Quantization](docs/source_en/LLM/LLM-quantization.md).
<details><summary>More</summary>

- 2024.02.22: Support gemma series: gemma-2b, [gemma-2b-instruct](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/gemma_2b_instruct), gemma-7b, gemma-7b-instruct.
- 2024.02.16: Support deepseek-math series: deepseek-math-7b, deepseek-math-7b-instruct, deepseek-math-7b-chat.  
- 🔥2024.02.05: Support **Qwen1.5** series models, see [model list](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.md#%E6%A8%A1%E5%9E%8B) for all supported Qwen1.5 models. Provide fine-tuning scripts for [qwen1half-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen1half_7b_chat), [qwen1half-7b-chat-int8](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen1half_7b_chat_int8).
- 2024.02.05: Support training of diffusion models such as **SDXL**, **SD**, **ControlNet**, as well as **DreamBooth** training. See corresponding [training scripts](https://github.com/modelscope/swift/tree/main/examples/pytorch/sdxl/scripts) for details.  
- 2024.02.01: Support minicpm series: [minicpm-2b-sft-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/minicpm_2b_sft_chat), minicpm-2b-chat.
- 🔥2024.02.01: Support dataset mixing to reduce **catastrophic forgetting**. Use `--train_dataset_mix_ratio 2.0` to enable training! We also open sourced the general knowledge dataset [ms-bench](https://www.modelscope.cn/datasets/iic/ms_bench/summary).
- 🔥2024.02.01: Support Agent training! Agent training algorithm is derived from this [paper](https://arxiv.org/pdf/2309.00986.pdf). We also added [ms-agent](https://www.modelscope.cn/datasets/iic/ms_agent/summary), a high-quality agent dataset. Use [this script](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/qwen_7b_chat/lora/sft.sh) to start Agent training!
- 🔥2024.02.01: Support adding SFT loss in DPO training to reduce repetitive generation caused by KL divergence loss. 
- 2024.02.01: Support using AdaLoRA and IA3 adapters in training.
- 2024.02.01: Support `--merge_lora` parameter in AnimateDiff training.
- 2024.01.30: Support [internlm-xcomposer2-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/internlm_xcomposer2_7b_chat).
- 🔥2024.01.30: Support [ZeRO-3](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat/full_ddp_zero3/), simply specify `--deepspeed default-zero3`.
- 2024.01.29: Support internlm2-math series: internlm2-math-7b, internlm2-math-7b-chat, internlm2-math-20b, internlm2-math-20b-chat.
- 🔥2024.01.26: Support [yi-vl-6b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yi_vl_6b_chat), yi-vl-34b-chat.
- 2024.01.24: Support codefuse-codegeex2-6b-chat, codefuse-qwen-14b-chat.
- 2024.01.23: Support orion series: orion-14b, [orion-14b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/orion_14b_chat).
- 2024.01.20: Support [xverse-13b-256k](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/xverse_13b_256k), xverse-65b-v2, xverse-65b-chat.
- 🔥2024.01.17: Support internlm2 series: internlm2-7b-base, internlm2-7b, [internlm2-7b-sft-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/internlm2_7b_sft_chat), internlm2-7b-chat, internlm2-20b-base, internlm2-20b, internlm2-20b-sft-chat, internlm2-20b-chat.
- 2024.01.15: Support yuan series: yuan2-2b-instruct, [yuan2-2b-janus-instruct](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yuan2_2b_janus_instruct), yuan2-51b-instruct, yuan2-102b-instruct.  
- 🔥2024.01.12: Support **deepseek-moe** series: deepseek-moe-16b, [deepseek-moe-16b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/deepseek_moe_16b_chat).
- 🔥2024.01.04: Support **VLLM deployment**, compatible with **OpenAI API** style, see [VLLM Inference Acceleration and Deployment](https://github.com/modelscope/swift/blob/main/docs/source/LLM/VLLM推理加速与部署.md#部署) for details. 
- 2024.01.04: Update [Benchmark](https://github.com/modelscope/swift/blob/main/docs/source/LLM/Benchmark.md) for convenient viewing of training speed and memory usage of different models.
- 🔥2023.12.29: Support web-ui for sft training and inference, use `swift web-ui` after installing ms-swift to start.
- 🔥2023.12.29: Support DPO RLHF (Reinforcement Learning from Human Feedback) and three datasets for this task: AI-ModelScope/stack-exchange-paired, AI-ModelScope/hh-rlhf and AI-ModelScope/hh_rlhf_cn. See [documentation](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E4%BA%BA%E7%B1%BB%E5%AF%B9%E9%BD%90%E8%AE%AD%E7%BB%83%E6%96%87%E6%A1%A3.md) to start training!
- 🔥2023.12.28: Support SCEdit! This tuner can significantly reduce memory usage in U-Net and support low-memory controllable image generation (replacing ControlNet), read the section below to learn more.  
- 2023.12.23: Support [codegeex2-6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/codegeex2_6b).
- 2023.12.19: Support [phi2-3b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/phi2_3b).
- 2023.12.18: Support VLLM for inference acceleration. 
- 2023.12.15: Support deepseek, deepseek-coder series: deepseek-7b, deepseek-7b-chat, deepseek-67b, deepseek-67b-chat, openbuddy-deepseek-67b-chat, deepseek-coder-1_3b, deepseek-coder-1_3b-instruct, deepseek-coder-6_7b, deepseek-coder-6_7b-instruct, deepseek-coder-33b, deepseek-coder-33b-instruct.
- 2023.12.13: Support mistral-7b-instruct-v2, [mixtral-moe-7b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/mixtral_7b_moe), [mixtral-moe-7b-instruct](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/mixtral_7b_moe_instruct).  
- 2023.12.09: Support `freeze_parameters` parameter as a compromise between lora and full-parameter training. Corresponding sh can be found in [full_freeze_ddp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/full_freeze_ddp). Support `disable_tqdm`, `lazy_tokenize`, `preprocess_num_proc` parameters, see [command line arguments](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.md) for details.
- 2023.12.08: Support [sus-34b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/sus_34b_chat), support yi-6b-200k, yi-34b-200k.
- 2023.12.07: Support [Multi-Node DDP training](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E5%BE%AE%E8%B0%83%E6%96%87%E6%A1%A3.md#%E4%BD%BF%E7%94%A8cli).
- 2023.12.05: Support models: zephyr-7b-beta-chat, openbuddy-zephyr-7b-chat. Support datasets: hc3-zh, hc3-en.  
- 🔥2023.12.02: [Self-cognition fine-tuning best practices](docs/source_en/LLM/Self-cognition-best-practice.md), **10 minutes to fine-tune a large model for self-cognition**, create your own unique large model.
- 🔥2023.11.30: Support training and inference of **qwen-1_8b**, **qwen-72b**, **qwen-audio** series models. Corresponding sh scripts can be found in [qwen_1_8b_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_1_8b_chat), [qwen_72b_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_72b_chat), [qwen_audio_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_audio_chat)
- 🔥2023.11.29: Support training and inference of **AnimateDiff** 
- 🔥2023.11.24: Support **yi-34b-chat**, **codefuse-codellama-34b-chat** models. Corresponding sh scripts can be found in [yi_34b_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yi_34b_chat), [codefuse_codellama_34b_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/codefuse_codellama_34b_chat).
- 🔥2023.11.18: Support **tongyi-finance-14b** series models: tongyi-finance-14b, tongyi-finance-14b-chat, tongyi-finance-14b-chat-int4. Corresponding sh scripts can be found in [tongyi_finance_14b_chat_int4](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/tongyi_finance_14b_chat_int4).
- 2023.11.16: Support **flash attn** for more models: qwen series, qwen-vl series, llama series, openbuddy series, mistral series, yi series, ziya series. Please use `use_flash_attn` parameter.
- 🔥2023.11.11: Support **NEFTune**, simply use `Swift.prepare_model(model, NEFTuneConfig())` to enable.  
- 🔥2023.11.11: Support training and inference by **command line** and inference by **Web-UI**, see `Usage with Swift CLI` section below for details.
- 🔥2023.11.10: Support **bluelm** series models: bluelm-7b, bluelm-7b-chat, bluelm-7b-32k, bluelm-7b-chat-32k. Corresponding sh scripts can be found in [bluelm_7b_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/bluelm_7b_chat).
- 🔥2023.11.08: Support training and inference of **xverse-65b** model, script at [xverse_65b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/xverse_65b).  
- 🔥2023.11.07: Support training and inference of **yi-6b**, **yi-34b** models, scripts at [yi_6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yi_6b), [yi_34b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yi_34b).
- 🔥2023.10.30: Support two new tuners: **QA-LoRA** and **LongLoRA**.
- 🔥2023.10.30: Support editing models using **ROME** (Rank One Model Editing) to infuse new knowledge into models without training!  
- 2023.10.30: Support **skywork-13b** series models: skywork-13b, skywork-13b-chat. Corresponding sh scripts can be found in [skywork_13b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/skywork_13b).
- 🔥2023.10.27: Support **chatglm3** series models: chatglm3-6b-base, chatglm3-6b, chatglm3-6b-32k. Corresponding sh scripts can be found in [chatglm3_6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/chatglm3_6b).
- 🔥2023.10.17: Support SFT of **int4**, **int8** models: qwen-7b-chat-int4, qwen-14b-chat-int4, qwen-vl-chat-int4, baichuan2-7b-chat-int4, baichuan2-13b-chat-int4, qwen-7b-chat-int8, qwen-14b-chat-int8.
- 2023.10.15: Support **ziya2-13b** series models: ziya2-13b, ziya2-13b-chat.  
- 2023.10.12: Support **mistral-7b** series models: openbuddy-mistral-7b-chat, mistral-7b, mistral-7b-instruct.
- 🔥2023.10.07: Support **DeepSpeed ZeRO-2**, enabling lora (not just qlora) to run DDP on dual A10 cards.
- 2023.10.04: Support more math, law, SQL, code domain datasets: blossom-math-zh, school-math-zh, text2sql-en, sql-create-context-en, lawyer-llama-zh, tigerbot-law-zh, leetcode-python-en.
- 🔥2023.09.25: Support **qwen-14b** series: qwen-14b, qwen-14b-chat.
- 2023.09.18: Support **internlm-20b** series: internlm-20b, internlm-20b-chat. 
- 2023.09.12: Support **MP+DDP** to accelerate full-parameter training.
- 2023.09.05: Support **openbuddy-llama2-70b-chat**.
- 2023.09.03: Support **baichuan2** series: baichuan2-7b, baichuan2-7b-chat, baichuan2-13b, baichuan2-13b-chat.  
</details>

## 🛠️ Installation

SWIFT runs in the Python environment. Please ensure your Python version is higher than 3.8.

- Method 1: Install SWIFT using pip command:

```shell
# Full capabilities
pip install ms-swift[all] -U 
# LLM only
pip install ms-swift[llm] -U
# AIGC only 
pip install ms-swift[aigc] -U
# Adapters only
pip install ms-swift -U
```

- Method 2: Install SWIFT through source code (convenient for running training and inference scripts), please run the following commands:

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .[llm]
```

SWIFT depends on torch>=1.13, recommend torch>=2.0.0.

- Method 3: Use SWIFT in our Docker image 

```shell
# China-Hangzhou image
docker pull registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.1.2-tf2.14.0-1.13.1
# US-west image
docker pull registry.us-west-1.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.1.2-tf2.14.0-1.13.1  
```

## 🚀 Getting Started

This section introduces basic usage, see the [Documentation](#-documentation) section for more ways to use.

### web-ui

```shell
swift web-ui
```

### Training

#### Supported Training Processes

| Training Process | Training Method                                                                 |
|------------------|----------------------------------------------------------------------------------|
| Pretraining      | Continuation                                                                    |
| Fine-tuning      | Single-turn/Multi-turn/Agent Training/Self-cognition/Multi-modal QA/Speech QA      |
| Human Alignment  | DPO                                                                             |
| Text-to-Image    | DreamBooth, etc.                                                                |
| Text-to-Video    | -                                                                               |

#### Single GPU Training

Start single GPU fine-tuning with the following command:

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

#### Model Parallel Training

Model parallel training modifies the `CUDA_VISIBLE_DEVICES` environment variable based on the above command:

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

#### Data Parallel Training

Data parallel training modifies the `NPROC_PER_NODE` environment variable based on the above command:

```shell
# If the number of CUDA_VISIBLE_DEVICES is an integer multiple of NPROC_PER_NODE (greater than 1), data parallel is launched according to NPROC_PER_NODE, and model parallel is launched according to CUDA_VISIBLE_DEVICES number/NPROC_PER_NODE
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

#### Deepspeed Training

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

### Inference

```shell
swift infer --model_type qwen1half-7b-chat --stream true
```

##### VLLM Inference

```shell
swift infer --model_type qwen1half-7b-chat --infer_backend vllm --stream true
```

### Evaluation

```shell
swift eval --model_type qwen1half-7b-chat --eval_dataset mmlu ceval
```

### Export

```shell
swift export --model_type qwen1half-7b-chat --quant_bits 4 --quant_method awq
```

### Deployment

```shell
swift deploy --model_type qwen1half-7b-chat --infer_backend vllm --max_model_len 8192  
```

### Supported Models

#### LLM Models

| Model Type                                     | Model Introduction                                                     | Language           | Model Size                             | Model Type                                 |
|------------------------------------------------|------------------------------------------------------------------------|--------------------|----------------------------------------|------------------------------------------- |
| Qwen/Qwen1.5                                   | [Tongyi Qwen 1.0 and 1.5 series models](https://github.com/QwenLM)  | Chinese/English    | 1.8B-72B, including quantized versions | base model/chat model                      |
| ChatGLM2/ChatGLM3/Codegeex2                    | [Zhipu ChatGLM series models](https://github.com/THUDM/)               | Chinese/English    | 6B                                     | base model/chat model  |
| Baichuan/Baichuan2                             | [Baichuan 1 and Baichuan 2](https://github.com/baichuan-inc)           | Chinese/English    | 7B-13B                                 | base model/chat model                       |
| Yuan2                                          | [Langchao Yuan series models](https://github.com/IEIT-Yuan)             | Chinese/English    | 2B-102B                                | chat model                                 |
| XVerse                                         | [XVerse series models](https://github.com/xverse-ai)                    | Chinese/English    | 7B-65B                                 | base model/chat model                       |
| LLaMA2                                         | [LLaMA2 series models](https://github.com/facebookresearch/llama)       | English            | 7B-70B, including quantized versions   | base model/chat model                       |
| Mistral/Mistral-MoE                            | [Mistral series models](https://github.com/mistralai/mistral-src)       | English            | 7B, including quantized and MoE versions | base model/chat model                     |
| YI                                             | [01AI's YI series models](https://github.com/01-ai)                     | Chinese/English    | 6B-34B                                 | base model/chat model                       |
| InternLM/InternLM2/InternLM2-Math              | [Pujiang AI Lab InternLM series models](https://github.com/InternLM/InternLM) | Chinese/English | 1.8B-20B                            | base model/chat model/math model            |
| DeepSeek/DeepSeek-Coder/DeepSeek-Math          | [DeepSeek series models](https://github.com/deepseek-ai)       | Chinese/English    | 1.3B-67B                               | base model/chat model/code generation model/math model |
| MAMBA                                          | [MAMBA temporal convolution model](https://github.com/state-spaces/mamba) | English          | 130M-2.8B                              | base model                                 |
| Gemma                                          | [Google Gemma series models](https://github.com/google/gemma_pytorch)   | English            | 2B-7B                                  | base model/chat model                       |
| MiniCPM                                        | [OpenBmB MiniCPM series models](https://github.com/OpenBMB/MiniCPM)     | Chinese/English    | 2B-3B                                  | chat model                                 |
| OpenBuddy                                      | [OpenBuddy series models](https://github.com/OpenBuddy/OpenBuddy)       | Chinese/English    | 7B-67B                                 | base model/chat model                       |
| Orion                                          | [OrionStar AI series models](https://github.com/OrionStarAI)            | Chinese/English    | 14B                                    | base model/chat model                       |
| BlueLM                                         | [VIVO BlueLM large model](https://github.com/vivo-ai-lab/BlueLM)        | Chinese/English    | 7B                                     | base model/chat model                       |
| Ziya2                                          | [Fengshenbang series models](https://github.com/IDEA-CCNL/Fengshenbang-LM) | Chinese/English  | 13B                                    | base model/chat model                       |
| Skywork                                        | [Skywork series models](https://github.com/SkyworkAI/Skywork) | Chinese/English    | 13B                                    | base model/chat model                       |
| Zephyr                                         | Zephyr series models based on Mistral                                  | English            | 7B                                     | chat model                                 |
| PolyLM                                         | [Tongyi Lab self-developed PolyLM series models](https://github.com/DAMO-NLP-MT/PolyLM) | Multilingual | 13B                                 | base model                                 |
| SeqGPT                                         | [Tongyi Lab self-developed text understanding model for information extraction and text classification](https://github.com/Alibaba-NLP/SeqGPT) | Chinese | 560M                               | semantic understanding model                |
| SUS                                            | [Southern University of Science and Technology model fine-tuned on YI](https://github.com/SUSTech-IDEA/SUS-Chat) | Chinese/English | 34B                              | chat model                                 |
| Tongyi-Finance                                 | [Tongyi finance series models](https://github.com/QwenLM/Qwen)          | Chinese/English    | 13B                                    | finance domain base model/chat model        |
| CodeFuse-CodeLLaMA/CodeFuse-Codegeex2/CodeFuse-Qwen | [Ant CodeFuse series models](https://github.com/codefuse-ai)        | Chinese/English    | 6B-34B                                 | code generation model                      |
| phi2                                           | Microsoft's PHI2 model                                                 | English            | 3B                                     | generation model                           |

#### MLLM Models

| Model Type       | Model Introduction                                                     | Language           | Model Size        | Model Type         |
|------------------|------------------------------------------------------------------------|--------------------|-------------------|------------------- |
| Qwen-VL          | [Tongyi Qwen vision model](https://github.com/QwenLM)               | Chinese/English    | 7B, including quantized versions | base model/chat model |
| Qwen-Audio       | [Tongyi Qwen speech model](https://github.com/QwenLM)               | Chinese/English    | 7B                | base model/chat model |
| YI-VL            | [01AI's YI series vision models](https://github.com/01-ai)             | Chinese/English    | 6B-34B            | chat model         |
| xcomposer2       | [Pujiang AI Lab InternLM vision model](https://github.com/InternLM/InternLM) | Chinese/English | 7B              | chat model         |
| DeepSeek-VL      | [DeepSeek series vision models](https://github.com/deepseek-ai) | Chinese/English    | 1.3B-7B           | chat model         |
| MiniCPM-VL       | [OpenBmB MiniCPM vision model](https://github.com/OpenBMB/MiniCPM)     | Chinese/English    | 3B                | chat model         |
| CogAgent/CogVLM  | [Zhipu ChatGLM visual QA and Agent model](https://github.com/THUDM/)   | Chinese/English    | 17B-18B           | chat model         |

#### Diffusion Models

| Model Type          | Model Introduction                                                    | Language | Model Type        |
|---------------------|----------------------------------------------------------------------|----------|------------------ | 
| AnimateDiff         | [AnimateDiff animation model](https://github.com/guoyww/AnimateDiff) | English  | text-to-video     |
| SD1.5/SD2.0/SDXL    | [StabilityAI series diffusion models](https://github.com/Stability-AI) | English | text-to-image    |

### Supported Open Source Datasets

| Dataset Type | Training Task  | Documentation                                                 | 
|--------------|:---------------|--------------------------------------------------------------- |
| General      | Fine-tuning    | 🔥ms-bench, 🔥ms-bench-mini, 🔥alpaca-en(gpt4), 🔥alpaca-zh(gpt4), multi-alpaca-all, instinwild-en, instinwild-zh, cot-en, cot-zh, firefly-all-zh, instruct-en, gpt4all-en, sharegpt-en, sharegpt-zh, tulu-v2-sft-mixture, wikipedia-zh, open-orca, open-orca-gpt4, sharegpt-gpt4, 🔥sharegpt-gpt4-mini. |
| Agent        | Fine-tuning    | 🔥ms-agent, damo-mini-agent-zh, damo-agent-zh, agent-instruct-all-en. |
| General      | Human Alignment | 🔥hh-rlhf-cn, stack-exchange-paired, hh-rlhf-harmless-base, hh-rlhf-helpful-base, hh-rlhf-helpful-online, hh-rlhf-helpful-rejection-sampled, hh-rlhf-red-team-attempts, hh-rlhf-cn-harmless-base-cn, hh-rlhf-cn-helpful-base-cn, hh-rlhf-cn-harmless-base-en, hh-rlhf-cn-helpful-base-en. |
| Code         | Fine-tuning    | code-alpaca-en, 🔥leetcode-python-en, 🔥codefuse-python-en, 🔥codefuse-evol-instruction-zh. |
| Medical      | Fine-tuning    | medical-en, medical-zh, medical-mini-zh, 🔥disc-med-sft-zh.   |
| Legal        | Fine-tuning    | lawyer-llama-zh, tigerbot-law-zh, 🔥disc-law-sft-zh.          |
| Math         | Fine-tuning    | 🔥blossom-math-zh, school-math-zh, open-platypus-en.          |
| SQL          | Fine-tuning    | text2sql-en, 🔥sql-create-context-en.                         |
| Text Generation | Fine-tuning | 🔥advertise-gen-zh, 🔥dureader-robust-zh.                     |
| Classification | Fine-tuning  | cmnli-zh, 🔥cmnli-mini-zh, 🔥jd-sentiment-zh, 🔥hc3-zh, 🔥hc3-en. |
| Quantization Assist | Quantization | pileval.                                                  | 
| Other        | Fine-tuning    | finance-en, poetry-zh, webnovel-zh, generated-chat-zh, cls-fudan-news-zh, ner-jave-zh. |
| Vision       | Fine-tuning    | coco-en, 🔥coco-mini-en, coco-mini-en-2, capcha-images.       |
| Audio        | Fine-tuning    | aishell1-zh, 🔥aishell1-mini-zh.                              |

### Supported Technologies

| Technology Name                                               |
|--------------------------------------------------------------- |
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
| Tuners provided by [PEFT](https://github.com/huggingface/peft), such as IA3, AdaLoRA, etc. |

### Supported Hardware 

| Hardware Environment       | Notes                                    |
|----------------------------|------------------------------------------ |
| CPU                        |                                          |
| RTX 20/30/40 series, etc.  | After 30 series, BF16 and FlashAttn can be used |
| Computing cards A10/A100, etc. | Support BF16 and FlashAttn            |
| Huawei Ascend NPU          |                                          |

### Benchmark

## 📃 Documentation

### Deep Learning Tutorials 

| Tutorial Name                                                |
|-------------------------------------------------------------- |
| [Introduction to Deep Learning](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/A.%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8%E4%BB%8B%E7%BB%8D.md) |
| [Large Model Basics](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/B.%E9%AD%94%E6%90%AD%E7%A4%BE%E5%8C%BA%E5%92%8CLLM%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86.md) |
| [Prompt Engineering](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/C.%E6%8F%90%E7%A4%BA%E8%AF%8D%E5%B7%A5%E7%A8%8B-prompt%20engineering.md) |
| [Transformer Architecture Introduction](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/D.Transformer%E7%BB%93%E6%9E%84.md) |
| [Training Technique Selection](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/E.%E6%8A%80%E6%9C%AF%E9%80%89%E5%9E%8B.md) |
| [Data Preprocessing](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/F.%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86.md) |
| [Quantization](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/G.%E9%87%8F%E5%8C%96.md) | 
| [Training](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/H.%E8%AE%AD%E7%BB%83.md) |
| [Inference](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/I.LLM%E5%92%8C%E5%A4%9A%E6%A8%A1%E6%80%81%E6%A8%A1%E5%9E%8B%E9%AB%98%E6%95%88%E6%8E%A8%E7%90%86%E5%AE%9E%E8%B7%B5.md) |
| [Deployment](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/J.%E9%83%A8%E7%BD%B2.md) |
| [Evaluation](https://github.com/modelscope/modelscope-classroom/blob/main/LLM-tutorial/K.%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%87%AA%E5%8A%A8%E8%AF%84%E4%BC%B0%E7%90%86%E8%AE%BA%E5%92%8C%E5%AE%9E%E6%88%98--LLM%20Automatic%20Evaluation.md) |

### Usage Documentation

| Document Name                                                |
| ------------------------------------------------------------ |
| [Using web-ui](docs/source_en/GetStarted/Web-ui.md)          |
| [Using tuners](docs/source_en/GetStarted/Tuners.md)          |
| [LLM Fine-tuning](docs/source_en/LLM/LLM-fine-tuning.md)     |
| [LLM Inference](docs/source_en/LLM/LLM-inference.md)         |
| [LLM Quantization](docs/source_en/LLM/LLM-quantization.md)   |
| [LLM Inference Acceleration and Deployment](docs/source_en/LLM/VLLM-inference-acceleration-and-deployment.md) |
| [Command Line Arguments](docs/source_en/LLM/Command-line-parameters.md) |
| [Supported Models and Datasets](docs/source_en/LLM/Supported-models-datasets.md) |
| [Customizing New Models and Datasets](docs/source_en/LLM/Customization.md) |
| [Agent Fine-Tuning Best Practices](docs/source_en/LLM/Agent-best-practice.md) [Self-Cognition Fine-Tuning Best Practices](docs/source_en/LLM/Self-cognition-best-practice) [Qwen1.5 Best Practices](https://github.com/modelscope/swift/blob/main/docs/source/LLM/Qwen1.5%E5%85%A8%E6%B5%81%E7%A8%8B%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.md) [Multi-Modal Model Training Best Practices](docs/source_en/Multi-Modal) |
| [DPO Human Alignment Training](docs/source_en/LLM/RLHF.md)   |
| [AnimateDiff Training](docs/source_en/AIGC/AnimateDiff-train-infer.md) |

## 🔍 Learn More

- [ModelScope Library](https://github.com/modelscope/modelscope/) The ModelScope library is the model library of the ModelScope project, containing popular deep learning models for various modalities.

- [Contribute Your Own Models to ModelScope](https://modelscope.cn/docs/ModelScope%E6%A8%A1%E5%9E%8B%E6%8E%A5%E5%85%A5%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88)

## 🏛 License

This framework is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE). For models and datasets, please refer to the original resource page and follow the corresponding License.

## 📎 Citation

```bibtex
@Misc{swift,
  title = {SWIFT},
  author = "{The ModelScope Team}",
  howpublished = {\url{https://github.com/modelscope/swift}},
  year = {2024}
}
```

## ☎ Contact Us

You can contact us and communicate with us by adding our WeChat group:

<p align="left">
<img src="asset/wechat.png" width="250" style="display: inline-block;">
</p>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/swift&type=Date)](https://star-history.com/#modelscope/swift&Date)
