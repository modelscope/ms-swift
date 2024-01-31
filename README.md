# SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning)

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>
<p align="center">
<a href="https://modelscope.cn/home">ModelScope Community</a>
<br>
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish&nbsp ｜ &nbsp<a href="https://github.com/modelscope/swift/blob/main/docs/source/GetStarted/%E5%BF%AB%E9%80%9F%E4%BD%BF%E7%94%A8.md">Docs</a>
</p>


<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.9.5-5D91D4.svg"></a>
<a href="https://pypi.org/project/ms-swift/"><img src="https://badge.fury.io/py/ms-swift.svg"></a>
<a href="https://github.com/modelscope/swift/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/swift"></a>
<a href="https://pepy.tech/project/ms-swift"><img src="https://pepy.tech/badge/ms-swift"></a>
<a href="https://github.com/modelscope/swift/"><img src="https://img.shields.io/badge/ms--swift-Build from source-6FEBB9.svg"></a>
</p>

## 📖 Table of Contents
- [Introduction](#-introduction)
- [News](#-news)
- 🔥[LLM Training and Inference](#-llm-training-and-inference)
- 🔥[SCEdit Tuner](#-SCEdit)
- [Installation](#-installation)
- [Getting Started](#-getting-started)
- [Learn More](#-learn-more)
- [License](#license)
- [Contact Us](#-contact-us)

## 📝 Introduction
SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning) is an extensible framwork designed to faciliate lightweight model fine-tuning and inference. It integrates implementations for various efficient fine-tuning methods,  by embracing approaches that is parameter-efficient, memory-efficient, and time-efficient. SWIFT integrates seamlessly into ModelScope ecosystem and offers the capabilities to finetune various models, with a primary emphasis on LLMs and vision models. Additionally, SWIFT is fully compatible with [PEFT](https://github.com/huggingface/peft), enabling users to  leverage the familiar Peft interface to finetune ModelScope models.

Currently supported approches (and counting):

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685)
2. 🔥SCEdit: [SCEdit: Efficient and Controllable Image Diffusion Generation via Skip Connection Editing](https://arxiv.org/abs/2312.11392)  < [arXiv](https://arxiv.org/abs/2312.11392)  |  [Project Page](https://scedit.github.io/) >
3. NEFTune: [Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914)
4. QA-LoRA:[Quantization-Aware Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2309.14717).
5. LongLoRA: [Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307)
6. ROME: [Rank-One Editing of Encoder-Decoder Models](https://arxiv.org/abs/2211.13317)
7. Adapter: [Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)
8. Prompt Tuning: [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)
9. Side: [Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks](https://arxiv.org/abs/1912.13503)
10. Res-Tuning: [Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone](https://arxiv.org/abs/2310.19859)  < [arXiv](https://arxiv.org/abs/2310.19859)  |  [Project Page](https://res-tuning.github.io/)  |  [Usage](docs/source/GetStarted/ResTuning.md) >
11. All tuners offered on [PEFT](https://github.com/huggingface/peft), like IA3, AdaLoRA

Key features:

1. By integrating the ModelScope library, models can be readily obatined via a model-id.
2. Tuners provided by SWIFT can be combined together to allow exploration of multiple tuners on a model for best result.
3. Support calling `activate_adapter` or `deactivate_adapter` or `set_active_adapters`  to activate/deactivate tuners. User can inference with one model and multiple tuners in different threads independently.
4. Support training and inference with scripts/CLI，meanwhile support inference with Web-UI.
5. Support model deployment(vllm/chatglm.cpp/xinference)，Check [Official documentation](./docs/source/GetStarted/部署指南.md) for details.

Users can check the [documentation of SWIFT](docs/source/GetStarted/快速使用.md) to get detail tutorials.


## 🎉 News
- 2024.1.30: Support [internlm-xcomposer2-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/internlm_xcomposer2_7b_chat).
- 2024.1.30: Support [ZeRO-3](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat/full_ddp_zero3/), just need to specify `--deepspeed_config_path default-zero3`.
- 2024.1.29: Support internlm2-math series: internlm2-math-7b, internlm2-math-7b-chat, internlm2-math-20b, internlm2-math-20b-chat.
- 🔥2024.1.26: Support [yi-vl-6b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yi_vl_6b_chat), yi-vl-34b-chat.
- 2024.1.24: Support codefuse-codegeex2-6b-chat, codefuse-qwen-14b-chat.
- 2024.1.23: Support orion series: orion-14b, [orion-14b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/orion_14b_chat).
- 2024.1.20: Support [xverse-13b-256k](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/xverse_13b_256k), xverse-65b-v2, xverse-65b-chat.
- 🔥2024.1.17: Support **internlm2** series: internlm2-7b-base, internlm2-7b, [internlm2-7b-sft-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/internlm2_7b_sft_chat), internlm2-7b-chat, internlm2-20b-base, internlm2-20b, internlm2-20b-sft-chat, internlm2-20b-chat.
- 2024.1.15: Support yuan series: yuan2-2b-instruct, [yuan2-2b-janus-instruct](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yuan2_2b_janus_instruct), yuan2-51b-instruct, yuan2-102b-instruct.
- 🔥2024.1.12: Support **deepseek-moe** series: deepseek-moe-16b, [deepseek-moe-16b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/deepseek_moe_16b_chat).
- 🔥2024.1.4: Support for **VLLM deployment**, compatible with the **OpenAI API** style. For more details, please refer to [VLLM Inference Acceleration and Deployment](https://github.com/modelscope/swift/blob/main/docs/source/LLM/VLLM推理加速与部署.md#部署)
- 2024.1.4: Update [Benchmark](https://github.com/modelscope/swift/blob/main/docs/source/LLM/Benchmark.md) to facilitate viewing the training speed and GPU memory required for different models.
<details><summary>More</summary>

- 🔥 2023.12.29: Support web-ui for training and inference, use `swift web-ui` after the installation of ms-swift.
- 🔥 2023.12.29: Support DPO RLHF(Reinforcement Learning from Human Feedback) and two datasets: AI-ModelScope/stack-exchange-paired and AI-ModelScope/hh-rlhf for this task. Check [this documentation](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E4%BA%BA%E7%B1%BB%E5%AF%B9%E9%BD%90%E8%AE%AD%E7%BB%83%E6%96%87%E6%A1%A3.md) to start training!
- 🔥 2023.12.28: Support SCEdit! This framework can easily reduce memory usage in training and inference, and replace ControlNet for controllable image generating scenarios, view the following chapter for details.
- 2023.12.23: Support [codegeex2-6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/codegeex2_6b).
- 2023.12.19: Support [phi2-3b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/phi2_3b).
- 2023.12.18: Support for VLLM for inference acceleration.
- 2023.12.15: Support deepseek, deepseek-coder series: deepseek-7b, deepseek-7b-chat, deepseek-67b, deepseek-67b-chat, openbuddy-deepseek-67b-chat, deepseek-coder-1_3b, deepseek-coder-1_3b-instruct, deepseek-coder-6_7b, deepseek-coder-6_7b-instruct, deepseek-coder-33b, deepseek-coder-33b-instruct.
- 2023.12.13: Support mistral-7b-instruct-v2, [mixtral-moe-7b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/mixtral_moe_7b), [mixtral-moe-7b-instruct](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/mixtral_moe_7b_instruct).
- 2023.12.9: Support the `freeze_parameters` parameter as a compromise between LoRA and full parameter. Corresponding shell scripts can be found at [full_freeze_ddp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/full_freeze_ddp). Support `disable_tqdm`, `lazy_tokenize`, `preprocess_num_proc` parameters, for details please refer to [Command-Line parameters](https://github.com/modelscope/swift/blob/main/docs/source/LLM/命令行参数.md).
- 2023.12.8: Support [sus-34b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/sus_34b_chat), support yi-6b-200k, yi-34b-200k.
- 2023.12.7: Support [Multi-Node DDP training](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E5%BE%AE%E8%B0%83%E6%96%87%E6%A1%A3.md#%E4%BD%BF%E7%94%A8cli).
- 2023.12.4: Supported models: zephyr-7b-beta-chat, openbuddy-zephyr-7b-chat. Supported datasets: hc3-zh, hc3-en.
- 🔥 2023.12.2: [Best Practices for Self-cognition Fine-tuning](https://github.com/modelscope/swift/blob/main/docs/source/LLM/自我认知微调最佳实践.md), **10 minutes for self-cognition fine-tuning for LLM**, creating a LLM that is specific to oneself.
- 🔥 2023.11.30: Support for training and inference of the **qwen-1_8b**, **qwen-72b**, and **qwen-audio** model series. The corresponding shell scripts can be viewed at [qwen_1_8b_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_1_8b_chat), [qwen_72b_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_72b_chat), [qwen_audio_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_audio_chat).
- 🔥 2023.11.29: Support the training and inference for **AnimateDiff**
- 🔥 2023.11.24: Support for **yi-34b-chat**, **codefuse-codellama-34b-chat**: The corresponding shell script can be found in [yi_34b_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yi_34b_chat), [codefuse_codellama_34b_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/codefuse_codellama_34b_chat).
- 🔥 2023.11.18: Support for **tongyi-finance-14b** series models: tongyi-finance-14b, tongyi-finance-14b-chat, tongyi-finance-14b-chat-int4. The corresponding shell script can be found in [tongyi_finance_14b_chat_int4](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/tongyi_finance_14b_chat_int4).
- 2023.11.16: Added support for more models in **flash attn**: qwen series, qwen-vl series, llama series, openbuddy series, mistral series, yi series, ziya series. Please use the `use_flash_attn` parameter.
- 🔥 2023.11.11: **NEFTune** Supported, Use is with `Swift.prepare_model(model, NEFTuneConfig())`
- 🔥 2023.11.11: Support training and inference with **CLI**, and inference with **Web-UI**. Check the [Run using Swift CLI](https://github.com/modelscope/swift/tree/main#run-using-swift-cli) chapter for details.
- 🔥 2023.11.11: Support model **deployment**(vllm/chatglm.cpp/xinference)，Check [Official documentation](./docs/source/GetStarted/部署指南.md) for details.
- 🔥 2023.11.10: Support for **bluelm** series models: bluelm-7b, bluelm-7b-chat, bluelm-7b-32k, bluelm-7b-chat-32k. The corresponding shell script can be found in [bluelm_7b_chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/bluelm_7b_chat).
- 🔥 2023.11.08: Support the finetuning of **xverse-65b** model, scripts can be found at: [xverse_65b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/xverse_65b).
- 🔥 2023.11.07: Support the finetuning of **yi-6b**, **yi-34b** model, scripts can be found at: [yi_6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yi_6b), [yi_34b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yi_34b).
- 🔥 2023.10.30: Support **QA-LoRA** and **LongLoRA** to decrease memory usage in training.
- 🔥 2023.10.30: Support **ROME**(Rank One Model Editing) to add/modify knowledges, training is not needed!
- 2023.10.30: Support for **skywork-13b** series models: skywork-13b, skywork-13b-chat. The corresponding shell script can be found in [skywork_13b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/skywork_13b).
- 🔥 2023.10.27: Support for **chatglm3** series models: chatglm3-6b-base, chatglm3-6b, chatglm3-6b-32k. The corresponding shell script can be found in [chatglm3_6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/chatglm3_6b).
- 🔥 2023.10.17: Supported **int4**, **int8** models: qwen-7b-chat-int4, qwen-14b-chat-int4, qwen-vl-chat-int4, baichuan2-7b-chat-int4, baichuan2-13b-chat-int4, qwen-7b-chat-int8, qwen-14b-chat-int8.
- 2023.10.15: Supported **ziya2-13b** model series: ziya2-13b, ziya2-13b-chat.
- 2023.10.12: Supported **mistral-7b** model series: openbuddy-mistral-7b-chat, mistral-7b, mistral-7b-instruct.
- 🔥 2023.10.7: Supported **DeepSpeed ZeRO-2**, enabling LoRA (not just QLoRA) to run DDP on 2*A10.
- 2023.10.4: Supported datasets in the fields of mathematics, law, SQL, and coding: blossom-math-zh, school-math-zh, text2sql-en, sql-create-context-en, lawyer-llama-zh, tigerbot-law-zh, leetcode-python-en.
- 🔥 2023.9.25: Supported **qwen-14b** model series: qwen-14b, qwen-14b-chat.
- 2023.9.18: Supported **internlm-20b** model series: internlm-20b, internlm-20b-chat.
- 2023.9.12: Supported training with **MP+DDP** to accelerate full-parameter fine-tuning speed.
- 2023.9.5: Supported **openbuddy-llama2-70b-chat** model.
- 2023.9.3: Supported **baichuan2** model series: baichuan2-7b, baichuan2-7b-chat, baichuan2-13b, baichuan2-13b-chat.
</details>


## ✨ LLM Training and Inference
### WEB UI training and inference

After installation, you can use web-ui training/inference like:

```shell
SWIFT_UI_LANG=en swift web-ui
```

> Supported environment variables:
>
> WEBUI_SHARE=1 Share the gradio or not
> SWIFT_UI_LANG=en/zh The language of radio
> WEBUI_SERVER server_name， web-ui host ip，0.0.0.0 means all routes are allowed，127.0.0.1 means only localhost can visit the web
> WEBUI_PORT The port of web-ui

Here is a simple introduction of web-ui:

[![Watch the video](docs/source/cources/resources/20240119160942.jpg)](https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/SWIFT%E8%A7%86%E9%A2%91_%E8%B0%83%E6%95%B4%E5%B0%81%E9%9D%A2.mp4)

### Simple Usage

- **Self-cognition fine-tuning** for large models in **10 minutes**, creating a personalized large model, please refer to [Best Practices for Self-cognition Fine-tuning](https://github.com/modelscope/swift/blob/main/docs/source/LLM/自我认知微调最佳实践.md).
- Quickly perform **inference** on LLM and build a **Web-UI**, see the [LLM Inference Documentation](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM推理文档.md).
- Rapidly **fine-tune** and perform inference on LLM, and build a Web-UI, see the [LLM Fine-tuning Documentation](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM微调文档.md).
- Using **interface** to fine-tuning and perform inference, see the [WEB-UI Documentation](https://github.com/modelscope/swift/blob/main/docs/source/GetStarted/%E7%95%8C%E9%9D%A2%E8%AE%AD%E7%BB%83%E6%8E%A8%E7%90%86.md).
- **DPO training** supported, start by using [this script](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/dpo/lora_ddp_mp/dpo.sh).
- Utilize VLLM for **inference acceleration** and **deployment(OpenAI API)**. Please refer to [VLLM Inference Acceleration and Deployment](https://github.com/modelscope/swift/blob/main/docs/source/LLM/VLLM推理加速与部署.md) for more information.
- View the models and datasets supported by Swift. You can check [supported models and datasets](https://github.com/modelscope/swift/blob/main/docs/source/LLM/支持的模型和数据集.md).
- Expand and customize models, datasets, and dialogue templates in Swift, see [Customization and Expansion](https://github.com/modelscope/swift/blob/main/docs/source/LLM/自定义与拓展.md).
- Check command-line parameters for fine-tuning and inference, see [Command-Line parameters](https://github.com/modelscope/swift/blob/main/docs/source/LLM/命令行参数.md).
- View the training time and training GPU memory comparison under different parameters, you can check [Benchmark](https://github.com/modelscope/swift/blob/main/docs/source/LLM/Benchmark.md).


### Quick Start
```python
# pip install ms-swift -U

# Experimental environment: A10, 3090, V100, ...
# 12GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments,
    infer_main, sft_main, app_ui_main, merge_lora_main
)

model_type = ModelType.qwen_1_8b
sft_args = SftArguments(
    model_type=model_type,
    train_dataset_sample=2000,
    dataset=[DatasetName.jd_sentiment_zh],
    output_dir='output')
result = sft_main(sft_args)
best_model_checkpoint = result['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')
torch.cuda.empty_cache()

infer_args = InferArguments(
    ckpt_dir=best_model_checkpoint,
    load_dataset_config=True,
    val_dataset_sample=10)
# merge_lora_main(infer_args)
result = infer_main(infer_args)
torch.cuda.empty_cache()

app_ui_main(infer_args)
```


### Features
- Supported SFT Methods: [lora](https://arxiv.org/abs/2106.09685), [qlora](https://arxiv.org/abs/2305.14314), [longlora](https://arxiv.org/abs/2309.12307), [qalora](https://arxiv.org/abs/2309.14717), full parameter fine-tuning, partial parameter fine-tuning.
- Supported Features: quantization, DDP, model parallelism, gradient checkpointing, pushing to modelscope hub, custom datasets, multimodal and agent SFT, mutli-round chat, DPO, self-cognition fine-tuning, ...
- Supported Models: [[Detailed Info]](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.md#%E6%A8%A1%E5%9E%8B)
  - Multi-Modal:
    - [qwen-vl](https://github.com/QwenLM/Qwen-VL) series: qwen-vl, qwen-vl-chat, qwen-vl-chat-int4.
    - [qwen-audio](https://github.com/QwenLM/Qwen-Audio) series: qwen-audio, qwen-audio-chat.
    - [yi-vl](https://github.com/01-ai/Yi) series: yi-vl-6b-chat, yi-vl-34b-chat.
    - [cogagent](https://github.com/THUDM/CogVLM) series: cogagent-18b-chat, cogagent-18b-instruct.
    - [internlm-xcomposer2](https://github.com/InternLM/InternLM-XComposer) series: internlm-xcomposer2-7b-chat.
  - General:
    - [qwen](https://github.com/QwenLM/Qwen) series: qwen-1_8b, qwen-1_8b-chat, qwen-1_8b-chat-int4, qwen-1_8b-chat-int8, qwen-7b, qwen-7b-chat, qwen-7b-chat-int4, qwen-7b-chat-int8, qwen-14b, qwen-14b-chat, qwen-14b-chat-int4, qwen-14b-chat-int8, qwen-72b, qwen-72b-chat, qwen-72b-chat-int4, qwen-72b-chat-int8.
    - [chatglm](https://github.com/THUDM/ChatGLM-6B) series: chatglm2-6b, chatglm2-6b-32k, chatglm3-6b-base, chatglm3-6b, chatglm3-6b-32k.
    - [llama](https://github.com/facebookresearch/llama) series: llama2-7b, llama2-7b-chat, llama2-13b, llama2-13b-chat, llama2-70b, llama2-70b-chat.
    - [yi](https://github.com/01-ai/Yi) series: yi-6b, yi-6b-200k, yi-6b-chat, yi-34b, yi-34b-200k, yi-34b-chat.
    - [internlm](https://github.com/InternLM/InternLM) series: internlm-7b, internlm-7b-chat, internlm-7b-chat-8k, internlm-20b, internlm-20b-chat, internlm2-7b-base, internlm2-7b, internlm2-7b-sft-chat, internlm2-7b-chat, internlm2-20b-base, internlm2-20b, internlm2-20b-sft-chat, internlm2-20b-chat.
    - [deepseek](https://github.com/deepseek-ai/deepseek-LLM) series: deepseek-7b, deepseek-7b-chat, deepseek-67b, deepseek-67b-chat, deepseek-moe-16b, deepseek-moe-16b-chat.
    - [openbuddy](https://github.com/OpenBuddy/OpenBuddy) series: openbuddy-llama2-13b-chat, openbuddy-llama-65b-chat, openbuddy-llama2-70b-chat, openbuddy-mistral-7b-chat, openbuddy-zephyr-7b-chat, openbuddy-deepseek-67b-chat.
    - [mistral](https://github.com/mistralai/mistral-src) series: mistral-7b, mistral-7b-instruct, mistral-7b-instruct-v2.
    - [mixtral](https://github.com/mistralai/mistral-src) series: mixtral-moe-7b, mixtral-moe-7b-instruct.
    - [baichuan](https://github.com/baichuan-inc/Baichuan2) series: baichuan-7b, baichuan-13b, baichuan-13b-chat, baichuan2-7b, baichuan2-7b-chat, baichuan2-13b, baichuan2-13b-chat, baichuan2-7b-chat-int4, baichuan2-13b-chat-int4.
    - [yuan](https://github.com/IEIT-Yuan/Yuan-2.0) series: yuan2-2b-instruct, yuan2-2b-janus-instruct, yuan2-51b-instruct, yuan2-102b-instruct.
    - [xverse](https://github.com/xverse-ai/XVERSE-13B) series: xverse-7b, xverse-7b-chat, xverse-13b, xverse-13b-chat, xverse-65b, xverse-65b-v2, xverse-65b-chat, xverse-13b-256k.
    - [orion](https://github.com/OrionStarAI/OrionStar-Yi-34B-Chat) series: orion-14b, orion-14b-chat.
    - [bluelm](https://github.com/vivo-ai-lab/BlueLM) series: bluelm-7b, bluelm-7b-chat, bluelm-7b-32k, bluelm-7b-chat-32k.
    - [zephyr](https://github.com/huggingface/alignment-handbook) series: zephyr-7b-beta-chat.
    - [ziya](https://github.com/IDEA-CCNL/Fengshenbang-LM) series: ziya2-13b, ziya2-13b-chat.
    - [skywork](https://github.com/SkyworkAI/Skywork) series: skywork-13b, skywork-13b-chat.
    - other: [polylm-13b](https://github.com/DAMO-NLP-MT/PolyLM), [seqgpt-560m](https://github.com/Alibaba-NLP/SeqGPT), [sus-34b-chat](https://github.com/SUSTech-IDEA/SUS-Chat), [openbmb-minicpm-2b-chat](https://github.com/OpenBMB/mlc-MiniCPM).
  - Financial:
    - [tongyi-finance](https://github.com/QwenLM/Qwen) series: tongyi-finance-14b, tongyi-finance-14b-chat, tongyi-finance-14b-chat-int4.
  - Coding:
    - [codefuse](https://github.com/codefuse-ai) series: codefuse-codellama-34b-chat, codefuse-codegeex2-6b-chat, codefuse-qwen-14b-chat.
    - [deepseek-coder](https://github.com/deepseek-ai/DeepSeek-Coder) series: deepseek-coder-1_3b, deepseek-coder-1_3b-instruct, deepseek-coder-6_7b, deepseek-coder-6_7b-instruct, deepseek-coder-33b, deepseek-coder-33b-instruct.
    - [codegeex2](https://github.com/THUDM/CodeGeeX2) series: codegeex2-6b.
    - [phi](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) series: phi2-3b.
  - Math:
    - [internlm2-math](https://github.com/InternLM/InternLM-Math) series: internlm2-math-7b, internlm2-math-7b-chat, internlm2-math-20b, internlm2-math-20b-chat.
- Supported Datasets: [[Detailed Info]](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.md#%E6%95%B0%E6%8D%AE%E9%9B%86)
  - NLP:
    - General: 🔥alpaca-en(gpt4), 🔥alpaca-zh(gpt4), multi-alpaca-all, instinwild-en, instinwild-zh, cot-en, cot-zh, firefly-all-zh, instruct-en, gpt4all-en, sharegpt-en, sharegpt-zh, tutu-v2-sft-mixture, wikipedia-zh, open-orca, open-orca-gpt4, sharegpt-gpt4.
    - Agent: damo-agent-zh, 🔥agent-instruct-all-en.
    - RLHF: 🔥hh-rlhf, stack-exchange-paired.
    - Coding: code-alpaca-en, 🔥leetcode-python-en, 🔥codefuse-python-en, 🔥codefuse-evol-instruction-zh.
    - Medical: medical-en, medical-zh, medical-mini-zh.
    - Law: 🔥lawyer-llama-zh, tigerbot-law-zh.
    - Math: 🔥blossom-math-zh, school-math-zh, open-platypus-en.
    - SQL: text2sql-en, 🔥sql-create-context-en.
    - Text Generation: 🔥advertise-gen-zh, 🔥dureader-robust-zh.
    - Classification: cmnli-zh, 🔥cmnli-mini-zh, 🔥jd-sentiment-zh, 🔥hc3-zh, 🔥hc3-en.
    - RLHF: 🔥hh-rlhf, stack-exchange-paired.
    - Other: finance-en, poetry-zh, webnovel-zh, generated-chat-zh, cls-fudan-news-zh, ner-jave-zh.
  - Multi-Modal:
    - Vision: coco-en, 🔥coco-mini-en, coco-mini-en-2, capcha-images.
    - Audio: aishell1-zh, 🔥aishell1-mini-zh.
  - Custom Dataset
- Supported Templates:
  - Text Generation: default-generation, default-generation-bos, chatglm-generation.
  - Chat: default, qwen, baichuan, chatglm2, chatglm3, llama, openbuddy, internlm, internlm2, yi, yuan, xverse, ziya, skywork, bluelm, zephyr, sus, deepseek, deepseek-coder, codefuse-codellama, codefuse, cogagent-chat, cogagent-instruct, yi-vl, internlm-xcomposer2, openbmb.


## 🔥SCEdit

SCEdit is an efficient generative fine-tuning framework proposed by Alibaba TongYi Vision Intelligence Lab. This framework enhances the fine-tuning capabilities for text-to-image generation downstream tasks and enables quick adaptation to specific generative scenarios, **saving 30%-50% of training memory costs compared to LoRA**. Furthermore, it can be directly extended to controllable image generation tasks, **requiring only 7.9% of the parameters that ControlNet needs for conditional generation and saving 30% of memory usage**. It supports various conditional generation tasks including edge maps, depth maps, segmentation maps, poses, color maps, and image completion.

We using 3D style data from the [style transfer dataset](https://modelscope.cn/datasets/damo/style_custom_dataset/dataPeview) for training, and testing with the same `Prompt: A boy in a camouflage jacket with a scarf`. The qualitative and quantitative results are as follows:

| Method    | bs   | ep   | Target Module | Param. (M)    | Mem. (MiB) | 3D style                                                     |
| --------- | ---- | ---- | ------------- | ------------- | ---------- | ------------------------------------------------------------ |
| LoRA/r=64 | 1    | 50   | q/k/v/out/mlp | 23.94 (2.20%) | 8440MiB    | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703665229562-0f33bbb0-c492-41b4-9f37-3ae720dca80d.png" alt="img" style="zoom:20%;" /> |
| SCEdit    | 1    | 50   | up_blocks     | 19.68 (1.81%) | 7556MiB    | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703665933913-74b98741-3b57-46a4-9871-539df3a0112c.png" alt="img" style="zoom:20%;" /> |
| LoRA/r=64 | 10   | 100  | q/k/v/out/mlp | 23.94 (2.20%) | 26300MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703750608529-de20d0e7-bf9c-4928-8e59-73cc54f2c8d7.png" alt="img" style="zoom:20%;" /> |
| SCEdit    | 10   | 100  | up_blocks     | 19.68 (1.81%) | 18634MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703663033092-94492e44-341f-4259-9df4-13c168e3b5d6.png" alt="img" style="zoom:20%;" /> |
| LoRA/r=64 | 30   | 200  | q/k/v/out/mlp | 23.94 (2.20%) | 69554MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703750626635-2e368d7b-5e99-4a06-b189-8615f302bcd7.png" alt="img" style="zoom:20%;" /> |
| SCEdit    | 30   | 200  | up_blocks     | 19.68 (1.81%) | 43350MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703662246942-1102b1f4-93ab-4653-b943-3302f2a5259e.png" alt="img" style="zoom:20%;" /> |

The benchmark listed above can be reproduced by：

```shell
# Install swift by the next chapter
cd examples/pytorch/multi_modal/notebook
python text_to_image_synthesis.py
```


## 🛠️ Installation

SWIFT is running in Python environment. Please make sure your python version is higher than 3.8.

- Install SWIFT by the `pip` command:

```shell
# full ability
pip install ms-swift[all] -U
# only use llm
pip install ms-swift[llm] -U
# only use aigc
pip install ms-swift[aigc] -U
# only use adapters
pip install ms-swift -U
```

- Install SWIFT by source code(for running sft/infer examples), please run:

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .[llm]
```

SWIFT requires torch>=1.13.

- Use SWIFT in our docker image:

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.8.0-py38-torch2.0.1-tf2.13.0-1.9.1
```

## 🚀 Getting Started

SWIFT supports multiple tuners, as well as tuners provided by [PEFT](https://github.com/huggingface/peft). To use these tuners, simply call:

```python
from swift import Swift, LoRAConfig
config = LoRAConfig(...)
model = Swift.prepare_model(model, config, extra_state_keys=['...'])
```

The code snippet above initialized the tuner randomly. The input model is an instance of `torch.nn.Module`, the config is a subclass instance of `SwiftConfig` or `PeftConfig`. extra_state_keys is
the extra module weights(like the linear head) to be trained and stored in the output dir.

You may combine multiple tuners by:

```python
from swift import Swift, LoRAConfig, PromptConfig
model = Swift.prepare_model(model, {'lora': LoRAConfig(...), 'prompt': PromptConfig(...)})
```

Call `save_pretrained` and `push_to_hub` after finetuning:

```python
from swift import push_to_hub
model.save_pretrained('some-output-folder')
push_to_hub('my-group/some-repo-id-modelscope', 'some-output-folder', token='some-ms-token')
```
Assume `my-group/some-repo-id-modelscope` is the model-id in the hub, and `some-ms-token` is the token for uploading.

Using the model-id to do later inference:

```python
from swift import Swift
model = Swift.from_pretrained(model, 'my-group/some-repo-id-modelscope')
```

Here shows a runnable example:

```python
import os
import tempfile

# Please install modelscope by `pip install modelscope`
from modelscope import Model

from swift import LoRAConfig, SwiftModel, Swift, push_to_hub

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)


model = Model.from_pretrained('modelscope/Llama-2-7b-ms', device_map='auto')
lora_config = LoRAConfig(target_modules=['q_proj', 'k_proj', 'v_proj'])
model: SwiftModel = Swift.prepare_model(model, lora_config)
# Do some finetuning here
model.save_pretrained(tmp_dir)

push_to_hub('my-group/swift_llama2', output_dir=tmp_dir)
model = Model.from_pretrained('modelscope/Llama-2-7b-ms', device_map='auto')
model = SwiftModel.from_pretrained(model, 'my-group/swift_llama2', device_map='auto')
```

This is a example that uses transformers for model creation uses SWIFT for efficient tuning.

```python
from swift import Swift, LoRAConfig, AdapterConfig, PromptConfig
from transformers import AutoModelForImageClassification

# init vit model
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

# init lora tuner config
lora_config = LoRAConfig(
    r=10,  # the rank of the LoRA module
    target_modules=['query', 'key', 'value'],  # the modules to be replaced with the end of the module name
    merge_weights=False  # whether to merge weights
)

# init adapter tuner config
adapter_config = AdapterConfig(
    dim=768,  # the dimension of the hidden states
    hidden_pos=0,  # the position of the hidden state to passed into the adapter
    target_modules=r'.*attention.output.dense$',  # the modules to be replaced with regular expression
    adapter_length=10  # the length of the adapter length
)

# init prompt tuner config
prompt_config = PromptConfig(
    dim=768,  # the dimension of the hidden states
    target_modules=r'.*layer\.\d+$',  # the modules to be replaced with regular expression
    embedding_pos=0,    # the position of the embedding tensor
    prompt_length=10,   # the length of the prompt tokens
    attach_front=False  # Whether prompt is attached in front of the embedding
)

# create model with swift. In practice, you can use any of these tuners or a combination of them.
model = Swift.prepare_model(model, {"lora_tuner": lora_config, "adapter_tuner": adapter_config, "prompt_tuner": prompt_config})

# get the trainable parameters of model
model.get_trainable_parameters()
# 'trainable params: 838,776 || all params: 87,406,432 || trainable%: 0.9596273189597764'
```

You can use the features offered by Peft in SWIFT:

```python
from swift import LoraConfig, Swift
from peft import TaskType
lora_config = LoraConfig(target_modules=['query', 'key', 'value'], task_type=TaskType.CAUSAL_LM)
model_wrapped = Swift.prepare_model(model, lora_config)

# or call from_pretrained to load weights in the modelhub
model_wrapped = Swift.from_pretrained(model, 'some-id-in-the-modelscope-modelhub')
```


The saving strategy between Swift tuners and Peft tuners are slightly different. You can name a tuner by:

```python
model = Swift.prepare_model(model, {'default': LoRAConfig(...)})
model.save_pretrained('./output')
```

In the output dir, you will have a dir structure like this:

```text
output
    |-- default
        |-- adapter_config.json
        |-- adapter_model.bin
    |-- adapter_config.json
    |-- adapter_model.bin
```

The config/weights stored in the output dir is the config of `extra_state_keys` and the weights of it. This is different from PEFT, which stores the weights and config of the `default` tuner.


## 🔍 Learn More

- [ModelScope library](https://github.com/modelscope/modelscope/)

  ModelScope Library is the model library of ModelScope project, which contains a large number of popular models.

- [Contribute your own model to ModelScope](https://modelscope.cn/docs/ModelScope%E6%A8%A1%E5%9E%8B%E6%8E%A5%E5%85%A5%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88)

## License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).


## ☎ Contact Us
You can contact and communicate with us by joining our WeChat Group:

<p align="left">
<img src="asset/wechat.png" width="250" style="display: inline-block;">
</p>


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/swift&type=Date)](https://star-history.com/#modelscope/swift&Date)
