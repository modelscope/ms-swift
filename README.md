# SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning)

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>

<p align="center">
<a href="https://modelscope.cn/home">ModelScope Hub</a>
<br>
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish
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
- [LLM Training and Inference](#-llm-training-and-inference)
- [Installation](#-installation)
- [Getting Started](#-getting-started)
- [Learn More](#-learn-more)
- [License](#license)
- [Contact Us](#-contact-us)

## 📝 Introduction
SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning) is an extensible framwork designed to faciliate lightweight model fine-tuning and inference. It integrates implementations for various efficient fine-tuning methods,  by embracing approaches that is parameter-efficient, memory-efficient, and time-efficient. SWIFT integrates seamlessly into ModelScope ecosystem and offers the capabilities to finetune various models, with a primary emphasis on LLMs and vision models. Additionally, SWIFT is fully compatible with [PEFT](https://github.com/huggingface/peft), enabling users to  leverage the familiar Peft interface to finetune ModelScope models.

Currently supported approches (and counting):

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685)
2. QA-LoRA:[Quantization-Aware Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2309.14717).
3. LongLoRA: [Efficient Fine-tuning of Long-Context Large Language Models](https://arxiv.org/abs/2309.12307)
4. Adapter: [Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)
5. Prompt Tuning: [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)
6. Side: [Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks](https://arxiv.org/abs/1912.13503)
7. Res-Tuning: [Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone](https://arxiv.org/abs/2310.19859)  < [arXiv](https://arxiv.org/abs/2310.19859)  |  [Project Page](https://res-tuning.github.io/)  |  [Usage](docs/source/GetStarted/ResTuning.md) >
8. ROME: [Rank-One Editing of Encoder-Decoder Models](https://arxiv.org/abs/2211.13317)
9. NEFTune: [Noisy Embeddings Improve Instruction Finetuning](https://arxiv.org/abs/2310.05914)
10. All tuners offered on [PEFT](https://github.com/huggingface/peft)

Key features:

1. By integrating the ModelScope library, models can be readily obatined via a model-id.
2. Tuners provided by SWIFT can be combined together to allow exploration of multiple tuners on a model for best result.
3. Support calling `activate_adapter` or `deactivate_adapter` or `set_active_adapters`  to activate/deactivate tuners. User can inference with one model and multiple tuners in different threads independently.
4. Support training and inference with scripts/CLI，meanwhile support inference with Web-UI.
5. Support model deployment(vllm/chatglm.cpp/xinference)，Check [Official documentation](./docs/source/GetStarted/部署指南.md) for details.

Users can check the [documentation of SWIFT](docs/source/GetStarted/快速使用.md) to get detail tutorials.


## 🎉 News
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
<details><summary>More</summary>

- 🔥 2023.10.30: Support **QA-LoRA** and **LongLoRA** to decrease memory usage in training.
- 🔥 2023.10.30: Support **ROME**(Rank One Model Editing) to add/modify knowledges, training is not needed!
- 2023.10.30: Support for **skywork-13b** series models: skywork-13b, skywork-13b-chat. The corresponding shell script can be found in [skywork_13b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/skywork_13b).
- 🔥 2023.10.27: Support for **chatglm3** series models: chatglm3-6b-base, chatglm3-6b, chatglm3-6b-32k. The corresponding shell script can be found in [chatglm3_6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/chatglm3_6b).
- 🔥 2023.10.17: Supported **int4**, **int8** models: qwen-7b-chat-int4, qwen-14b-chat-int4, qwen-vl-chat-int4, baichuan2-7b-chat-int4, baichuan2-13b-chat-int4, qwen-7b-chat-int8, qwen-14b-chat-int8.
- 2023.10.15: Supported **ziya2-13b** model series: ziya2-13b, ziya2-13b-chat.
- 2023.10.12: Supported **mistral-7b** model series: openbuddy-mistral-7b-chat, mistral-7b, mistral-7b-chat.
- 🔥 2023.10.7: Supported **DeepSpeed ZeRO-2**, enabling LoRA (not just QLoRA) to run DDP on 2*A10.
- 2023.10.4: Supported datasets in the fields of mathematics, law, SQL, and coding: blossom-math-zh, school-math-zh, text2sql-en, sql-create-context-en, lawyer-llama-zh, tigerbot-law-zh, leetcode-python-en.
- 🔥 2023.9.25: Supported **qwen-14b** model series: qwen-14b, qwen-14b-chat.
- 2023.9.18: Supported **internlm-20b** model series: internlm-20b, internlm-20b-chat.
- 2023.9.12: Supported training with **MP+DDP** to accelerate full-parameter fine-tuning speed.
- 2023.9.5: Supported **openbuddy-llama2-70b-chat** model.
- 2023.9.3: Supported **baichuan2** model series: baichuan2-7b, baichuan2-7b-chat, baichuan2-13b, baichuan2-13b-chat.
</details>


## ✨ LLM Training and Inference
### Simple Usage
- [Must Read] **Self-cognitionfine-tuning** for large models in **10 minutes**, creating a personalized large model, please refer to [Best Practices for Self-cognition Fine-tuning](https://github.com/modelscope/swift/blob/main/docs/source/LLM/自我认知微调最佳实践.md).
- Quickly perform **inference** on LLM and build a **Web-UI**, see the [LLM Inference Documentation](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM推理文档.md).
- Rapidly **fine-tune** and perform inference on LLM, and build a Web-UI. See the [LLM Fine-tuning Documentation](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM微调文档.md).
- View the models and datasets supported by Swift. You can check [supported models and datasets](https://github.com/modelscope/swift/blob/main/docs/source/LLM/支持的模型和数据集.md).
- Expand and customize models, datasets, and dialogue templates in Swift, see [Customization and Expansion](https://github.com/modelscope/swift/blob/main/docs/source/LLM/自定义与拓展.md).
- Check command-line parameters for fine-tuning and inference, see [Command-Line parameters](https://github.com/modelscope/swift/blob/main/docs/source/LLM/命令行参数.md).


### Features
- Supported SFT Methods: [lora](https://arxiv.org/abs/2106.09685), [qlora](https://arxiv.org/abs/2305.14314), full(full parameter fine-tuning)
- Supported Features: quantization, DDP, model parallelism, gradient checkpointing, pushing to modelscope hub, custom datasets, multimodal and agent SFT, mutli-round chat, ...
- Supported Models:
  - Multi-Modal:
    - qwen-vl series: [qwen-vl](https://modelscope.cn/models/qwen/Qwen-VL/summary), [qwen-vl-chat](https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary), [qwen-vl-chat-int4](https://modelscope.cn/models/qwen/Qwen-VL-Chat-Int4/summary)
    - qwen-audio series: [qwen-audio](https://modelscope.cn/models/qwen/Qwen-Audio/summary), [qwen-audio-chat](https://modelscope.cn/models/qwen/Qwen-Audio-Chat/summary)
  - General:
    - qwen series: [qwen-1_8b-chat](https://modelscope.cn/models/qwen/Qwen-1_8B/summary), [qwen-1_8b-chat-int4](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat-Int4/summary), [qwen-1_8b-chat-int8](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat-Int8/summary), [qwen-7b](https://modelscope.cn/models/qwen/Qwen-7B/summary), [qwen-7b-chat](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary), [qwen-7b-chat-int4](https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int4/summary), [qwen-7b-chat-int8](https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int8/summary), [qwen-14b](https://modelscope.cn/models/qwen/Qwen-14B/summary), [qwen-14b-chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary), [qwen-14b-chat-int4](https://modelscope.cn/models/qwen/Qwen-14B-Chat-Int4/summary), [qwen-14b-chat-int8](https://modelscope.cn/models/qwen/Qwen-14B-Chat-Int8/summary), [qwen-72b](https://modelscope.cn/models/qwen/Qwen-72B/summary), [qwen-72b-chat](https://modelscope.cn/models/qwen/Qwen-72B-Chat/summary), [qwen-72b-chat-int4](https://modelscope.cn/models/qwen/Qwen-72B-Chat-Int4/summary), [qwen-72b-chat-int8](https://modelscope.cn/models/qwen/Qwen-72B-Chat-Int8/summary)
    - chatglm series: [chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary), [chatglm2-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm2-6b-32k/summary), [chatglm3-6b-base](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-base/summary), [chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary), [chatglm3-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-32k/summary)
    - baichuan series: [baichuan-7b](https://modelscope.cn/models/baichuan-inc/baichuan-7B/summary), [baichuan-13b](https://modelscope.cn/models/baichuan-inc/Baichuan-13B-Base/summary), [baichuan-13b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan-13B-Chat/summary), [baichuan2-7b](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Base/summary), [baichuan2-7b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat/summary), [baichuan2-13b](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Base/summary), [baichuan2-13b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat/summary), [baichuan2-7b-chat-int4](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat-4bits/summary), [baichuan2-13b-chat-int4](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat-4bits/summary)
    - llama series: [llama2-7b](https://modelscope.cn/models/modelscope/Llama-2-7b-ms/summary), [llama2-7b-chat](https://modelscope.cn/models/modelscope/Llama-2-7b-chat-ms/summary), [llama2-13b](https://modelscope.cn/models/modelscope/Llama-2-13b-ms/summary), [llama2-13b-chat](https://modelscope.cn/models/modelscope/Llama-2-13b-chat-ms/summary), [llama2-70b](https://modelscope.cn/models/modelscope/Llama-2-70b-ms/summary), [llama2-70b-chat](https://modelscope.cn/models/modelscope/Llama-2-70b-chat-ms/summary)
    - yi series: [yi-6b](https://modelscope.cn/models/01ai/Yi-6B/summary), [yi-6b-200k](https://modelscope.cn/models/01ai/Yi-6B-200K/summary), [yi-6b-chat](https://modelscope.cn/models/01ai/Yi-6B-Chat/summary), [yi-34b](https://modelscope.cn/models/01ai/Yi-34B/summary), [yi-34b-200k](https://modelscope.cn/models/01ai/Yi-34B-200K/summary), [yi-34b-chat](https://modelscope.cn/models/01ai/Yi-34B-Chat/summary)
    - openbuddy series: [openbuddy-llama2-13b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama2-13b-v8.1-fp16/summary), [openbuddy-llama-65b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama-65b-v8-bf16/summary), [openbuddy-llama2-70b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama2-70b-v10.1-bf16/summary), [openbuddy-mistral-7b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-mistral-7b-v13.1/summary), [openbuddy-zephyr-7b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-zephyr-7b-v14.1/summary)
    - internlm series: [internlm-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-7b/summary), [internlm-7b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-v1_1/summary), [internlm-7b-chat-8k](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-8k/summary), [internlm-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-20b/summary), [internlm-20b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-20b/summary)
    - xverse series: [xverse-7b](https://modelscope.cn/models/xverse/XVERSE-7B/summary), [xverse-7b-chat](https://modelscope.cn/models/xverse/XVERSE-7B-Chat/summary), [xverse-13b](https://modelscope.cn/models/xverse/XVERSE-13B/summary), [xverse-13b-chat](https://modelscope.cn/models/xverse/XVERSE-13B-Chat/summary), [xverse-65b](https://modelscope.cn/models/xverse/XVERSE-65B/summary)
    - bluelm series: [bluelm-7b](https://modelscope.cn/models/vivo-ai/BlueLM-7B-Base/summary), [bluelm-7b-chat](https://modelscope.cn/models/vivo-ai/BlueLM-7B-Chat/summary), [bluelm-7b-32k](https://modelscope.cn/models/vivo-ai/BlueLM-7B-Base-32K/summary), [bluelm-7b-chat-32k](https://modelscope.cn/models/vivo-ai/BlueLM-7B-Chat-32K/summary)
    - mistral series: [mistral-7b](https://modelscope.cn/models/AI-ModelScope/Mistral-7B-v0.1/summary), [mistral-7b-chat](https://modelscope.cn/models/AI-ModelScope/Mistral-7B-Instruct-v0.1/summary)
    - zephyr series: [zephyr-7b-beta-chat](https://modelscope.cn/models/modelscope/zephyr-7b-beta/summary)
    - ziya series: [ziya2-13b](https://modelscope.cn/models/Fengshenbang/Ziya2-13B-Base/summary), [ziya2-13b-chat](https://modelscope.cn/models/Fengshenbang/Ziya2-13B-Chat/summary)
    - skywork series: [skywork-13b](https://modelscope.cn/models/skywork/Skywork-13B-base/summary), [skywork-13b-chat](https://modelscope.cn/models/skywork/Skywork-13B-chat/summary)
    - sus series: [sus-34b-chat](https://modelscope.cn/models/SUSTC/SUS-Chat-34B/summary)
    - other: [polylm-13b](https://modelscope.cn/models/damo/nlp_polylm_13b_text_generation/summary), [seqgpt-560m](https://modelscope.cn/models/damo/nlp_seqgpt-560m/summary)
  - Financial:
    - tongyi-finance series: [tongyi-finance-14b](https://modelscope.cn/models/TongyiFinance/Tongyi-Finance-14B/summary), [tongyi-finance-14b-chat](https://modelscope.cn/models/TongyiFinance/Tongyi-Finance-14B-Chat/summary), [tongyi-finance-14b-chat-int4](https://modelscope.cn/models/TongyiFinance/Tongyi-Finance-14B-Chat-Int4/summary)
  - Coding:
    - codefuse series: [codefuse-codellama-34b-chat](https://modelscope.cn/models/codefuse-ai/CodeFuse-CodeLlama-34B/summary)
- Supported Datasets:
  - NLP:
    - General: 🔥[alpaca-en](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-en/summary)(gpt4), 🔥[alpaca-zh](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-zh/summary)(gpt4), [multi-alpaca-all](https://www.modelscope.cn/datasets/damo/nlp_polylm_multialpaca_sft/summary), [instinwild-en](https://www.modelscope.cn/datasets/wyj123456/instinwild/summary), [instinwild-zh](https://www.modelscope.cn/datasets/wyj123456/instinwild/summary), [cot-en](https://www.modelscope.cn/datasets/YorickHe/CoT/summary), [cot-zh](https://www.modelscope.cn/datasets/YorickHe/CoT/summary), [firefly-all-zh](https://www.modelscope.cn/datasets/wyj123456/firefly/summary), [instruct-en](https://www.modelscope.cn/datasets/wyj123456/instruct/summary), [gpt4all-en](https://www.modelscope.cn/datasets/wyj123456/GPT4all/summary), [sharegpt-en](https://www.modelscope.cn/datasets/huangjintao/sharegpt/summary), [sharegpt-zh](https://www.modelscope.cn/datasets/huangjintao/sharegpt/summary), [tutu-v2-sft-mixture](https://modelscope.cn/datasets/AI-ModelScope/tulu-v2-sft-mixture/summary), [wikipedia-zh](https://modelscope.cn/datasets/AI-ModelScope/wikipedia-cn-20230720-filtered/summary), [open-orca](https://modelscope.cn/datasets/AI-ModelScope/OpenOrca/summary), [open-orca-gpt4](https://modelscope.cn/datasets/AI-ModelScope/OpenOrca/summary), [sharegpt-gpt4](https://modelscope.cn/datasets/AI-ModelScope/sharegpt_gpt4/summary)
    - Agent: [damo-agent-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary), 🔥[damo-agent-mini-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary), 🔥[agent-instruct-all-en](https://modelscope.cn/datasets/ZhipuAI/AgentInstruct/summary)
    - Coding: [code-alpaca-en](https://www.modelscope.cn/datasets/wyj123456/code_alpaca_en/summary), 🔥[leetcode-python-en](https://modelscope.cn/datasets/AI-ModelScope/leetcode-solutions-python/summary), 🔥[codefuse-python-en](https://modelscope.cn/datasets/codefuse-ai/CodeExercise-Python-27k/summary), 🔥[codefuse-evol-instruction-zh](https://modelscope.cn/datasets/codefuse-ai/Evol-instruction-66k/summary)
    - Medical: [medical-en](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary), [medical-zh](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary), [medical-mini-zh](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary)
    - Law: 🔥[lawyer-llama-zh](https://modelscope.cn/datasets/AI-ModelScope/lawyer_llama_data/summary), [tigerbot-law-zh](https://modelscope.cn/datasets/AI-ModelScope/tigerbot-law-plugin/summary)
    - Math: 🔥[blossom-math-zh](https://modelscope.cn/datasets/AI-ModelScope/blossom-math-v2/summary), [school-math-zh](https://modelscope.cn/datasets/AI-ModelScope/school_math_0.25M/summary), [open-platypus-en](https://modelscope.cn/datasets/AI-ModelScope/Open-Platypus/summary)
    - SQL: [text2sql-en](https://modelscope.cn/datasets/AI-ModelScope/texttosqlv2_25000_v2/summary), 🔥[sql-create-context-en](https://modelscope.cn/datasets/AI-ModelScope/sql-create-context/summary)
    - Text Generation: 🔥[advertise-gen-zh](https://modelscope.cn/datasets/lvjianjin/AdvertiseGen/summary), 🔥[dureader-robust-zh](https://modelscope.cn/datasets/modelscope/DuReader_robust-QG/summary)
    - Classification: [cmnli-zh](https://www.modelscope.cn/datasets/modelscope/clue/summary), 🔥[cmnli-mini-zh](https://www.modelscope.cn/datasets/modelscope/clue/summary), 🔥[jd-sentiment-zh](https://modelscope.cn/datasets/DAMO_NLP/jd/summary), [🔥hc3-zh](https://modelscope.cn/datasets/simpleai/HC3-Chinese/summary), [🔥hc3-en](https://modelscope.cn/datasets/simpleai/HC3/summary)
    - Other: [finance-en](https://www.modelscope.cn/datasets/wyj123456/finance_en/summary), [poetry-zh](https://www.modelscope.cn/datasets/modelscope/chinese-poetry-collection/summary), [webnovel-zh](https://modelscope.cn/datasets/AI-ModelScope/webnovel_cn/summary), [generated-chat-zh](https://modelscope.cn/datasets/AI-ModelScope/generated_chat_0.4M/summary), [cls-fudan-news-zh](https://modelscope.cn/datasets/damo/zh_cls_fudan-news/summary), [ner-jave-zh](https://modelscope.cn/datasets/damo/zh_ner-JAVE/summary)
  - Multi-Modal:
    - Vision: [coco-en](https://modelscope.cn/datasets/modelscope/coco_2014_caption/summary), 🔥[coco-mini-en](https://modelscope.cn/datasets/modelscope/coco_2014_caption/summary)
    - Audio: [aishell1-zh](https://modelscope.cn/datasets/speech_asr/speech_asr_aishell1_trainsets/summary), 🔥[aishell1-mini-zh](https://modelscope.cn/datasets/speech_asr/speech_asr_aishell1_trainsets/summary)
  - Custom Dataset
- Supported Templates:
  - Text Generation: default-generation, default-generation-bos, chatglm-generation
  - Chat: default, chatml(qwen), baichuan, chatglm2, chatglm3, llama, openbuddy, internlm, xverse, ziya, skywork, bluelm, yi, zephyr


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
