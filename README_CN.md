# SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning)

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
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
<a href="https://github.com/modelscope/swift/"><img src="https://img.shields.io/badge/ms--swift-Build from source-6FEBB9.svg"></a>
</p>

##  📖 目录
- [简介](#-简介)
- [新闻](#-新闻)
- 🔥[大模型训练推理](#-大模型训练推理)
- 🔥[SCEdit](#-SCEdit)
- [安装](#-安装)
- [快速开始](#-快速开始)
- [了解更多](#-了解更多)
- [License](#license)
- [联系我们](#-联系我们)

## 📝 简介
SWIFT（Scalable lightWeight Infrastructure for Fine-Tuning）是一个可扩展的轻量级一站式训练、推理深度学习框架。它集成了各种高效的微调方法，如LoRA、QLoRA、阿里云自研的ResTuning-Bypass等，以及开箱即用的训练推理脚本，使开发者可以在单张商业级显卡上微调推理LLM&AIGC模型。此外，SWIFT与[PEFT](https://github.com/huggingface/peft)完全兼容，使开发者可以在ModelScope模型体系中使用PEFT的能力。

目前支持的方法：

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
11. [PEFT](https://github.com/huggingface/peft)提供的tuners, 如IA3, AdaLoRA等

主要能力：
1. 可以通过model-id使SWIFT或PEFT的方法使用ModelScope Hub中的模型
2. 在单次训练或推理中可以使用多个tuners
3. 支持调用`activate_adapter`或`deactivate_adapter`或`set_active_adapters`来使部分tuner激活或失活，用户可以在推理时同时加载多个独立的tuners在不同线程中并行使用。
4. 支持通过脚本方式和命令行方式开启训练和推理，同时支持Web-UI方式进行推理。
5. 支持模型训练后的部署链路(vllm/chatglm.cpp/xinference)，详情可以查看[官方文档](./docs/source/GetStarted/部署指南.md)。

用户可以查看 [SWIFT官方文档](docs/source/GetStarted/快速使用.md) 来了解详细信息。

## 🎉 新闻
- 🔥2023.1.17: 支持internlm2系列: internlm2-7b-base, internlm2-7b, [internlm2-7b-sft-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/internlm2_7b_sft_chat), internlm2-7b-chat, internlm2-20b-base, internlm2-20b, internlm2-20b-sft-chat, internlm2-20b-chat.
- 2023.1.15: 支持yuan系列: yuan2-2b-instruct, [yuan2-2b-janus-instruct](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yuan2_2b_janus_instruct), yuan2-51b-instruct, yuan2-102b-instruct.
- 🔥2023.1.12: 支持**deepseek-moe**系列: deepseek-moe-16b, [deepseek-moe-16b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/deepseek_moe_16b_chat).
- 🔥2023.1.4: 支持**VLLM部署**, 兼容**OpenAI API**样式, 具体可以查看[VLLM推理加速与部署](https://github.com/modelscope/swift/blob/main/docs/source/LLM/VLLM推理加速与部署.md#部署).
- 2023.1.4: 更新[Benchmark](https://github.com/modelscope/swift/blob/main/docs/source/LLM/Benchmark.md), 方便查看不同模型训练的速度和所需显存.
- 🔥 2023.12.29: 支持web-ui进行sft训练和推理，安装ms-swift后使用`swift web-ui`开启
- 🔥 2023.12.29: 支持 DPO RLHF(Reinforcement Learning from Human Feedback) 和两个用于此任务的数据集: AI-ModelScope/stack-exchange-paired 以及 AI-ModelScope/hh-rlhf. 查看[文档](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E4%BA%BA%E7%B1%BB%E5%AF%B9%E9%BD%90%E8%AE%AD%E7%BB%83%E6%96%87%E6%A1%A3.md)开启训练！
- 🔥 2023.12.28: 支持SCEdit! 该tuner可显著降低U-Net中的显存占用，并支持低显存可控图像生成（取代ControlNet），阅读下面的章节来了解详细信息
- 2023.12.23: 支持[codegeex2-6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/codegeex2_6b).
- 2023.12.19: 支持[phi2-3b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/phi2_3b).
- 2023.12.18: 支持VLLM进行推理加速.
- 2023.12.15: 支持deepseek, deepseek-coder系列: deepseek-7b, deepseek-7b-chat, deepseek-67b, deepseek-67b-chat, openbuddy-deepseek-67b-chat, deepseek-coder-1_3b, deepseek-coder-1_3b-instruct, deepseek-coder-6_7b, deepseek-coder-6_7b-instruct, deepseek-coder-33b, deepseek-coder-33b-instruct.
- 2023.12.13: 支持mistral-7b-instruct-v2, [mixtral-moe-7b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/mixtral_7b_moe), [mixtral-moe-7b-instruct](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/mixtral_7b_moe_instruct).
- 2023.12.9: 支持`freeze_parameters`参数, 作为lora和全参数训练的折中方案. 对应的sh可以查看[full_freeze_ddp](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/full_freeze_ddp). 支持`disable_tqdm`, `lazy_tokenize`, `preprocess_num_proc`参数, 具体可以查看[命令行参数](https://github.com/modelscope/swift/blob/main/docs/source/LLM/命令行参数.md).
- 2023.12.8: 支持[sus-34b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/sus_34b_chat), 支持yi-6b-200k, yi-34b-200k.
- 2023.12.7: 支持[Multi-Node DDP训练](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E5%BE%AE%E8%B0%83%E6%96%87%E6%A1%A3.md#%E4%BD%BF%E7%94%A8cli).
- 2023.12.5: 支持模型: zephyr-7b-beta-chat, openbuddy-zephyr-7b-chat. 支持数据集: hc3-zh, hc3-en.
- 🔥 2023.12.2: [自我认知微调最佳实践](https://github.com/modelscope/swift/blob/main/docs/source/LLM/自我认知微调最佳实践.md), **10分钟对大模型进行自我认知微调**, 创建专属于自己的大模型.
<details><summary>更多</summary>

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
- 🔥 2023.10.7: 支持**DeepSpeed ZeRO-2**, 使得lora(不仅仅是qlora)可以在双卡A10上运行DDP.
- 2023.10.4: 支持更多数学, 法律, SQL, 代码领域的数据集: blossom-math-zh, school-math-zh, text2sql-en, sql-create-context-en, lawyer-llama-zh, tigerbot-law-zh, leetcode-python-en.
- 🔥 2023.9.25: 支持**qwen-14b**系列: qwen-14b, qwen-14b-chat.
- 2023.9.18: 支持**internlm-20b**系列: internlm-20b, internlm-20b-chat.
- 2023.9.12: 支持**MP+DDP**对全参数训练进行加速.
- 2023.9.5: 支持**openbuddy-llama2-70b-chat**.
- 2023.9.3: 支持**baichuan2**系列: baichuan2-7b, baichuan2-7b-chat, baichuan2-13b, baichuan2-13b-chat.
</details>


## ✨ 大模型训练推理
### WEB UI训练推理

安装SWIFT之后，用如下方式启动界面训练推理：

```shell
swift web-ui
```

> 支持的环境变量：
>
> WEBUI_SHARE=1 控制gradio是否是share状态
> SWIFT_UI_LANG=en/zh 控制web-ui界面语言
> WEBUI_SERVER server_name参数， web-ui host ip，0.0.0.0代表所有ip均可访问，127.0.0.1代表只允许本机访问
> WEBUI_PORT web-ui的端口号

下面是一个web-ui的简单视频介绍：



[![Watch the video](docs/source/cources/resources/20240119160942.jpg)](https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/SWIFT%E8%A7%86%E9%A2%91_%E8%B0%83%E6%95%B4%E5%B0%81%E9%9D%A2.mp4)

### 简单使用

- **10分钟**对大模型进行**自我认知微调**, 创建专属于自己的大模型, 可以查看[自我认知微调最佳实践](https://github.com/modelscope/swift/blob/main/docs/source/LLM/自我认知微调最佳实践.md).
- 快速对LLM进行**推理**, 搭建**Web-UI**, 可以查看[LLM推理文档](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM推理文档.md).
- 快速对LLM进行**微调**, 推理并搭建Web-UI, 可以查看[LLM微调文档](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM微调文档.md).
- 使用**界面**方式进行微调和推理, 可以查看[WEB-UI文档](https://github.com/modelscope/swift/blob/main/docs/source/GetStarted/%E7%95%8C%E9%9D%A2%E8%AE%AD%E7%BB%83%E6%8E%A8%E7%90%86.md).
- 支持**DPO训练**, 使用[这个脚本](https://github.com/modelscope/swift/blob/main/examples/pytorch/llm/scripts/dpo/lora_ddp_mp/dpo.sh)开启训练
- 使用VLLM进行**推理加速**和**部署(OpenAI API)**. 可以查看[VLLM推理加速与部署](https://github.com/modelscope/swift/blob/main/docs/source/LLM/VLLM推理加速与部署.md).
- 查看swift支持的模型和数据集. 可以查看[支持的模型和数据集](https://github.com/modelscope/swift/blob/main/docs/source/LLM/支持的模型和数据集.md).
- 对swift中的模型, 数据集, 对话模板进行**拓展**, 可以查看[自定义与拓展](https://github.com/modelscope/swift/blob/main/docs/source/LLM/自定义与拓展.md).
- 查询微调和推理的命令行参数, 可以查看[命令行参数](https://github.com/modelscope/swift/blob/main/docs/source/LLM/命令行参数.md).
- 查看不同参数下的训练时间和训练显存对比, 可以查看[Benchmark](https://github.com/modelscope/swift/blob/main/docs/source/LLM/Benchmark.md).


### 特性
- 支持的SFT方法: [lora](https://arxiv.org/abs/2106.09685), [qlora](https://arxiv.org/abs/2305.14314), [longlora](https://arxiv.org/abs/2309.12307), [qalora](https://arxiv.org/abs/2309.14717), 全参数微调, 部分参数微调.
- 支持的特性: 模型量化, DDP, 模型并行, gradient checkpointing, 支持推送ModelScope Hub, 自定义数据集, 多模态和Agent SFT, 多轮对话, DPO, 自我认知微调, ...
- 支持的模型: [[详细]](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.md#%E6%A8%A1%E5%9E%8B)
  - 多模态:
    - qwen-vl 系列: [qwen-vl](https://modelscope.cn/models/qwen/Qwen-VL/summary), [qwen-vl-chat](https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary), [qwen-vl-chat-int4](https://modelscope.cn/models/qwen/Qwen-VL-Chat-Int4/summary)
    - qwen-audio 系列: [qwen-audio](https://modelscope.cn/models/qwen/Qwen-Audio/summary), [qwen-audio-chat](https://modelscope.cn/models/qwen/Qwen-Audio-Chat/summary)
    - cogagent 系列: [cogagent-chat](https://www.modelscope.cn/models/ZhipuAI/cogagent-chat/summary), [cogagent-vqa](https://www.modelscope.cn/models/ZhipuAI/cogagent-vqa/summary)
  - 通用:
    - qwen 系列: [qwen-1_8b-chat](https://modelscope.cn/models/qwen/Qwen-1_8B/summary), [qwen-1_8b-chat-int4](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat-Int4/summary), [qwen-1_8b-chat-int8](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat-Int8/summary), [qwen-7b](https://modelscope.cn/models/qwen/Qwen-7B/summary), [qwen-7b-chat](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary), [qwen-7b-chat-int4](https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int4/summary), [qwen-7b-chat-int8](https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int8/summary), [qwen-14b](https://modelscope.cn/models/qwen/Qwen-14B/summary), [qwen-14b-chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary), [qwen-14b-chat-int4](https://modelscope.cn/models/qwen/Qwen-14B-Chat-Int4/summary), [qwen-14b-chat-int8](https://modelscope.cn/models/qwen/Qwen-14B-Chat-Int8/summary), [qwen-72b](https://modelscope.cn/models/qwen/Qwen-72B/summary), [qwen-72b-chat](https://modelscope.cn/models/qwen/Qwen-72B-Chat/summary), [qwen-72b-chat-int4](https://modelscope.cn/models/qwen/Qwen-72B-Chat-Int4/summary), [qwen-72b-chat-int8](https://modelscope.cn/models/qwen/Qwen-72B-Chat-Int8/summary)
    - chatglm 系列: [chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary), [chatglm2-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm2-6b-32k/summary), [chatglm3-6b-base](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-base/summary), [chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary), [chatglm3-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-32k/summary)
    - llama 系列: [llama2-7b](https://modelscope.cn/models/modelscope/Llama-2-7b-ms/summary), [llama2-7b-chat](https://modelscope.cn/models/modelscope/Llama-2-7b-chat-ms/summary), [llama2-13b](https://modelscope.cn/models/modelscope/Llama-2-13b-ms/summary), [llama2-13b-chat](https://modelscope.cn/models/modelscope/Llama-2-13b-chat-ms/summary), [llama2-70b](https://modelscope.cn/models/modelscope/Llama-2-70b-ms/summary), [llama2-70b-chat](https://modelscope.cn/models/modelscope/Llama-2-70b-chat-ms/summary)
    - yi 系列: [yi-6b](https://modelscope.cn/models/01ai/Yi-6B/summary), [yi-6b-200k](https://modelscope.cn/models/01ai/Yi-6B-200K/summary), [yi-6b-chat](https://modelscope.cn/models/01ai/Yi-6B-Chat/summary), [yi-34b](https://modelscope.cn/models/01ai/Yi-34B/summary), [yi-34b-200k](https://modelscope.cn/models/01ai/Yi-34B-200K/summary), [yi-34b-chat](https://modelscope.cn/models/01ai/Yi-34B-Chat/summary)
    - internlm 系列: [internlm-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-7b/summary), [internlm-7b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-v1_1/summary), [internlm-7b-chat-8k](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-8k/summary), [internlm-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-20b/summary), [internlm-20b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-20b/summary), [internlm2-7b-base](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-7b/summary), [internlm2-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-7b/summary), [internlm2-7b-sft-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b-sft/summary), [internlm2-7b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-7b/summary), [internlm2-20b-base](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-base-20b/summary), [internlm2-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-20b/summary), [internlm2-20b-sft-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b-sft/summary), [internlm2-20b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm2-chat-20b/summary)
    - deepseek 系列: [deepseek-7b](https://modelscope.cn/models/deepseek-ai/deepseek-llm-7b-base/summary), [deepseek-7b-chat](https://modelscope.cn/models/deepseek-ai/deepseek-llm-7b-chat/summary), [deepseek-67b](https://modelscope.cn/models/deepseek-ai/deepseek-llm-67b-base/summary), [deepseek-67b-chat](https://modelscope.cn/models/deepseek-ai/deepseek-llm-67b-chat/summary), [deepseek-moe-16b](https://modelscope.cn/models/deepseek-ai/deepseek-moe-16b-base/summary), [deepseek-moe-16b-chat|[deepseek-ai/deepseek-moe-16b-chat](https://modelscope.cn/models/deepseek-ai/deepseek-moe-16b-chat/summary)
    - openbuddy 系列: [openbuddy-llama2-13b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama2-13b-v8.1-fp16/summary), [openbuddy-llama-65b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama-65b-v8-bf16/summary), [openbuddy-llama2-70b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama2-70b-v10.1-bf16/summary), [openbuddy-mistral-7b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-mistral-7b-v13.1/summary), [openbuddy-zephyr-7b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-zephyr-7b-v14.1/summary), [openbuddy-deepseek-67b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-deepseek-67b-v15.2/summary)
    - mistral 系列: [mistral-7b](https://modelscope.cn/models/AI-ModelScope/Mistral-7B-v0.1/summary), [mistral-7b-instruct](https://modelscope.cn/models/AI-ModelScope/Mistral-7B-Instruct-v0.1/summary), [mistral-7b-instruct-v2](https://modelscope.cn/models/AI-ModelScope/Mistral-7B-Instruct-v0.2/summary), [mixtral-moe-7b](https://modelscope.cn/models/AI-ModelScope/Mixtral-8x7B-v0.1/summary), [mixtral-moe-7b-instruct](https://modelscope.cn/models/AI-ModelScope/Mixtral-8x7B-Instruct-v0.1/summary)
    - baichuan 系列: [baichuan-7b](https://modelscope.cn/models/baichuan-inc/baichuan-7B/summary), [baichuan-13b](https://modelscope.cn/models/baichuan-inc/Baichuan-13B-Base/summary), [baichuan-13b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan-13B-Chat/summary), [baichuan2-7b](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Base/summary), [baichuan2-7b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat/summary), [baichuan2-13b](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Base/summary), [baichuan2-13b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat/summary), [baichuan2-7b-chat-int4](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat-4bits/summary), [baichuan2-13b-chat-int4](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat-4bits/summary)
    - yuan 系列: [yuan2-2b-instruct](https://modelscope.cn/models/YuanLLM/Yuan2.0-2B-hf/summary), [yuan2-2b-janus-instruct](https://modelscope.cn/models/YuanLLM/Yuan2-2B-Janus-hf/summary), [yuan2-51b-instruct](https://modelscope.cn/models/YuanLLM/Yuan2.0-51B-hf/summary), [yuan2-102b-instruct](https://modelscope.cn/models/YuanLLM/Yuan2.0-102B-hf/summary)
    - xverse 系列: [xverse-7b](https://modelscope.cn/models/xverse/XVERSE-7B/summary), [xverse-7b-chat](https://modelscope.cn/models/xverse/XVERSE-7B-Chat/summary), [xverse-13b](https://modelscope.cn/models/xverse/XVERSE-13B/summary), [xverse-13b-chat](https://modelscope.cn/models/xverse/XVERSE-13B-Chat/summary), [xverse-65b](https://modelscope.cn/models/xverse/XVERSE-65B/summary)
    - bluelm 系列: [bluelm-7b](https://modelscope.cn/models/vivo-ai/BlueLM-7B-Base/summary), [bluelm-7b-chat](https://modelscope.cn/models/vivo-ai/BlueLM-7B-Chat/summary), [bluelm-7b-32k](https://modelscope.cn/models/vivo-ai/BlueLM-7B-Base-32K/summary), [bluelm-7b-chat-32k](https://modelscope.cn/models/vivo-ai/BlueLM-7B-Chat-32K/summary)
    - zephyr 系列: [zephyr-7b-beta-chat](https://modelscope.cn/models/modelscope/zephyr-7b-beta/summary)
    - ziya 系列: [ziya2-13b](https://modelscope.cn/models/Fengshenbang/Ziya2-13B-Base/summary), [ziya2-13b-chat](https://modelscope.cn/models/Fengshenbang/Ziya2-13B-Chat/summary)
    - skywork 系列: [skywork-13b](https://modelscope.cn/models/skywork/Skywork-13B-base/summary), [skywork-13b-chat](https://modelscope.cn/models/skywork/Skywork-13B-chat/summary)
    - sus 系列: [sus-34b-chat](https://modelscope.cn/models/SUSTC/SUS-Chat-34B/summary)
    - other: [polylm-13b](https://modelscope.cn/models/damo/nlp_polylm_13b_text_generation/summary), [seqgpt-560m](https://modelscope.cn/models/damo/nlp_seqgpt-560m/summary)
  - 金融:
    - tongyi-finance 系列: [tongyi-finance-14b](https://modelscope.cn/models/TongyiFinance/Tongyi-Finance-14B/summary), [tongyi-finance-14b-chat](https://modelscope.cn/models/TongyiFinance/Tongyi-Finance-14B-Chat/summary), [tongyi-finance-14b-chat-int4](https://modelscope.cn/models/TongyiFinance/Tongyi-Finance-14B-Chat-Int4/summary)
  - 代码:
    - codefuse 系列: [codefuse-codellama-34b-chat](https://modelscope.cn/models/codefuse-ai/CodeFuse-CodeLlama-34B/summary)
    - deepseek-coder 系列: [deepseek-coder-1_3b](https://modelscope.cn/models/deepseek-ai/deepseek-coder-1.3b-base/summary), [deepseek-coder-1_3b-instruct](https://modelscope.cn/models/deepseek-ai/deepseek-coder-1.3b-instruct/summary), [deepseek-coder-6_7b](https://modelscope.cn/models/deepseek-ai/deepseek-coder-6.7b-base/summary), [deepseek-coder-6_7b-instruct](https://modelscope.cn/models/deepseek-ai/deepseek-coder-6.7b-instruct/summary), [deepseek-coder-33b](https://modelscope.cn/models/deepseek-ai/deepseek-coder-33b-base/summary), [deepseek-coder-33b-instruct](https://modelscope.cn/models/deepseek-ai/deepseek-coder-33b-instruct/summary)
    - phi 系列: [phi2-3b](https://modelscope.cn/models/AI-ModelScope/phi-2/summary)
- 支持的数据集: [[详细]](https://github.com/modelscope/swift/blob/main/docs/source/LLM/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.md#%E6%95%B0%E6%8D%AE%E9%9B%86)
  - NLP:
    - 通用: 🔥[alpaca-en](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-en/summary)(gpt4), 🔥[alpaca-zh](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-zh/summary)(gpt4), [multi-alpaca-all](https://www.modelscope.cn/datasets/damo/nlp_polylm_multialpaca_sft/summary), [instinwild-en](https://www.modelscope.cn/datasets/wyj123456/instinwild/summary), [instinwild-zh](https://www.modelscope.cn/datasets/wyj123456/instinwild/summary), [cot-en](https://www.modelscope.cn/datasets/YorickHe/CoT/summary), [cot-zh](https://www.modelscope.cn/datasets/YorickHe/CoT/summary), [firefly-all-zh](https://www.modelscope.cn/datasets/wyj123456/firefly/summary), [instruct-en](https://www.modelscope.cn/datasets/wyj123456/instruct/summary), [gpt4all-en](https://www.modelscope.cn/datasets/wyj123456/GPT4all/summary), [sharegpt-en](https://www.modelscope.cn/datasets/huangjintao/sharegpt/summary), [sharegpt-zh](https://www.modelscope.cn/datasets/huangjintao/sharegpt/summary), [tutu-v2-sft-mixture](https://modelscope.cn/datasets/AI-ModelScope/tulu-v2-sft-mixture/summary), [wikipedia-zh](https://modelscope.cn/datasets/AI-ModelScope/wikipedia-cn-20230720-filtered/summary), [open-orca](https://modelscope.cn/datasets/AI-ModelScope/OpenOrca/summary), [open-orca-gpt4](https://modelscope.cn/datasets/AI-ModelScope/OpenOrca/summary), [sharegpt-gpt4](https://modelscope.cn/datasets/AI-ModelScope/sharegpt_gpt4/summary)
    - Agent: [damo-agent-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary), 🔥[damo-agent-mini-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary), 🔥[agent-instruct-all-en](https://modelscope.cn/datasets/ZhipuAI/AgentInstruct/summary)
    - RLHF: 🔥[hh-rlhf](https://www.modelscope.cn/datasets/AI-ModelScope/hh-rlhf/summary), [stack-exchange-paired](https://www.modelscope.cn/datasets/AI-ModelScope/stack-exchange-paired/summary)
    - 代码: [code-alpaca-en](https://www.modelscope.cn/datasets/wyj123456/code_alpaca_en/summary), 🔥[leetcode-python-en](https://modelscope.cn/datasets/AI-ModelScope/leetcode-solutions-python/summary), 🔥[codefuse-python-en](https://modelscope.cn/datasets/codefuse-ai/CodeExercise-Python-27k/summary), 🔥[codefuse-evol-instruction-zh](https://modelscope.cn/datasets/codefuse-ai/Evol-instruction-66k/summary)
    - 医疗: [medical-en](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary), [medical-zh](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary), [medical-mini-zh](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary)
    - 法律: 🔥[lawyer-llama-zh](https://modelscope.cn/datasets/AI-ModelScope/lawyer_llama_data/summary), [tigerbot-law-zh](https://modelscope.cn/datasets/AI-ModelScope/tigerbot-law-plugin/summary)
    - 数学: 🔥[blossom-math-zh](https://modelscope.cn/datasets/AI-ModelScope/blossom-math-v2/summary), [school-math-zh](https://modelscope.cn/datasets/AI-ModelScope/school_math_0.25M/summary), [open-platypus-en](https://modelscope.cn/datasets/AI-ModelScope/Open-Platypus/summary)
    - SQL: [text2sql-en](https://modelscope.cn/datasets/AI-ModelScope/texttosqlv2_25000_v2/summary), 🔥[sql-create-context-en](https://modelscope.cn/datasets/AI-ModelScope/sql-create-context/summary)
    - 文本生成: 🔥[advertise-gen-zh](https://modelscope.cn/datasets/lvjianjin/AdvertiseGen/summary), 🔥[dureader-robust-zh](https://modelscope.cn/datasets/modelscope/DuReader_robust-QG/summary)
    - 分类: [cmnli-zh](https://www.modelscope.cn/datasets/modelscope/clue/summary), 🔥[cmnli-mini-zh](https://www.modelscope.cn/datasets/modelscope/clue/summary), 🔥[jd-sentiment-zh](https://modelscope.cn/datasets/DAMO_NLP/jd/summary), [🔥hc3-zh](https://modelscope.cn/datasets/simpleai/HC3-Chinese/summary), [🔥hc3-en](https://modelscope.cn/datasets/simpleai/HC3/summary)
    - 其他: [finance-en](https://www.modelscope.cn/datasets/wyj123456/finance_en/summary), [poetry-zh](https://www.modelscope.cn/datasets/modelscope/chinese-poetry-collection/summary), [webnovel-zh](https://modelscope.cn/datasets/AI-ModelScope/webnovel_cn/summary), [generated-chat-zh](https://modelscope.cn/datasets/AI-ModelScope/generated_chat_0.4M/summary), [cls-fudan-news-zh](https://modelscope.cn/datasets/damo/zh_cls_fudan-news/summary), [ner-jave-zh](https://modelscope.cn/datasets/damo/zh_ner-JAVE/summary)
  - 多模态:
    - 视觉: [coco-en](https://modelscope.cn/datasets/modelscope/coco_2014_caption/summary), 🔥[coco-mini-en](https://modelscope.cn/datasets/modelscope/coco_2014_caption/summary)
    - 音频: [aishell1-zh](https://modelscope.cn/datasets/speech_asr/speech_asr_aishell1_trainsets/summary), 🔥[aishell1-mini-zh](https://modelscope.cn/datasets/speech_asr/speech_asr_aishell1_trainsets/summary)
  - 自定义数据集
- 支持的对话模板:
  - 文本生成: default-generation, default-generation-bos, chatglm-generation
  - 对话: default, qwen, baichuan, chatglm2, chatglm3, llama, openbuddy, internlm, internlm2, yi, yuan, xverse, ziya, skywork, bluelm, zephyr, sus, deepseek, deepseek-coder, codefuse-codellama, cogagent


## 🔥SCEdit

SCEdit由阿里巴巴通义实验室视觉智能团队(Alibaba TongYi Vision Intelligence Lab)所提出，是一个高效的生成式微调框架。该框架不仅支持文生图下游任务的微调能力，**相比LoRA节省30%-50%的训练显存开销**，实现快速迁移到特定的生成场景中；而且还可以**直接扩展到可控图像生成任务中，仅需ControlNet条件生成7.9%的参数量并节省30%的显存开销**，支持边缘图、深度图、分割图、姿态、颜色图、图像补全等条件生成任务。

我们使用了[风格迁移数据集](https://modelscope.cn/datasets/damo/style_custom_dataset/dataPeview)中的3D风格数据进行了训练，并使用相同的`Prompt: A boy in a camouflage jacket with a scarf`进行测试，具体的定性和定量的结果如下：

| Method    | bs   | ep   | Target Module | Param. (M)    | Mem. (MiB) | 3D style                                                     |
| --------- | ---- | ---- | ------------- | ------------- | ---------- | ------------------------------------------------------------ |
| LoRA/r=64 | 1    | 50   | q/k/v/out/mlp | 23.94 (2.20%) | 8440MiB    | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703665229562-0f33bbb0-c492-41b4-9f37-3ae720dca80d.png" alt="img" style="zoom:20%;" /> |
| SCEdit    | 1    | 50   | up_blocks     | 19.68 (1.81%) | 7556MiB    | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703665933913-74b98741-3b57-46a4-9871-539df3a0112c.png" alt="img" style="zoom:20%;" /> |
| LoRA/r=64 | 10   | 100  | q/k/v/out/mlp | 23.94 (2.20%) | 26300MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703750608529-de20d0e7-bf9c-4928-8e59-73cc54f2c8d7.png" alt="img" style="zoom:20%;" /> |
| SCEdit    | 10   | 100  | up_blocks     | 19.68 (1.81%) | 18634MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703663033092-94492e44-341f-4259-9df4-13c168e3b5d6.png" alt="img" style="zoom:20%;" /> |
| LoRA/r=64 | 30   | 200  | q/k/v/out/mlp | 23.94 (2.20%) | 69554MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703750626635-2e368d7b-5e99-4a06-b189-8615f302bcd7.png" alt="img" style="zoom:20%;" /> |
| SCEdit    | 30   | 200  | up_blocks     | 19.68 (1.81%) | 43350MiB   | <img src="https://intranetproxy.alipay.com/skylark/lark/0/2023/png/167218/1703662246942-1102b1f4-93ab-4653-b943-3302f2a5259e.png" alt="img" style="zoom:20%;" /> |

使用SCEdit执行训练任务并复现上述结果：

```shell
# 先执行下面章节的安装步骤
cd examples/pytorch/multi_modal/notebook
python text_to_image_synthesis.py
```

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
# 仅使用adapters
pip install ms-swift -U
```

- 方法2：通过源代码安装SWIFT（方便运行训练推理脚本），请运行以下命令：

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .[llm]
```

SWIFT依赖torch>=1.13。

- 方法3：在我们的Docker镜像中使用SWIFT

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.8.0-py38-torch2.0.1-tf2.13.0-1.9.1
```

## 🚀 快速开始

SWIFT支持多个tuners，包括由[PEFT](https://github.com/huggingface/peft)提供的tuners。要使用这些tuners，只需调用:
```python
from swift import Swift, LoRAConfig
config = LoRAConfig(...)
model = Swift.prepare_model(model, config, extra_state_keys=['...'])
```
上面的代码片段随机初始化了tuner。输入model是torch.nn.Module的一个实例，config是SwiftConfig或PeftConfig的子类实例。extra_state_keys是要训练并存储在输出目录中的额外模块权重（如linear head）。

您可以通过以下方式组合多个tuners：
```python
from swift import Swift, LoRAConfig, PromptConfig
model = Swift.prepare_model(model, {'lora': LoRAConfig(...), 'prompt': PromptConfig(...)})
```

在微调之后，您可以调用save_pretrained和push_to_hub方法：

```python
from swift import push_to_hub
model.save_pretrained('some-output-folder')
push_to_hub('my-group/some-repo-id-modelscope', 'some-output-folder', token='some-ms-token')
```
假设`my-group/some-repo-id-modelscope`是Hub中的model-id，而`some-ms-token`是用于上传的令牌。

使用model-id进行后续推理：

```python
from swift import Swift
model = Swift.from_pretrained(model, 'my-group/some-repo-id-modelscope')
```

下面是一个可运行的示例：

```python
import os
import tempfile

# 请通过`pip install modelscope`安装modelscope
from modelscope import Model

from swift import LoRAConfig, SwiftModel, Swift, push_to_hub

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)


model = Model.from_pretrained('modelscope/Llama-2-7b-ms', device_map='auto')
lora_config = LoRAConfig(target_modules=['q_proj', 'k_proj', 'v_proj'])
model: SwiftModel = Swift.prepare_model(model, lora_config)
# 在这里进行一些微调操作
model.save_pretrained(tmp_dir)

push_to_hub('my-group/swift_llama2', output_dir=tmp_dir)
model = Model.from_pretrained('modelscope/Llama-2-7b-ms', device_map='auto')
model = SwiftModel.from_pretrained(model, 'my-group/swift_llama2', device_map='auto')
```

这是一个使用transformers库实例化模型，并使用SWIFT进行高效微调的示例。

```python
from swift import Swift, LoRAConfig, AdapterConfig, PromptConfig
from transformers import AutoModelForImageClassification

# 初始vit模型
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

# 初始化LoRA tuner配置
lora_config = LoRAConfig(
    r=10,  # LoRA模块的rank
    target_modules=['query', 'key', 'value'],  # 将要被替换的模块的模块名后缀
    merge_weights=False  # 是否合并权重
)

# 初始化adapter tuner配置
adapter_config = AdapterConfig(
    dim=768,  # hidden states的维度
    hidden_pos=0,  # 要传递到adapter的hidden state的位置
    target_modules=r'.*attention.output.dense$',  # 要使用正则表达式替换的模块
    adapter_length=10  # adapter长度
)

# 初始化prompt tuner配置
prompt_config = PromptConfig(
    dim=768,  # hidden states的维度
    target_modules=r'.*layer\.\d+$',  # 要使用正则表达式替换的模块
    embedding_pos=0,    # embedding张量的位置
    prompt_length=10,   # 提示符token的长度
    attach_front=False  # 是否将提示符附加在embedding前面
)

# 使用swift创建模型。在实践中，您可以使用其中任何一个调谐器或它们的组合。
model = Swift.prepare_model(model, {"lora_tuner": lora_config, "adapter_tuner": adapter_config, "prompt_tuner": prompt_config})

# 获取模型的可训练参数。
model.get_trainable_parameters()
# 'trainable params: 838,776 || all params: 87,406,432 || trainable%: 0.9596273189597764'
```

可以在SWIFT中使用PEFT提供的功能：

```python
from swift import LoraConfig, Swift
from peft import TaskType
lora_config = LoraConfig(target_modules=['query', 'key', 'value'], task_type=TaskType.CAUSAL_LM)
model_wrapped = Swift.prepare_model(model, lora_config)

# 或者使用from_pretrained从modelscope hub中加载权重。
model_wrapped = Swift.from_pretrained(model, 'some-id-in-the-modelscope-modelhub')
```

Swift tuners和Peft tuners之间的保存策略略有不同。可以通过以下方式为Swift tuners命名：

```python
model = Swift.prepare_model(model, {'default': LoRAConfig(...)})
model.save_pretrained('./output')
```

在output目录中将会得到以下类似的目录结构：

```text
output
    |-- default
        |-- adapter_config.json
        |-- adapter_model.bin
    |-- adapter_config.json
    |-- adapter_model.bin
```

存储在output目录中的config/weights是extra_state_keys的配置和权重。这与Peft不同，Peft存储了`default` tuner的config/weights。

## 🔍 了解更多

- [ModelScope库](https://github.com/modelscope/modelscope/)

  ModelScope库是ModelScope项目的模型库，包含了各模态热门的深度学习模型。

- [将自己的模型贡献给ModelScope](https://modelscope.cn/docs/ModelScope%E6%A8%A1%E5%9E%8B%E6%8E%A5%E5%85%A5%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88)

## License

本项目使用[Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE)进行许可。


## ☎ 联系我们

您可以通过加我们的微信群, 来和我们联系和交流:

<p align="left">
<img src="asset/wechat.png" width="250" style="display: inline-block;">
</p>


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/swift&type=Date)](https://star-history.com/#modelscope/swift&Date)
