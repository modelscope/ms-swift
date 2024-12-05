# SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning)

<p align="center">
    <br>
    <img src="resources/banner.png"/>
    <br>
<p>
<p align="center">
<a href="https://modelscope.cn/home">魔搭社区官网</a>
<br>
        中文&nbsp ｜ &nbsp<a href="README.md">English</a>&nbsp
</p>


<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.17-5D91D4.svg"></a>
<a href="https://pypi.org/project/ms-swift/"><img src="https://badge.fury.io/py/ms-swift.svg"></a>
<a href="https://github.com/modelscope/swift/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/swift"></a>
<a href="https://pepy.tech/project/ms-swift"><img src="https://pepy.tech/badge/ms-swift"></a>
<a href="https://github.com/modelscope/swift/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/6427" target="_blank"><img src="https://trendshift.io/api/badge/repositories/6427" alt="modelscope%2Fswift | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

##  📖 目录
- [简介](#-简介)
- [用户群](#-用户群)
- [新闻](#-新闻)
- [快速开始](#-快速开始)
- [License](#-license)
- [引用](#-引用)

## 📝 简介
🍲 ms-swift是魔搭社区官方提供的LLM与多模态LLM微调部署框架，现已支持400+LLM与100+多模态LLM的训练（预训练、微调、人类对齐）、推理、评测、量化与部署。其中LLM包括：Qwen2.5、Llama3.2、GLM4、Internlm2.5、Yi1.5、Mistral、DeepSeek、Baichuan2、Gemma2、TeleChat2等模型，多模态LLM包括：Qwen2-VL、Qwen2-Audio、Llama3.2-Vision、Llava、InternVL2、MiniCPM-V-2.6、GLM4v、Xcomposer2.5、Yi-VL、DeepSeek-VL、Phi3.5-Vision等模型。

🍔 除此之外，ms-swift汇集了最新的训练技术，包括LoRA、QLoRA、Llama-Pro、LongLoRA、GaLore、Q-GaLore、LoRA+、LISA、DoRA、FourierFt、ReFT、UnSloth、Megatron和Liger等。ms-swift支持使用vLLM和LMDeploy对推理、评测和部署模块进行加速。为了帮助研究者和开发者更轻松地微调和应用大模型，ms-swift还提供了基于Gradio的Web-UI界面及丰富的最佳实践。

SWIFT具有丰富全面的文档，请查看我们的文档网站:
<p align="center">
        <a href="https://arxiv.org/abs/2408.05517">论文</a> &nbsp ｜ <a href="https://swift.readthedocs.io/en/latest/">English Documentation</a> &nbsp ｜ &nbsp <a href="https://swift.readthedocs.io/zh-cn/latest/">中文文档</a> &nbsp
</p>
<p align="center">
        <a href="https://swift2x-en.readthedocs.io/en/latest/">Swift2.x En Doc</a> &nbsp ｜ &nbsp <a href="https://swift2x.readthedocs.io/zh-cn/latest/">Swift2.x中文文档</a> &nbsp
</p>

## ☎ 用户群

请扫描下面的二维码来加入我们的交流群：

[Discord Group](https://discord.com/invite/D27yfEFVz5)              |  微信群
:-------------------------:|:-------------------------:
<img src="asset/discord_qr.jpg" width="200" height="200">  |  <img src="asset/wechat.png" width="200" height="200">


## 🎉 新闻
- 🎁2024.12.04: SWIFT3.0大版本更新. 请查看[ReleaseNote和BreakChange](./docs/source/Instruction/ReleaseNote3.0.md).
- 2024.11.28: 支持模型qwq-32b-preview, marco-o1, 支持数据集open-o1.
- 2024.10.09: 支持LLM和MLLM的reward modeling、PPO训练.
- 2024.09.26: 支持llama3.2、llama3.2-vision系列模型的训练到部署.
- 🔥2024.09.19: 支持qwen2.5、qwen2.5-math、qwen2.5-coder系列模型. 支持qwen2-vl-72b系列模型. 最佳实践可以查看[这里](https://github.com/modelscope/ms-swift/issues/2064).
- 🔥2024.08.30: 支持qwen2-vl系列模型的推理与微调: qwen2-vl-2b-instruct, qwen2-vl-7b-instruct.
- 🔥2024.08.26: 支持[Liger](https://github.com/linkedin/Liger-Kernel), 该内核支持LLaMA、Qwen、Mistral等模型, 并大幅减少显存使用(10%~60%), 使用`--use_liger true`开启训练.
- 🔥2024.08.22: 支持[ReFT](https://github.com/stanfordnlp/pyreft), 该tuner可以以LoRA的1/15~1/65的参数量达到和LoRA匹配或更好的效果, 使用`--sft_type reft`开始训练!
- 🔥2024.08.12: 🎉 SWIFT论文已经发布到arXiv上，可以点击[这个链接](https://arxiv.org/abs/2408.05517)阅读.
<details><summary>More</summary>

- 🔥2024.08.12: 支持packing和flash-attention时不污染attention_mask, 使用`--packing`开启。详情见[PR](https://github.com/huggingface/transformers/pull/31629/files).
- 🔥2024.08.09: 支持qwen2-audio模型的推理与微调. 最佳实践可以查看[这里](https://github.com/modelscope/ms-swift/issues/1653).
- 🔥2024.08.05: 支持多模态数据集的评测！命令行完全一致，新增了许多[多模态数据集](https://swift.readthedocs.io/zh-cn/latest/Instruction/LLM%E8%AF%84%E6%B5%8B%E6%96%87%E6%A1%A3.html#id2).
- 🔥2024.07.19: 支持[Q-Galore](https://arxiv.org/abs/2407.08296)算法, 该算法可以减少显存使用约60% (qwen-7b-chat, full, 80G -> 35G), 使用命令行:`swift sft --model_type xxx --use_galore true --galore_quantization true`来开始训练!
- 🔥2024.07.16: 支持ollama和bitsandbytes导出. 可以使用命令: `swift export --model_type xxx --to_ollama true`或者`swift export --model_type xxx --quant_method bnb --quant_bits 4`.
- 🔥2024.06.29: 支持[eval-scope](https://github.com/modelscope/eval-scope)&[open-compass](https://github.com/open-compass/opencompass)评测! 我们支持了包含`BoolQ, ocnli, humaneval, math, ceval, mmlu, gsk8k, ARC_e`等50+标准数据集在内的评测流程。
- 🔥2024.06.07: 支持**Qwen2**系列LLM, 包括0.5B、1.5B、7B、72B的Base和Instruct模型, 以及对应的gptq-int4、gptq-int8、awq-int4量化版本. 使用双卡80GiB A100对Qwen2-72B-Instruct进行自我认知微调并推理部署的最佳实践可以查看[这里](https://github.com/modelscope/swift/issues/1092).
</details>

## 🛠️ 安装


## 🚀 快速开始

本章节介绍基本使用，更丰富的使用方式请查看[文档部分](https://swift.readthedocs.io/zh-cn/latest/)。

### Web-UI

Web-UI是基于gradio界面技术的**零门槛**训练部署界面方案。Web-UI配置简单，且完美支持多卡训练和部署：

```shell
swift web-ui
```
![image.png](./docs/resources/web-ui.png)

### 训练



## 🏛 License

本框架使用[Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE)进行许可。模型和数据集请查看原资源页面并遵守对应License。

## 📎 引用

```bibtex
@misc{zhao2024swiftascalablelightweightinfrastructure,
      title={SWIFT:A Scalable lightWeight Infrastructure for Fine-Tuning},
      author={Yuze Zhao and Jintao Huang and Jinghan Hu and Xingjun Wang and Yunlin Mao and Daoze Zhang and Zeyinzi Jiang and Zhikai Wu and Baole Ai and Ang Wang and Wenmeng Zhou and Yingda Chen},
      year={2024},
      eprint={2408.05517},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.05517},
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=modelscope/swift&type=Date)](https://star-history.com/#modelscope/ms-swift&Date)
