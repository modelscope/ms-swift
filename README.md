# SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning)

<p align="center">
    <br>
    <img src="asset/banner.png"/>
    <br>
<p>
<p align="center">
<a href="https://modelscope.cn/home">ModelScope Community Website</a>
<br>
        <a href="README_CN.md">中文</a> &nbsp ｜ &nbsp English &nbsp
</p>

<p align="center">
<img src="https://img.shields.io/badge/python-3.10-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.19-5D91D4.svg"></a>
<a href="https://pypi.org/project/ms-swift/"><img src="https://badge.fury.io/py/ms-swift.svg"></a>
<a href="https://github.com/modelscope/ms-swift/blob/main/LICENSE"><img src="https://img.shields.io/github/license/modelscope/ms-swift"></a>
<a href="https://pepy.tech/project/ms-swift"><img src="https://pepy.tech/badge/ms-swift"></a>
<a href="https://github.com/modelscope/ms-swift/pulls"><img src="https://img.shields.io/badge/PR-welcome-55EB99.svg"></a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/6427" target="_blank"><img src="https://trendshift.io/api/badge/repositories/6427" alt="modelscope%2Fswift | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
        <a href="https://arxiv.org/abs/2408.05517">Paper</a> &nbsp ｜ <a href="https://swift.readthedocs.io/en/latest/">English Documentation</a> &nbsp ｜ &nbsp <a href="https://swift.readthedocs.io/zh-cn/latest/">中文文档</a> &nbsp
</p>

## 📖 Table of Contents
- [Groups](#-Groups)
- [Introduction](#-introduction)
- [News](#-news)
- [Installation](#%EF%B8%8F-installation)
- [Quick Start](#-quick-Start)
- [Usage](#-Usage)
- [License](#-License)
- [Citation](#-citation)


## ☎ Groups

You can contact us and communicate with us by adding our group:


[Discord Group](https://discord.com/invite/D27yfEFVz5)              |  WeChat Group
:-------------------------:|:-------------------------:
<img src="asset/discord_qr.jpg" width="200" height="200">  |  <img src="asset/wechat.png" width="200" height="200">


## 📝 Introduction
🍲 ms-swift is an official framework provided by the ModelScope community for fine-tuning and deploying large language models and multi-modal large models. It currently supports the training (pre-training, fine-tuning, human alignment), inference, evaluation, quantization, and deployment of 500+ large models and 200+ multi-modal large models. These large language models (LLMs) include models such as Qwen3, Qwen3-MoE, Qwen2.5, InternLM3, GLM4, Mistral, DeepSeek-R1, Yi1.5, TeleChat2, Baichuan2, and Gemma2. The multi-modal LLMs include models such as Qwen2.5-VL, Qwen2-Audio, Llama4, Llava, InternVL3, MiniCPM-V-2.6, GLM4v, Xcomposer2.5, Yi-VL, DeepSeek-VL2, Phi3.5-Vision, and GOT-OCR2.

🍔 Additionally, ms-swift incorporates the latest training technologies, including lightweight techniques such as LoRA, QLoRA, Llama-Pro, LongLoRA, GaLore, Q-GaLore, LoRA+, LISA, DoRA, FourierFt, ReFT, UnSloth, and Liger, as well as human alignment training methods like DPO, GRPO, RM, PPO, GKD, KTO, CPO, SimPO, and ORPO. ms-swift supports acceleration of inference, evaluation, and deployment modules using vLLM, SGLang and LMDeploy, and it supports model quantization with technologies like GPTQ, AWQ, and BNB. Furthermore, ms-swift offers a Gradio-based Web UI and a wealth of best practices.

**Why choose ms-swift?**

- 🍎 **Model Types**: Supports 500+ pure text large models, **200+ multi-modal large models**, as well as All-to-All multi-modal models, sequence classification models, and embedding models, **covering the entire process from training to deployment**.
- **Dataset Types**: Comes with 150+ pre-training, fine-tuning, human alignment, multi-modal datasets, and supports custom datasets.
- **Hardware Support**: Compatible with CPU, RTX series, T4/V100, A10/A100/H100, Ascend NPU, MPS, etc.
- **Lightweight Training**: Supports lightweight fine-tuning methods like LoRA, QLoRA, DoRA, LoRA+, ReFT, RS-LoRA, LLaMAPro, Adapter, GaLore, Q-Galore, LISA, UnSloth, Liger-Kernel.
- **Distributed Training**: Supports distributed data parallel (DDP), device_map simple model parallelism, DeepSpeed ZeRO2/ZeRO3, FSDP, Megatron, and other distributed training techniques.
- **Quantization Training**: Supports training quantized models like BNB, AWQ, GPTQ, AQLM, HQQ, EETQ.
- 🍊 **RLHF Training**: Supports human alignment training methods such as DPO, GRPO, RM, PPO, GKD, KTO, CPO, SimPO, ORPO for both pure text and multi-modal large models.
- 🍓 **Multi-Modal Training**: Supports training on different modalities like images, videos, and audio, for tasks like VQA, captioning, OCR, and grounding.
- 🥥 **Megatron Parallelism**: Supports accelerating CPT/SFT/DPO using Megatron parallelism techniques, currently compatible with 200+ large language models.
- **Interface Training**: Provides capabilities for training, inference, evaluation, quantization through an interface, completing the whole large model pipeline.
- **Plugin and Extension**: Supports custom model and dataset extensions, as well as customization of components like loss, metric, trainer, loss-scale, callback, optimizer.
- 🍉 **Toolbox Capabilities**: Offers not only training support for large models and multi-modal large models but also covers the entire process of inference, evaluation, quantization, and deployment.
- **Inference Acceleration**: Supports inference acceleration engines like PyTorch, vLLM, SGLang, LmDeploy, and provides OpenAI API for accelerating inference, deployment, and evaluation modules.
- **Model Evaluation**: Uses EvalScope as the evaluation backend and supports evaluation on 100+ datasets for both pure text and multi-modal models.
- **Model Quantization**: Supports AWQ, GPTQ, and BNB quantized exports, with models that can use vLLM/SGLang/LmDeploy for inference acceleration and continue training.


## 🎉 News
- 🎁 2025.06.23: Fine-tuning of reranker models is supported. Training scripts can be found here: [Reranker](https://github.com/modelscope/ms-swift/blob/main/examples/train/reranker/train_reranker.sh).
- 🎁 2025.06.18: Support for accelerating the ms-swift [inference](https://github.com/modelscope/ms-swift/blob/main/examples/infer/sglang), deployment, evaluation, and UI modules using the [sglang](https://github.com/sgl-project/sglang) inference acceleration engine. Simply set `--infer_backend sglang` to enable it.
- 🎁 2025.06.15: Support for GKD training on both pure text large models and multimodal models. Training scripts can be found here: [Pure Text](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/gkd), [Multimodal](https://github.com/modelscope/ms-swift/blob/main/examples/train/multimodal/rlhf/gkd).
- 🎁 2025.06.11: Support for using Megatron parallelism techniques for RLHF training. The training script can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/megatron/rlhf).
- 🎁 2025.05.29: Support sequence parallel in pt, sft, dpo and grpo, check script [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/long_text).
- 🎁 2025.05.11: GRPO now supports custom processing logic for reward models. See the GenRM example [here](./docs/source_en/Instruction/GRPO/DeveloperGuide/reward_model.md).
- 🎁 2025.04.15: The ms-swift paper has been accepted by AAAI 2025. You can find the paper at [this link](https://ojs.aaai.org/index.php/AAAI/article/view/35383).
- 🎁 2025.03.23: Multi-round GRPO is now supported for training multi-turn dialogue scenarios (e.g., agent tool calling). Please refer to the [doc](./docs/source_en/Instruction/GRPO/DeveloperGuide/multi_turn.md).
- 🎁 2025.03.16: Support for Megatron's parallel training techniques is now available. Please see the [Megatron-SWIFT training documentation](https://swift.readthedocs.io/en/latest/Instruction/Megatron-SWIFT-Training.html).
- 🎁 2025.03.15: Fine-tuning of embedding models for both pure text and multimodal models is supported. Please check the [training script](examples/train/embedding).
- 🎁 2025.03.05: The hybrid mode for GRPO is supported, with a script for training a 72B model on 4 GPUs (4*80G) available [here](examples/train/grpo/internal/vllm_72b_4gpu.sh). Tensor parallelism with vllm is also supported, with the training script available [here](examples/train/grpo/internal).
- 🎁 2025.02.21: The GRPO algorithm now supports LMDeploy, with the training script available [here](examples/train/grpo/internal/full_lmdeploy.sh). Additionally, the performance of the GRPO algorithm has been tested, achieving a training speed increase of up to 300% using various tricks. Please check the WanDB table [here](https://wandb.ai/tastelikefeet/grpo_perf_test?nw=nwuseryuzezyz).
- 🎁 2025.02.21: The `swift sample` command is now supported. The reinforcement fine-tuning script can be found [here](docs/source_en/Instruction/Reinforced-Fine-tuning.md), and the large model API distillation sampling script is available [here](examples/sampler/distill/distill.sh).
- 🔥 2025.02.12: Support for the GRPO (Group Relative Policy Optimization) training algorithm has been added. Documentation is available [here](docs/source_en/Instruction/GRPO/GetStarted/GRPO.md).
- 🎁 2024.12.04: Major update to **ms-swift 3.0**. Please refer to the [release notes and changes](docs/source_en/Instruction/ReleaseNote3.0.md).
<details><summary>More</summary>

- 🎉 2024.08.12: The ms-swift paper has been published on arXiv and can be read [here](https://arxiv.org/abs/2408.05517).
- 🔥 2024.08.05: Support for using [evalscope](https://github.com/modelscope/evalscope/) as a backend for evaluating large models and multimodal models.
- 🔥 2024.07.29: Support for using [vllm](https://github.com/vllm-project/vllm) and [lmdeploy](https://github.com/InternLM/lmdeploy) to accelerate inference for large models and multimodal models. When performing infer/deploy/eval, you can specify `--infer_backend vllm/lmdeploy`.
- 🔥 2024.07.24: Support for human preference alignment training for multimodal large models, including DPO/ORPO/SimPO/CPO/KTO/RM/PPO.
- 🔥 2024.02.01: Support for Agent training! The training algorithm is derived from [this paper](https://arxiv.org/pdf/2309.00986.pdf).
</details>

## 🛠️ Installation
To install using pip:
```shell
pip install ms-swift -U
```

To install from source:
```shell
# pip install git+https://github.com/modelscope/ms-swift.git

git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

Running Environment:

|              | Range        | Recommended | Notes                                     |
| ------------ |--------------| ----------- | ----------------------------------------- |
| python       | >=3.9        | 3.10        |                                           |
| cuda         |              | cuda12      | No need to install if using CPU, NPU, MPS |
| torch        | >=2.0        |             |                                           |
| transformers | >=4.33       | 4.51.3      |                                           |
| modelscope   | >=1.23       |             |                                           |
| peft | >=0.11,<0.16 | ||
| trl | >=0.13,<0.19 | 0.18 |RLHF|
| deepspeed    | >=0.14       | 0.16.9 | Training                                  |
| vllm         | >=0.5.1      | 0.8.5.post1       | Inference/Deployment/Evaluation           |
| sglang |     | 0.4.6.post5 | Inference/Deployment/Evaluation |
| lmdeploy     | >=0.5,<0.9        | 0.8       | Inference/Deployment/Evaluation           |
| evalscope | >=0.11       |  | Evaluation |

For more optional dependencies, you can refer to [here](https://github.com/modelscope/ms-swift/blob/main/requirements/install_all.sh).


## 🚀 Quick Start

10 minutes of self-cognition fine-tuning of Qwen2.5-7B-Instruct on a single 3090 GPU:

### Command Line Interface

```shell
# 22GB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
```

Tips:

- If you want to train with a custom dataset, you can refer to [this guide](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html) to organize your dataset format and specify `--dataset <dataset_path>`.
- The `--model_author` and `--model_name` parameters are only effective when the dataset includes `swift/self-cognition`.
- To train with a different model, simply modify `--model <model_id/model_path>`.
- By default, ModelScope is used for downloading models and datasets. If you want to use HuggingFace, simply specify `--use_hf true`.

After training is complete, use the following command to infer with the trained weights:

- Here, `--adapters` should be replaced with the last checkpoint folder generated during training. Since the adapters folder contains the training parameter file `args.json`, there is no need to specify `--model`, `--system` separately; Swift will automatically read these parameters. To disable this behavior, you can set `--load_args false`.

```shell
# Using an interactive command line for inference.
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048

# merge-lora and use vLLM for inference acceleration
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --merge_lora true \
    --infer_backend vllm \
    --max_model_len 8192 \
    --temperature 0 \
    --max_new_tokens 2048
```

Finally, use the following command to push the model to ModelScope:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<your-model-id>' \
    --hub_token '<your-sdk-token>' \
    --use_hf false
```


### Web-UI
The Web-UI is a **zero-threshold** training and deployment interface solution based on Gradio interface technology. For more details, you can check [here](https://swift.readthedocs.io/en/latest/GetStarted/Web-UI.html).

```shell
SWIFT_UI_LANG=en swift web-ui
```

![image.png](./docs/resources/web-ui-en.jpg)

### Using Python

ms-swift also supports training and inference using Python. Below is pseudocode for training and inference. For more details, you can refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/notebook/qwen2_5-self-cognition/self-cognition-sft.ipynb).

Training:

```python
# Retrieve the model and template, and add a trainable LoRA module
model, tokenizer = get_model_tokenizer(model_id_or_path, ...)
template = get_template(model.model_meta.template, tokenizer, ...)
model = Swift.prepare_model(model, lora_config)

# Download and load the dataset, and encode the text into tokens
train_dataset, val_dataset = load_dataset(dataset_id_or_path, ...)
train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)

# Train the model
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=template.data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    template=template,
)
trainer.train()
```
Inference:

```python
# Perform inference using the native PyTorch engine
engine = PtEngine(model_id_or_path, adapters=[lora_checkpoint])
infer_request = InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}])
request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)

resp_list = engine.infer([infer_request], request_config)
print(f'response: {resp_list[0].choices[0].message.content}')
```

## ✨ Usage
Here is a minimal example of training to deployment using ms-swift. For more details, you can check the [examples](https://github.com/modelscope/ms-swift/tree/main/examples).

- If you want to use other models or datasets (including multimodal models and datasets), you only need to modify `--model` to specify the corresponding model's ID or path, and modify `--dataset` to specify the corresponding dataset's ID or path.
- By default, ModelScope is used for downloading models and datasets. If you want to use HuggingFace, simply specify `--use_hf true`.

|   Useful Links |
| ------ |
|   [🔥Command Line Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html)   |
|   [Supported Models and Datasets](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html)   |
|   [Custom Models](https://swift.readthedocs.io/en/latest/Customization/Custom-model.html), [🔥Custom Datasets](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html)   |
|   [LLM Tutorial](https://github.com/modelscope/modelscope-classroom/tree/main/LLM-tutorial)   |

### Training

Supported Training Methods:

| Method                             | Full-Parameter                                               | LoRA                                                                                        | QLoRA                                                        | Deepspeed                                                    | Multi-Node                                                   | Multi-Modal                                                                                  |
|------------------------------------|--------------------------------------------------------------|---------------------------------------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| Pre-training                       | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/pretrain/train.sh) | ✅                                                                                           | ✅                                                            | ✅                                                            | ✅                                                            | ✅                                                                                            |
| Instruction Supervised Fine-tuning | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/full/train.sh) | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/lora_sft.sh)            | [✅](https://github.com/modelscope/ms-swift/tree/main/examples/train/qlora) | [✅](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-gpu/deepspeed) | [✅](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node)                                                            | [✅](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal)              |
| DPO Training                       | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/dpo)            | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/dpo) | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/multimodal/rlhf/dpo)  |
| GRPO Training                      | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/internal) | ✅                                                                                           | ✅                                                            | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/external)                                    | ✅                                                                                            |
| Reward Model Training              | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/rm.sh)             | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/rm.sh) | ✅                                                            | ✅                                                                                            |
| PPO Training                       | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/ppo)            | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/ppo) | ✅                                                            | ❌                                                                                            |
| GKD Training                       | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/gkd)            | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/gkd) | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/multimodal/rlhf/gkd)  |
| KTO Training                       | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/kto.sh)            | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/kto.sh) | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/multimodal/rlhf/kto.sh)  |
| CPO Training                       | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/cpo.sh)            | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/cpo.sh) | ✅                                                            | ✅                                                                                            |
| SimPO Training                     | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/simpo.sh)          | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/simpo.sh) | ✅                                                            | ✅                                                                                            |
| ORPO Training                      | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/orpo.sh)           | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/orpo.sh) | ✅                                                            | ✅                                                                                            |
| Classification Model Training      | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/seq_cls/qwen2_5/sft.sh) | ✅                                                            | ✅                                                            | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/seq_cls/qwen2_vl/sft.sh) |
| Embedding Model Training           | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/embedding/train_gte.sh) | ✅                                                            | ✅                                                            | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/embedding/train_gme.sh)  |
| Reranker Model Training | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |


Pre-training:
```shell
# 8*A100
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift pt \
    --model Qwen/Qwen2.5-7B \
    --dataset swift/chinese-c4 \
    --streaming true \
    --train_type full \
    --deepspeed zero2 \
    --output_dir output \
    --max_steps 10000 \
    ...
```

Fine-tuning:
```shell
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset AI-ModelScope/alpaca-gpt4-data-en \
    --train_type lora \
    --output_dir output \
    ...
```

RLHF:
```shell
CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
    --train_type lora \
    --output_dir output \
    ...
```


### Inference
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --stream true \
    --infer_backend pt \
    --max_new_tokens 2048

# LoRA
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapters swift/test_lora \
    --stream true \
    --infer_backend pt \
    --temperature 0 \
    --max_new_tokens 2048
```

### Interface Inference
```shell
CUDA_VISIBLE_DEVICES=0 swift app \
    --model Qwen/Qwen2.5-7B-Instruct \
    --stream true \
    --infer_backend pt \
    --max_new_tokens 2048
```

### Deployment
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend vllm
```

### Sampling
```shell
CUDA_VISIBLE_DEVICES=0 swift sample \
    --model LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --sampler_engine pt \
    --num_return_sequences 5 \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh#5
```

### Evaluation
```shell
CUDA_VISIBLE_DEVICES=0 swift eval \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend lmdeploy \
    --eval_backend OpenCompass \
    --eval_dataset ARC_c
```

### Quantization
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --model Qwen/Qwen2.5-7B-Instruct \
    --quant_bits 4 --quant_method awq \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh \
    --output_dir Qwen2.5-7B-Instruct-AWQ
```

### Push Model
```shell
swift export \
    --model <model-path> \
    --push_to_hub true \
    --hub_model_id '<model-id>' \
    --hub_token '<sdk-token>'
```

## 🏛 License

This framework is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE). For models and datasets, please refer to the original resource page and follow the corresponding License.

## 📎 Citation

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
