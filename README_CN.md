<h1>SWIFT(Scalable lightWeight Infrastructure for Fine-Tuning)</h1>

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>

<p align="center">
<a href="https://modelscope.cn/home">魔搭社区</a>
<br>
        中文&nbsp ｜ &nbsp<a href="README.md">English</a>
</p>

# 简介
SWIFT（Scalable lightWeight Infrastructure for Fine-Tuning）是一个可扩展的框架，旨在促进轻量级模型的微调。它集成了各种高效的微调方法的实现，采用了参数高效、内存高效和时间高效的方法。SWIFT可以无缝地集成到ModelScope生态系统中，并提供微调各种模型的能力，主要侧重于LLMs和视觉模型。此外，SWIFT与[Peft](https://github.com/huggingface/peft)完全兼容，使用户能够利用熟悉的Peft接口对ModelScope模型进行微调。

目前支持的方法（数量持续增加）：

1. LoRA：[LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685)
2. Adapter：[Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)
3. Prompt Tuning: [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)
4. Side: [Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks](https://arxiv.org/abs/1912.13503)
5. ResTuning-Bypass
6. 所有在[Peft](https://github.com/huggingface/peft)上提供的tuners

关键特点：
1. 通过集成ModelScope库，可以通过model id轻松获取模型。
2. SWIFT提供的tuners可以组合在一起，以便在模型上探索多个tuners，以获得最佳结果。
3. 支持调用`activate_adapter`或`deactivate_adapter`来使tuner激活或失活，用户可以在推理时用一个模型在不同线程中使用多种tuners而互不干扰。

用户可以查看 [Swift官方文档](./docs/Get Started/1.Introduction.md) 来了解详细信息。

## 大模型微调的例子
[code link](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm)

1. 支持的SFT方法: [lora](https://arxiv.org/abs/2106.09685), [qlora](https://arxiv.org/abs/2305.14314), 全参数微调
2. 支持的模型:
   1. qwen 系列: qwen-7b, [qwen-7b-chat](https://github.com/QwenLM/Qwen-7B)
   2. qwen-vl 系列: qwen-vl, [qwen-vl-chat](https://github.com/QwenLM/Qwen-VL)
   3. baichuan 系列: baichuan-7b, baichuan-13b, baichuan-13b-chat, baichuan2-7b, baichuan2-7b-chat, baichuan2-13b, baichuan2-13b-chat
   4. chatglm2 系列: chatglm2-6b, chatglm2-6b-32k
   5. llama 系列: llama2-7b, llama2-7b-chat, llama2-13b, llama2-13b-chat, llama2-70b, llama2-70b-chat
   6. openbuddy-llama 系列: openbuddy-llama2-13b, openbuddy-llama-65b, openbuddy-llama2-70b
   7. internlm 系列: internlm-7b, internlm-7b-chat, internlm-7b-chat-8k
   8. other: polylm-13b, seqgpt-560m
3. 支持的特性: 模型量化, DDP, 模型并行(device_map), gradient checkpointing, 梯度累加, 支持推送ModelScope Hub, 自定义数据集, 多模态和Agent SFT, 多轮对话, ...
4. 支持的数据集:
   1. NLP: alpaca-en(gpt4), alpaca-zh(gpt4), finance-en, multi-alpaca-all, code-en, instinwild-en, instinwild-zh, cot-en, cot-zh, firefly-all-zh, poetry-zh, instruct-en, gpt4all-en, cmnli-zh
   2. agent: [damo-agent-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary), damo-agent-mini-zh
   3. 多模态: coco-en
5. 支持的对话模板: chatml(qwen), baichuan, chatglm2, llama, openbuddy-llama, default, default-generation

# 安装

SWIFT在Python环境中运行。请确保您的Python版本高于3.8。

请使用pip命令安装SWIFT：

```shell
pip install ms-swift -U
```

如果您想通过源代码安装SWIFT，请运行以下命令：

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .
```

如果您在使用源代码，请记得通过以下方式安装所需的依赖项：
```shell
pip install -r requirements/framework.txt
```

SWIFT requires torch>=1.13.

我们还建议在我们的Docker镜像中使用SWIFT
```shell
docker pull registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0
```

# 快速开始
SWIFT支持多个tuners，包括由[Peft](https://github.com/huggingface/peft)提供的调谐器。要使用这些调谐器，只需调用:
```python
from swift import Swift
model = Swift.prepare_model(model, config, extra_state_keys=['...'])
```
上面的代码片段随机初始化了tuner。输入模型是torch.nn.Module的一个实例，配置是SwiftConfig或PeftConfig的子类实例。extra_state_keys是要训练并存储在输出目录中的额外模块权重（如linear head）。

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

使用model-id进行后续推断：

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

这是一个使用transformers库创建模型，并使用SWIFT进行高效微调的示例。

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

您可以在SWIFT中使用Peft提供的功能：

```python
from swift import LoraConfig, Swift
from peft import TaskType
lora_config = LoraConfig(target_modules=['query', 'key', 'value'], task_type=TaskType.CAUSAL_LM)
model_wrapped = Swift.prepare_model(model, lora_config)

# 或者使用from_pretrained从modelscope hub中加载权重。
model_wrapped = Swift.from_pretrained(model, 'some-id-in-the-modelscope-modelhub')
```

或者：

```python
from swift import LoraConfig, get_peft_model, PeftModel
from peft import TaskType
lora_config = LoraConfig(target_modules=['query', 'key', 'value'], task_type=TaskType.CAUSAL_LM)
model_wrapped = get_peft_model(model, lora_config)

# 或者使用from_pretrained从modelscope hub中加载权重。
model_wrapped = PeftModel.from_pretrained(model, 'some-id-in-the-modelscope-modelhub')
```

Swift tuners和Peft tuners之间的保存策略略有不同。您可以通过以下方式为Swift tuners命名：

```python
model = Swift.prepare_model(model, {'default': LoRAConfig(...)})
model.save_pretrained('./output')
```

在输出目录中，您将会得到以下类似的目录结构：

```text
output
    |-- default
        |-- adapter_config.json
        |-- adapter_model.bin
    |-- adapter_config.json
    |-- adapter_model.bin
```

存储在输出目录中的config/weights是extra_state_keys的配置和权重。这与Peft不同，Peft存储了default调谐器的权重和配置。


# Learn More

- [ModelScope库](https://github.com/modelscope/modelscope/)

  ModelScope库是ModelScope项目的模型库，包含大量热门模型。

- [将自己的模型贡献给ModelScope](https://modelscope.cn/docs/ModelScope%E6%A8%A1%E5%9E%8B%E6%8E%A5%E5%85%A5%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88)

# License

本项目使用[Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE)进行许可。
