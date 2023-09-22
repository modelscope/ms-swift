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
SWIFT（Scalable lightWeight Infrastructure for Fine-Tuning）是一个可扩展的轻量级一站式训练、推理深度学习框架。它集成了各种高效的微调方法，如LoRA、QLoRA、阿里云自研的ResTuning-Bypass等，以及开箱即用的训练推理脚本，使开发者可以在单张商业级显卡上微调推理LLM&AIGC模型。此外，SWIFT与[PEFT](https://github.com/huggingface/peft)完全兼容，使开发者可以在ModelScope模型体系中使用PEFT的能力。

目前支持的方法：

1. LoRA：[LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685)
2. Adapter：[Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)
3. Prompt: [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)
4. Side: [Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks](https://arxiv.org/abs/1912.13503)
5. ResTuning-Bypass
6. 所有在[PEFT](https://github.com/huggingface/peft)上提供的tuners

主要能力：
1. 可以通过model-id使SWIFT或PEFT的方法使用ModelScope Hub中的模型
2. 在单次训练或推理中可以使用多个tuners
3. 支持调用`activate_adapter`或`deactivate_adapter`或`set_active_adapters`来使部分tuner激活或失活，用户可以在推理时同时加载多个独立的tuners在不同线程中并行使用。

用户可以查看 [Swift官方文档](docs/source/GetStarted/Introduction.md) 来了解详细信息。

## 大模型微调的例子
[code link](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm)

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
   2. agent: [damo-agent-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary), damo-agent-mini-zh
   3. 多模态: [coco-en](https://modelscope.cn/datasets/modelscope/coco_2014_caption/summary)
   4. 其他: [cls-fudan-news-zh](https://modelscope.cn/datasets/damo/zh_cls_fudan-news/files), [ner-jave-zh](https://modelscope.cn/datasets/damo/zh_ner-JAVE/summary)
5. 支持的对话模板: chatml(qwen), baichuan, chatglm2, llama, openbuddy-llama, default, default-generation

# 安装

SWIFT在Python环境中运行。请确保您的Python版本高于3.8。

- 方法1：使用pip命令安装SWIFT：

```shell
pip install ms-swift -U
```

- 方法2：通过源代码安装SWIFT（方便运行训练推理脚本），请运行以下命令：

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .
```

SWIFT依赖torch>=1.13。

- 方法3：在我们的Docker镜像中使用SWIFT

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.8.0-py38-torch2.0.1-tf2.13.0-1.9.1
```

# 快速开始
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


# Learn More

- [ModelScope库](https://github.com/modelscope/modelscope/)

  ModelScope库是ModelScope项目的模型库，包含了各模态热门的深度学习模型。

- [将自己的模型贡献给ModelScope](https://modelscope.cn/docs/ModelScope%E6%A8%A1%E5%9E%8B%E6%8E%A5%E5%85%A5%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88)

# License

本项目使用[Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE)进行许可。
