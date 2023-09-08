<h1>SWIFT(Scalable lightWeight Infrastructure for Fine-Tuning)</h1>

<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
<p>

<p align="center">
<a href="https://modelscope.cn/home">Modelscope Hub</a>
<br>
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish
</p>

# Introduction

SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning) is an extensible framwork designed to faciliate lightweight model fine-tuning. It integrates implementations for various efficient fine-tuning methods,  by embracing approaches that is parameter-efficient, memory-efficient, and time-efficient. SWIFT integrates seamlessly into ModelScope ecosystem and offers the capabilities to finetune various modles, with a primary emphasis on LLMs and vision models. Additionally, SWIFT is fully compatible with [Peft](https://github.com/huggingface/peft), enabling users to  leverage the familiar Peft interface to finetune ModelScope models.

Currently supported approches (and counting):

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685)
2. Adapter: [Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)
3. Prompt Tuning: [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)
4. All tuners offered on [Peft](https://github.com/huggingface/peft).

Key features:

1. By integrating the ModelScope library, models can be readily obatined via a model-id.
2. Tuners provided by SWIFT be combined together to allow exploration of multiple tuners on a model for best result.

## LLM SFT Example
[code link](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm)

1. supported SFT methods: [lora](https://arxiv.org/abs/2106.09685), [qlora](https://arxiv.org/abs/2305.14314), full(full parameter fine-tuning)
2. supported models: qwen-7b, [qwen-7b-chat](https://github.com/QwenLM/Qwen-7B), qwen-vl, [qwen-vl-chat](https://github.com/QwenLM/Qwen-VL), baichuan-7b, baichuan-13b, baichuan-13b-chat, chatglm2-6b, chatglm2-6b-32k, llama2-7b, llama2-7b-chat, llama2-13b, llama2-13b-chat, llama2-70b, llama2-70b-chat, openbuddy-llama2-13b, openbuddy-llama-65b, openbuddy-llama2-70b, polylm-13b, baichuan2-7b, baichuan2-7b-chat, baichuan2-13b, baichuan2-13b-chat, seqgpt-560m
3. supported features: quantization, ddp, model parallelism(device map), gradient checkpointing, gradient accumulation, pushing to modelscope hub, custom datasets, multimodal and agent SFT, mutli-round chat, ...
4. supported datasets:
   1. NLP: alpaca-en(gpt4), alpaca-zh(gpt4), finance-en, multi-alpaca-all, code-en, instinwild-en, instinwild-zh, cot-en, cot-zh, firefly-all-zh, poetry-zh, instruct-en, gpt4all-en, cmnli-zh
   2. agent: [damo-agent-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary), damo-agent-mini-zh
   3. multi-modal: coco-en
5. supported templates: chatml(qwen), baichuan, chatglm2, llama, openbuddy-llama, default, default-generation

# Installation

SWIFT is running in Python environment. Please make sure your python version is higher than 3.8.

Please install SWIFT by the `pip` command:

```shell
pip install ms-swift -U
```

If you want to install SWIFT by source code, please run:

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .
```

If you are using source code, please remember install requirements by:
```shell
pip install -r requirements/framework.txt
```

SWIFT requires torch>=1.13.

We also recommend to use SWIFT in our docker image:
```shell
docker pull registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.7.1-py38-torch2.0.1-tf1.15.5-1.8.0
```

# Getting Started

SWIFT supports multiple tuners, as well as tuners provided by [Peft](https://github.com/huggingface/peft). To use the these tuners, simply call:

```python
from swift import Swift
model = Swift.prepare_model(model, config, extra_state_keys=['...'])
```

The code snippet above initialized the tuner randomly. The input model is an instance of `torch.nn.Module`, config is a subclass instance of `SwiftConfig` or `PeftConfig`. extra_state_keys is
the extra module weights(like the linear head) to be trained and stored in the output dir.

You may combine multiple tuners by:

```python
from swift import Swift, LoRAConfig, PromptConfig
model = Swift.prepare_model(model, {'lora': LoRAConfig(...), 'prompt': PromptConfig(...)})
```

You can all `save_pretrained` and `push_to_hub` after finetuning:

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

or:

```python
from swift import LoraConfig, get_peft_model, PeftModel
from peft import TaskType
lora_config = LoraConfig(target_modules=['query', 'key', 'value'], task_type=TaskType.CAUSAL_LM)
model_wrapped = get_peft_model(model, lora_config)

# or call from_pretrained to load weights in the modelhub
model_wrapped = PeftModel.from_pretrained(model, 'some-id-in-the-modelscope-modelhub')
```


The saving strategy between Swift tuners and Peft tuners are slightly different. You can name a tuner of a SWIFT by:

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

The config/weights stored in the output dir is the config of `extra_state_keys` and the weights of it. This is different from Peft, which stores the weights and config of the `default` tuner.


# Learn More

- [ModelScope library](https://github.com/modelscope/modelscope/)

  ModelScope Library is the model library of ModelScope project, which contains a large number of popular models.

- [Contribute your own model to ModelScope](https://modelscope.cn/docs/ModelScope%E6%A8%A1%E5%9E%8B%E6%8E%A5%E5%85%A5%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88)

# License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).
