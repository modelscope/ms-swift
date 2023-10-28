<h1>SWIFT(Scalable lightWeight Infrastructure for Fine-Tuning)</h1>

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

# Introduction

SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning) is an extensible framwork designed to faciliate lightweight model fine-tuning and inference. It integrates implementations for various efficient fine-tuning methods,  by embracing approaches that is parameter-efficient, memory-efficient, and time-efficient. SWIFT integrates seamlessly into ModelScope ecosystem and offers the capabilities to finetune various models, with a primary emphasis on LLMs and vision models. Additionally, SWIFT is fully compatible with [PEFT](https://github.com/huggingface/peft), enabling users to  leverage the familiar Peft interface to finetune ModelScope models.

Currently supported approches (and counting):

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/abs/2106.09685)
2. Adapter: [Parameter-Efficient Transfer Learning for NLP](http://arxiv.org/abs/1902.00751)
3. Prompt Tuning: [Visual Prompt Tuning](https://arxiv.org/abs/2203.12119)
4. Side: [Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks](https://arxiv.org/abs/1912.13503)
5. ResTuning-Bypass
7. All tuners offered on [PEFT](https://github.com/huggingface/peft)

Key features:

1. By integrating the ModelScope library, models can be readily obatined via a model-id.
2. Tuners provided by SWIFT can be combined together to allow exploration of multiple tuners on a model for best result.
3. Support calling `activate_adapter` or `deactivate_adapter` or `set_active_adapters`  to activate/deactivate tuners. User can inference with one model and multiple tuners in different threads independently.

Users can check the [documentation of Swift](docs/source/GetStarted/Introduction.md) to get detail tutorials.

## LLM SFT Example
Press [this link](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm) to view the detail documentation of these examples.

### Basic Usage
```bash
git clone https://github.com/modelscope/swift.git
cd swift
pip install .[llm]
```

```python
# Experimental environment: A10, 3090, A100, ...
# 16GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

from swift.llm import DatasetName, InferArguments, ModelType, SftArguments
from swift.llm.run import infer_main, sft_main

model_type = ModelType.qwen_7b_chat_int4
sft_args = SftArguments(
    model_type=model_type,
    eval_steps=50,
    train_dataset_sample=2000,
    dataset=[DatasetName.leetcode_python_en],
    output_dir='output',
    gradient_checkpointing=True)
best_ckpt_dir = sft_main(sft_args)
print(f'best_ckpt_dir: {best_ckpt_dir}')
torch.cuda.empty_cache()
infer_args = InferArguments(
    model_type=sft_args.model_type,
    ckpt_dir=best_ckpt_dir,
    dataset=sft_args.dataset,
    stream=True,
    show_dataset_sample=5)
infer_main(infer_args)
```


### Features
- Supported SFT Methods: [lora](https://arxiv.org/abs/2106.09685), [qlora](https://arxiv.org/abs/2305.14314), full(full parameter fine-tuning)
- Supported Features: quantization, DDP, model parallelism, gradient checkpointing, pushing to modelscope hub, custom datasets, multimodal and agent SFT, mutli-round chat, ...
- Supported Models:
  - 🔥 qwen series: [qwen-7b](https://modelscope.cn/models/qwen/Qwen-7B/summary), [qwen-7b-chat](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary), [qwen-14b](https://modelscope.cn/models/qwen/Qwen-14B/summary), [qwen-14b-chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary), [qwen-7b-chat-int4](https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int4/summary), [qwen-14b-chat-int4](https://modelscope.cn/models/qwen/Qwen-14B-Chat-Int4/summary), [qwen-7b-chat-int8](https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int8/summary), [qwen-14b-chat-int8](https://modelscope.cn/models/qwen/Qwen-14B-Chat-Int8/summary)
  - 🔥 qwen-vl series: [qwen-vl](https://modelscope.cn/models/qwen/Qwen-VL/summary), [qwen-vl-chat](https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary), [qwen-vl-chat-int4](https://modelscope.cn/models/qwen/Qwen-VL-Chat-Int4/summary)
  - baichuan series: [baichuan-7b](https://modelscope.cn/models/baichuan-inc/baichuan-7B/summary), [baichuan-13b](https://modelscope.cn/models/baichuan-inc/Baichuan-13B-Base/summary), [baichuan-13b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan-13B-Chat/summary), [baichuan2-7b](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Base/summary), [baichuan2-7b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat/summary), [baichuan2-13b](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Base/summary), [baichuan2-13b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat/summary), [baichuan2-7b-chat-int4](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat-4bits/summary), [baichuan2-13b-chat-int4](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat-4bits/summary)
  - chatglm series: [chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary), [chatglm2-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm2-6b-32k/summary), [chatglm3-6b-base](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-base/summary), [chatglm3-6b](https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary), [chatglm3-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm3-6b-32k/summary)
  - llama series: [llama2-7b](https://modelscope.cn/models/modelscope/Llama-2-7b-ms/summary), [llama2-7b-chat](https://modelscope.cn/models/modelscope/Llama-2-7b-chat-ms/summary), [llama2-13b](https://modelscope.cn/models/modelscope/Llama-2-13b-ms/summary), [llama2-13b-chat](https://modelscope.cn/models/modelscope/Llama-2-13b-chat-ms/summary), [llama2-70b](https://modelscope.cn/models/modelscope/Llama-2-70b-ms/summary), [llama2-70b-chat](https://modelscope.cn/models/modelscope/Llama-2-70b-chat-ms/summary)
  - openbuddy series: [openbuddy-llama2-13b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama2-13b-v8.1-fp16/summary), [openbuddy-llama-65b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama-65b-v8-bf16/summary), [openbuddy-llama2-70b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama2-70b-v10.1-bf16/summary), [openbuddy-mistral-7b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-mistral-7b-v13.1/summary)
  - internlm series: [internlm-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-7b/summary), [internlm-7b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-v1_1/summary), [internlm-7b-chat-8k](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-8k/summary), [internlm-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-20b/summary), [internlm-20b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-20b/summary)
  - xverse series: [xverse-7b](https://modelscope.cn/models/xverse/XVERSE-7B/summary), [xverse-7b-chat](https://modelscope.cn/models/xverse/XVERSE-7B-Chat/summary), [xverse-13b](https://modelscope.cn/models/xverse/XVERSE-13B/summary), [xverse-13b-chat](https://modelscope.cn/models/xverse/XVERSE-13B-Chat/summary)
  - mistral series: [mistral-7b](https://modelscope.cn/models/AI-ModelScope/Mistral-7B-v0.1/summary), [mistral-7b-chat](https://modelscope.cn/models/AI-ModelScope/Mistral-7B-Instruct-v0.1/summary)
  - ziya series: [ziya2-13b](https://modelscope.cn/models/Fengshenbang/Ziya2-13B-Base/summary), [ziya2-13b-chat](https://modelscope.cn/models/Fengshenbang/Ziya2-13B-Chat/summary)
  - other: [polylm-13b](https://modelscope.cn/models/damo/nlp_polylm_13b_text_generation/summary), [seqgpt-560m](https://modelscope.cn/models/damo/nlp_seqgpt-560m/summary)
- Supported Datasets:
  - NLP:
    - General: 🔥[alpaca-en](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-en/summary)(gpt4), 🔥[alpaca-zh](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-zh/summary)(gpt4), [multi-alpaca-all](https://www.modelscope.cn/datasets/damo/nlp_polylm_multialpaca_sft/summary), [instinwild-en](https://www.modelscope.cn/datasets/wyj123456/instinwild/summary), [instinwild-zh](https://www.modelscope.cn/datasets/wyj123456/instinwild/summary), [cot-en](https://www.modelscope.cn/datasets/YorickHe/CoT/summary), [cot-zh](https://www.modelscope.cn/datasets/YorickHe/CoT/summary), [firefly-all-zh](https://www.modelscope.cn/datasets/wyj123456/firefly/summary), [instruct-en](https://www.modelscope.cn/datasets/wyj123456/instruct/summary), [gpt4all-en](https://www.modelscope.cn/datasets/wyj123456/GPT4all/summary), [sharegpt-en](https://www.modelscope.cn/datasets/huangjintao/sharegpt/summary), [sharegpt-zh](https://www.modelscope.cn/datasets/huangjintao/sharegpt/summary)
    - Agent: [damo-agent-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary), 🔥[damo-agent-mini-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary)
    - Coding: [code-alpaca-en](https://www.modelscope.cn/datasets/wyj123456/code_alpaca_en/summary), [code-python-zh](https://modelscope.cn/datasets/codefuse-ai/CodeExercise-Python-27k/summary), 🔥[leetcode-python-en](https://modelscope.cn/datasets/AI-ModelScope/leetcode-solutions-python/summary)
    - Medical: [medical-en](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary), [medical-zh](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary), [medical-mini-zh](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary)
    - Law: 🔥[lawyer-llama-zh](https://modelscope.cn/datasets/AI-ModelScope/lawyer_llama_data/summary), [tigerbot-law-zh](https://modelscope.cn/datasets/AI-ModelScope/tigerbot-law-plugin/summary)
    - Math: 🔥[blossom-math-zh](https://modelscope.cn/datasets/AI-ModelScope/blossom-math-v2/summary), [school-math-zh](https://modelscope.cn/datasets/AI-ModelScope/school_math_0.25M/summary)
    - SQL: [text2sql-en](https://modelscope.cn/datasets/AI-ModelScope/texttosqlv2_25000_v2/summary), 🔥[sql-create-context-en](https://modelscope.cn/datasets/AI-ModelScope/sql-create-context/summary)
    - Text Generation: 🔥[advertise-gen-zh](https://modelscope.cn/datasets/lvjianjin/AdvertiseGen/summary), 🔥[dureader-robust-zh](https://modelscope.cn/datasets/modelscope/DuReader_robust-QG/summary)
    - Classification: [cmnli-zh](https://www.modelscope.cn/datasets/modelscope/clue/summary), [jd-sentiment-zh](https://modelscope.cn/datasets/DAMO_NLP/jd/summary)
    - Other: [finance-en](https://www.modelscope.cn/datasets/wyj123456/finance_en/summary), [poetry-zh](https://www.modelscope.cn/datasets/modelscope/chinese-poetry-collection/summary), [cls-fudan-news-zh](https://modelscope.cn/datasets/damo/zh_cls_fudan-news/summary), [ner-jave-zh](https://modelscope.cn/datasets/damo/zh_ner-JAVE/summary)
  - Multi-Modal: 🔥[coco-en](https://modelscope.cn/datasets/modelscope/coco_2014_caption/summary)
  - Custom Dataset
- Supported Templates:
  - Text Generation: default-generation, chatglm-generation
  - Chat: chatml(qwen), baichuan, chatglm2, chatglm3, llama, openbuddy-llama, default, internlm, xverse


### News
- 🔥 2023.10.27: Support for chatglm3 series models: chatglm3-6b-base, chatglm3-6b, chatglm3-6b-32k. The corresponding shell script can be found in `scripts/chatglm3_6b_32k`.
- 🔥 2023.10.24: Use the registration mechanism to add models, datasets, and chat templates. To customize models, datasets, and chat templates, refer to the "User Guide" section. The corresponding Python file can be found in `custom.py`, and the corresponding shell script can be found in `scripts/custom/tigerbot_13b_chat`.
- 🔥 2023.10.17: Supported int4, int8 models: qwen-7b-chat-int4, qwen-14b-chat-int4, qwen-vl-chat-int4, baichuan2-7b-chat-int4, baichuan2-13b-chat-int4, qwen-7b-chat-int8, qwen-14b-chat-int8. The corresponding shell script can be found at `scripts/qwen_7b_chat_int4`, `scripts/qwen_14b_chat_int4`, `scripts/qwen_vl_chat_int4`, `scripts/qwen_7b_chat_int8`, `scripts/qwen_14b_chat_int8`.
- 2023.10.15: Supported ziya2-13b model series: ziya2-13b, ziya2-13b-chat. The corresponding shell script can be found at `scripts/ziya2_13b_chat`.
- 2023.10.12: Supported mistral-7b model series: openbuddy-mistral-7b-chat, mistral-7b, mistral-7b-chat. The corresponding shell script can be found at `scripts/openbuddy_mistral_7b_chat`, `scripts/mistral_7b_chat`.
- 🔥 2023.10.7: Supported DeepSpeed ZeRO-2, enabling LoRA (not just QLoRA) to run DDP on 2*A10. The corresponding shell script can be found at `scripts/qwen_7b_chat/lora_ddp_ds/sft.sh`.
- 2023.10.4: Supported datasets in the fields of mathematics, law, SQL, and coding: blossom-math-zh, school-math-zh, text2sql-en, sql-create-context-en, lawyer-llama-zh, tigerbot-law-zh, leetcode-python-en.
- 🔥 2023.9.25: Supported qwen-14b model series: qwen-14b, qwen-14b-chat. The corresponding shell script can be found at `scripts/qwen_14b`, `scripts/qwen_14b_chat`.
- 2023.9.18: Supported internlm-20b model series: internlm-20b, internlm-20b-chat. The corresponding shell script can be found at `scripts/internlm_20b`, `scripts/internlm_20b_chat`.
- 2023.9.12: Supported training with MP+DDP to accelerate full-parameter fine-tuning speed. The corresponding shell script can be found at `scripts/qwen_7b_chat/full_mp_ddp/sft.sh`.


# Installation

SWIFT is running in Python environment. Please make sure your python version is higher than 3.8.

- Install SWIFT by the `pip` command:

```shell
pip install ms-swift -U
```

- Install SWIFT by source code(for running sft/infer examples), please run:

```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .
```

SWIFT requires torch>=1.13.

- Use SWIFT in our docker image:

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.8.0-py38-torch2.0.1-tf2.13.0-1.9.1
```

# Getting Started

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


# Learn More

- [ModelScope library](https://github.com/modelscope/modelscope/)

  ModelScope Library is the model library of ModelScope project, which contains a large number of popular models.

- [Contribute your own model to ModelScope](https://modelscope.cn/docs/ModelScope%E6%A8%A1%E5%9E%8B%E6%8E%A5%E5%85%A5%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88)

# License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).
