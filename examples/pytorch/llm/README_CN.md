<h1 align="center">大模型微调的例子</h1>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.9.2-5D91D4.svg"></a>
<a href="https://github.com/modelscope/swift/"><img src="https://img.shields.io/badge/ms--swift-Build from source-6FEBB9.svg"></a>
</p>


<p align="center">
<a href="https://modelscope.cn/home">魔搭社区</a>
<br>
        中文&nbsp ｜ &nbsp<a href="README.md">English</a>
</p>


## 特性
- 支持的SFT方法: [lora](https://arxiv.org/abs/2106.09685), [qlora](https://arxiv.org/abs/2305.14314), 全参数微调
- 支持的特性: 模型量化, DDP, 模型并行, gradient checkpointing, 梯度累加, 支持推送ModelScope Hub, 自定义数据集, 多模态和Agent SFT, 多轮对话, ...
- 支持的模型
  - 🔥 qwen 系列: [qwen-7b](https://modelscope.cn/models/qwen/Qwen-7B/summary), [qwen-7b-chat](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary), [qwen-14b](https://modelscope.cn/models/qwen/Qwen-14B/summary), [qwen-14b-chat](https://modelscope.cn/models/qwen/Qwen-14B-Chat/summary), [qwen-7b-chat-int4](https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int4/summary), [qwen-14b-chat-int4](https://modelscope.cn/models/qwen/Qwen-14B-Chat-Int4/summary), [qwen-7b-chat-int8](https://modelscope.cn/models/qwen/Qwen-7B-Chat-Int8/summary), [qwen-14b-chat-int8](https://modelscope.cn/models/qwen/Qwen-14B-Chat-Int8/summary)
  - 🔥 qwen-vl 系列: [qwen-vl](https://modelscope.cn/models/qwen/Qwen-VL/summary), [qwen-vl-chat](https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary), [qwen-vl-chat-int4](https://modelscope.cn/models/qwen/Qwen-VL-Chat-Int4/summary)
  - baichuan 系列: [baichuan-7b](https://modelscope.cn/models/baichuan-inc/baichuan-7B/summary), [baichuan-13b](https://modelscope.cn/models/baichuan-inc/Baichuan-13B-Base/summary), [baichuan-13b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan-13B-Chat/summary), [baichuan2-7b](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Base/summary), [baichuan2-7b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat/summary), [baichuan2-13b](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Base/summary), [baichuan2-13b-chat](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat/summary), [baichuan2-7b-chat-int4](https://modelscope.cn/models/baichuan-inc/Baichuan2-7B-Chat-4bits/summary), [baichuan2-13b-chat-int4](https://modelscope.cn/models/baichuan-inc/Baichuan2-13B-Chat-4bits/summary)
  - chatglm2 系列: [chatglm2-6b](https://modelscope.cn/models/ZhipuAI/chatglm2-6b/summary), [chatglm2-6b-32k](https://modelscope.cn/models/ZhipuAI/chatglm2-6b-32k/summary)
  - llama 系列: [llama2-7b](https://modelscope.cn/models/modelscope/Llama-2-7b-ms/summary), [llama2-7b-chat](https://modelscope.cn/models/modelscope/Llama-2-7b-chat-ms/summary), [llama2-13b](https://modelscope.cn/models/modelscope/Llama-2-13b-ms/summary), [llama2-13b-chat](https://modelscope.cn/models/modelscope/Llama-2-13b-chat-ms/summary), [llama2-70b](https://modelscope.cn/models/modelscope/Llama-2-70b-ms/summary), [llama2-70b-chat](https://modelscope.cn/models/modelscope/Llama-2-70b-chat-ms/summary)
  - openbuddy 系列: [openbuddy-llama2-13b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama2-13b-v8.1-fp16/summary), [openbuddy-llama-65b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama-65b-v8-bf16/summary), [openbuddy-llama2-70b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-llama2-70b-v10.1-bf16/summary), [openbuddy-mistral-7b-chat](https://modelscope.cn/models/OpenBuddy/openbuddy-mistral-7b-v13.1/summary)
  - internlm 系列: [internlm-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-7b/summary), [internlm-7b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-v1_1/summary), [internlm-7b-chat-8k](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-7b-8k/summary), [internlm-20b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-20b/summary), [internlm-20b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-chat-20b/summary)
  - xverse 系列: [xverse-7b](https://modelscope.cn/models/xverse/XVERSE-7B/summary), [xverse-7b-chat](https://modelscope.cn/models/xverse/XVERSE-7B-Chat/summary), [xverse-13b](https://modelscope.cn/models/xverse/XVERSE-13B/summary), [xverse-13b-chat](https://modelscope.cn/models/xverse/XVERSE-13B-Chat/summary)
  - mistral 系列: [mistral-7b](https://modelscope.cn/models/AI-ModelScope/Mistral-7B-v0.1/summary), [mistral-7b-chat](https://modelscope.cn/models/AI-ModelScope/Mistral-7B-Instruct-v0.1/summary)
  - ziya 系列: [ziya2-13b](https://modelscope.cn/models/Fengshenbang/Ziya2-13B-Base/summary), [ziya2-13b-chat](https://modelscope.cn/models/Fengshenbang/Ziya2-13B-Chat/summary)
  - other: [polylm-13b](https://modelscope.cn/models/damo/nlp_polylm_13b_text_generation/summary), [seqgpt-560m](https://modelscope.cn/models/damo/nlp_seqgpt-560m/summary)
- 支持的数据集:
  - NLP:
    - 通用: 🔥[alpaca-en](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-en/summary)(gpt4), 🔥[alpaca-zh](https://modelscope.cn/datasets/AI-ModelScope/alpaca-gpt4-data-zh/summary)(gpt4), [multi-alpaca-all](https://www.modelscope.cn/datasets/damo/nlp_polylm_multialpaca_sft/summary), [instinwild-en](https://www.modelscope.cn/datasets/wyj123456/instinwild/summary), [instinwild-zh](https://www.modelscope.cn/datasets/wyj123456/instinwild/summary), [cot-en](https://www.modelscope.cn/datasets/YorickHe/CoT/summary), [cot-zh](https://www.modelscope.cn/datasets/YorickHe/CoT/summary), [firefly-all-zh](https://www.modelscope.cn/datasets/wyj123456/firefly/summary), [instruct-en](https://www.modelscope.cn/datasets/wyj123456/instruct/summary), [gpt4all-en](https://www.modelscope.cn/datasets/wyj123456/GPT4all/summary), [sharegpt-en](https://www.modelscope.cn/datasets/huangjintao/sharegpt/summary), [sharegpt-zh](https://www.modelscope.cn/datasets/huangjintao/sharegpt/summary)
    - Agent: [damo-agent-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary), 🔥[damo-agent-mini-zh](https://modelscope.cn/datasets/damo/MSAgent-Bench/summary)
    - 代码: [code-alpaca-en](https://www.modelscope.cn/datasets/wyj123456/code_alpaca_en/summary), [code-python-zh](https://modelscope.cn/datasets/codefuse-ai/CodeExercise-Python-27k/summary), 🔥[leetcode-python-en](https://modelscope.cn/datasets/AI-ModelScope/leetcode-solutions-python/summary)
    - 医疗: [medical-en](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary), [medical-zh](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary), [medical-mini-zh](https://www.modelscope.cn/datasets/huangjintao/medical_zh/summary)
    - 法律: 🔥[lawyer-llama-zh](https://modelscope.cn/datasets/AI-ModelScope/lawyer_llama_data/summary), [tigerbot-law-zh](https://modelscope.cn/datasets/AI-ModelScope/tigerbot-law-plugin/summary)
    - 数学: 🔥[blossom-math-zh](https://modelscope.cn/datasets/AI-ModelScope/blossom-math-v2/summary), [school-math-zh](https://modelscope.cn/datasets/AI-ModelScope/school_math_0.25M/summary)
    - SQL: [text2sql-en](https://modelscope.cn/datasets/AI-ModelScope/texttosqlv2_25000_v2/summary), 🔥[sql-create-context-en](https://modelscope.cn/datasets/AI-ModelScope/sql-create-context/summary)
    - 文本生成: 🔥[advertise-gen-zh](https://modelscope.cn/datasets/lvjianjin/AdvertiseGen/summary), 🔥[dureader-robust-zh](https://modelscope.cn/datasets/modelscope/DuReader_robust-QG/summary)
    - 分类: [cmnli-zh](https://www.modelscope.cn/datasets/modelscope/clue/summary), [jd-sentiment-zh](https://modelscope.cn/datasets/DAMO_NLP/jd/summary)
    - 其他: [finance-en](https://www.modelscope.cn/datasets/wyj123456/finance_en/summary), [poetry-zh](https://www.modelscope.cn/datasets/modelscope/chinese-poetry-collection/summary), [cls-fudan-news-zh](https://modelscope.cn/datasets/damo/zh_cls_fudan-news/summary), [ner-jave-zh](https://modelscope.cn/datasets/damo/zh_ner-JAVE/summary)
  - 多模态: 🔥[coco-en](https://modelscope.cn/datasets/modelscope/coco_2014_caption/summary)
  - 自定义数据集
- 支持的对话模板:
  - 文本生成: default-generation, chatglm2-generation
  - 对话: chatml(qwen), baichuan, chatglm2, llama, openbuddy-llama, default, internlm, xverse


## 新闻
- 🔥 2023.10.24: 使用注册机制来新增模型, 数据集和对话模板. 如何自定义模型, 数据集和对话模板可以查看`使用文档`部分, 其对应的py文件可以查看`custom.py`, 其对应的sh脚本可以查看`scripts/custom/tigerbot_13b_chat`.
- 2023.10.17: 支持int8模型的SFT: qwen-7b-chat-int8, qwen-14b-chat-int8. 对应的sh脚本可以查看`scripts/qwen_7b_chat_int8`, `scripts/qwen_14b_chat_int8`.
- 🔥 2023.10.16: 支持int4模型的SFT: qwen-7b-chat-int4, qwen-14b-chat-int4, qwen-vl-chat-int4, baichuan2-7b-chat-int4, baichuan2-13b-chat-int4. 对应的sh脚本可以查看`scripts/qwen_7b_chat_int4`, `scripts/qwen_14b_chat_int4`, `scripts/qwen_vl_chat_int4`, `scripts/baichuan2_7b_chat_int4`, `scripts/baichuan2_13b_chat_int4`.
- 2023.10.15: 支持ziya2-13b系列模型: ziya2-13b, ziya2-13b-chat. 对应的sh脚本可以查看`scripts/ziya2_13b_chat`.
- 2023.10.12: 支持mistral-7b系列模型: openbuddy-mistral-7b-chat, mistral-7b, mistral-7b-chat. 对应的sh脚本可以查看`scripts/openbuddy_mistral_7b_chat`, `scripts/mistral_7b_chat`.
- 🔥 2023.10.7: 支持DeepSpeed ZeRO-2, 使得lora(不仅仅是qlora)可以在双卡A10上运行DDP. 对应的sh脚本可以查看`scripts/qwen_7b_chat/lora_ddp_ds/sft.sh`.
- 2023.10.4: 支持更多数学, 法律, SQL, 代码领域的数据集: blossom-math-zh, school-math-zh, text2sql-en, sql-create-context-en, lawyer-llama-zh, tigerbot-law-zh, leetcode-python-en.
- 🔥 2023.9.25: 支持**qwen-14b**系列模型: qwen-14b, qwen-14b-chat. 对应的sh脚本可以查看`scripts/qwen_14b`, `scripts/qwen_14b_chat`.
- 2023.9.18: 支持internlm-20b系列模型: internlm-20b, internlm-20b-chat. 对应的sh脚本可以查看`scripts/internlm_20b`, `scripts/internlm_20b_chat`.
- 🔥 2023.9.12: 支持MP+DDP的方式训练, 加快全参数微调的速度, 对应的sh脚本可以查看`scripts/qwen_7b_chat/full_mp_ddp/sft.sh`.
- 2023.9.5: 支持训练只保存模型权重, 而不保存断点续训所需的优化器权重等中间状态, 避免全参数微调保存checkpoint所需时间过长和空间过大的问题. 可以查看`sft.sh`中的命令行参数: `--only_save_model`.
- 2023.9.5: 支持openbuddy-llama2-70b-chat模型. 对应的sh脚本可以查看`scripts/openbuddy_llama2_70b_chat`.
- 2023.9.3: 支持baichuan2系列模型: baichuan2-7b, baichuan2-7b-chat, baichuan2-13b, baichuan2-13b-chat. 对应的sh脚本可以查看`scripts/baichuan2_7b`, `scripts/baichuan2_7b_chat`, `scripts/baichuan2_13b_chat`.


## 准备实验环境
实验环境: A10, 3090, V100, A100均可.
```bash
# 设置pip全局镜像和安装相关的python包
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
git clone https://github.com/modelscope/swift.git
cd swift
pip install .[llm]
cd examples/pytorch/llm

# 如果你想要使用deepspeed.
pip install deepspeed -U

# 如果你想要使用基于auto_gptq的qlora训练. (推荐, 效果优于bnb)
# 使用auto_gptq的模型: qwen-7b-chat-int4, qwen-14b-chat-int4, qwen-7b-chat-int8, qwen-14b-chat-int8
pip install auto_gptq optimum -U

# 如果你想要使用基于bnb的qlora训练.
pip install bitsandbytes -U
```


## 简单的使用
以下案例可以用于测试环境. 请确保您已经阅读了`准备实验环境`部分.
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


## 微调和推理
性能: full(优) > lora > qlora

训练显存: qlora(低,3090) > lora > full(2*A100)

**提示**:
- 你可以在训练时设置`--gradient_checkpointing true`来节约显存, 但这会略微降低训练速度. 如果你需要在消费级显卡中训练大模型, 这很有用, 例如: 3090.
- 如果你想要使用量化参数`quantization_bit`, 你需要先安装bnb: `pip install bitsandbytes -U`.
- 如果你想要使用基于auto_gptq的量化, 你需要先安装auto_gptq: `pip install auto_gptq -U`.
  使用auto_gptq的模型包含: `qwen-7b-chat-int4`, `qwen-14b-chat-int4`, `qwen-7b-chat-int8`, `qwen-14b-chat-int8`.
  如果脚本提供了非量化模型和int4/int8模型的多个版本的qlora SFT版本, 推荐使用int4/int8模型版本的脚本.
- 如果你想要使用deepspeed, 你需要`pip install deepspeed -U`. 使用deepspeed可以节约显存, 但可能会略微降低训练速度.
- 如果你使用的是V100等较老的GPU, 你需要设置`--dtype fp16`, 因为其不支持bf16.
- 如果你的机器是A100等高性能显卡, 且使用的是qwen系列模型, 推荐你安装[flash-attn](https://github.com/Dao-AILab/flash-attention), 这将会加快训练和推理的速度以及显存占用(A10, 3090, V100等显卡不支持flash-attn进行训练).
- 如果你要进行二次预训练而不是SFT, 你可以参考`DatasetName.tigerbot_law_zh`数据集和其对于的sh文件: `scripts/qwen_7b/qlora_ddp`.
- 如果你想在训练时, 将权重push到ModelScope Hub中, 你需要设置`--push_to_hub true`.
- 如何你想要在推理时, 合并LoRA权重并保存，你需要设置`--merge_lora_and_save true`. 不推荐对量化的模型进行merge, 这会存在精度损失, 即qlora.
- 以下提供了可以直接运行的`qwen_7b_chat`的sh脚本(你只需要在推理时指定`ckpt_dir`即可顺利执行). 更多模型的scripts脚本, 可以查看`scripts`文件夹. 如果你想要自定义sh脚本, 推荐你参考`scripts/qwen_7b_chat`中的脚本进行书写.
```bash
# 微调(qlora)+推理 qwen-7b-chat-int8, 需要16GB显存.
# 推荐的实验环境: V100, A10, 3090
bash scripts/qwen_7b_chat_int8/qlora/sft.sh
bash scripts/qwen_7b_chat_int8/qlora/infer.sh

# 微调(qlora+ddp+deepspeed)+推理 qwen-7b-chat-int8, 需要2卡*19GB显存.
# 推荐的实验环境: V100, A10, 3090
bash scripts/qwen_7b_chat_int8/qlora_ddp_ds/sft.sh
bash scripts/qwen_7b_chat_int8/qlora_ddp_ds/infer.sh

# 微调(qlora)+推理 qwen-7b-chat-int4, 需要13GB显存.
# 推荐的实验环境: V100, A10, 3090
bash scripts/qwen_7b_chat_int4/qlora/sft.sh
bash scripts/qwen_7b_chat_int4/qlora/infer.sh

# 微调(qlora+ddp+deepspeed)+推理 qwen-7b-chat-int4, 需要2卡*16GB显存.
# 推荐的实验环境: V100, A10, 3090
bash scripts/qwen_7b_chat_int4/qlora_ddp_ds/sft.sh
bash scripts/qwen_7b_chat_int4/qlora_ddp_ds/infer.sh

# 微调(lora)+推理 qwen-7b-chat, 需要60GB显存.
# 推荐的实验环境: A100
bash scripts/qwen_7b_chat/lora/sft.sh
bash scripts/qwen_7b_chat/lora/infer.sh

# 微调(lora+ddp)+推理 qwen-7b-chat, 需要2卡*60GB显存.
# 推荐的实验环境: A100
bash scripts/qwen_7b_chat/lora_ddp/sft.sh
bash scripts/qwen_7b_chat/lora_ddp/infer.sh

# 微调(lora+ddp+deepspeed)+推理 qwen-7b-chat, 需要2卡*18GB显存.
# 推荐的实验环境: A10, 3090
bash scripts/qwen_7b_chat/lora_ddp_ds/sft.sh
bash scripts/qwen_7b_chat/lora_ddp_ds/infer.sh

# 微调(lora+mp+ddp)+推理 qwen-7b-chat, 需要4卡*15GB显存.
# 推荐的实验环境: A10, 3090
bash scripts/qwen_7b_chat/lora_mp_ddp/sft.sh
bash scripts/qwen_7b_chat/lora_mp_ddp/infer.sh

# 微调(full+mp)+推理 qwen-7b-chat, 需要2卡*75G显存.
# 推荐的实验环境: A100
bash scripts/qwen_7b_chat/full_mp/sft.sh
bash scripts/qwen_7b_chat/full_mp/infer.sh

# 微调(full+mp+ddp)+推理 qwen-7b-chat, 需要4卡*75G显存.
# 推荐的实验环境: A100
bash scripts/qwen_7b_chat/full_mp_ddp/sft.sh
bash scripts/qwen_7b_chat/full_mp_ddp/infer.sh

# 微调(qlora)+推理 qwen-7b-chat, 需要13GB显存.
# 推荐的实验环境: A10, 3090
bash scripts/qwen_7b_chat/qlora/sft.sh
bash scripts/qwen_7b_chat/qlora/infer.sh

# 微调(qlora+ddp)+推理 qwen-7b-chat, 需要2卡*14GB显存.
# 推荐的实验环境: A10, 3090
bash scripts/qwen_7b_chat/qlora_ddp/sft.sh
bash scripts/qwen_7b_chat/qlora_ddp/infer.sh

# 微调(qlora+ddp+deepspeed)+推理 qwen-7b-chat, 需要2卡*16GB显存.
# 推荐的实验环境: A10, 3090
bash scripts/qwen_7b_chat/qlora_ddp_ds/sft.sh
bash scripts/qwen_7b_chat/qlora_ddp_ds/infer.sh
```


## 使用文档
### 自定义模型
以下是一个自定义模型的案例. 运行该自定义模型的sh可以查看`scripts/custom/tigerbot_13b_chat`.

```python
from swift.llm import (
    register_model, LoRATM, get_model_tokenizer_from_repo, get_model_tokenizer
)
import torch
from torch import dtype as Dtype
from typing import Dict, Any

class CustomModelType:
    tigerbot_13b_chat = 'tigerbot-13b-chat'

class CustomTemplateType:
    tigerbot = 'tigerbot'

@register_model(CustomModelType.tigerbot_13b_chat,
                'TigerResearch/tigerbot-13b-chat-v4', LoRATM.llama2,
                CustomTemplateType.tigerbot)
def get_tigerbot_model_tokenizer(model_dir: str,
                                 torch_dtype: Dtype,
                                 model_kwargs: Dict[str, Any],
                                 load_model: bool = True,
                                 **kwargs):
    use_flash_attn = kwargs.pop('use_flash_attn', False)
    if use_flash_attn:
        require_version('transformers>=4.34')
        logger.info('Setting use_flash_attention_2: True')
        model_kwargs['use_flash_attention_2'] = True
    return get_model_tokenizer_from_repo(model_dir, torch_dtype, model_kwargs,
                                         load_model, **kwargs)

# 不使用修饰器的用法:
# register_model(CustomModelType.tigerbot_13b_chat,
#                'TigerResearch/tigerbot-13b-chat-v4', LoRATM.llama2,
#                CustomTemplateType.tigerbot, get_tigerbot_model_tokenizer)

if __name__ == '__main__':
    model_kwargs = {'device_map': 'auto'}
    model, tokenizer = get_model_tokenizer(CustomModelType.tigerbot_13b_chat, torch.bfloat16, use_flash_attn=False)
    print(model, tokenizer)
```
`register_model`会在`MODEL_MAPPING`中注册模型, 该函数的参数含义如下:
- `model_type`: 必填项. 表示模型的名字, 也是唯一的id.
- `model_id_or_path`: 必填项. 表示模型在ModelScope Hub中的`model_id`, 或者是本地的模型目录`model_dir`.
- `lora_target_modules`: 默认为`None`. 表示在sh脚本中指定`--lora_target_modules AUTO`或未指定`--lora_target_modules`情况下默认使用的lora_target_modules.
- `template`: 默认为`TemplateType.default`. 表示在sh脚本中未指定`--template`情况下默认使用的chat template.
- `get_function`: 默认值为`None`. 获取model和tokenizer的函数. 如果传入None, 则使用修饰器方案进行模型注册, `register_model`函数将返回`Callable[[GetModelTokenizerFunction], GetModelTokenizerFunction]`, 该方案需要有一定python基础的用户使用. 如果传入一个函数, 则使用正常方案进行注册. 一般使用`get_model_tokenizer_from_repo`作为参数传入, 返回model和tokenizer. 如果出现需要对模型代码打补丁等情况, 则可以通过自定义该函数来实现.
- `requires`: 默认为`[]`. 表示模型所需要的区别于其他模型的依赖. 该参数一般不需要设置.
- `torch_dtype`: 默认为`None`. 表示模型所推荐使用的torch_dtype. 该参数一般不需要设置.
- `automodel_class`: 默认为`AutoModelForCausalLM`. 表示被调用from_pretrained的类. 如果你使用的是`roberta-base`等模型, 则需要修改该参数. 该参数一般不需要设置.
- `revision`: 默认为`'master'`. 用于指定模型的版本号. 如果`model_id_or_path`是本地的模型目录, 则该参数失效. 该参数一般不需要设置.
- `ignore_file_pattern`: 默认为`None`. 表示下载的时候需要忽略的文件名的正则pattern, 该参数会传递给`snapshot_download`. 例如`r'.+\.bin$'`, `r'.+\.savetensors$'`等. 该参数一般不需要设置.
- `max_length`: 默认为`None`. 用于注释模型的max_length. 该参数一般不需要设置.
- `function_kwargs`: 默认为`{}`, 用于传递给`get_function`, 用于支持修饰器情况下的`partial`功能. 该参数一般不需要设置.
- `**kwargs`: 其他用于注释模型能力的参数. 该参数一般不需要设置.


### 自定义数据集
以下是一个自定义数据集的案例. 运行该自定义数据集的sh可以查看`scripts/custom/tigerbot_13b_chat`.

```python
import ast
from swift.llm import register_dataset, get_dataset, preprocess_conversations
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets
from typing import List
from modelscope import MsDataset

class CustomDatasetName:
    agent_instruct_all_en = 'agent-instruct-all-en'

_agent_instruct_subset_list = [
    'alfworld', 'db', 'kg', 'mind2web', 'os', 'webshop'
]

@register_dataset(
    CustomDatasetName.agent_instruct_all_en,
    task='chat',
    function_kwargs={'subset_name_list': _agent_instruct_subset_list})
def get_agent_instruct_dataset(subset_name_list: List[str]) -> HfDataset:
    dataset_list: List[HfDataset] = []
    for subset_name in subset_name_list:
        dataset: HfDataset = MsDataset.load(
            'huangjintao/AgentInstruct_copy',
            subset_name=subset_name,
            split='train').to_hf_dataset()
        dataset_list.append(dataset)
    dataset = concatenate_datasets(dataset_list)

    def repair_conversations(s: str) -> str:
        s = s.replace('}\n {', '},\n {')
        return ast.literal_eval(s)

    return preprocess_conversations(
        dataset, 'human', 'gpt', repair_conversations=repair_conversations)

# 不使用修饰器的用法:
# register_dataset(
#     CustomDatasetName.agent_instruct_all_en,
#     get_agent_instruct_dataset,
#     task='chat',
#     function_kwargs={'subset_name_list': _agent_instruct_subset_list})

if __name__ == '__main__':
    train_dataset, _ = get_dataset([CustomDatasetName.agent_instruct_all_en],
                                   0.)
    print(train_dataset)
    print(train_dataset[0].keys())
```
`register_dataset`会在`DATASET_MAPPING`中注册数据集, 该函数的参数含义如下:
- `dataset_name`: 必填项, 表示数据集的名字, 也是数据集的唯一id.
- `get_function`: 默认值为`None`. 获取数据集的函数. 如果传入None, 则使用修饰器方案进行数据集注册, `register_dataset`函数将返回`Callable[[GetDatasetFunction], GetDatasetFunction]`, 该方案需要有一定python基础的用户使用. 如果传入一个函数, 则使用正常方案进行注册.
  `get_function`函数不用传入任何参数, 需要返回`HfDataset`或`Tuple[HfDataset, HfDataset]`. 第一种情况下, 数据集处理函数会切分一部分的数据集作为验证集 (根据命令行超参数`dataset_test_ratio`); 第二种情况下, 返回的两个数据集分别作为其训练集和验证集. 我们支持使用多个数据集进行微调. 我们会将各个子数据集的训练集和验证集部分分别进行拼接, 最终返回合并后的训练集和验证集.
  函数返回的`HfDataset`需要符合一定的规范. 如果是指令微调(单轮对话)的情况下, 需包含`query`, `response`字段, 分别代表指令微调的用户询问和AI助手的回答, 具体可以参考`alpaca-zh`数据集. 如果是多轮对话, 则需要额外加上`history`字段, 代表对话的历史信息, 具体可以参考`damo-agent-mini-zh`数据集. 如果每个数据集样例具有不同的`system`, 则需要额外加上system字段, 具体你也可以参考`damo-agent-mini-zh`数据集. 我们只会对`response`部分进行loss的计算和优化.
- `task`: 注释数据集用作的任务. 该参数一般不需要设置.
- `function_kwargs`: 默认为`{}`, 用于传递给`get_function`, 用于支持修饰器情况下的`partial`功能. 该参数一般不需要设置.
- `**kwargs`: 其他用于注释数据集的参数. 该参数一般不需要设置.

### 自定义对话模板
以下是一个自定义对话模板的案例. 运行该自定义对话模板的sh可以查看`scripts/custom/tigerbot_13b_chat`.

```python
from swift.llm import (
    register_template, Template, get_template, get_model_tokenizer, ModelType, inference
)
class CustomTemplateType:
    tigerbot = 'tigerbot'

# Ref: https://github.com/TigerResearch/TigerBot/blob/main/infer.py
register_template(
    CustomTemplateType.tigerbot,
    Template([], ['\n\n### Instruction:\n{{QUERY}}\n\n### Response:\n'], [],
             [['eos_token_id']]))

if __name__ == '__main__':
    # only for test
    _, tokenizer = get_model_tokenizer(ModelType.qwen_7b_chat, load_model=False)
    template = get_template(CustomTemplateType.tigerbot, tokenizer)
    inputs = {'query': '浙江的省会在哪里?', 'response': '杭州',
              'system': 'you are a helpful assistant!',
              'history': [('你好!', '你好! 我是AI智能助手. '),
                          ('1+1=?', '2')]}
    print(tokenizer.decode(template.encode(inputs)['input_ids']))
```
`register_template`会在`TEMPLATE_MAPPING`中注册对话模板, 该函数的参数含义如下:
- `template_type`: 必填项, 表示对话模板的名字, 也是template的唯一id.
- `template`: 必填项, 需要传入一个`Template`. 初始化`Template`需要传入4个参数: `prefix`, `prompt`, `chat_sep`, `suffix`.

模板初始化函数会根据这四个内容, 获取完整的chat template, 使其支持预训练, text generation式的SFT, 各种chat类型的SFT. 其中这四个配置内容的含义如下.
- `prefix`: 表示对话模板中的前缀部分, 一般为system部分及其相关格式, 前缀token, bos token等内容. 我们使用`{{SYSTEM}}`作为system部分的占位符.
- `prompt`: 表示对话模板中的一轮对话. 我们使用`{{QUERY}}`作为每轮对话中, human询问部分的占位符, `{{ROUND0}}`则表示本次对话是第几轮的占位符, 从0开始计数, `{{ROUND1}}`从1开始计数. AI助手的回复部分会拼接在`prompt`的后面, 因此我们没有设计其占位符.
- `chat_sep`: 如果需要进行多轮对话, `chat_sep`会作为每轮对话之间的分隔符, 例如: 换行等. 如果设置为None, 则该Template不支持多轮对话.
- `suffix`: 作为对话模板的后缀部分, 一般为eos token. 会拼接在最后一轮的对话后面. 只有最后一轮对话的reponse部分和`suffix`会计算loss并优化, 其余部分不计算损失.


### sft.sh 命令行参数
- `--model_type`: 表示你选择的模型类型, 默认是`None`, 即如果没有指定`model_id_or_path`, 则选择`'qwen-7b-chat'`, 如果指定了, 则会根据`model_id_or_path`以及`MODEL_MAPPING`推断`model_type`. 这两个参数不能同时指定. 可以选择的`model_type`可以查看`MODEL_MAPPING.keys()`.
- `--model_id_or_path`: 表示模型在ModelScope Hub中的`model_id`, 或者是本地的模型目录`model_dir`, 不区分大小写, 默认为`None`. 如果`--model_id_or_path`未被注册, 则会抛出异常. 你可以使用`model_type`的方式指定模型类型, 也可以通过`model_id_or_path`的方式指定模型类型.
- `--model_revision`: 表示模型在ModelScope Hub中对应`model_id`的版本号, 默认为`None`. 如果`model_id_or_path`使用本地的模型目录, 则该参数失效. model_revision指定为None, 则使用注册在`MODEL_MAPPING`中的revision. 否则强制使用model_revision.
- `model_cache_dir`: 默认为`None`. 如果模型在本地已经有缓存, 且缓存路径并非ModelScope默认cache路径, 可以通过指定该参数从cache_dir中导入model和tokenizer.
- `--sft_type`: 表示微调的方式, 默认是`'lora'`. 你可以选择的值包括: 'lora', 'full'. 如果你要使用lora或qlora, 你需要选择`--sft_type lora`. qlora需额外设置`--quantization_bit 4`. 如果你要使用全参数微调, 则需选择`--sft_type full`.
- `--tuner_backend`: 表示lora, qlora的后端支持, 默认是`'swift'`. 你可以选择的值包括: 'swift', 'peft'.
- `--template_type`: 表示使用的对话模板的类型, 默认是`None`, 即根据`model_type`查找`MODEL_MAPPING`中的`template`. 可以选择的`template_type`可以查看`TEMPLATE_MAPPING.keys()`.
- `--output_dir`: 表示ckpt存储的目录, 默认是`'output'`. 我们会在该目录后拼接`model_type`和微调版本号. 方便用户对不同模型进行多次对比实验, 而不需要改变`output_dir`命令行参数.
- `--add_output_dir_suffix`: 默认为`True`, 表示会在`output_dir`的目录后拼接上`model_type`和微调版本号的后缀. 如果要避免此行为, 你可以设置为`False`.
- `--ddp_backend`: 表示分布式的后端支持, 默认是`'nccl'`. 你可以选择的值包括: 'nccl', 'gloo', 'mpi', 'ccl'.
- `--seed`: 全局的seed, 默认使用42. 在分布式训练中, 为避免每个进程使用相同的dropout等情况, 我们会令`seed=seed+rank`.
- `--resume_from_checkpoint`: 用于断点续训, 默认为`None`. 你可以将其设置为checkpoint的路径, 例如: `'output/qwen-7b-chat/vx_xxx/checkpoint-xxx'`, 来进行断点续训.
- `--dtype`: 基模型载入时的torch_dtype, 默认为`None`, 即智能选择dtype: 如果机器不支持bf16, 则使用fp16, 如果`MODEL_MAPPING`中对应模型有指定torch_dtype, 则使用其对应dtype, 否则使用bf16. 你可以选择的值包括: 'bf16', 'fp16', 'fp32'.
- `--dataset`: 用于选择训练的数据集, 默认为`'blossom-math-zh'`. 可以选择的数据集可以查看`DATASET_MAPPING.keys()`. 如果需要使用多个数据集进行训练, 你可以使用','或者' '进行分割, 例如: `alpaca-en,alpaca-zh` or `alpaca-en alpaca-zh`.
- `--dataset_seed`: 用于指定数据集处理的seed, 默认为`42`. 以random_state形式存在, 不影响全局seed.
- `--dataset_test_ratio`: 用于指定子数据集切分成训练集和验证集的比例, 默认为`0.01`. 如果子数据集已经进行了训练集和验证集的切分, 则此参数无效. 当`dataset`中指定了多个子数据集时, 且获取子数据集的函数没有进行训练集和验证集的切分(即返回的是`HfDataset`而不是`Tuple[HfDataset, HfDataset]`), 则我们需要对该子数据集进行切分. 最后, 我们会将这些子数据集的训练集和验证集部分分别进行拼接, 生成完整微调数据集的训练集和验证集.
- `--train_dataset_sample`: 对完整训练集进行采样, 默认是`20000`, 用于加快训练的速度. 该参数是为了避免数据集过大, 单个epoch训练时间过长的问题. LoRA的收敛通常较快, 不需要过多数据样本的微调. 如果你指定为`-1`, 则使用完整的训练集进行训练, 该情况一般出现在全参数微调的设置下.
- `--system`: 对话模板中使用的system, 默认为`'you are a helpful assistant!'`.
- `--max_length`: token的最大长度, 默认为`2048`. 可以避免个别过长的数据样本造成OOM的问题. 如果某数据样本长度超过max_length, 我们会切除最前面的token: `input_ids[-max_length:]`. 如果设置为-1, 则无限制.
- `--quantization_bit`: 用于指定是否进行量化和量化的bit数, 默认为`0`, 即不进行量化. 量化情况下, 只支持lora的微调方式, 不支持全参数的微调方式.
- `--bnb_4bit_comp_dtype`: 在进行4bit量化时, 我们需要在模型的forward和backward时, 将其进行反量化. 该参数用于指定反量化后的torch_dtype. 默认为`None`, 即与`dtype`保持一致. 可选择的值包括: 'fp16', 'bf16', 'fp32'. 当quantization_bit为0时, 该参数无效.
- `--bnb_4bit_quant_type`: 4bit量化时的量化方式, 默认是`'nf4'`. 可选择的值包括: 'nf4', 'fp4'. 当quantization_bit为0时, 该参数无效.
- `--bnb_4bit_use_double_quant`: 是否在4bit量化时开启double量化, 默认为`True`. 当quantization_bit为0时, 该参数无效.
- `--lora_target_modules`: 指定lora模块, 默认为`None`. 如果lora_target_modules为None, 或者传入AUTO, 则根据`model_type`查找`MODEL_MAPPING`中的`lora_target_modules`(默认指定为qkv). 如果传入`ALL`, 则将所有的Linear层都指定为lora模块(不含head). 该参数只有当`sft_type`指定为'lora'时才生效.
- `--lora_rank`: 默认为`8`. 只有当`sft_type`指定为'lora'时才生效.
- `--lora_alpha`: 默认为`32`. 只有当`sft_type`指定为'lora'时才生效.
- `--lora_dropout_p`: 默认为`0.05`, 只有当`sft_type`指定为'lora'时才生效.
- `--gradient_checkpointing`: 是否开启gradient checkpointing, 默认为`False`. 该参数可以用于节约显存, 虽然这会略微降低训练速度. 该参数在max_length较大, batch_size较大时作用显著.
- `--deepspeed_config_path`: 用于指定deepspeed的配置文件的路径, 默认为`None`, 即不开启deepspeed. deepspeed可以节约显存. 我们书写了默认的ZeRO-2的配置文件: `ds_config/zero2.json`.
- `--batch_size`: 训练时的batch_size, 默认为`1`. 增大batch_size可以增加GPU的利用率, 但不一定会增加训练速度, 因为在一个batch中, 需要对较短的句子按该batch中最长句子的长度进行padding, 从而引入无效的计算量.
- `--eval_batch_size`: 评估时的batch_size, 默认为`None`, 即当`predict_with_generate`为True时, 设置为1, 为False时, 设置为`batch_size`.
- `--num_train_epochs`: 训练的epoch数, 默认为`1`. 如果`max_steps >= 0`, 则覆盖`num_train_epochs`.
- `--max_steps`: 训练的max_steps数, 默认为`-1`. 如果`max_steps >= 0`, 则覆盖`num_train_epochs`.
- `--optim`: 默认为`'adamw_torch'`.
- `--learning_rate`: 默认值为`None`, 即如果`sft_type`为lora, 则设置为1e-4, 如果`sft_type`为full, 则设置为2e-5.
- `--weight_decay`: 默认值为`0.01`.
- ` --gradient_accumulation_steps`: 梯度累加, 默认值为`16`. `total_batch_size =  batch_size * gradient_accumulation_steps * world_size`.
- `--max_grad_norm`: 梯度裁剪, 默认值为`1`.
- `--predict_with_generate`: 评估时是否使用生成式的方式, 默认为`False`. 如果设置为False, 则使用`loss`进行评估. 如果设置为True, 则使用`ROUGE-L`等指标进行评估. 使用生成式评估耗费的时间很长, 请谨慎选择.
- `--lr_scheduler_type`: 默认值为`'cosine'`.
- `--warmup_ratio`: warmup占用总的训练steps的比例, 默认为`0.05`.
- `--eval_steps`: 每训练多少steps进行评估, 默认为`50`.
- `--save_steps`: 每训练多少个steps进行保存, 默认为`None`, 即设置为`eval_steps`.
- `--only_save_model`: 是否只保存模型参数, 而不存储断点续训所需的中间状态, 默认为`None`, 即如果`sft_type`为'lora'并且不使用deepspeed(`deepspeed_config_path`为None), 设置为False, 否则设置为True(e.g. 使用了全参数微调或者使用了deepspeed).
- `--save_total_limit`: 保存的checkpoint的数量, 默认为`2`, 即保存best和last的checkpoint. 如果设置为-1, 则保存所有的checkpoint.
- `--logging_steps`: 每训练多少步打印训练信息(e.g. loss, learning_rate等), 默认为`5`.
- `--dataloader_num_workers`: 默认值为`1`.
- `--push_to_hub`: 是否将训练的checkpoint同步推送到ModelScope Hub中, 默认为`False`.
- `--hub_model_id`: 推送到的ModelScope Hub的model_id, 默认为`None`, 即设置为`f'{model_type}-{sft_type}'`. 你可以将其设置为model_id, 也可以设置为repo_name. 我们会根据hub_token推断出user_name. 推送的远程仓库如果不存在, 则会创建一个新的仓库, 如果存在, 则复用之前的仓库. 该参数只有在`push_to_hub`设置为True时才生效.
- `--hub_private_repo`: 推送的ModelScope Hub中的模型仓库的权限是否设置为私有, 默认为`True`. 该参数只有在`push_to_hub`设置为True时才生效.
- `--push_hub_strategy`: 推送策略, 默认为`'push_best'`. 可选择的值包括: 'end', 'push_best', 'push_last', 'checkpoint', 'all_checkpoints'. 'push_best'表示在每次保存权重时, 将最好的模型进行推送并覆盖之前的权重, 'push_last'表示在每次保存权重时, 将最后的权重进行推送并覆盖之前的权重. 该参数只有在`push_to_hub`设置为True时才生效.
- `--hub_token`: 推送时需要的SDK token. 可以从[https://modelscope.cn/my/myaccesstoken](https://modelscope.cn/my/myaccesstoken)获取, 默认为`None`, 即从环境变量`MODELSCOPE_API_TOKEN`中获取. 该参数只有在`push_to_hub`设置为True时才生效.
- `--test_oom_error`: 用于检测训练是否会发生OOM, 默认为`False`. 如果设置为True, 则会将训练集按max_length倒序进行排列, 方便OOM的测试. 该参数一般用于测试, 请谨慎设置.
- `--use_flash_attn`: 是否使用flash attn, 默认为`None`. 安装flash_attn的步骤可以查看[https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- `--ignore_args_error`: 是否忽略命令行传参错误抛出的Error, 默认为`False`. 如果需要拷贝代码到notebook中运行, 需要设置成True.
- `--logging_dir`: 默认为`None`. 即设置为`f'{self.output_dir}/runs'`, 表示tensorboard文件存储路径.
- `--max_new_tokens`: 默认为`2048`. 该参数只有在`predict_with_generate`设置为True的时候才生效.
- `--do_sample`: 默认为`True`. 该参数只有在`predict_with_generate`设置为True的时候才生效.
- `--temperature`: 默认为`0.9`. 该参数只有在`predict_with_generate`设置为True的时候才生效.
- `--top_k`: 默认为`20`. 该参数只有在`predict_with_generate`设置为True的时候才生效.
- `--top_p`: 默认为`0.9`. 该参数只有在`predict_with_generate`设置为True的时候才生效.
- `--repetition_penalty`: 默认为`1.05`. 该参数只有在`predict_with_generate`设置为True的时候才生效.


### infer.sh 命令行参数
- `--model_type`: 默认值为`None`, 具体的参数介绍可以在`sft.sh命令行参数`中查看.
- `--model_id_or_path`: 默认值为`None`, 具体的参数介绍可以在`sft.sh命令行参数`中查看. 推荐使用model_type的方式指定.
- `--model_revision`: 默认值为`None`. 具体的参数介绍可以在`sft.sh命令行参数`中查看. 如果`model_id_or_path`为None或者是本地的模型目录, 则该参数失效.
- `--sft_type`: 默认值为`'lora'`, 具体的参数介绍可以在`sft.sh命令行参数`中查看.
- `--template_type`: 默认值为`None`, 具体的参数介绍可以在`sft.sh命令行参数`中查看.
- `--ckpt_dir`: 必填项, 值为SFT阶段保存的checkpoint路径, e.g. `'/path/to/your/vx_xxx/checkpoint-xxx'`.
- `--eval_human`: 使用数据集中的验证集部分进行评估还是使用人工的方式评估, 默认值为`False`. 我们可以直观感受到微调后模型的效果.
- `--seed`: 默认值为`42`, 具体的参数介绍可以在`sft.sh命令行参数`中查看.
- `--dtype`: 默认值为`None`, 具体的参数介绍可以在`sft.sh命令行参数`中查看.
- `--dataset`: 默认值为`'blossom-math-zh'`, 具体的参数介绍可以在`sft.sh命令行参数`中查看. 该参数只有在`eval_human`设置为False时才生效.
- `--dataset_seed`: 默认值为`42`, 具体的参数介绍可以在`sft.sh命令行参数`中查看. 该参数只有在`eval_human`设置为False时才生效.
- `--dataset_test_ratio`: 默认值为`0.01`, 具体的参数介绍可以在`sft.sh命令行参数`中查看. 该参数只有在`eval_human`设置为False时才生效.
- `--show_dataset_sample`: 表示想要评估和展示的验证集的数量, 默认值为`10`. 该参数只有在`eval_human`设置为False时才生效.
- `--system`: 默认值为`'you are a helpful assistant!'`. 具体的参数介绍可以在`sft.sh命令行参数`中查看.
- `--max_length`: 默认值为`2048`. 具体的参数介绍可以在`sft.sh命令行参数`中查看.
- `--quantization_bit`: 默认值为0. 具体的参数介绍可以在`sft.sh命令行参数`中查看.
- `--bnb_4bit_comp_dtype`: 默认值为`None`.  具体的参数介绍可以在`sft.sh命令行参数`中查看. 若`quantization_bit`设置为0, 则该参数失效.
- `--bnb_4bit_quant_type`: 默认值为`'nf4'`.  具体的参数介绍可以在`sft.sh命令行参数`中查看. 若`quantization_bit`设置为0, 则该参数失效.
- `--bnb_4bit_use_double_quant`: 默认值为`True`.  具体的参数介绍可以在`sft.sh命令行参数`中查看. 若`quantization_bit`设置为0, 则该参数失效.
- `--max_new_tokens`: 生成新token的最大数量, 默认值为`2048`.
- `--do_sample`: 是使用贪婪生成的方式还是采样生成的方式, 默认值为`True`.
- `--temperature`: 默认值为`0.9`. 该参数只有在`do_sample`设置为True时才生效.
- `--top_k`: 默认值为`20`. 该参数只有在`do_sample`设置为True时才生效.
- `--top_p`: 默认值为`0.9`. 该参数只有在`do_sample`设置为True时才生效.
- `--repetition_penalty`: 默认值为`1.05`.
- `--use_flash_attn`: 默认值为`None`, 即为'auto'. 具体的参数介绍可以在`sft.sh命令行参数`中查看.
- `--ignore_args_error`: 默认值为`False`, 具体的参数介绍可以在`sft.sh命令行参数`中查看.
- `--stream`: 是否使用流式输出, 默认为`True`.
- `--merge_lora_and_save`: 是否将lora权重merge到基模型中, 并保存完整的权重, 默认为`False`. 权重会保存在`ckpt_dir`的同级目录中,  e.g. `'/path/to/your/vx_xxx/checkpoint-xxx-merged'`目录下.
- `--overwrite_generation_config`: 是否将评估所使用的generation_config保存成`generation_config.json`文件, 默认为`False`. 训练时保存的generate_config文件将被覆盖.
