# LLM微调文档
## 目录
- [环境准备](#环境准备)
- [微调](#微调)
- [Merge LoRA](#merge-lora)
- [推理](#推理)
- [Web-UI](#web-ui)

## 环境准备
GPU设备: A10, 3090, V100, A100均可.
```bash
# 设置pip全局镜像
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .[llm]

# 如果你想要使用deepspeed.
pip install deepspeed -U

# 如果你想要使用基于auto_gptq的qlora训练. (推荐, 效果优于bnb)
# 支持auto_gptq的模型: `https://github.com/modelscope/swift/blob/main/docs/source/LLM/支持的模型和数据集.md#模型`
# auto_gptq和cuda版本有对应关系，请按照`https://github.com/PanQiWei/AutoGPTQ#quick-installation`选择版本
pip install auto_gptq -U

# 如果你想要使用基于bnb的qlora训练.
pip install bitsandbytes -U

# 环境对齐 (如果你运行错误, 可以跑下面的代码, 仓库使用最新环境测试)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## 微调
### 使用python
```python
# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments,
    infer_main, sft_main, app_ui_main, merge_lora_main
)

model_type = ModelType.qwen_7b_chat
sft_args = SftArguments(
    model_type=model_type,
    train_dataset_sample=2000,
    dataset=[DatasetName.blossom_math_zh],
    output_dir='output')
result = sft_main(sft_args)
best_model_checkpoint = result['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')
torch.cuda.empty_cache()

infer_args = InferArguments(
    ckpt_dir=best_model_checkpoint,
    show_dataset_sample=10)
# merge_lora_main(infer_args)
result = infer_main(infer_args)
torch.cuda.empty_cache()

app_ui_main(infer_args)
```

### 使用CLI
```bash
# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset blossom-math-zh \
    --output_dir output \

# 使用自己的数据集
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --custom_train_dataset_path chatml.jsonl \
    --output_dir output \

# 使用DDP
# Experimental environment: 2 * 3090
# 2 * 23GB GPU memory
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset blossom-math-zh \
    --output_dir output \

# 多机多卡
# node0
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
NPROC_PER_NODE=4 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset blossom-math-zh \
    --output_dir output \
# node1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=xxx.xxx.xxx.xxx \
NPROC_PER_NODE=4 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset blossom-math-zh \
    --output_dir output \
```

### 更多sh脚本

更多sh脚本可以查看[这里](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts)

性能: full(优) > lora > qlora(auto_gptq) > qlora(bnb)

训练显存: qlora(低,3090) < lora < full(高,2*A100)

```bash
# 下面的脚本需要在此目录下执行
cd examples/pytorch/llm
```

**提示**:

- 我们默认在训练时设置`--gradient_checkpointing true`来**节约显存**, 这会略微降低训练速度.
- 如果你想要使用量化参数`--quantization_bit 4`, 你需要先安装[bnb](https://github.com/TimDettmers/bitsandbytes): `pip install bitsandbytes -U`. 这会减少显存消耗, 但通常会降低训练速度.
- 如果你想要使用基于**auto_gptq**的量化, 你需要先安装对应cuda版本的[auto_gptq](https://github.com/PanQiWei/AutoGPTQ): `pip install auto_gptq -U`.
  > 使用auto_gptq的模型可以查看[LLM支持的模型](https://github.com/modelscope/swift/blob/main/docs/source/LLM/支持的模型和数据集.md#模型). 建议使用auto_gptq, 而不是bnb.
- 如果你想要使用deepspeed, 你需要`pip install deepspeed -U`. 使用deepspeed可以**节约显存**, 但可能会略微降低训练速度.
- 如果你的训练涉及到**知识编辑**的内容, 例如: [自我认知微调](https://github.com/modelscope/swift/blob/main/docs/source/LLM/自我认知微调最佳实践.md), 你需要在MLP上也加上LoRA, 否则可能会效果不佳. 你可以简单传入参数`--lora_target_modules ALL`来对所有的linear(qkvo, mlp)加上lora, **这通常是效果最好的**.
- 如果你使用的是**V100**等较老的GPU, 你需要设置`--dtype AUTO`或者`--dtype fp16`, 因为其不支持bf16.
- 如果你的机器是A100等高性能显卡, 且使用的是qwen系列模型, 推荐你安装[**flash-attn**](https://github.com/Dao-AILab/flash-attention), 这将会加快训练和推理的速度以及显存占用(A10, 3090, V100等显卡不支持flash-attn进行训练). 支持flash-attn的模型可以查看[LLM支持的模型](https://github.com/modelscope/swift/blob/main/docs/source/LLM/支持的模型和数据集.md#模型)
- 如果你要进行**二次预训练**, **多轮对话**, 你可以参考[自定义与拓展](https://github.com/modelscope/swift/blob/main/docs/source/LLM/自定义与拓展.md#注册数据集的方式)
- 如果你需要断网进行训练, 请使用`--model_cache_dir`和设置`--check_model_is_latest false`. 具体参数含义请查看[命令行参数](https://github.com/modelscope/swift/blob/main/docs/source/LLM/命令行参数.md).
- 如果你想在训练时, 将权重push到ModelScope Hub中, 你需要设置`--push_to_hub true`.
- 如何你想要在推理时, 合并LoRA权重并保存，你需要设置`--merge_lora_and_save true`. **不推荐对qlora训练的模型进行merge**, 这会存在精度损失.
- 以下提供了可以直接运行的`qwen_7b_chat`的sh脚本(你只需要在推理时指定`--ckpt_dir`即可顺利执行). 更多模型的scripts脚本, 可以查看[scripts文件夹](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts). 如果你想要**自定义sh脚本**, 推荐你参考`scripts/qwen_7b_chat`中的脚本进行书写.

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

# 微调(lora)+推理 qwen-7b-chat, 需要18GB显存.
# 推荐的实验环境: V100, A10, 3090
bash scripts/qwen_7b_chat/lora/sft.sh
bash scripts/qwen_7b_chat/lora/infer.sh

# 微调(lora+ddp)+推理 qwen-7b-chat, 需要2卡*18GB显存.
# 推荐的实验环境: V100, A10, 3090
bash scripts/qwen_7b_chat/lora_ddp/sft.sh
bash scripts/qwen_7b_chat/lora_ddp/infer.sh

# 微调(lora+ddp+deepspeed)+推理 qwen-7b-chat, 需要2卡*18GB显存.
# 推荐的实验环境: V100, A10, 3090
bash scripts/qwen_7b_chat/lora_ddp_ds/sft.sh
bash scripts/qwen_7b_chat/lora_ddp_ds/infer.sh

# 微调(lora+mp+ddp)+推理 qwen-7b-chat, 需要4卡*20GB显存.
# 推荐的实验环境: V100, A10, 3090
bash scripts/qwen_7b_chat/lora_mp_ddp/sft.sh
bash scripts/qwen_7b_chat/lora_mp_ddp/infer.sh

# 微调(full+mp)+推理 qwen-7b-chat, 需要2卡*55G显存.
# 推荐的实验环境: A100
bash scripts/qwen_7b_chat/full_mp/sft.sh
bash scripts/qwen_7b_chat/full_mp/infer.sh

# 微调(full+mp+ddp)+推理 qwen-7b-chat, 需要4卡*55G显存.
# 推荐的实验环境: A100
bash scripts/qwen_7b_chat/full_mp_ddp/sft.sh
bash scripts/qwen_7b_chat/full_mp_ddp/infer.sh

# 以下基于bnb的qlora脚本已不再推荐使用. 请优先使用基于auto_gptq的qlora脚本.
# 微调(qlora)+推理 qwen-7b-chat, 需要18GB显存.
# 推荐的实验环境: A10, 3090
bash scripts/qwen_7b_chat/qlora/sft.sh
bash scripts/qwen_7b_chat/qlora/infer.sh

# 微调(qlora+ddp)+推理 qwen-7b-chat, 需要2卡*20GB显存.
# 推荐的实验环境: A10, 3090
bash scripts/qwen_7b_chat/qlora_ddp/sft.sh
bash scripts/qwen_7b_chat/qlora_ddp/infer.sh

# 微调(qlora+ddp+deepspeed)+推理 qwen-7b-chat, 需要2卡*16GB显存.
# 推荐的实验环境: A10, 3090
bash scripts/qwen_7b_chat/qlora_ddp_ds/sft.sh
bash scripts/qwen_7b_chat/qlora_ddp_ds/infer.sh
```

## Merge LoRA
提示: **暂时**不支持bnb和auto_gptq量化模型的merge lora.
```bash
swift merge-lora --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx'
```

## 推理
### 原始模型
**单样本推理**可以查看[LLM推理文档](./LLM推理文档.md#-推理)

使用**数据集**评估:
```bash
CUDA_VISIBLE_DEVICES=0 swift infer --model_id_or_path qwen/Qwen-7B-Chat --dataset blossom-math-zh
```
### 微调后模型
**单样本推理**

使用LoRA**增量**权重进行推理:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from swift.tuners import Swift
import torch

model_dir = 'vx_xxx/checkpoint-100'
model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16, {'device_map': 'auto'})

model = Swift.from_pretrained(model, model_dir, inference_mode=True)
template = get_template(template_type, tokenizer)
query = 'xxxxxx'
response, history = inference(model, template, query)
print(f'response: {response}')
print(f'history: {history}')
```

使用LoRA **merged**的权重进行推理:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
import torch

model_dir = 'vx_xxx/checkpoint-100-merged'
model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16, {'device_map': 'auto'},
                                       model_dir=model_dir)

template = get_template(template_type, tokenizer)
query = 'xxxxxx'
response, history = inference(model, template, query)
print(f'response: {response}')
print(f'history: {history}')
```

使用**数据集**评估:
```bash
# 直接推理
CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx'

# Merge LoRA增量权重并推理
swift merge-lora --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx'
CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx-merged'
```

## Web-UI
### 原始模型
使用原始模型的web-ui可以查看[LLM推理文档](./LLM推理文档.md#-Web-UI)

### 微调后模型
```bash
# 直接使用web-ui
CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx'

# merge LoRA增量权重并使用web-ui
swift merge-lora --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx'
CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx-merged'
```
