# LLM微调文档
## 目录
- [环境准备](#环境准备)
- [微调](#微调)
- [DPO](#dpo)
- [ORPO](#orpo)
- [Merge LoRA](#merge-lora)
- [量化](#量化)
- [推理](#推理)
- [Web-UI](#web-ui)
- [推送模型](#推送模型)

## 环境准备
GPU设备: A10, 3090, V100, A100均可.
```bash
# 设置pip全局镜像 (加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

# 如果你想要使用deepspeed.
pip install deepspeed -U

# 如果你想要使用基于auto_gptq的qlora训练. (推荐, 效果优于bnb)
# 支持auto_gptq的模型: `https://github.com/modelscope/swift/blob/main/docs/source/LLM/支持的模型和数据集.md#模型`
# auto_gptq和cuda版本有对应关系，请按照`https://github.com/PanQiWei/AutoGPTQ#quick-installation`选择版本
pip install auto_gptq -U

# 如果你想要使用基于bnb的qlora训练.
pip install bitsandbytes -U

# 环境对齐 (通常不需要运行. 如果你运行错误, 可以跑下面的代码, 仓库使用最新环境测试)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## 微调
如果你要使用界面的方式进行微调与推理, 可以查看[界面训练与推理文档](https://github.com/modelscope/swift/blob/main/docs/source/GetStarted/%E7%95%8C%E9%9D%A2%E8%AE%AD%E7%BB%83%E6%8E%A8%E7%90%86.md).

### 使用python
```python
# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments,
    infer_main, sft_main, app_ui_main
)

model_type = ModelType.qwen_7b_chat
sft_args = SftArguments(
    model_type=model_type,
    dataset=[f'{DatasetName.blossom_math_zh}#2000'],
    output_dir='output')
result = sft_main(sft_args)
best_model_checkpoint = result['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')
torch.cuda.empty_cache()

infer_args = InferArguments(
    ckpt_dir=best_model_checkpoint,
    load_dataset_config=True)
# merge_lora(infer_args, device_map='cpu')
result = infer_main(infer_args)
torch.cuda.empty_cache()

app_ui_main(infer_args)
```

### 使用CLI
```bash
# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset AI-ModelScope/blossom-math-v2 \
    --output_dir output \

# 使用自己的数据集
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset chatml.jsonl \
    --output_dir output \

# 使用DDP
# Experimental environment: 2 * 3090
# 2 * 23GB GPU memory
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset AI-ModelScope/blossom-math-v2 \
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
    --dataset AI-ModelScope/blossom-math-v2 \
    --output_dir output \
# node1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=xxx.xxx.xxx.xxx \
NPROC_PER_NODE=4 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset AI-ModelScope/blossom-math-v2 \
    --output_dir output \
```

### 更多sh脚本

更多sh脚本可以查看[这里](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts)

```bash
# 脚本需要在此目录下执行
cd examples/pytorch/llm
```

**提示**:

- 我们默认在训练时设置`--gradient_checkpointing true`来**节约显存**, 这会略微降低训练速度.
- 如果你想要使用量化参数`--quantization_bit 4`, 你需要先安装[bnb](https://github.com/TimDettmers/bitsandbytes): `pip install bitsandbytes -U`. 这会减少显存消耗, 但通常会降低训练速度.
- 如果你想要使用基于**auto_gptq**的量化, 你需要先安装对应cuda版本的[auto_gptq](https://github.com/PanQiWei/AutoGPTQ): `pip install auto_gptq -U`.
  > 使用auto_gptq的模型可以查看[LLM支持的模型](支持的模型和数据集.md#模型). 建议使用auto_gptq, 而不是bnb.
- 如果你想要使用deepspeed, 你需要`pip install deepspeed -U`. 使用deepspeed可以**节约显存**, 但可能会略微降低训练速度.
- 如果你的训练涉及到**知识编辑**的内容, 例如: [自我认知微调](自我认知微调最佳实践.md), 你需要在MLP上也加上LoRA, 否则可能会效果不佳. 你可以简单传入参数`--lora_target_modules ALL`来对所有的linear(qkvo, mlp)加上lora, **这通常是效果最好的**.
- 如果你使用的是**V100**等较老的GPU, 你需要设置`--dtype AUTO`或者`--dtype fp16`, 因为其不支持bf16.
- 如果你的机器是A100等高性能显卡, 且模型支持flash-attn, 推荐你安装[**flash-attn**](https://github.com/Dao-AILab/flash-attention), 这将会加快训练和推理的速度以及显存占用(A10, 3090, V100等显卡不支持flash-attn进行训练). 支持flash-attn的模型可以查看[LLM支持的模型](支持的模型和数据集.md#模型)
- 如果你要进行**二次预训练**, **多轮对话**, 你可以参考[自定义与拓展](自定义与拓展.md#注册数据集的方式)
- 如果你需要**断网**进行训练, 请使用`--model_id_or_path <model_dir>`和设置`--check_model_is_latest false`. 具体参数含义请查看[命令行参数](命令行参数.md).
- 如果你想在训练时, 将权重push到ModelScope Hub中, 你需要设置`--push_to_hub true`.
- 如果你想要在推理时, 合并LoRA权重并保存，你需要设置`--merge_lora true`. **不推荐对qlora训练的模型进行merge**, 这会存在精度损失. 因此**不建议使用qlora进行微调**, 部署生态不好.


**注意**:

- 由于曾用名问题, 以`xxx_ds`结尾的脚本的含义是: 使用deepspeed zero2进行训练. (e.g. `full_ddp_ds`).
- 除了以下列出的脚本, 其他脚本不一定进行维护.


如果你想要**自定义脚本**, 可以参考以下脚本进行修改: (以下脚本会**定期维护**)

- full: [qwen1half-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen1half_7b_chat/full) (A100), [qwen-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/full_mp) (2\*A100)
- full+ddp+zero2: [qwen-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/full_ddp_zero2) (4\*A100)
- full+ddp+zero3: [qwen-14b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat/full_ddp_zero3) (4\*A100)
- lora: [chatglm3-6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/chatglm3_6b/lora) (3090), [baichuan2-13b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/baichuan2_13b_chat/lora_mp) (2\*3090), [yi-34b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yi_34b_chat/lora) (A100), [qwen-72b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_72b_chat/lora_mp) (2\*A100)
- lora+ddp: [chatglm3-6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/chatglm3_6b/lora_ddp) (2\*3090)
- lora+ddp+zero3: [qwen-14b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat/lora_ddp_zero3) (4\*3090), [qwen-72b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_72b_chat/lora_ddp_zero3) (4\*A100)
- qlora(gptq-int4): [qwen-14b-chat-int4](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat_int4/qlora) (3090), [qwen1half-72b-chat-int4](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen1half_72b_chat_int4/qlora) (A100)
- qlora(gptq-int8): [qwen-14b-chat-int8](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat_int8/qlora) (3090)
- qlora(bnb-int4): [qwen-14b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat/qlora) (3090), [llama2-70b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/llama2_70b_chat/qlora_mp) (2 \* 3090)


## DPO
如果你要使用DPO进行人类对齐, 你可以查看[DPO训练文档](DPO训练文档.md).

## ORPO
如果你要使用ORPO进行人类对齐, 你可以查看[ORPO最佳实践](ORPO算法最佳实践.md).

## Merge LoRA
提示: **暂时**不支持bnb和auto_gptq量化模型的merge lora, 这会产生较大的精度损失.
```bash
# 如果你需要量化, 可以指定`--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true
```

## 量化

对微调后模型进行量化可以查看[LLM量化文档](LLM量化文档.md#微调后模型)

## 推理
如果你要使用VLLM进行推理加速, 可以查看[VLLM推理加速与部署](VLLM推理加速与部署.md#微调后的模型)

### 原始模型
**单样本推理**可以查看[LLM推理文档](LLM推理文档.md#推理)

使用**数据集**评估:
```bash
CUDA_VISIBLE_DEVICES=0 swift infer --model_id_or_path qwen/Qwen-7B-Chat --dataset AI-ModelScope/blossom-math-v2
```
### 微调后模型
**单样本推理**:

使用LoRA**增量**权重进行推理:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from swift.tuners import Swift

ckpt_dir = 'vx-xxx/checkpoint-100'
model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
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

ckpt_dir = 'vx-xxx/checkpoint-100-merged'
model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'},
                                       model_id_or_path=ckpt_dir)

template = get_template(template_type, tokenizer)
query = 'xxxxxx'
response, history = inference(model, template, query)
print(f'response: {response}')
print(f'history: {history}')
```

使用**数据集**评估:
```bash
# 直接推理
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' \
    --load_dataset_config true \

# Merge LoRA增量权重并推理
# 如果你需要量化, 可以指定`--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged' --load_dataset_config true
```

**人工**评估:
```bash
# 直接推理
CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx'

# Merge LoRA增量权重并推理
# 如果你需要量化, 可以指定`--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged'
```

## Web-UI
如果你要使用VLLM进行部署并提供**API**接口, 可以查看[VLLM推理加速与部署](VLLM推理加速与部署.md#部署)

### 原始模型
使用原始模型的web-ui可以查看[LLM推理文档](LLM推理文档.md#Web-UI)

### 微调后模型
```bash
# 直接使用app-ui
CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx'

# merge LoRA增量权重并使用app-ui
# 如果你需要量化, 可以指定`--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged'
```

## 推送模型
如果你想推送模型到ModelScope，可以参考[模型推送文档](LLM量化文档.md#推送模型)
