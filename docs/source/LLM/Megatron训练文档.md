# Megatron训练文档

支持使用megatron进行训练的模型可以查看[这里](支持的模型和数据集.md#模型)

## 目录
- [环境准备](#环境准备)
- [SFT案例](#SFT案例)
- [多机预训练案例](#多机预训练案例)
- [MegatronArguments与SftArguments的映射](#MegatronArguments与SftArguments的映射)


## 环境准备

```shell
# 设置pip全局镜像 (加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

# 安装megatron相关依赖 (你不需要安装megatron-ml等其他依赖库)
pip install pybind11
# transformer_engine (如果安装不成功请尝试: release_v1.7)
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
# apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

其他两个依赖库为[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)和[Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch). 会由swift进行git clone并安装, 不需要用户进行安装. 你也可以通过环境变量`MEGATRON_LM_PATH`, `PAI_MEGATRON_PATCH_PATH`指定已经下载好的repo路径.


## SFT案例
这里介绍可以很快跑通的使用megatron训练的案例，通过此案例，你可以熟悉magatron训练的全流程。使用HF Trainer进行微调的对应案例可以查看[自我认知微调最佳实践](自我认知微调最佳实践.md).

1. HF格式的权重转成megatron格式的权重:
```shell
# 默认输出路径: --megatron_output_dir {model_type}-tp{tp}-pp{pp}
CUDA_VISIBLE_DEVICES=0 swift export --model_type qwen2-7b-instruct \
    --to_megatron true --tp 2 --dtype bf16

# 如果使用qwen2-72b-instruct，转换命令如下:
CUDA_VISIBLE_DEVICES=0,1,2,3 swift export --model_type qwen2-72b-instruct \
    --to_megatron true --tp 8 --dtype bf16
```

2. 使用megatron格式权重进行微调，命令脚本如下:
```shell
# Experimental Environment: 4 * A100
# GPU Memory Requirement: 4 * 55GB
# TP=2, DP=2
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
    --resume_from_checkpoint qwen2-7b-instruct-tp2-pp1 \
    --dataset swift-mix:sharegpt#500 swift-mix:codefuse#250 swift-mix:metamathqa#250 self-cognition#500 \
    --max_length 2048 \
    --learning_rate 2e-6 \
    --output_dir output \
    --model_name 小黄 'Xiao Huang' \
    --model_author 魔搭 ModelScope \
    --train_backend megatron
```

3. 将megatron格式权重重新转成HF格式:
```shell
# 未微调模型
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir qwen2-7b-instruct-tp2-pp1 --to_hf true

# 微调后模型
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen2-7b-instruct-tp2-pp1/vx-xxx --to_hf true

# 如果使用qwen2-72b-instruct，转换命令如下:
CUDA_VISIBLE_DEVICES=0,1,2,3 swift export \
    --ckpt_dir qwen2-72b-instruct-tp8-pp1 --to_hf true
```

4. 对获得的权重进行推理测试，并使用vLLM进行加速:
```shell
# 未微调模型
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen2-7b-instruct \
    --model_id_or_path qwen2-7b-instruct-tp2-pp1/qwen2-7b-instruct-hf \

# 微调后模型
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen2-7b-instruct-tp2-pp1/vx-xxx/qwen2-7b-instruct-hf
```

微调后模型效果如下：
```python
"""
<<< 你是谁
我是小黄，由魔搭开发的人工智能聊天机器人。我的目标是通过文本交流提供帮助、信息和娱乐。如果您有任何问题或需要帮助，请随时向我提问。
--------------------------------------------------
<<< who are you
I am Xiao Huang, an artificial intelligence chatbot developed by ModelScope. My purpose is to provide assistance, information, and entertainment through text communication. If you have any questions or need help, please feel free to ask me at any time.
--------------------------------------------------
<<< 晚上睡不着觉怎么办
晚上睡不着觉可能是因为多种原因，例如压力、焦虑、不规律的作息时间、咖啡因摄入过多、睡眠环境不佳等。以下是一些可能有助于改善睡眠质量的建议：

1. 建立规律的作息时间：每天尽量在同一时间上床睡觉和起床，即使在周末也是如此。这有助于调整您的生物钟并改善睡眠质量。
2. 创造舒适的睡眠环境：确保您的卧室安静、黑暗、凉爽，并且床铺舒适。使用遮光窗帘、耳塞或白噪音机等设备可以帮助创造一个更舒适的睡眠环境。
3. 避免咖啡因和酒精：避免在睡前几小时内摄入咖啡因和酒精，因为它们可能会影响您的睡眠质量。
4. 放松身心：尝试进行深呼吸、冥想、瑜伽或其他放松技巧，以帮助您放松身心并准备入睡。
5. 避免使用电子设备：在睡前避免使用电子设备，因为屏幕发出的蓝光可能会影响您的睡眠质量。
6. 避免午睡：如果您在白天打盹，可能会影响您晚上的睡眠质量。尽量避免在晚上睡觉前几小时内打盹。
7. 限制晚上摄入的液体：在睡前几小时内避免摄入过多的液体，以减少夜间起床上厕所的次数。
8. 保持积极的心态：避免在睡前担心或焦虑，因为这可能会影响您的睡眠质量。尝试进行积极的思考，例如思考您期待的第二天的事情。
9. 尝试放松技巧：尝试进行深呼吸、冥想、瑜伽或其他放松技巧，以帮助您放松身心并准备入睡。
10. 如果您尝试了上述建议但仍然无法入睡，请考虑咨询医生或睡眠专家以获取更多建议。
"""
```

我们对训练完的HF模型进行评测：
```shell
pip install llmuses==0.4.0
# 原始模型
CUDA_VISIBLE_DEVICES=0 swift eval --model_type qwen2-7b-instruct \
    --eval_dataset ceval mmlu gsm8k arc --eval_backend Native

# 未微调模型
CUDA_VISIBLE_DEVICES=0 swift eval --model_type qwen2-7b-instruct \
    --model_id_or_path qwen2-7b-instruct-tp2-pp1/qwen2-7b-instruct-hf \
    --eval_dataset ceval mmlu gsm8k arc --eval_backend Native

# 微调后模型
CUDA_VISIBLE_DEVICES=0 swift eval \
    --ckpt_dir output/qwen2-7b-instruct-tp2-pp1/vx-xxx/qwen2-7b-instruct-hf \
    --eval_dataset ceval mmlu gsm8k arc --eval_backend Native
```

评测结果：
|     |  ceval    | mmlu   | gsm8k    | arc   |
| ---- | ---- | ---- | ---- | ---- |
|  原始模型  |    0.6642  |  0.6909    |    0.787  |  0.8507    |
|  未微调  |    0.6642  |  0.6909    |    0.787  |  0.8507    |
|  微调后  |   0.7392   |    0.6878  |  0.8241    |    0.8481  |


**多机微调**：
```shell
# node0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
NPROC_PER_NODE=8 \
swift sft \
    --resume_from_checkpoint qwen2-7b-instruct-tp2-pp1 \
    --dataset swift-mix:sharegpt#20000 swift-mix:codefuse#10000 swift-mix:metamathqa#10000 self-cognition#500 \
    --max_length 8192 \
    --learning_rate 2e-6 \
    --sft_type full \
    --output_dir output \
    --model_name 小黄 'Xiao Huang' \
    --model_author 魔搭 ModelScope \
    --train_backend megatron

# node1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=xxx.xxx.xxx.xxx \
NPROC_PER_NODE=8 \
swift sft \
    --resume_from_checkpoint qwen2-7b-instruct-tp2-pp1 \
    --dataset swift-mix:sharegpt#20000 swift-mix:codefuse#10000 swift-mix:metamathqa#10000 self-cognition#500 \
    --max_length 8192 \
    --learning_rate 2e-6 \
    --sft_type full \
    --output_dir output \
    --model_name 小黄 'Xiao Huang' \
    --model_author 魔搭 ModelScope \
    --train_backend megatron
```

**阿里云-DLC多机训练**（通配符不用改）:
```shell
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
swift sft \
    --resume_from_checkpoint qwen2-7b-instruct-tp2-pp1 \
    --dataset swift-mix:sharegpt#20000 swift-mix:codefuse#10000 swift-mix:metamathqa#10000 self-cognition#500 \
    --max_length 8192 \
    --learning_rate 2e-6 \
    --sft_type full \
    --output_dir output \
    --model_name 小黄 'Xiao Huang' \
    --model_author 魔搭 ModelScope \
    --train_backend megatron
```


## 多机预训练案例
敬请期待...


## MegatronArguments与SftArguments的映射
|  MegatronArguments    |  SftArguments |
| ---- | ---- |
|   optimizer   | optim |
|   lr_decay_style   | lr_scheduler_type |
|  weight_decay  | weight_decay |
| clip_grad   |  max_grad_norm |
|   adam_beta1 | adam_beta1 |
|  adam_beta2  | adam_beta2 |
| adam_eps  | adam_epsilon |
|  lr  | learning_rate |
|  min_lr  | min_lr |
|   fp16<br> apply_query_key_layer_scaling | fp16 |
|  bf16  | bf16 |
|  tensor_model_parallel_size  | tp |
|  pipeline_model_parallel_size  | pp |
|  seed  | seed |
|  load  | resume_from_checkpoint |
|  save  | output_dir |
|  tensorboard_dir  | logging_dir |
|  log_interval  | logging_steps |
|  eval_interval  | eval_steps |
|  save_interval  | save_steps |
|  micro_batch_size  | batch_size |
|  global_batch_size  | batch_size * gradient_accumulation_steps * world_size |
|  sequence_parallel  | sequence_parallel |
|  num_workers  | dataloader_num_workers |
|  use_flash_attn  | use_flash_attn |
|  train_iters  | int(math.ceil(len(train_dataset) * num_train_epochs / global_batch_size)) |
|  eval_iters  | int(math.ceil(len(val_dataset) / global_batch_size)) |
|  lr_warmup_iters  |  warmup_steps if warmup_steps > 0 else math.ceil(train_iters * warmup_ratio) |
|  no_save_optim<br>no_save_rng  | save_only_model |
