# Megatron Training Documentation (Beta)

## Table of Contents
- [Environment Preparation](#Environment-Preparation)
- [SFT Example](#SFT-Example)
- [Mapping between MegatronArguments and SftArguments](#Mapping-between-MegatronArguments-and-SftArguments)


## Environment-Preparation

```shell
# Install ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

# Install Megatron-related dependencies (You do not need to install megatron-ml or other dependency libraries)
# transformer_engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
# apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```


## SFT-Example
Here we present a quick-start example of training with Megatron. Through this example, you can get familiar with the entire Megatron training workflow. For a corresponding example of fine-tuning using HF Trainer, please refer to [Self-cognition-best-practice](Self-cognition-best-practice.md).

1. Converting weights from HF format to Megatron format:
```shell
# Default output path: --megatron_output_dir {model_type}-tp{tp}-pp{pp}
CUDA_VISIBLE_DEVICES=0 swift export --model_type qwen2-7b-instruct --to_megatron true --tp 2 --dtype bf16
```

2. Fine-tuning using Megatron format weights, the command script is as follows:
```shell
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

3. Converting weights from Megatron format back to HF format:
```shell
# Unfine-tuned model
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir qwen2-7b-instruct-tp2-pp1 --to_hf true

# fine-tuned model
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen2-7b-instruct-tp2-pp1/v7-20240723-195011 --to_hf true
```

4. Perform inference testing on the obtained weights and accelerate using vLLM:
```shell
# Unfine-tuned model
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen2-7b-instruct \
    --model_id_or_path qwen2-7b-instruct-tp2-pp1/qwen2-7b-instruct-hf \

# fine-tuned model
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen2-7b-instruct-tp2-pp1/v7-20240723-195011/qwen2-7b-instruct-hf
```

The performance of the fine-tuned model is as follows:
```python
"""
<<< 你是谁
我是一个名为小黄的人工智能，由魔搭开发。我被设计成能够理解和生成自然语言文本，以便更好地与人类进行交流并回答问题。请问有什么我可以帮助您的吗？
--------------------------------------------------
<<< who are you
I am an artificial intelligence named Xiao Huang, developed by ModelScope. I am designed to understand and generate natural language text in order to better communicate with humans and answer their questions. How can I assist you?
--------------------------------------------------
<<< 晚上睡不着觉怎么办
如果您晚上睡不着觉，可以尝试以下方法来帮助您入睡：

1. 保持规律的作息时间，每天按时上床睡觉和起床。
2. 避免在睡前使用电子设备，如手机、电脑和电视，因为这些设备发出的蓝光会抑制褪黑素的分泌，影响睡眠。
3. 在睡前进行放松活动，如深呼吸、冥想或听轻音乐。
4. 保持卧室安静、黑暗和凉爽。
5. 避免在睡前摄入咖啡因和酒精。
6. 尝试进行轻度锻炼，如散步或瑜伽，但避免在睡前进行剧烈运动。
7. 如果以上方法都无效，可以考虑咨询医生或睡眠专家，以获取更专业的建议。

希望这些建议能帮助您改善睡眠质量。
"""
```

We evaluate the trained HF model:
```shell
pip install llmuses==0.4.0
# Original model
CUDA_VISIBLE_DEVICES=0 swift eval --model_type qwen2-7b-instruct \
    --eval_dataset ceval mmlu gsm8k arc --eval_backend Native

# Unfine-tuned model
CUDA_VISIBLE_DEVICES=1 swift eval --model_type qwen2-7b-instruct --model_id_or_path /mnt/nas2/huangjintao.hjt/work/swift/qwen2-7b-instruct-tp2-pp1/qwen2-7b-instruct-hf \
    --eval_dataset ceval mmlu gsm8k arc --eval_backend Native

# fine-tuned model
CUDA_VISIBLE_DEVICES=1 swift eval --ckpt_dir /mnt/nas2/huangjintao.hjt/work/swift/output/qwen2-7b-instruct-tp2-pp1/v7-20240723-195011/qwen2-7b-instruct-hf \
    --eval_dataset ceval mmlu gsm8k arc --eval_backend Native
```




## Mapping between MegatronArguments and SftArguments
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

