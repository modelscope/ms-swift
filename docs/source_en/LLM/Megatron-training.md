# Megatron Training Documentation

Models that support training with Megatron can be found [here](../Instruction/Supported-models-datasets.md#models).

## Table of Contents
- [Environment Preparation](#Environment-Preparation)
- [SFT Example](#SFT-Example)
- [Multi-Node Pre-Training Example](#Multi-Node-Pre-Training-Example)
- [Mapping between MegatronArguments and SftArguments](#Mapping-between-MegatronArguments-and-SftArguments)


## Environment-Preparation

```shell
# Install ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

# Install Megatron-related dependencies (You do not need to install megatron-ml or other dependency libraries)
pip install pybind11
# transformer_engine (If the installation is unsuccessful, please try: release_v1.7)
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
# apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

The other two dependency libraries are [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch). They will be cloned and installed via swift, so no user installation is required. You can also specify the paths to the already downloaded repositories using the environment variables `MEGATRON_LM_PATH` and `PAI_MEGATRON_PATCH_PATH`.


## SFT-Example
Here we present a quick-start example of training with Megatron. Through this example, you can get familiar with the entire Megatron training workflow. For a corresponding example of fine-tuning using HF Trainer, please refer to [Self-cognition-best-practice](Self-cognition-best-practice.md).

1. Converting weights from HF format to Megatron format:
```shell
# Default output path: --megatron_output_dir {model_type}-tp{tp}-pp{pp}
CUDA_VISIBLE_DEVICES=0 swift export --model_type qwen2-7b-instruct \
    --to_megatron true --tp 2 --dtype bf16

# If using qwen2-72b-instruct, the conversion command is as follows:
CUDA_VISIBLE_DEVICES=0,1,2,3 swift export --model_type qwen2-72b-instruct \
    --to_megatron true --tp 8 --dtype bf16
```

2. Fine-tuning using Megatron format weights, the command script is as follows:
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

3. Converting weights from Megatron format back to HF format:
```shell
# Unfine-tuned model
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir qwen2-7b-instruct-tp2-pp1 --to_hf true

# fine-tuned model
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen2-7b-instruct-tp2-pp1/vx-xxx --to_hf true

# If using qwen2-72b-instruct, the conversion command is as follows:
CUDA_VISIBLE_DEVICES=0,1,2,3 swift export \
    --ckpt_dir qwen2-72b-instruct-tp8-pp1 --to_hf true
```

4. Perform inference testing on the obtained weights and accelerate using vLLM:
```shell
# Unfine-tuned model
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen2-7b-instruct \
    --model_id_or_path qwen2-7b-instruct-tp2-pp1/qwen2-7b-instruct-hf \

# fine-tuned model
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen2-7b-instruct-tp2-pp1/vx-xxx/qwen2-7b-instruct-hf
```

The performance of the fine-tuned model is as follows:
```python
"""
<<< 你是谁
我是小黄，由魔搭开发的人工智能聊天机器人。我的目标是通过文本交流提供帮助、信息和娱乐。如果您有任何问题或需要帮助，请随时向我提问。
--------------------------------------------------
<<< who are you
I am Xiao Huang, an artificial intelligence chatbot developed by ModelScope. My purpose is to provide assistance, information, and entertainment through text communication. If you have any questions or need help, please feel free to ask me at any time.
--------------------------------------------------
<<< What should I do if I can't sleep at night?
Lack of sleep at night can be caused by various factors, such as stress, anxiety, irregular sleep patterns, caffeine or alcohol consumption, or an uncomfortable sleep environment. Here are some suggestions that may help improve your sleep quality:

1. Establish a regular sleep schedule: Try to go to bed and wake up at the same time every day, even on weekends. This helps adjust your body clock and improve your sleep quality.
2. Create a comfortable sleep environment: Ensure that your bedroom is quiet, dark, and cool, and that your bed is comfortable. Use blackout curtains, earplugs, or white noise machines to create a more comfortable sleep environment.
3. Avoid caffeine and alcohol: Avoid consuming caffeine and alcohol in the hours leading up to bedtime, as they can affect your sleep quality.
4. Relax your mind and body: Try deep breathing, meditation, yoga, or other relaxation techniques to help you relax and prepare for sleep.
5. Avoid using electronic devices: Avoid using electronic devices before bedtime, as the blue light emitted by screens can affect your sleep quality.
6. Avoid napping during the day: If you take naps during the day, it may affect your sleep quality at night. Try to avoid napping for several hours before bedtime.
7. Limit your fluid intake before bedtime: Avoid drinking too much liquid before bedtime to reduce the number of times you need to get up to use the bathroom.
8. Maintain a positive mindset: Avoid worrying or being anxious before bedtime, as this can affect your sleep quality. Try to think positively about the next day.
9. Try relaxation techniques: Try deep breathing, meditation, yoga, or other relaxation techniques to help you relax and prepare for sleep.
10. If you have tried the above suggestions but still cannot sleep, consider consulting a doctor or sleep expert for more advice.
"""
```

We evaluate the trained HF model:
```shell
pip install llmuses==0.4.0
# Original model
CUDA_VISIBLE_DEVICES=0 swift eval --model_type qwen2-7b-instruct \
    --eval_dataset ceval mmlu gsm8k arc --eval_backend Native

# Unfine-tuned model
CUDA_VISIBLE_DEVICES=0 swift eval --model_type qwen2-7b-instruct \
    --model_id_or_path qwen2-7b-instruct-tp2-pp1/qwen2-7b-instruct-hf \
    --eval_dataset ceval mmlu gsm8k arc --eval_backend Native

# fine-tuned model
CUDA_VISIBLE_DEVICES=0 swift eval \
    --ckpt_dir output/qwen2-7b-instruct-tp2-pp1/vx-xxx/qwen2-7b-instruct-hf \
    --eval_dataset ceval mmlu gsm8k arc --eval_backend Native
```


Evaluation results:
|     |  ceval    | mmlu   | gsm8k    | arc   |
| ---- | ---- | ---- | ---- | ---- |
|  Original Model  |    0.6642  |  0.6909    |    0.787  |  0.8507    |
|  Unfine-tuned  |    0.6642  |  0.6909    |    0.787  |  0.8507    |
|  Fine-tuned  |   0.7392   |    0.6878  |  0.8241    |    0.8481  |


**Multi-Node**:
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


**Alibaba Cloud DLC Multi-Node Training** (No need to modify the wildcard):
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


## Multi-Node Pre-Training Example
Comming soon...


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
