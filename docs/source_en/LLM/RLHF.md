
# LLM Human Alignment Training Documentation
## Table of Contents
- [Environment Preparation](#environment-preparation)
- [Human Alignment Training](#human-alignment-training)

## Environment Preparation
GPU devices: A10, 3090, V100, A100 are all acceptable. For GPUs with memory <=24G, at least a dual-card environment is required. Since human alignment training loads two models on one card, it occupies more memory than fine-tuning due to an additional inference model's memory consumption.
```bash
# Install ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

# Environment alignment (usually not necessary. If you encounter errors, you can run the following code, the repository uses the latest environment for testing)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## Human Alignment Training
The following shell script runs a human alignment training. First, you need to switch to the runtime directory:

```shell
cd examples/pytorch/llm
```

Run the following command:

```shell
# Experimental environment: 4*A100
# Memory usage: 4 * 20G, dual-card device_map * 2ddp
nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
swift dpo \
    --model_type  yi-6b-chat \
    --ref_model_type  yi-6b-chat \
    --model_revision  master \
    --sft_type  lora \
    --tuner_backend  swift \
    --dtype  AUTO  \
    --output_dir  output  \
    --dataset  hh-rlhf-cn:harmless_base_cn  \
    --num_train_epochs  3  \
    --max_length  1024  \
    --max_prompt_length  512  \
    --check_dataset_strategy  none  \
    --lora_rank  8  \
    --lora_alpha  32  \
    --lora_dropout_p  0.05  \
    --lora_target_modules  ALL  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --weight_decay  0.1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  $(expr 16 / $nproc_per_node)  \
    --max_grad_norm  1.0  \
    --warmup_ratio  0.03  \
    --eval_steps  2000  \
    --save_steps  2000  \
    --save_total_limit  2  \
    --logging_steps  10 \
```

### Shell Script

The sh script can be viewed [here](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/dpo).

```bash
# The following script needs to be executed in this directory
cd examples/pytorch/llm
```

**Tips**:

- We default to setting `--gradient_checkpointing true` during training to **save memory**, which will slightly reduce training speed.
- If you are using older GPUs such as **V100**, you need to set `--dtype AUTO` or `--dtype fp16`, because they do not support bf16.
- If your machine has high-performance graphics cards like A100 and you are using the qwen series models, we recommend installing [**flash-attn**](https://github.com/Dao-AILab/flash-attention), which will speed up training and inference as well as reduce memory usage (A10, 3090, V100, etc. graphics cards do not support training with flash-attn). Models that support flash-attn can be viewed in [LLM Supported Models](Supported-models-datasets.md#models)
- If you need to train offline, please use `--model_id_or_path <model_dir>` and set `--check_model_is_latest false`. For specific parameter meanings, please see [Command Line Arguments](Command-line-parameters.md).
- If you want to push weights to the ModelScope Hub during training, you need to set `--push_to_hub true`.

```bash
# dpo training for mistral-7b max_length=1024, bs=1
# Recommended experimental environment: V100, A10, 3090, 2 cards, 4 cards or 8 cards
bash scripts/dpo/lora_ddp_mp/dpo.sh
bash scripts/dpo/lora_ddp_mp/infer.sh
```

Since DPO training will result in a complete model or adapter weights, the steps for LoRA merging and inference are the same as for fine-tuning, so please refer to the corresponding steps in the [Fine-tuning Documentation](LLM-fine-tuning.md#merge-lora).
