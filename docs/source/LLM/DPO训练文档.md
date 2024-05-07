# DPO训练文档
## 目录
- [环境准备](#环境准备)
- [人类对齐训练](#人类对齐训练)

## 环境准备
GPU设备: A10, 3090, V100, A100均可，如果是显存<=24G的GPU最少需要双卡环境。由于人类对齐训练在一张卡上加载两个模型，因此比微调的显存多占用一个推理模型的显存使用量。
```bash
# 设置pip全局镜像 (加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

# 环境对齐 (通常不需要运行. 如果你运行错误, 可以跑下面的代码, 仓库使用最新环境测试)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## 人类对齐训练
下面的shell脚本运行了一个人类对齐训练。首先需要切换到运行目录：

```shell
cd examples/pytorch/llm
```

运行下面的命令：

```shell
# Experimental environment: 4*A100
# Memory usage: 4 * 20G，双卡device_map * 2ddp
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

### sh脚本

sh脚本可以查看[这里](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/dpo)。

```bash
# 下面的脚本需要在此目录下执行
cd examples/pytorch/llm
```

**提示**:

- 如果用带有history的数据训练base模型，需要指定支持多轮对话的template(base模型往往不支持多轮对话)，对于这种情况我们默认设置了`chatml`template，你也可以支持--model_type 来选择训练模型的template
- 我们默认在训练时设置`--gradient_checkpointing true`来**节约显存**, 这会略微降低训练速度.
- 如果你使用的是**V100**等较老的GPU, 你需要设置`--dtype AUTO`或者`--dtype fp16`, 因为其不支持bf16.
- 如果你的机器是A100等高性能显卡, 且使用的是qwen系列模型, 推荐你安装[**flash-attn**](https://github.com/Dao-AILab/flash-attention), 这将会加快训练和推理的速度以及显存占用(A10, 3090, V100等显卡不支持flash-attn进行训练). 支持flash-attn的模型可以查看[LLM支持的模型](支持的模型和数据集.md#模型)
- 如果你需要断网进行训练, 请使用`--model_id_or_path <model_dir>`和设置`--check_model_is_latest false`. 具体参数含义请查看[命令行参数](命令行参数.md).
- 如果你想在训练时, 将权重push到ModelScope Hub中, 你需要设置`--push_to_hub true`.

```bash
# dpo训练 mistral-7b max_length=1024，bs=1
# 推荐的实验环境: V100, A10, 3090，2卡4卡或8卡
bash scripts/dpo/lora_ddp_mp/dpo.sh
bash scripts/dpo/lora_ddp_mp/infer.sh
```

由于DPO训练后会得到一个完整模型或者adapter的weights，因此LoRA合并、推理的步骤和微调步骤相同，因此请参考[微调文档](LLM微调文档.md#merge-lora)对应的步骤。
