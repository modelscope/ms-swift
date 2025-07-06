# Qwen3最佳实践

讨论区：[issue 4030](https://github.com/modelscope/ms-swift/issues/4030)

Qwen文档: [https://qwen.readthedocs.io/en/latest/training/ms_swift.html](https://qwen.readthedocs.io/en/latest/training/ms_swift.html)

## 推理

思考模式：
```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3-8B \
    --infer_backend vllm \
    --stream true \
    --max_new_tokens 2048 \
    --max_model_len 8192
```

```text
<<< who are you?
<think>
Okay, the user is asking "who are you?" Let me start by introducing myself as Qwen, the large language model developed by Alibaba Cloud. I should mention my capabilities, like answering questions, creating content, and engaging in conversations. But I need to keep it concise. Also, the user might want to know how I can assist them. Maybe I should ask how I can help them today. Let me check if there's anything else important to include. Oh, I should make sure the tone is friendly and approachable. Alright, that should cover it.
</think>

Hello! I am Qwen, a large language model developed by Alibaba Cloud. I can assist with a wide range of tasks, such as answering questions, creating content, writing stories, coding, and more. How can I help you today? 😊
<<< clear
<<< who are you? /no_think
<think>

</think>

I am Qwen, a large language model developed by Alibaba Cloud. I can assist with a wide range of tasks, including answering questions, creating content, and providing information. How can I help you today?
```

非思考模式：
- 其中`--response_prefix`代表模型的输出会在其前缀后继续生成。等价于enable_thinking设置为False。
```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3-8B \
    --infer_backend vllm \
    --stream true \
    --max_new_tokens 2048 \
    --max_model_len 8192 \
    --response_prefix '<think>\n\n</think>\n\n'
```

```text
<<< who are you?
<think>

</think>

I am Qwen, a large-scale language model developed by Alibaba Cloud. I am designed to assist with a wide range of tasks, including answering questions, creating content, and providing information. How can I assist you today?
```

## 训练

在开始训练之前，请确保您的环境已正确配置。

```bash
pip install ms-swift -U
pip install transformers

pip install deepspeed # 多GPU训练
pip install liger-kernel # 节约显存资源
pip install flash-attn --no-build-isolation  # packing需要
```

## 监督微调 (SFT)

### 数据准备

使用 ms-swift 进行 SFT 的自定义数据集格式如下（system 字段是可选的）。您可以将其组织为 JSON、JSONL 或 CSV 格式。在训练脚本中指定 `--dataset <dataset_path>`。有关完整的数据集格式指南，请参考[自定义数据集文档](../Customization/自定义数据集.md)。

```text
# 通用格式
{"messages": [
    {"role": "system", "content": "<system-prompt>"},
    {"role": "user", "content": "<query1>"},
    {"role": "assistant", "content": "<response1>"}
]}
# 带think的格式
{"messages": [
    {"role": "user", "content": "Where is the capital of Zhejiang?"},
    {"role": "assistant", "content": "<think>\n...\n</think>\n\nThe capital of Zhejiang is Hangzhou."}
]}
```

如果您想使用不含思维链的数据进行训练，同时保留模型的推理能力，可以通过以下两种方法尽量减少微调的影响：

**选项 1**：【推荐】在训练期间，指定 `--loss_scale ignore_empty_think`，以忽略对 `<think>\n\n</think>\n\n` 的损失计算，从而避免推理能力的丧失。训练脚本参考[这里](https://github.com/modelscope/ms-swift/blob/main/examples/train/think_model/qwen3_demo1.sh)。该方式同样适用于deepseek-r1等模型。自定义数据集格式如下：

```json
{"messages": [
    {"role": "user", "content": "Where is the capital of Zhejiang?"},
    {"role": "assistant", "content": "<think>\n\n</think>\n\nThe capital of Zhejiang is Hangzhou."}
]}
```

**选项 2**：在数据集的查询中添加 `/no_think`，以避免推理能力的丧失。训练脚本请参考[这里](https://github.com/modelscope/ms-swift/blob/main/examples/train/think_model/qwen3_demo2.sh)。自定义数据集格式如下：

```json
{"messages": [
    {"role": "user", "content": "Where is the capital of Zhejiang? /no_think"},
    {"role": "assistant", "content": "<think>\n\n</think>\n\nThe capital of Zhejiang is Hangzhou."}
]}
```

你可以使用以下命令获取蒸馏的推理数据集，在训练时，与不含思维链数据集进行混合，进一步缓解推理能力的丧失：
- 其中`--val_dataset`的选择任意。推理产生的`result_path`，可以直接在训练时指定`--dataset distill_dataset.jsonl`使用。
- 该思路同样适用于其他推理模型，例如deepseek-r1。
```shell
# 4 * 80GiB
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --model Qwen/Qwen3-32B \
    --infer_backend vllm \
    --val_dataset 'AI-ModelScope/alpaca-gpt4-data-en#5000' 'AI-ModelScope/alpaca-gpt4-data-zh#5000' \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 2 \
    --max_model_len 8192 \
    --max_new_tokens 4096 \
    --write_batch_size 1000 \
    --result_path distill_dataset.jsonl
```

### 30分钟自我认知微调

本节将介绍30分钟对 Qwen3-8B 进行自我认知微调。所需GPU显存为 22GB，可以在 ModelScope 提供的[免费算力](https://modelscope.cn/my/mynotebook) A10 中运行。

训练后，模型将不再认为自己是由“阿里云”训练的“Qwen”，而是由“swift”训练的“swift-robot”。

如果需要在离线环境下进行训练，可以手动下载模型和数据集，并指定 `--model <model-path>` 和 `--dataset <dataset-dir>`。数据集可以在 [Modelscope Hub](https://modelscope.cn/datasets/swift/self-cognition)上找到。对`swift/self-cognition`数据集的预处理函数可以查看[这里](https://github.com/modelscope/ms-swift/blob/36fdf381e5e88cb8a71c9d69c1d8936a989318cc/swift/llm/dataset/dataset/llm.py#L882)。

关于训练脚本中各参数的含义，请参考[命令行参数文档](../Instruction/命令行参数.md)。

```bash
# 显存占用：22GB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3-8B \
    --train_type lora \
    --dataset 'swift/Qwen3-SFT-Mixin#2000' \
              'swift/self-cognition:qwen3#600' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
```

微调完成后，可以使用以下脚本来测试微调结果。注意，`--adapters` 部分需要修改为最后保存检查点的目录路径：

```bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

```text
<<< who are you?
<think>
Okay, the user asked, "who are you?" I need to introduce myself. Let me start by stating my name, swift-robot. Then, I should mention that I'm an AI assistant developed by swift. I should explain my purpose, which is to provide information and assistance. I should also highlight my capabilities, like answering questions, generating text, and engaging in conversation. It's important to keep the tone friendly and approachable. Maybe add something about being here to help and encourage the user to ask anything. Let me check if I covered all the key points: name, developer, purpose, capabilities, and a welcoming statement. Yeah, that should do it. Now, let me put that into a concise and friendly response.
</think>

Hello! I am swift-robot, an artificial intelligence assistant developed by swift. My purpose is to provide information and assistance to users like you. I can answer questions, generate text, and engage in conversations on a wide range of topics. I am here to help, so feel free to ask me anything you need!
```

默认情况下，ms-swift 会使用 ModelScope 社区下载模型和数据集。如果想使用 HuggingFace 社区，则需要额外指定 `--use_hf true`。

合并 LoRA 权重：

```shell
swift export \
    --adapters output/checkpoint-xxx \
    --merge_lora true
```

推送模型到 ModelScope/HuggingFace：

```bash
# 如果是推送完整的权重，需要修改`--adapters`为`--model`.
# Modelscope的hub_token可以在这里找到: https://modelscope.cn/my/myaccesstoken
swift export \
    --adapters output/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<hub-model-id>' \
    --hub_token '<hub-token>' \
    --use_hf false
```

如果要使用多 GPU 进行训练，以下提供了多 GPU 训练的示例：

```bash
# 4 * 60GB
# 你可以通过设置`--dataset AI-ModelScope/alpaca-gpt4-data-en`跑通实验
# 注意：如果你指定了`--packing true`, 你必须额外设置`--attn_impl flash_attn`

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen3-8B \
    --train_type full \
    --dataset '<your-dataset>' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --packing true \
    --eval_steps 100 \
    --save_steps 100 \
    --logging_steps 5 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir output \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --attn_impl flash_attn
```

## 强化学习 (RL)

ms-swift 支持 DPO、GRPO、DAPO、PPO、KTO、GKD 等 RLHF 方法。本章将着重介绍使用 ms-swift 对 Qwen3-8B 进行 GRPO 训练。更多关于GRPO的内容，可以参考[GRPO文档](../Instruction/GRPO/GetStarted/GRPO.md)。更多RLHF训练脚本，参考[examples/train/rlhf](https://github.com/modelscope/ms-swift/tree/main/examples/train/rlhf)。

### 环境设置

除了安装上述介绍的 ms-swift 相关依赖项外，还需要安装以下依赖项：
```
pip install "math_verify==0.5.2"
pip install vllm==0.8.5.post1
```

### 数据准备

使用 ms-swift 进行 GRPO 训练的数据集格式与 SFT 类似，但不需要最后一轮的 assistant 部分。如果使用 accuracy 作为奖励，则需要额外的 `solution` 列来计算准确率。

示例数据集格式：

```jsonl
{"messages": [{"role": "user", "content": "Tell me tomorrow's weather"}]}
{"messages": [{"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}]}
{"messages": [{"role": "user", "content": "What is your name?"}]}
```

关于其他 RLHF 算法的数据集准备，请参考[自定义数据集文档](../Customization/自定义数据集.md#rlhf)。

数据集要求的注意事项：

- **奖励函数计算**：数据集格式取决于所使用的奖励函数。可能需要额外的列来支持特定的奖励计算。例如：

  - 当使用内置的 accuracy 或 cosine 奖励时，数据集必须包含一个 `solution` 列以计算回复的准确性。
  - 数据集中的其他列将作为 ``**kwargs`` 传递给奖励函数以实现进一步的自定义。

- **自定义奖励函数**：为了根据您的具体需求调整奖励函数，可以参考链接：[外部奖励插件](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin)。该插件提供了实现自定义奖励函数的示例和模板。

我们使用使 AI-MO/NuminaMath-TIR 作为数据集，并使用accuracy函数计算模型回答的准确率奖励。

在训练过程中，使用 vLLM 加速采样过程。

```bash
# 70G*8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-8B \
    --train_type full \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-6 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_length 4096 \
    --max_completion_length 4096 \
    --vllm_max_model_len 8192 \
    --reward_funcs accuracy \
    --num_generations 16 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.4 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --deepspeed zero3 \
    --tensor_parallel_size 1 \
    --temperature 1.0 \
    --top_p 0.85 \
    --log_completions true \
    --overlong_filter true
```

## Megatron-SWIFT

ms-swift 引入了 Megatron 并行技术以加速大模型的CPT/SFT/DPO。支持的模型可以在[支持的模型文档](../Instruction/支持的模型和数据集.md)中找到。

关于环境准备以及 HF 和 MCore 模型权重的转换，可以参考[Megatron-SWIFT训练文档](../Instruction/Megatron-SWIFT训练.md)。

我们将使用阿里云 DLC 启动训练。训练环境由2台配备8卡 80GiB A800 GPU 组成。关于多节点启动方法的更多信息，请参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node)。

```bash
# https://help.aliyun.com/zh/pai/user-guide/general-environment-variables
# 请确保两个节点上的权重保存路径`--save`和packing缓存路径`--packing_cache`相同且共享。
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
megatron sft \
    --load Qwen3-30B-A3B-Base-mcore \
    --dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT' \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --packing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --train_iters 2000 \
    --eval_iters 50 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-30B-A3B-Base \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 8192 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --use_flash_attn true
```

训练loss图（部分）：

<img width="910" alt="Image" src="https://github.com/user-attachments/assets/9fe393aa-8299-4659-aa2f-be5d44f0730b" />

效果截图：

<img width="1066" alt="Image" src="https://github.com/user-attachments/assets/1a924130-1954-43e9-9093-b019aeef5949" />


自定义数据集格式与`swift sft`相同，详见之前章节。只需指定 `--dataset <dataset_path>` 即可。

使用 `megatron sft` 和 `swift sft` 在对 Qwen3-30B-A3B 模型进行全参数微调的训练速度和 GPU 显存使用对比情况如下：

|          | Megatron-LM | DeepSpeed-ZeRO2 | DeepSpeed-ZeRO3 |
| -------- | ----------- | --------------- | --------------- |
| 训练速度 | 9.6s/it     | -               | 91.2s/it        |
| 显存使用 | 16 * 60GiB  | OOM             | 16 * 80GiB      |
