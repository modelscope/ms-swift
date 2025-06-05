# Qwen3 Best Practice

讨论区：[https://github.com/modelscope/ms-swift/issues/4030](https://github.com/modelscope/ms-swift/issues/4030)

**中文版 notebook**: [https://modelscope.cn/notebook/share/ipynb/d4d8765f/qwen3.ipynb](https://modelscope.cn/notebook/share/ipynb/d4d8765f/qwen3.ipynb)

Qwen文档: [https://qwen.readthedocs.io/en/latest/training/ms_swift.html](https://qwen.readthedocs.io/en/latest/training/ms_swift.html)


ModelScope SWIFT (**ms-swift**) 是由 `ModelScope 社区 <https://modelscope.cn/>`__ 提供的大模型和多模态大模型训练部署框架。

GitHub 仓库：`ms-swift <https://github.com/modelscope/ms-swift>`__

使用 ms-swift 训练大语言模型的特性：

- **模型类型**：支持 500+ 纯文本大模型和 200+ 多模态大模型，覆盖从训练到部署全流程。
- **硬件支持**：兼容 CPU、RTX 系列 GPU、T4/V100、A10/A100/H100、Ascend NPU、MPS 等。
- **训练方法**：支持全参数微调、LoRA、QLoRA、DoRA 等技术。
- **分布式训练**：支持分布式训练技术，如 DDP、device_map、DeepSpeed ZeRO-2/ZeRO-3、FSDP，并集成 Megatron 的并行技术，包括张量并行、流水线并行、序列并行和专家并行。
- **RLHF 训练**：支持纯文本和多模态大模型的人类对齐方法，如 DPO、GRPO、DAPO、RM、PPO、KTO 等。

本文将介绍可运行的训练示例，并提供自定义数据集的格式。内容包括如何使用 ms-swift 对 Qwen3-8B 进行 SFT 和 GRPO，以及使用 Megatron-SWIFT（ms-swift 集成的 Megatron-LM）对 Qwen3-30B-A3B 进行 SFT。通过专家并行技术，MoE 模型的训练速度可以提升近 10 倍。

在开始微调之前，请确保您的环境已正确配置。

```bash
pip install ms-swift -U
# Install from source
pip install git+https://github.com/modelscope/ms-swift.git

pip install transformers -U

# Optional packages
pip install deepspeed # multi-GPU training
pip install liger-kernel # save GPU memory resources
pip install flash-attn --no-build-isolation
```

## 监督微调 (SFT)

### 数据准备

使用 ms-swift 进行 SFT 的自定义数据集格式如下（system 字段是可选的）。您可以将其组织为 JSON、JSONL 或 CSV 格式。在训练脚本中指定 ``--dataset <dataset_path>``。

有关完整的数据集格式指南，请参考：`自定义数据集文档 <https://swift.readthedocs.io/zh-cn/latest/Customization/%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86.html>`__

```json
# General format
{"messages": [
    {"role": "system", "content": "<system-prompt>"},
    {"role": "user", "content": "<query1>"},
    {"role": "assistant", "content": "<response1>"}
]}
# Format with think
{"messages": [
    {"role": "user", "content": "Where is the capital of Zhejiang?"},
    {"role": "assistant", "content": "<think>\n...\n</think>\n\nThe capital of Zhejiang is Hangzhou."}
]}
```

如果您想使用不含思维链的数据进行训练，同时保留模型的推理能力，可以通过以下两种方法尽量减少微调期间的干扰：

**选项 1**：在训练期间，指定 ``--loss_scale ignore_empty_think``，以忽略对 ``<think>\\n\\n</think>\\n\\n`` 的损失计算，从而避免推理能力的丧失。训练脚本请参考 `这里 <https://github.com/modelscope/ms-swift/blob/main/examples/train/think_model/qwen3_demo1.sh>`__。自定义数据集格式如下：

```json
{"messages": [
    {"role": "user", "content": "Where is the capital of Zhejiang?"},
    {"role": "assistant", "content": "<think>\n\n</think>\n\nThe capital of Zhejiang is Hangzhou."}
]}
```

**选项 2**：在数据集的查询中添加 ``/no_think``，以避免推理能力的丧失。训练脚本请参考 `这里 <https://github.com/modelscope/ms-swift/blob/main/examples/train/think_model/qwen3_demo2.sh>`__。自定义数据集格式如下：

```
{"messages": [
    {"role": "user", "content": "Where is the capital of Zhejiang? /no_think"},
    {"role": "assistant", "content": "<think>\n\n</think>\n\nThe capital of Zhejiang is Hangzhou."}
]}
```

### 30分钟自我认知微调

本节将介绍30分钟对 Qwen3-8B 进行自我认知微调。所需GPU显存为 22GB，可以在 ModelScope 提供的 `免费算力 <https://modelscope.cn/my/mynotebook>`__ A10 中运行。

训练后，模型将不再认为自己是由“阿里云”训练的“Qwen”，而是由“swift”训练的“swift-robot”。

如果需要在离线环境下进行训练，可以手动下载模型和数据集，并指定 ``--model <model-path>`` 和 ``--dataset <dataset-dir>``。数据集可以在 `Modelscope Hub <https://modelscope.cn/datasets/swift/self-cognition>`__ 上找到。

For the meaning of each parameter in the training script, please refer to the `Command-line 关于训练脚本中各参数的含义，请参考 `命令行参数文档 <https://swift.readthedocs.io/zh-cn/latest/Instruction/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.html>`__。

```bash
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

微调完成后，可以使用以下脚本来测试微调结果。注意，``--adapters`` 部分需要修改为最后保存检查点的目录路径：

```bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

```
<<< who are you?
<think>
Okay, the user asked, "who are you?" I need to introduce myself. Let me start by stating my name, swift-robot. Then, I should mention that I'm an AI assistant developed by swift. I should explain my purpose, which is to provide information and assistance. I should also highlight my capabilities, like answering questions, generating text, and engaging in conversation. It's important to keep the tone friendly and approachable. Maybe add something about being here to help and encourage the user to ask anything. Let me check if I covered all the key points: name, developer, purpose, capabilities, and a welcoming statement. Yeah, that should do it. Now, let me put that into a concise and friendly response.
</think>

Hello! I am swift-robot, an artificial intelligence assistant developed by swift. My purpose is to provide information and assistance to users like you. I can answer questions, generate text, and engage in conversations on a wide range of topics. I am here to help, so feel free to ask me anything you need!
```

默认情况下，ms-swift 会使用 ModelScope 社区下载模型和数据集。如果想使用 HuggingFace 社区，则需要额外指定 ``--use_hf true``。

合并 LoRA 权重：

```
swift export \
    --adapters output/checkpoint-xxx \
    --merge_lora true
```

推送模型到 ModelScope/HuggingFace：

```bash
# If you are pushing the complete weights, you need to change `--adapters` to `--model`.
# The Modelscope hub_token can be found here: https://modelscope.cn/my/myaccesstoken
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
# You can run the experiment by setting `--dataset AI-ModelScope/alpaca-gpt4-data-en`.
# Note: If you want to specify `--packing true`, you must additionally set `--attn_impl flash_attn`.

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen3-8B \
    --train_type full \
    --dataset '<your-dataset>' \
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

ms-swift 支持 DPO、GRPO、DAPO、PPO、KTO 等 RLHF 方法。本章将着重介绍使用 ms-swift 对 Qwen3-8B 进行 GRPO 训练。

有关详细的 RLHF 支持信息，请参考：`支持的功能 <https://swift.readthedocs.io/zh-cn/latest/Instruction/%E9%A2%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E5%BE%AE%E8%B0%83.html>`__。

### 环境设置

除了安装上述介绍的 ms-swift 相关依赖项外，还需要安装以下依赖项：
```
pip install "math_verify==0.5.2"
pip install vllm
```

### 数据准备

使用 ms-swift 进行 GRPO 训练的数据集格式与 SFT 类似，但不需要最后一轮的 assistant 部分。如果使用 accuracy 作为奖励，则需要一个 ``solution`` 列来计算准确率。

示例数据集格式：

```jsonl
{"messages": [{"role": "user", "content": "Tell me tomorrow's weather"}]}
{"messages": [{"role": "user", "content": "What is 1 + 1?"}, {"role": "assistant", "content": "It equals 2"}, {"role": "user", "content": "What about adding 1?"}]}
{"messages": [{"role": "user", "content": "What is your name?"}]}
```

关于其他 RLHF 算法的数据集准备，请参考：`自定义数据集文档 <https://swift.readthedocs.io/zh-cn/latest/Customization/%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86.html#rlhf>`__。

数据集要求的注意事项：

- **奖励函数计算**：数据集格式取决于所使用的奖励函数。可能需要额外的列来支持特定的奖励计算。例如：

  - 当使用内置的 accuracy 或 cosine 奖励时，数据集必须包含一个 ``solution`` 列以计算回复的准确性。
  - 数据集中的其他列将作为 ``**kwargs`` 传递给奖励函数以实现进一步的自定义。

- **自定义奖励函数**：为了根据您的具体需求调整奖励函数，可以参考链接：`外部奖励插件 <https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin>`__。该插件提供了实现自定义奖励函数的示例和模板。

在训练过程中，我们使用 vLLM 加速采样过程。通过设置 ``num_infer_workers=8`` ，我们为每个设备部署一个 vLLM 引擎以加快采样速度。

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
    --max_completion_length 4096 \
    --vllm_max_model_len 8192 \
    --reward_funcs accuracy \
    --num_generations 16 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.4 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --gc_collect_after_offload true \
    --deepspeed zero3 \
    --num_infer_workers 8 \
    --tensor_parallel_size 1 \
    --temperature 1.0 \
    --top_p 0.85 \
    --log_completions true \
    --overlong_filter true
```

## Megatron-SWIFT

ms-swift 引入了 Megatron 并行技术以加速大模型的训练。支持的模型可以在 `支持的模型文档 <https://swift.readthedocs.io/zh-cn/latest/Instruction/%E6%94%AF%E6%8C%81%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%92%8C%E6%95%B0%E6%8D%AE%E9%9B%86.html>`__ 中找到。

关于环境准备以及 HF 和 MCore 模型权重的转换，可以参考 `Megatron-SWIFT训练文档 <https://swift.readthedocs.io/zh-cn/latest/Instruction/Megatron-SWIFT%E8%AE%AD%E7%BB%83.html>`__。这里不展开介绍。

我们将使用阿里云 DLC 启动训练。训练环境由2台配备8卡 80GiB A800 GPU 组成。关于多节点启动方法的更多信息，请参考 `这里 <https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node>`__。

```bash
# https://help.aliyun.com/zh/pai/user-guide/general-environment-variables
# Ensure that the weight-saving paths on the two nodes are identical.
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
megatron sft \
    --load Qwen3-30B-A3B-Base-mcore \
    --dataset 'liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT' \
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
    --lr_warmup_iters 100 \
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

自定义数据集格式与 ``swift sft`` 相同，详见之前章节。只需指定 ``--dataset <dataset_path>`` 即可。

使用 ``megatron sft`` 和 ``swift sft`` 在对 Qwen3-30B-A3B 模型进行全参数微调的训练速度和 GPU 显存使用对比情况如下：

+------------------+-------------+------------------+------------------+
|                  | Megatron-LM | DeepSpeed-ZeRO2  |  DeepSpeed-ZeRO3 |
+==================+=============+==================+==================+
| 训练速度   |   9.6s/it   |        -         |    91.2s/it      |
+------------------+-------------+------------------+------------------+
| 显存使用 | 16 * 60GiB  |       OOM        |   16 * 80GiB     |
+------------------+-------------+------------------+------------------+

## 总结

以上为使用 ms-swift 训练 Qwen3 系列模型的最佳实践。如果在使用过程中遇到任何困难，请在 `此 issue <https://github.com/modelscope/ms-swift/issues/4030>`__ 中参与讨论。
