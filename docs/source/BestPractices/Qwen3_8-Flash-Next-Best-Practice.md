# Qwen3.8-Flash-Next 最佳实践

[Qwen3.8-Flash-Next](https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next) 是一个多模态超稀疏 MoE 模型，共 125B 参数（其中包含一张额外的 51B N-gram 嵌入表），每 token 激活约 6B。它的架构结合了三个关键设计：

![Qwen3.8-Flash-Next 模型结构](https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.8-Flash-Next/architecture.png)

- **GDN + QSA**：每四层中有三层用 Gated DeltaNet 压缩历史，第四层用 Qwen Sparse Attention 做长范围精确检索。QSA 的预算是 2048 token。
- **Gated Residual**（Hyper-Connections）：每层输入扩成 4 路残差分支，动态控制跳层读写。
- **N-gram Embedding**：一张 51B 的查询记忆表，用极小的单 token 计算量换来容量，可以 offload 到主机内存。


checkpoint 原生支持 262,144 token 上下文。

## 环境设置

```shell
pip install -U ms-swift
pip install -U "transformers>=5.16" "qwen_vl_utils>=0.0.14"

# Megatron
pip install -U mcore-bridge
pip install --no-build-isolation transformer_engine[pytorch]

# flash-linear-attention
pip install -U "flash-linear-attention>=0.5.2" --no-build-isolation

# causal_conv1d
pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation

# vllm 安装需包含 https://github.com/vllm-project/vllm/pull/53896
# 目前安装源码 (>0.28.0)
# 参考 https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source
```

## 微调（Megatron SFT）

**显存优化：N-gram 表 offload**

N-gram 嵌入表占了 51B 参数、bf16 下约 **95GiB**，而且只能沿 TP 切分，在 8 卡 TP2 下每卡仍要背十几 GiB。

设置`PLE_CPU_OFFLOAD`环境变量卸载PLE权重

```shell
PLE_CPU_OFFLOAD=1 megatron sft ...
```

注意，开启后
- **该表将会冻结训练**。LoRA 训练本来就不更新它，无影响；但全参训练下这部分权重会静默地不参与训练。
- **主机内存至少需要 96GiB**。

8 卡 LoRA 微调，序列长度 8192：

```shell
PLE_CPU_OFFLOAD=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron sft \
    --model Qwen/Qwen3.8-Flash-Next \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT#2000' \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules in_proj out_proj linear_proj linear_qkv \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --decoder_first_pipeline_num_layers 12 \
    --sequence_parallel true \
    --moe_permute_fusion true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --padding_free true \
    --max_length 8192 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --lr 1e-4 \
    --save_steps 500 \
    --output_dir output
```

> **`--decoder_first_pipeline_num_layers 12`**：PLE 层（第 2 层）落在 PP stage 0，它的激活与查表开销都压在该 stage，因此按 24/24 均分会两边不均。可以通过给 stage 0 分配更少的层拉平

### 实测数据

测试环境为 8卡，LoRA（rank 8）、TP2/EP4/PP2（并行策略组除外），`micro_batch_size=1`、`global_batch_size=8`，数据集为每条长度精确相等的合成集。显存每卡峰值，耗时以 **seq=2048 基线**为 1.00×。

| 分组 | 配置 | seq | PF | RC | PLE offload | 显存 (GiB) | Δ Mem | 耗时 |
|:--|:--|--:|:-:|:-:|:-:|--:|--:|--:|
| **基线** | seq=2048 | 2048 | – | full | – | **75.6** | – | **1.00×** |
| **序列长度** | – | 4096 | – | full | – | 77.8 | +2.9% | 1.03× |
| | – | 8192 | – | full | – | 93.3 | +23.4% | 1.32× |
| **显存开关** | PLE offload | 4096 | – | full | **✓** | **63.6** | **−15.9%** | 1.04× |
| | PLE offload | 8192 | – | full | **✓** | **77.7** | +2.8% | 1.27× |
| | PLE offload + PF | 8192 | ✓ | full | **✓** | **72.5** | **−4.1%** | 1.26× |
| | padding_free | 4096 | ✓ | full | – | 77.8 | +2.9% | 1.16× |
| | recompute selective | 4096 | – | **selective** | – | 92.1 | +21.8% | **0.75×** |
| | recompute selective | 8192 | – | **selective** | – | 123.5 | +63.4% | **0.87×** |
| | 关 recompute | 4096 | – | **none** | – | 92.1 | +21.8% | **0.74×** |
| **并行策略** | TP4/EP2/PP2 | 8192 | – | full | – | 105.7 | +39.8% | 2.58× |
| | TP2/EP4/PP1 | 8192 | – | full | – | 142.5 | +88.5% | **0.80×** |

<sub>PF = padding_free，RC = recompute_granularity，PLE offload = `PLE_CPU_OFFLOAD=1`；“–” 表示关闭/默认。**耗时列是单步时长的相对值，数值越大越慢**（如 1.32× = 每步耗时是基线的 1.32 倍）。</sub>


- **序列 2048 → 4096 几乎不涨显存**（+2.9%），4096 → 8192 才明显。QSA 的 budget 是 2048：全长 ≤ 2048 时稀疏选择覆盖全部可见 key、等价 dense，超过后 attention 不再随 s² 增长。
- **PLE offload 是性价比最高的开关**（4096 省 14.2GiB、8192 省 15.6GiB，耗时基本持平）。
- **`padding_free` 在长序列叠加 PLE offload 时额外省 5.2GiB**（8192 下 77.7 → 72.5）。单用（4096 下 77.8）在等长数据上无显存收益且略慢。
- **关掉 recompute 快 26%，但多吃 14.3GiB**，80GiB 卡建议保持开启。`selective` 对本模型无效：显存与完全关闭相同（4096 下均为 92.1GiB），只慢 1.5%。
- **不要盲目加 TP**：hidden 只有 2560，TP4 切得过碎，通信开销压过收益（TP4/EP2/PP2：显存反而 +39.8%、慢 2.6 倍）。
- **80GiB 单卡可跑 8192 的只有 PLE offload + padding_free**（72.5GiB）。

> 除两行 selective（后续单独补测）外，表中各行为同一批测得。需要注意的是，**耗时列的噪声较大**：同一配置在不同批次重测，单步耗时差异可达 7%。因此 5% 以内的耗时差异不具备区分度，只有 seq=8192、recompute 与并行策略这类数十百分点的差异才是真实趋势。

## 强化学习（GRPO）

8 卡 GRPO LoRA 训练，rollout 使用 colocate 模式的 vLLM，`max_completion_length` 为 8192：

```shell
# 8*135GiB
SYSTEM_PROMPT="Please reason step by step, and put your final answer within \\boxed{}."

PLE_CPU_OFFLOAD=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.8-Flash-Next \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --system "$SYSTEM_PROMPT" \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules in_proj out_proj linear_proj linear_qkv \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --decoder_first_pipeline_num_layers 12 \
    --sequence_parallel true \
    --moe_permute_fusion true \
    --moe_aux_loss_coeff 1e-3 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --padding_free true \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_enable_lora false \
    --vllm_gpu_memory_utilization 0.35 \
    --vllm_tensor_parallel_size 8 \
    --vllm_max_model_len 9216 \
    --sleep_level 2 \
    --offload_model true \
    --offload_optimizer true \
    --offload_bridge false \
    --num_train_epochs 1 \
    --global_batch_size 8 \
    --micro_batch_size 1 \
    --steps_per_generation 1 \
    --num_generations 4 \
    --reward_funcs accuracy \
    --max_length 1024 \
    --max_completion_length 8192 \
    --temperature 1.0 \
    --loss_type grpo \
    --beta 0.0 \
    --lr 5e-5 \
    --save_safetensors true \
    --merge_lora false \
    --logging_steps 1 \
    --log_completions true \
    --output_dir output
```

两个 vLLM 侧开关的说明：

- **`--vllm_enable_lora false`**：本文的实测数据均在该设置下取得（每步把 LoRA 合并进 base 后全量同步），权重同步已验证正常。设为 `true` 让 vLLM 原生服务 adapter 理论上更快，且本模型的 vLLM 模型类确实声明了 `SupportsLoRA`、`packed_modules_mapping` 也覆盖了上面训练的模块，**但我们没有实测过这条路径**；vLLM 的 LoRA 还有第二道门（MoE expert kernel 需支持 LoRA），需要自行验证后再用于正式训练。
- **`--vllm_enforce_eager`** 无需开启（默认关闭即可，保留 CUDA graph 以获得更快的 rollout）。本文数据是在开启的情况下测的，因此表中的耗时偏保守。

### 显存占用

上面这份配置在 8 × B200 上实测：

| 配置 | padding_free | 峰值显存 |
|:--|:-:|--:|
| 上述推荐配置 | – | **135 GiB** |
| 上述推荐配置 | ✓ | **135 GiB** |

> ⚠️ **这份配置需要 > 80GiB 的卡（如 H20/H200/B200）。** colocate 模式下训练器与 vLLM 引擎共享同一张卡，而 `max_completion_length=8192` 又要求 `vllm_max_model_len` 达 9216，KV cache 占用较大。
>
> **80GiB 卡的调整方向**（未逐一实测，需自行验证）：降低 `--max_completion_length`（并同步降 `--vllm_max_model_len`）是最直接的手段；其次是降 `--vllm_gpu_memory_utilization`、确保 `--sleep_level 2` 与三个 `--offload_*` 均开启。如果仍不够，考虑改用 server 模式把 rollout 引擎放到单独的卡上。

### 权重同步是否正确

GRPO 的常见故障是 rollout 引擎拿到了过期或映射错误的权重，此时训练不报错但学不动。用 `rollout_correction/kl` 判断（同一批 token 上 KL(π_rollout ‖ π_training)）：

```shell
grep -oE "'rollout_correction/kl': [0-9.e+-]+" 训练日志 | head -20
```

- **kl < 0.01**：正常，rollout 与训练侧一致。
- **kl 很大或持续增长**：权重同步有问题。

同时建议关注 `completions/mean_length`（rollout 是否真的产出 token）与 `reward_std`（组内是否有区分度；恒为 0 说明没有有效的 advantage 信号）。

## 已知限制

- **`sequence_parallel` 必须开启**，框架会拒绝 false。
- **`context_parallel_size > 1` 时需设置 `cp_comm_type=allgather`**：QSA 的选择必须在 attention 之前看到全部 key，ring/p2p 无法提供。
- **`apply_rope_fusion` 不支持开启**（ms-swift 默认已为 false，一般无需关心）。开启后传给 QSA 的是原始 rotary 表而非逐 token 的角度，indexer 无法得到正确位置；此时会直接报 `RuntimeError` 并提示关闭该项，不会静默算错。
- **fp8 未支持**：N-gram 层的 kernel 指针类型固定为 bf16。
- **MTP 未支持**：设置 `mtp_num_layers` 会报 `NotImplementedError`。
