# Qwen3.8-Flash-Next Best Practice

[Qwen3.8-Flash-Next](https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next) is a multimodal, ultra-sparse MoE model with 125B parameters in total (including an additional 51B N-gram embedding table), activating about 6B per token. Its architecture combines three key designs:

![Qwen3.8-Flash-Next architecture](https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.8-Flash-Next/architecture.png)

- **GDN + QSA**: three of every four layers use Gated DeltaNet to compress history, while the fourth uses Qwen Sparse Attention for precise long-range retrieval. QSA's budget is 2048 tokens.
- **Gated Residual** (Hyper-Connections): each layer's input is expanded into 4 residual branches that dynamically control cross-layer reads and writes.
- **N-gram Embedding**: a 51B lookup-memory table that trades an extremely small per-token compute cost for capacity, and can be offloaded to host memory.


The checkpoint natively supports a 262,144-token context.

## Environment Setup

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

# The vllm build must include https://github.com/vllm-project/vllm/pull/53896
# For now, install from source (>0.28.0)
# See https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source
```

The training backend uses Megatron. For environment preparation, please refer to the [Megatron-SWIFT Quick Start guide](../Megatron-SWIFT/Quick-start.md).

## Fine-tuning (Megatron SFT)

**Memory optimization: N-gram table offload**

The N-gram embedding table holds 51B parameters (~**95GiB** in bf16), and can only be sharded along TP.

Set the `PLE_CPU_OFFLOAD` environment variable to offload the PLE weights:

```shell
PLE_CPU_OFFLOAD=1 megatron sft ...
```

Note that once enabled:
- **The table is frozen during training.** LoRA training does not update it anyway, so there is no impact; under full-parameter training, however, these weights silently stop participating in training.

8-GPU LoRA fine-tuning:

```shell
# 8*70G
PLE_CPU_OFFLOAD=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron sft \
    --model Qwen/Qwen3.8-Flash-Next \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT#2000' \
    --num_train_epochs 1 \
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

> **`--decoder_first_pipeline_num_layers 12`**: the PLE layer (layer 2) lands on PP stage 0, so both its activations and its lookup cost fall on that stage; an even 24/24 split would therefore be unbalanced between the two stages. Assigning fewer layers to stage 0 evens it out.

### Measured data

Measured on 8 GPUs with LoRA (rank 8), TP2/EP4/PP2 (except for the "parallelism" group), `micro_batch_size=1`, `global_batch_size=8`, on a synthetic dataset where every sample is exactly the same length. Memory is the per-GPU peak; step time is relative to the **seq=2048 baseline** = 1.00×.

| Group | Config | seq | PF | RC | PLE offload | Mem (GiB) | Δ Mem | Time |
|:--|:--|--:|:-:|:-:|:-:|--:|--:|--:|
| **baseline** | seq=2048 | 2048 | – | full | – | **75.6** | – | **1.00×** |
| **sequence length** | – | 4096 | – | full | – | 77.8 | +2.9% | 1.03× |
| | – | 8192 | – | full | – | 93.3 | +23.4% | 1.32× |
| **memory switches** | PLE offload | 4096 | – | full | **✓** | **63.6** | **−15.9%** | 1.04× |
| | PLE offload | 8192 | – | full | **✓** | **77.7** | +2.8% | 1.27× |
| | PLE offload + PF | 8192 | ✓ | full | **✓** | **72.5** | **−4.1%** | 1.26× |
| | padding_free | 4096 | ✓ | full | – | 77.8 | +2.9% | 1.16× |
| | recompute selective | 4096 | – | **selective** | – | 92.1 | +21.8% | **0.75×** |
| | recompute selective | 8192 | – | **selective** | – | 123.5 | +63.4% | **0.87×** |
| | recompute off | 4096 | – | **none** | – | 92.1 | +21.8% | **0.74×** |
| **parallelism** | TP4/EP2/PP2 | 8192 | – | full | – | 105.7 | +39.8% | 2.58× |
| | TP2/EP4/PP1 | 8192 | – | full | – | 142.5 | +88.5% | **0.80×** |

<sub>PF = padding_free, RC = recompute_granularity, PLE offload = `PLE_CPU_OFFLOAD=1`; "–" means off/default. **The Time column is the relative single-step duration — higher is slower** (e.g. 1.32× means each step takes 1.32× as long as the baseline).</sub>


## Reinforcement Learning (GRPO)

8-GPU GRPO LoRA training, with rollout using vLLM in colocate mode and `max_completion_length` of 8192:

```shell
# 8*135G
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
