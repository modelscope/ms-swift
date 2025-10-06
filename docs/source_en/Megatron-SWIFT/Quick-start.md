# Quick Start

ms-swift incorporates Megatron's parallelization techniques to accelerate the training of large models, including data parallelism, tensor parallelism, pipeline parallelism, sequence parallelism, context parallelism, and expert parallelism. It supports CPT/SFT/DPO for models such as Qwen3, [Qwen3-MoE](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/qwen3_moe.sh), Qwen2.5, Llama3, Deepseek-R1 and GLM4.5 series. For a complete list of supported models, please refer to the [Supported Models and Datasets documentation](../Instruction/Supported-models-and-datasets.md). We recommend using Megatron-SWIFT for MoE training; it can typically achieve a 10x speedup in training.


| Method                             | Full-parameter | LoRA | MoE  | Multimodal |
| ---------------------------------- | -------------- | ---- | ---- | ---------- |
| Pretraining                        | ✅              | ✅    | ✅    | ✅          |
| Instruction-supervised fine-tuning | ✅              | ✅    | ✅    | ✅          |
| DPO                                | ✅              | ✅    | ✅    | ✅          |
| Classification tasks               | ✅              | ✅    | ✅    | ✅          |

## Environment Setup

To use Megatron-SWIFT, in addition to installing the `swift` dependencies, you also need to install the following:

```shell
# Recommended PyTorch version: 2.5 / 2.6
pip install pybind11

# transformer_engine
# If an installation error occurs, you can refer to this issue for resolution: https://github.com/modelscope/ms-swift/issues/3793
pip install --no-build-isolation transformer_engine[pytorch]
# Or install using the following command
# pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.5#egg=transformer_engine[pytorch]

# apex
git clone https://github.com/NVIDIA/apex
cd apex
# https://github.com/modelscope/ms-swift/issues/4176
git checkout e13873debc4699d39c6861074b9a3b2a02327f92
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# megatron-core
pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_r0.13.0

# If you are using multi-node training, please additionally set the `MODELSCOPE_CACHE` environment variable to a shared storage path.
# This will ensure that the dataset cache is shared, thereby speeding up preprocessing.
# Note: This step is crucial; otherwise multi-machine training may hang due to data inconsistencies caused by randomness in data preprocessing.
export MODELSCOPE_CACHE='/xxx/shared'

# Megatron-LM
# The training module in the dependent library Megatron-LM will be cloned and installed by swift via `git clone`. Alternatively, you can use the environment variable `MEGATRON_LM_PATH` to point to the path of an already downloaded repository (in offline environments, use the [core_r0.13.0 branch](https://github.com/NVIDIA/Megatron-LM/tree/core_r0.13.0)).
git clone --branch core_r0.13.0 https://github.com/NVIDIA/Megatron-LM.git
export MEGATRON_LM_PATH='/xxx/Megatron-LM'

# flash_attn
# Choose an appropriate version to install: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1
# Note: Do not install a version higher than the maximum supported by transformer_engine: https://github.com/NVIDIA/TransformerEngine/blob/release_v2.6/transformer_engine/pytorch/attention/dot_product_attention/utils.py#L109
```

Alternatively, you can also use the image:
```
modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-modelscope1.29.2-swift3.8.3
modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-modelscope1.29.2-swift3.8.3
modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.6.3-py311-torch2.7.1-vllm0.10.1.1-modelscope1.29.2-swift3.8.3
```

Recommended Operating Environment:

|        | Range | Recommended | Notes |
|--------------|--------------|-------------|--------------------|
| python       | >=3.9        | 3.10/3.11        |                    |
| cuda         |              | cuda12      |                    |
| torch        | >=2.0        | 2.6.0/2.7.1    |                    |
| transformer_engine    | >=2.3       |         |                  |
| apex |   |  0.1 | |
| megatron_core    | >=0.12       | 0.13      |                  |
| flash_attn    |        | 2.7.4.post1/3.0.0b1   |                  |
| transformers | >=4.33       | 4.56.2      |                    |
| modelscope   | >=1.23       |             |                    |
| peft         | >=0.11,<0.18 |             |      LoRA          |
| trl          | >=0.15,<0.21 |       |      RLHF        |


## Quick Start Example

This section introduces a quick start example for fine-tuning the self-awareness of the Qwen2.5-7B-Instruct model using two 80GiB A100 GPUs. The following best practices can be completed within 10 minutes.

First, we need to convert the weights from HF (Hugging Face) format to Megatron format:
- Multi-GPU weight conversion: Remove `CUDA_VISIBLE_DEVICES=0` to enable multi-GPU weight conversion.
- Conversion precision test: `--test_convert_precision true` will test the conversion precision. For large MoE model conversions, this option takes longer and consumes more memory, so you may omit it as needed.
```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen2.5-7B-Instruct \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir Qwen2.5-7B-Instruct-mcore \
    --test_convert_precision true
```

Next, use the following script to start training. The required GPU memory resources are 2*80GiB:
- If using multi-machine training, it is recommended to share a disk and specify the same path for `--save`.
```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --load Qwen2.5-7B-Instruct-mcore \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 16 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --max_epochs 1 \
    --save megatron_output/Qwen2.5-7B-Instruct \
    --save_interval 100 \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --model_author swift \
    --model_name swift-robot
```

Finally, convert the Megatron format weights back to HF format:
- Note: Please point `--mcore_model` to the parent directory of `iter_xxx`. By default, the corresponding checkpoint from `latest_checkpointed_iteration.txt` will be used.
- If OOM (Out of Memory) occurs, simply remove `CUDA_VISIBLE_DEVICES=0`. If you encounter insufficient memory, please remove `--test_convert_precision true`.

```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --mcore_model megatron_output/Qwen2.5-7B-Instruct/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx-hf \
    --test_convert_precision true
```

We then perform inference on the generated HF format weights:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model megatron_output/Qwen2.5-7B-Instruct/vx-xxx-hf \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

The inference results are as follows:

```
<<< who are you?
I am a language model developed by swift, you can call me swift-robot. How can I assist you?
```

- For pretraining, you can use `megatron pt` instead of `megatron sft`, which will use a generative template for training.
- Megatron-SWIFT uses the same dataset and template processing modules as ms-swift, thus supporting techniques such as packing, loss scale, and agent training. For custom dataset formats, please refer to the [Custom Dataset Documentation](../Customization/Custom-dataset.md).
- **More Examples**: Including packing, multi-node training, 32K context length, DPO, MoE models, and pre-training, can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/megatron).

Training tips:
- Ways to increase training throughput: use packing, increase DP (data parallelism), reduce recomputation, and increase computation-communication overlap.
- Parallelism choices:
  - Megatron-SWIFT uses ZeRO-1 (use_distributed_optimizer enabled by default) combined with various parallelism techniques.
  - DP is the fastest but consumes the most memory; use other parallel techniques to reduce memory usage.
  - TP/EP involve heavy communication, so keep them within the NVLink domain when possible; for cross-domain setups prefer PP/DP. For expert layers, prefer EP over ETP — ETP saves memory but is slower.
  - MoE parallel folding: separate MoE parallel groups from Dense groups. Attention uses tp-cp-dp-pp groups, while MoE uses etp-ep-dp-pp groups.
- Choosing parallelism for weight conversion: Megatron-SWIFT uses the torch_dist storage format on the MCore side; you can adjust parallelism at training time and do not need to specify it during weight conversion.


## Benchmark
The training speed comparison for full-parameter dense models with 8K context length, using `megatron sft` and `swift sft`, under a single-node, eight-GPU A800 environment is as follows:

**Dense** Qwen2.5-14B:


|                  | Megatron-LM | Deepspeed-ZeRO2 | Deepspeed-ZeRO3 |
| ---------------- | ----------- | --------------- | --------------- |
| Training Speed   | 9.04s/it    | 10.32s/it       | 10.56s/it       |
| GPU Memory Usage | 8\*64GB      | 8\*80GB          | 8\*58GB          |


The training speed comparison for full-parameter MoE models with 8K context length, using `megatron sft` and `swift sft`, under a two-node, 16-GPU A800 environment is as follows:

**MoE** Qwen3-30B-A3B:

|                  | Megatron-LM | Deepspeed-ZeRO2 | Deepspeed-ZeRO3 |
| ---------------- | ----------- | --------------- | --------------- |
| Training Speed   | 9.6s/it    | -        | 91.2s/it       |
| GPU Memory Usage | 16 * 60GiB  | OOM             | 16 * 80GiB      |
