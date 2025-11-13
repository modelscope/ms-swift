# Mcore Bridge

Megatron is renowned for its excellent training speed and rich parallelism techniques, but this also brings a relatively high barrier to entry. Therefore, mcore-bridge was created to make Megatron training as simple and easy to use as transformers. With Mcore-Bridge, users can:

1. Directly load model weights in safetensors format and seamlessly use Megatron for efficient training. Save training weights directly in safetensors format without additional conversion.
2. Support bidirectional conversion compatible with LoRA incremental weights.
3. Support `Megatron->vLLM` weight synchronization for algorithms like GRPO/GKD.
4. Support multi-machine conversion of ultra-large-scale models.

Mcore-Bridge is compatible with various model architectures including Dense/MoE/multimodal. After training is complete, the converted models can be directly deployed using mainstream inference frameworks such as transformers, vLLM, SGLang, etc.

## Seamless Training

Currently, Mcore-Bridge supports parallelism techniques including TP/PP/EP/ETP/VPP and all model architectures supported by Megatron-SWIFT. Refer to [Supported Models Documentation](../Instruction/Supported-models-and-datasets.md). The following introduces Mcore-Bridge's seamless training capabilities, covering both Dense and MoE models.

### Dense Models

Below is an example of training the multimodal model Qwen3-VL:

```shell
# 2 * 76GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
    --load_from_cache_file true \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --packing true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --split_dataset_ratio 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --max_epochs 1 \
    --save megatron_output/Qwen3-VL-8B-Instruct \
    --save_interval 200 \
    --vit_gradient_checkpointing false \
    --max_length 2048 \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 8
```

Then we perform inference on the validation set:

```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model megatron_output/Qwen3-VL-8B-Instruct/vx-xxx/checkpoint-xxx \
    --load_data_args true \
    --stream true
```

### MoE Models

Below is an example of CoT training for the text-only model Qwen3-Moe:

```shell
# 8 * 76GiB, 3s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron sft \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --load_safetensors true \
    --save_safetensors true \
    --dataset 'swift/Chinese-Qwen3-235B-Thinking-2507-Distill-data-110k-SFT#20000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --moe_permute_fusion true \
    --pipeline_model_parallel_size 2 \
    --decoder_first_pipeline_num_layers 25 \
    --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 4 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507 \
    --eval_interval 500 \
    --save_interval 500 \
    --max_length 8192 \
    --packing true \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --moe_expert_capacity_factor 2 \
    --attention_backend flash
```

Perform inference on the trained weights:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx/checkpoint-xxx \
    --stream true \
    --max_new_tokens 1024
```

## LoRA Export

In addition to supporting full parameter import/export, Mcore-Bridge also supports separate import/export of LoRA incremental models.

Below is an example of self-cognition training using LoRA for the text-only model Qwen3-Moe:

- If you want to export merged weights instead of LoRA delta weights, please set `--merge_lora true`. Setting `--merge_lora true` has better compatibility and supports all model series.
- Note: Since the model structures of transformers and Megatron are not necessarily identical (for example, the expert part of transformers' Qwen3-VL-Moe is not implemented as Linear, but as Parameters), some models cannot convert LoRA delta weights (however, if Qwen3-VL-Moe only sets linear_proj and linear_qkv for LoRA training, conversion is also supported). But most models support LoRA conversion, such as: Qwen3-Moe, Qwen3-Omni-Moe, GLM4.5-V, etc.

```shell
# 50GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --load_safetensors true \
    --save_safetensors true \
    --merge_lora false \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT#2000' \
              'swift/self-cognition#1000' \
    --load_from_cache_file true \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --split_dataset_ratio 0.01 \
    --moe_permute_fusion true \
    --expert_model_parallel_size 2 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 8 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507 \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 2048 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --moe_expert_capacity_factor 2 \
    --attention_backend flash \
    --model_author swift \
    --model_name swift-robot
```

Perform inference on the exported LoRA weights:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapters megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx/checkpoint-xxx \
    --stream true
```

## Export and Conversion Precision Testing

In addition to supporting safetensors conversion and saving during training, Mcore-Bridge also supports the `megatron export` command for standalone weight export. `megatron export` supports conversion precision testing during weight conversion, which is very helpful for verifying accuracy when integrating new models. Typically, models already integrated into Megatron-SWIFT will not have precision misalignment issues, so you can confidently set `--test_convert_precision false`.
- Note: For multimodal models, please focus on the `mean_diff (with loss)` field. The `mean_diff` may show a large difference because it includes image tokens, and loss is not calculated for that portion.

Full parameter weights:

```shell
# safetensors -> torch_dist
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --save Qwen3-30B-A3B-Instruct-2507-mcore \
    --to_mcore true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true
```

```shell
# torch_dist -> safetensors
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --load Qwen3-30B-A3B-Instruct-2507-mcore \
    --save Qwen3-30B-A3B-Instruct-2507-hf \
    --to_hf true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true
```

LoRA weights:

```shell
# torch_dist -> safetensors
# If you need to perform merge-lora and test precision alignment after merge-lora, simply set `--merge_lora true`
# You can also change `--model safetensors-path` to `--load torch-dist-path`. These two methods are equivalent, and mcore-bridge will handle it automatically.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapter_load megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx-lora \
    --merge_lora false \
    --to_hf true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true
```

```shell
# safetensors -> torch_dist
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapters megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx-lora \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx-mcore \
    --merge_lora false \
    --to_mcore true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true
```

Merge-LoRA:
```shell
# torch_dist -> torch_dist
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapter_load megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx-merged \
    --merge_lora true \
    --to_mcore true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2
```


## Using Code

You need to create the following file (test.py), then run `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test.py`. Below is sample code for loading, exporting, and saving weights using Mcore-Bridge.

```python
import torch

from swift.megatron import MegatronArguments, convert_hf_config, get_megatron_model_meta
from swift.llm import get_model_tokenizer
from megatron.training.initialize import initialize_megatron

model_id = 'Qwen/Qwen3-4B-Instruct-2507'
_, processor = get_model_tokenizer(model_id, load_model=False, download_model=True)
model_info = processor.model_info
megatron_model_meta = get_megatron_model_meta(model_info.model_type)
config_kwargs = convert_hf_config(model_info.config)
megatron_args = MegatronArguments(
    model=model_id,
    tensor_model_parallel_size=2,
    torch_dtype=torch.bfloat16,
    **config_kwargs,
)
extra_args = megatron_args.parse_to_megatron()
initialize_megatron(args_defaults=extra_args)
mg_model = megatron_model_meta.model_provider()
bridge = megatron_model_meta.bridge_cls()
# Load weights
bridge.load_weights(mg_model, model_info.model_dir)
# Export weights
for name, parameters in bridge.export_weights([mg_model]):
    pass
# Save weights
bridge.save_weights([mg_model], 'output/Qwen3-4B-Instruct-2507-new')
```

Inference with the newly generated weights:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model output/Qwen3-4B-Instruct-2507-new \
    --model_type qwen3_nothinking \
    --stream true
```

Loading, exporting, and saving LoRA weights follows the same pattern. Run `CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 test.py`

```python
import torch

from swift.megatron import (
    MegatronArguments, convert_hf_config, get_megatron_model_meta, prepare_mcore_model
)
from swift.llm import get_model_tokenizer
from megatron.training.initialize import initialize_megatron

model_id = 'Qwen/Qwen3-30B-A3B-Instruct-2507'
_, processor = get_model_tokenizer(model_id, load_model=False, download_model=True)
model_info = processor.model_info
megatron_model_meta = get_megatron_model_meta(model_info.model_type)
config_kwargs = convert_hf_config(model_info.config)
megatron_args = MegatronArguments(
    model=model_id,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
    expert_model_parallel_size=2,
    sequence_parallel=True,
    torch_dtype=torch.bfloat16,
    train_type='lora',
    **config_kwargs,
)
extra_args = megatron_args.parse_to_megatron()
initialize_megatron(args_defaults=extra_args)
mg_model = megatron_model_meta.model_provider()
# Load weights
bridge = megatron_model_meta.bridge_cls()
bridge.load_weights(mg_model, model_info.model_dir)
# Prepare LoRA and load
peft_model = prepare_mcore_model(mg_model)
print(f'peft_model: {peft_model}')
# bridge.load_weights(mg_model, 'adapter-path', is_peft_format=True)
# Export weights
for name, parameters in bridge.export_weights([mg_model], is_peft_format=True):
    pass
bridge.save_weights([mg_model], 'output/Qwen3-30B-A3B-Instruct-2507-lora', is_peft_format=True)
```

Inference with the newly generated weights:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapters output/Qwen3-30B-A3B-Instruct-2507-lora \
    --stream true
```
