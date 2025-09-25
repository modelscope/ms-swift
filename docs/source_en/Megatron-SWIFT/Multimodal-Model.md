# Multimodal Models

ms-swift introduces Megatron's parallelization techniques to accelerate the training of large multimodal models. Currently, it supports CPT/SFT/DPO for models such as Qwen3-VL, Qwen3-Omni, Qwen2.5-VL, Qwen2.5-Omni, InternVL3.5, GLM4.5v, Kimi-VL. For a complete list of supported models, please refer to the [Supported Models and Datasets documentation](../Instruction/Supported-models-and-datasets.md).

For environment setup, please refer to the Megatron-SWIFT [Quick Start guide](./Quick-start.md).

## Dense Model

This section demonstrates fine-tuning the Qwen2.5-VL-7B-Instruct model on the LaTeX-OCR task using two 80GiB A100 GPUs, with both full-parameter fine-tuning and LoRA. The best practices described below can be completed within 10 minutes.

First, we need to convert the model weights from Hugging Face format to Megatron format:
```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir Qwen2.5-VL-7B-Instruct-mcore \
    --test_convert_precision true
```

### Full

The full-parameter training script is as follows:
```shell
# 2 * 72GiB; 4.1s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --load Qwen2.5-VL-7B-Instruct-mcore \
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
    --save megatron_output/Qwen2.5-VL-7B-Instruct \
    --save_interval 200 \
    --vit_gradient_checkpointing true \
    --max_length 2048 \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 8
```

Convert Megatron-format weights saved with full parameters to Hugging Face format:

- Note: `--mcore_model` should point to the parent directory of `iter_xxx`. By default, the checkpoint specified in `latest_checkpointed_iteration.txt` will be used.

```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --mcore_model megatron_output/Qwen2.5-VL-7B-Instruct/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen2.5-VL-7B-Instruct/vx-xxx-hf \
    --test_convert_precision true
```

### LoRA

The LoRA training script is as follows:
```shell
# 2 * 23GiB; 2.3s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --load Qwen2.5-VL-7B-Instruct-mcore \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
    --load_from_cache_file true \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --tensor_model_parallel_size 1 \
    --sequence_parallel true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --packing true \
    --split_dataset_ratio 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_epochs 1 \
    --save megatron_output/Qwen2.5-VL-7B-Instruct \
    --save_interval 200 \
    --vit_gradient_checkpointing true \
    --max_length 2048 \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 8
```

Merge the LoRA-saved incremental weights and convert them to Hugging Face format:
```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --mcore_adapters megatron_output/Qwen2.5-VL-7B-Instruct/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen2.5-VL-7B-Instruct/vx-xxx-hf \
    --test_convert_precision true
```


Finally, we use the generated Hugging Face format weights to perform inference on the validation set:
```shell
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model megatron_output/Qwen2.5-VL-7B-Instruct/vx-xxx-hf \
    --attn_impl flash_attn \
    --stream true \
    --load_data_args true \
    --temperature 0 \
    --max_new_tokens 512
```

The inference results are as follows:
```
[QUERY] Using LaTeX to perform OCR on the image.
[LABELS] \forall x \in X , ( \alpha f ) ( x ) = \alpha f ( x )
[RESPONSE] \forall x \in X , ( \alpha f ) ( x ) = \alpha f ( x )
--------------------------------------------------
[QUERY] Using LaTeX to perform OCR on the image.
[LABELS] \pi \int _ { c } ^ { d } \{ g ( y ) \} ^ { 2 } d y
[RESPONSE] \pi \int _ { c } ^ { d } \{ g ( y ) \} ^ { 2 } d y
--------------------------------------------------
[QUERY] Using LaTeX to perform OCR on the image.
[LABELS] [ \frac 2 3 x ^ { \frac 3 2 } ] _ { 0 } ^ { 1 }
[RESPONSE] [ \frac 2 3 x ^ { \frac 3 2 } ] _ { 0 } ^ { 1 }
```

## MoE Model

The model conversion steps for MoE models are the same as those for Dense models (please refer to the Dense model section for modifications). Below is the training script for LoRA fine-tuning of the OpenGVLab/InternVL3_5-30B-A3B-mcore model.

- During MoE model conversion, the precision test via `--test_convert_precision true` takes a long time; consider removing it as appropriate.

```bash
# 2 * 43GiB, 8s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --load InternVL3_5-30B-A3B-mcore \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
    --load_from_cache_file true \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --sequence_parallel true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --packing true \
    --split_dataset_ratio 0.01 \
    --expert_model_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_epochs 1 \
    --save megatron_output/InternVL3_5-30B-A3B \
    --eval_interval 200 \
    --save_interval 200 \
    --vit_gradient_checkpointing true \
    --max_length 2048 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash
```

After training is completed, we use the generated Hugging Face format weights to perform inference on the validation set:
```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model megatron_output/InternVL3_5-30B-A3B/vx-xxx-hf \
    --attn_impl flash_attn \
    --stream true \
    --load_data_args true \
    --temperature 0 \
    --max_new_tokens 512
```
