# Flexible Quantization with EntroPack

Weight quantization methods commonly provide fixed formats such as 4-bit and 8-bit. When a quality or memory budget falls between two formats, users must choose one of the fixed options. EntroPack instead takes bits per parameter (bpp) as the compression target. `target_bpp` accepts fractional values such as 3.5 or 5.2, allowing finer control over weight storage. EntroPack quantizes weight vectors onto the E8 lattice and entropy-encodes the quantized representation with rANS. The encoder searches for a quantization scale that brings the resulting rate close to `target_bpp`. This process only uses the weights themselves and requires no calibration dataset, activation statistics, or additional fine-tuning. It can run online while loading a model or produce a compressed checkpoint for later use.

ms-swift supports three EntroPack linear implementations: `default`, `fp8`, and `int8`. The `default` implementation reconstructs the weight and performs a regular linear operation. The `fp8` and `int8` implementations combine sub-8-bpp weight storage with the corresponding low-precision computation. Storage rate and compute format can therefore be selected independently instead of being tied to one fixed bit width. FP8 requires GPU compute capability 8.9 or later, and INT8 requires 8.0 or later.

This guide uses `Qwen/Qwen3.8-27B` to demonstrate online inference, weight export, and LoRA training. The commands use 4.0 bpp only as an example; replace it according to the available memory and required model quality. The theoretical storage of compressed linear weights is approximately:

```text
number of compressed parameters × target_bpp ÷ 8
```

`target_bpp` is a rate-control target. Actual usage also includes encoding metadata, uncompressed weights, activations, and the KV cache.

## Installation

Until the EntroPack integration is included in a Transformers release, install the integration branch:

```shell
pip install -U ms-swift
pip install "transformers @ git+https://github.com/mi804/transformers.git@entropack-integration"
```

Install the EntroPack package matching the local CUDA version:

```shell
# CUDA 12
pip install "entropack[cuda12]>=0.1.3"

# CUDA 13
pip install "entropack[cuda13]>=0.1.3"
```

## Online quantized inference

Online quantization builds EntroPack weights directly from the original checkpoint, so no quantized model needs to be prepared in advance. Use `entropack_target_bpp` to set the storage budget and `entropack_linear_type` to select the compute implementation.

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3.8-27B \
    --quant_method entropack \
    --entropack_target_bpp 4.0 \
    --entropack_linear_type default \
    --infer_backend transformers \
    --stream true \
    --max_new_tokens 512
```

Online quantization reads the original model and does not create a new checkpoint. Quantization runs again each time the model is loaded.

`entropack_linear_type` accepts `default`, `fp8`, and `int8`. `default` supports rates in `[0.001, 11]` bpp. FP8 and INT8 require a target rate below 8 bpp.

## Exporting quantized weights

`swift export` saves the online quantization result as a complete model directory. The directory can be copied to another machine or passed to `--model` for later training and inference.

```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen3.8-27B \
    --quant_method entropack \
    --entropack_target_bpp 4.0 \
    --entropack_linear_type default \
    --torch_dtype bfloat16 \
    --output_dir Qwen3.8-27B-EntroPack
```

The exported model can be loaded directly for inference:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen3.8-27B-EntroPack \
    --infer_backend transformers \
    --stream true \
    --max_new_tokens 512
```

The quantization settings are stored in the model configuration, so `quant_method` and `entropack_target_bpp` do not need to be passed again.

## LoRA training

LoRA training can either quantize the original model online or load an exported EntroPack checkpoint. Both workflows train the same type of adapter. The first performs quantization during startup, while the second reads the compressed weights directly.

### Online quantization from the original model

The following command loads the original Qwen3.8-27B weights, quantizes them to a target of 4.0 bpp, and starts LoRA training:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3.8-27B \
    --quant_method entropack \
    --entropack_target_bpp 4.0 \
    --entropack_linear_type default \
    --tuner_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output/Qwen3.8-27B-EntroPack \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
```

The output directory contains the LoRA adapter and training state. The original model is not modified.

### Training from exported quantized weights

Point `--model` to the directory produced by `swift export`:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen3.8-27B-EntroPack \
    --tuner_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output/Qwen3.8-27B-EntroPack \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
```

This workflow does not quantize the original weights again and directly reuses the exported EntroPack model.

## Inference with a LoRA adapter

A LoRA checkpoint contains only adapter parameters, not the EntroPack base model. Passing the adapter path alone lets ms-swift restore the base model and settings saved during training:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/Qwen3.8-27B-EntroPack/vx-xxx/checkpoint-xxx \
    --infer_backend transformers \
    --stream true \
    --max_new_tokens 512
```

An adapter trained with EntroPack can also be loaded on the original dense base model. Explicitly select the original model and set `load_args=false` to avoid restoring the EntroPack settings used during training:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3.8-27B \
    --adapters output/Qwen3.8-27B-EntroPack/vx-xxx/checkpoint-xxx \
    --load_args false \
    --infer_backend transformers \
    --stream true \
    --max_new_tokens 512
```

EntroPack can therefore be used only to reduce base-weight memory during LoRA training, while deployment can use either the original or the EntroPack base model. Because the two base models have different weights, evaluate both options before deployment.

## Multi-GPU LoRA

An exported EntroPack checkpoint can also be used for multi-GPU training. With DDP, each GPU loads a copy of the compressed base model and synchronizes LoRA gradients:

```shell
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model Qwen3.8-27B-EntroPack \
    --tuner_type lora \
    --dataset train.jsonl \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --output_dir output/Qwen3.8-27B-EntroPack-DDP
```

Enable FSDP2 with `--fsdp fsdp2`:

```shell
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model Qwen3.8-27B-EntroPack \
    --tuner_type lora \
    --dataset train.jsonl \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --fsdp fsdp2 \
    --output_dir output/Qwen3.8-27B-EntroPack-FSDP2
```

EntroPack does not currently support DeepSpeed ZeRO. FSDP2 requires PyTorch 2.7 or later.

## Training results

### Setup

We fine-tuned Qwen3.8-27B on `AI-ModelScope/LaTeX_OCR:human_handwrite` with LoRA. The experiment used 1,200 training samples and 50 validation samples on two NVIDIA H20 GPUs. The per-device batch size was 8, the global batch size was 16, and training ran for one epoch (75 optimizer steps). LoRA rank was 8, alpha was 32, and the learning rate was 1e-4. All settings except quantization were identical.

The compared configurations were:

- EntroPack 2.0 bpp;
- EntroPack 3.0 bpp;
- EntroPack 4.0 bpp;
- BitsAndBytes NF4 4-bit with BF16 compute and double quantization.

### Results

#### Loss and accuracy curves

<p align="center">
  <img src="../../resources/entropack/latex_ocr_train_loss.png" width="49%" alt="LaTeX OCR train loss">
  <img src="../../resources/entropack/latex_ocr_train_token_acc.png" width="49%" alt="LaTeX OCR train token accuracy">
</p>

All four runs converged, with validation token accuracy above 99.5% for every run.

#### Metrics

Weight reconstruction error was computed across all 606 quantized linear layers and 24.806B weights. Quantized model size is the total byte size of every tensor in the model `state_dict`.

| Configuration | Weight relative error | Quantized model size | Stable time per step | Peak memory per GPU |
|---|---:|---:|---:|---:|
| EntroPack 2.0 bpp | 28.69% | **10.57 GiB** | 3.125 s | **19.43 GiB** |
| EntroPack 3.0 bpp | 14.37% | 13.49 GiB | 3.125 s | 22.26 GiB |
| EntroPack 4.0 bpp | **7.19%** | 16.40 GiB | 3.156 s | 25.43 GiB |
| BNB NF4 | 9.29% | 16.67 GiB | **3.063 s** | 25.26 GiB |

EntroPack 4.0 bpp has a similar model size to BNB NF4 with lower weight reconstruction error. Lowering target bpp continuously reduces model size and training memory. EntroPack 2.0 bpp uses about 5.8 GiB less memory per GPU than BNB NF4, at the cost of higher reconstruction error.

#### LoRA inference

After training, all four adapters were evaluated on the same two OCR examples with `temperature=0`. Every configuration produced the correct formula.

| Image | Ground truth | EntroPack 2.0 bpp | EntroPack 3.0 bpp | EntroPack 4.0 bpp | BNB NF4 |
|---|---|---|---|---|---|
| ![](../../resources/entropack/latex_ocr_sample_1.png) | `\frac { \tan \alpha - \tan \beta } { 1 + \tan \alpha \tan \beta }` | Correct | Correct | Correct | Correct |
| ![](../../resources/entropack/latex_ocr_sample_2.png) | `\sqrt { b ^ { 2 } - 4 a c }` | Correct | Correct | Correct | Correct |

## Limitations

The current EntroPack integration in ms-swift supports CUDA devices only.

EntroPack stores weights with entropy coding, so entropy information must be decoded before a linear operation. For compute-heavy workloads such as QLoRA training and LLM prefill, large matrix multiplications can amortize this cost. During token-by-token autoregressive decoding, fewer tokens are processed per step and entropy decoding accounts for a larger portion of runtime. Generation can therefore be slower than with uncompressed weights or fixed-width quantization. For latency-sensitive workloads, benchmark decoding speed with the target model, batch size, and context length.

See the [command-line arguments](../Instruction/Command-line-parameters.md#quantization-arguments) for the complete parameter reference.
