# v5 SFT with the Liger fused kernels (causal_lm, single GPU).
#
# `--use_liger_kernel true` patches the HF decoder's per-layer ops (rms_norm / rotary / swiglu / ...) in
# place via twinkle.kernel.kernelize. Adding `fused_linear_cross_entropy` to `--liger_kernel_config`
# (a JSON object) additionally fuses the lm_head GEMM into the loss, so the forward never materialises
# the full-vocab logits -- a large memory win. dict-valued Config fields are passed as a JSON string.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --use_liger_kernel true \
    --liger_kernel_config '{"fused_linear_cross_entropy": true}' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --logging_steps 5 \
    --save_steps 100 \
    --output_dir output \
    --report_to tensorboard
