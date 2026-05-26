# The generated LoRA delta weights cannot be merged into an FP8 base model via Merge-LoRA.
# Due to the limited precision of FP8, the LoRA delta will be rounded to 0.
# However, you can use BF16 weights to perform Merge-LoRA.

# Although the model passed in here is BF16, it will be converted to FP8
# after being loaded as a Megatron model
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
megatron sft \
    --model Qwen/Qwen3.5-4B \
    --save_safetensors true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
    --model_author swift \
    --model_name swift-robot \
    --merge_lora false \
    --linear_decoupled_in_proj true \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --fp8_recipe blockwise \
    --fp8_format e4m3 \
    --fp8_param_gather true \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --tensor_model_parallel_size 2 \
    --micro_batch_size 1 \
    --global_batch_size 2 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --packing true \
    --finetune true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --output_dir megatron_output/Qwen3.5-4B \
    --eval_steps 200 \
    --save_steps 200 \
    --max_length 4096 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --mtp_num_layers 1 \
    --attention_backend flash

# Merge-LoRA
# FP8 base model + BF16 LoRA inference requires inference framework support
# Alternatively, you can use BF16 base model + BF16 LoRA for inference
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
megatron export \
    --model Qwen/Qwen3.5-4B \
    --adapters megatron_output/Qwen3.5-4B/vx-xxx/checkpoint-xxx \
    --output_dir megatron_output/Qwen3.5-4B/vx-xxx/checkpoint-xxx-merged \
    --to_hf true \
    --linear_decoupled_in_proj true \
    --mtp_num_layers 1 \
    --merge_lora true


# Inference with merged weights
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model megatron_output/Qwen3.5-4B/vx-xxx/checkpoint-xxx-merged \
    --stream true \
    --enable_thinking false
