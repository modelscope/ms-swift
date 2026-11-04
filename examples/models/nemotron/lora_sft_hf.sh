# Key constraints for this model:
#   1. target_modules: Do NOT use `all-linear`. The Mamba mixer's in_proj/out_proj
#      cause NaN in backward when wrapped with LoRA. Use only attention + MLP modules.
#   2. DeepSpeed
#      ZeRO-3 + GA>1 corrupts LoRA adapters after step 1 (loss/token_acc → 0).
#      ZeRO-3 + GA=1 is OK; ZeRO-2 + GA>1 is OK.
#   3. experts_impl: Use `grouped_mm` for MoE efficiency.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model nv-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
    --tuner_type lora \
    --target_modules q_proj k_proj v_proj o_proj \
    --dataset 'swift/self-cognition#1000' \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --router_aux_loss_coef 1e-3 \
    --experts_impl grouped_mm \
    --gradient_accumulation_steps 4 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --attn_impl flash_attn \
    --padding_free true
