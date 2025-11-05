# If you don't want to train the router, set:
# `--target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`

# Note: If you need to use DeepSpeed ZeRO-2/ZeRO-3 but encounter hangs
# try using transformers==4.51.3

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --train_type lora \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT#2000' \
              'swift/self-cognition#1000' \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --router_aux_loss_coef 1e-3 \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
