# 14GiB * 2
nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --config_file "./examples/train/fsdp_qlora/fsdp_offload.json" \
    swift/cli/sft.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset 'swift/self-cognition#1000' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --max_length 2048 \
    --quant_bits 4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_quant_storage bfloat16 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_checkpointing true \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --model_author swift \
    --model_name swift-robot
