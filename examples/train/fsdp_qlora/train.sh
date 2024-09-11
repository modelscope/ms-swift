nproc_per_node=2

PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --config_file "./fsdp_offload.json" \
    swift/cli/sft.py \
    --model_id_or_path LLM-Research/Meta-Llama-3.1-70B-Instruct \
    --sft_type lora \
    --dtype bf16 \
    --dataset self-cognition#1000 \
    --num_train_epochs 1 \
    --max_length 2048 \
    --quantization_bit 4 \
    --bnb_4bit_comp_dtype AUTO \
    --bnb_4bit_quant_storage bfloat16 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --model_author swift \
    --model_name swift-robot
