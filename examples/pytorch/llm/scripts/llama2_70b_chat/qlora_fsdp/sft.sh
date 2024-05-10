# 2 GPU * 24G
# bitsandbytes>=0.43.0 needed
nproc_per_node=2

PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --config_file "./scripts/llama2_70b_chat/qlora_fsdp/fsdp_offload.json" \
    llm_sft.py \
    --model_type llama2-70b-chat \
    --model_revision master \
    --sft_type lora \
    --tuner_backend peft \
    --template_type AUTO \
    --dtype bf16 \
    --output_dir output \
    --dataset leetcode-python-en \
    --train_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 2048 \
    --check_dataset_strategy warning \
    --quantization_bit 4 \
    --bnb_4bit_comp_dtype AUTO \
    --bnb_4bit_quant_storage bfloat16 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dtype AUTO \
    --lora_dropout_p 0.05 \
    --lora_target_modules DEFAULT \
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
