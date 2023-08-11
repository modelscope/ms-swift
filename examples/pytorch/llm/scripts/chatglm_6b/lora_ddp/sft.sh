# # 4 * 15G
nproc_per_node=4
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    src/llm_sft.py \
    --model_type chatglm2-6b \
    --sft_type lora \
    --output_dir runs \
    --ddp_backend gloo \
    --dataset alpaca-en,alpaca-zh \
    --dataset_sample 20000 \
    --max_length 1024 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.1 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn false \
