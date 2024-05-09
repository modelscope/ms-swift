# Experimental environment: 2 * 3090
# 2 * 22GB GPU memory
nproc_per_node=4

NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type chatglm3-6b-32k \
    --model_revision master \
    --sft_type lora \
    --tuner_backend peft \
    --dtype AUTO \
    --output_dir output \
    --ddp_backend nccl \
    --dataset long-alpaca-12k \
    --train_dataset_sample -1 \
    --num_train_epochs 1 \
    --max_length 16000 \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 10000 \
    --save_steps 10000 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --save_only_model true \
    --sequence_parallel_size 4 \
    --pack_to_max_length false \
