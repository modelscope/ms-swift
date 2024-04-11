# Experimental environment: 4 * A100
# 4 * 78GB GPU memory
nproc_per_node=4

NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_id_or_path qwen/Qwen-14B-Chat \
    --model_revision master \
    --sft_type full \
    --tuner_backend peft \
    --template_type AUTO \
    --dtype AUTO \
    --output_dir output \
    --ddp_backend nccl \
    --dataset blossom-math-zh \
    --train_dataset_sample -1 \
    --num_train_epochs 5 \
    --max_length 2048 \
    --check_dataset_strategy warning \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 64 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn true \
    --deepspeed 'default-zero3' \
    --save_only_model true \
