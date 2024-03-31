# Experimental environment: 2 * 3090
# 2 * 24GB GPU memory
nproc_per_node=2

PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    llm_sft.py \
    --model_id_or_path qwen/Qwen-1_8B-Chat \
    --model_revision master \
    --sft_type full \
    --template_type AUTO \
    --dtype AUTO \
    --output_dir output \
    --ddp_backend nccl \
    --dataset jd-sentiment-zh \
    --train_dataset_sample -1 \
    --val_dataset_sample 1000 \
    --num_train_epochs 1 \
    --max_length 2048 \
    --check_dataset_strategy warning \
    --gradient_checkpointing true \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_only_model true \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn false \
