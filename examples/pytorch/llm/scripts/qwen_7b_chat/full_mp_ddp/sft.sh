# Experimental environment: 4 * A100
# 4 * 55GB GPU memory (use flash_attn)
nproc_per_node=2

PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
    --nproc_per_node=$nproc_per_node \
    --master_port 29500 \
    llm_sft.py \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --model_revision master \
    --sft_type full \
    --template_type AUTO \
    --dtype AUTO \
    --output_dir output \
    --dataset medical-en medical-zh \
    --train_dataset_sample 200000 \
    --num_train_epochs 1 \
    --max_length 8192 \
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
    --use_flash_attn true \
    --preprocess_num_proc 4 \
