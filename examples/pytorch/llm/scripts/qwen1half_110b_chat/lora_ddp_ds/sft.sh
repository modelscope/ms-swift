# Experimental environment: A100
# 8*60GB GPU memory
# modify 'model_id_or_path' in /swift/swift/llm/utils/model.py （line 1522）
# Alternatively, you can add a new parameter --model_id_or_path and set it to the local path.

PYTHONPATH=/path/to/swift \
nproc_per_node=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
swift sft \
    --model_type qwen1half-110b-chat \
    --sft_type lora \
    --tuner_backend peft \
    --dtype AUTO \
    --output_dir output \
    --ddp_backend nccl \
    --dataset ms-bench-mini \
    --train_dataset_sample 5000 \
    --num_train_epochs 2 \
    --max_length 1024 \
    --check_dataset_strategy warning \
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
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --use_flash_attn false \
    --self_cognition_sample 1000 \
    --model_name 卡卡罗特 \
    --model_author 陶白白 \
    --use_flash_attn false \
    --deepspeed default-zero3
