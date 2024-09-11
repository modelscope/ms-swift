nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type orpo \
    --model_id_or_path LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --sft_type full \
    --dataset shareai-llama3-dpo-zh-en-emoji \
    --streaming true \
    --num_train_epochs 1 \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5