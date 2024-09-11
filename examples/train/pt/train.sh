nproc_per_node=2

PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0,1 \
swift pt \
    --model_id_or_path LLM-Research/Meta-Llama-3.1-8B \
    --sft_type full \
    --dataset chinese-c4 \
    --num_train_epochs 1 \
    --batch_size 1 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps $(expr 256 / $nproc_per_node) \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --streaming true \
    --save_total_limit 2 \
    --logging_steps 5
