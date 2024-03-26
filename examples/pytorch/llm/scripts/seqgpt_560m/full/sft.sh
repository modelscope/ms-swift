# Experimental environment: A10
# 12GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_sft.py \
    --model_id_or_path damo/nlp_seqgpt-560m \
    --model_revision master \
    --sft_type full \
    --template_type default-generation \
    --dtype AUTO \
    --output_dir output \
    --dataset ner-jave-zh \
    --train_dataset_sample -1 \
    --num_train_epochs 3 \
    --max_length 1024 \
    --check_dataset_strategy warning \
    --gradient_checkpointing true \
    --batch_size 4 \
    --weight_decay 0.1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_only_model false \
    --save_total_limit 2 \
    --logging_steps 10 \
