# If `num_labels` is provided, it will be considered a classification task,
# and AutoModelForSequenceClassification will be used to load the model.
# The BERT model does not require templates, so it can usually be used without registration.
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model AI-ModelScope/bert-base-chinese \
    --train_type lora \
    --dataset 'DAMO_NLP/jd:cls#2000' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 512 \
    --truncation_strategy right \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --num_labels 2 \
    --task_type seq_cls
