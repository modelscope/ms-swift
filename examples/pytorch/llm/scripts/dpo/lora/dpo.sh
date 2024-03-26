# Experimental environment: 2*A100
# Memory usage: 2 * 20G
PYTHONPATH=../../.. \
python llm_dpo.py \
    --model_type  yi-6b-chat \
    --ref_model_type  yi-6b-chat \
    --model_revision  master \
    --sft_type  lora \
    --tuner_backend  swift \
    --dtype  AUTO  \
    --output_dir  output  \
    --dataset  hh-rlhf-cn-harmless-base-cn  \
    --train_dataset_sample  -1  \
    --truncation_strategy  truncation_left  \
    --val_dataset_sample  2000  \
    --num_train_epochs  3  \
    --max_length  1024  \
    --max_prompt_length  512  \
    --check_dataset_strategy  none  \
    --lora_rank  8  \
    --lora_alpha  32  \
    --lora_dropout_p  0.05  \
    --lora_target_modules  ALL  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --weight_decay  0.1  \
    --learning_rate  5e-5  \
    --gradient_accumulation_steps  16  \
    --max_grad_norm  1.0  \
    --warmup_ratio  0.03  \
    --eval_steps  2000  \
    --save_steps  2000  \
    --save_total_limit  2  \
    --logging_steps  10 \
