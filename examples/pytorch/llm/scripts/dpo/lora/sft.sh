# Experimental environment: A10, 3090
# 16GB GPU memory
# Recommended to use `qwen_14b_chat_int4`
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_dpo.py \
    --model_type  qwen-7b \
    --ref_model_type  qwen-7b \
    --model_revision  master \
    --sft_type  lora \
    --tuner_backend  swift \
    --template_type  chatml \
    --dtype  AUTO  \
    --output_dir  output  \
    --dataset  stack-exchange-paired  \
    --train_dataset_sample  20000  \
    --truncation_strategy  truncation_left  \
    --val_dataset_sample  2000  \
    --num_train_epochs  1  \
    --max_length  2048  \
    --check_dataset_strategy  none  \
    --lora_rank  8  \
    --lora_alpha  32  \
    --lora_dropout_p  0.05  \
    --lora_target_modules  ALL  \
    --gradient_checkpointing  true  \
    --batch_size  1  \
    --weight_decay  0.01  \
    --learning_rate  1e-4  \
    --gradient_accumulation_steps  16  \
    --max_grad_norm  1.0  \
    --warmup_ratio  0.03  \
    --eval_steps  10000  \
    --save_steps  500  \
    --save_total_limit  2  \
    --logging_steps  10 \
