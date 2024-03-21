# Experimental environment: A10, A100
# 16GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_sft.py \
    --model_type  yi-6b-chat \
    --lora_lr_ratio 16.0 \
    --sft_type  lora \
    --tuner_backend  swift \
    --dtype  AUTO \
    --output_dir  output \
    --dataset  ms-agent \
    --use_loss_scale  true \
    --train_dataset_sample  50000 \
    --train_dataset_mix_ratio  2.0 \
    --num_train_epochs 2 \
    --max_length  2048 \
    --check_dataset_strategy  warning \
    --lora_rank  8 \
    --lora_alpha  32 \
    --lora_dropout_p  0.05 \
    --lora_target_modules  ALL \
    --gradient_checkpointing  true \
    --batch_size  1 \
    --weight_decay  0.1 \
    --learning_rate  1e-4 \
    --gradient_accumulation_steps  16 \
    --max_grad_norm  0.5 \
    --warmup_ratio  0.03 \
    --eval_steps  500 \
    --save_steps  500 \
    --save_total_limit  2 \
    --logging_steps  10 \
