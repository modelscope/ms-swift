# Experiment env: A100
# 1 * 30GB GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_sft.py \
  --model_type telechat-12b \
  --dataset dureader-robust-zh \
  --batch_size 1 \
  --max_length 1024 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --use_flash_attn true \
  --eval_steps 1000 \
  --save_steps 1000 \
  --train_dataset_sample -1 \
  --num_train_epochs 2 \
  --check_dataset_strategy none \
  --gradient_checkpointing true \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.03 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --sft_type lora \
  --lora_target_modules DEFAULT \
  --lora_rank 8 \
  --lora_alpha 32 \
  --dtype fp16
