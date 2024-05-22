# Experiment env: A10, RTX3090/4090, A100
CUDA_VISIBLE_DEVICES=0 \
swift sft \
  --model_type codeqwen1half-7b-chat-awq \
  --dataset leetcode-python-en \
  --batch_size 4 \
  --max_length 2048 \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --use_flash_attn true \
  --eval_steps 2000 \
  --save_steps 2000 \
  --num_train_epochs 3 \
  --check_dataset_strategy none \
  --gradient_checkpointing true \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.03 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --sft_type lora \
  --lora_target_modules ALL \
  --lora_rank 8 \
  --lora_alpha 32
