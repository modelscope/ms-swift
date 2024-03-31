# Experiment env: A10, RTX3090/4090, A100
# 1 * 17G GPU memory
PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python llm_sft.py \
  --model_type qwen1half-7b-chat-awq \
  --dataset ms-agent \
  --train_dataset_mix_ratio 3 \
  --batch_size 4 \
  --max_length 1024 \
  --use_loss_scale true \
  --gradient_accumulation_steps 2 \
  --learning_rate 5e-5 \
  --use_flash_attn true \
  --eval_steps 2000 \
  --save_steps 2000 \
  --train_dataset_sample -1 \
  --num_train_epochs 1 \
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
