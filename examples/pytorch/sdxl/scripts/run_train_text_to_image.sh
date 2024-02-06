PYTHONPATH=../../../ \
accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
  --pretrained_model_name_or_path="AI-ModelScope/stable-diffusion-v1-5" \
  --dataset_name="AI-ModelScope/pokemon-blip-captions" \
  --use_ema \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --checkpointing_steps=500 \
  --checkpoints_total_limit 2 \
  --output_dir="train_text_to_image" \
