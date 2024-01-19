PYTHONPATH=../../../ \
accelerate launch train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="AI-ModelScope/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="AI-ModelScope/sdxl-vae-fp16-fix" \
  --dataset_name="AI-ModelScope/pokemon-blip-captions" \
  --caption_column="text" \
  --resolution=1024 \
  --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=2 \
  --checkpointing_steps=500 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="train_text_to_image_lora_sdxl" \
  --validation_prompt="cute dragon creature" \
  --report_to="tensorboard" \
