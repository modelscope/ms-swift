PYTHONPATH=../../../ \
accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="AI-ModelScope/stable-diffusion-v1-5" \
  --dataset_name="AI-ModelScope/pokemon-blip-captions" \
  --caption_column="text" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 \
  --checkpointing_steps=5000 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --seed=42 \
  --output_dir="train_text_to_image_lora" \
  --validation_prompt="cute dragon creature" \
  --report_to="tensorboard" \
