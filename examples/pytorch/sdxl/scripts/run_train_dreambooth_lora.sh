PYTHONPATH=../../.. \
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="AI-ModelScope/stable-diffusion-v1-5" \
  --instance_data_dir="./dog-example" \
  --output_dir="train_dreambooth_lora" \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=50 \
  --seed="0" \
