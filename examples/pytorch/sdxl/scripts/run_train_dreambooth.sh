PYTHONPATH=../../.. \
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="AI-ModelScope/stable-diffusion-v1-5" \
  --instance_data_dir="./dog-example" \
  --output_dir="train_dreambooth" \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
