PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python infer_dreambooth_lora_sdxl.py \
    --base_model_path "AI-ModelScope/stable-diffusion-xl-base-1.0" \
    --lora_model_path "train_dreambooth_lora_sdxl" \
    --prompt "A picture of a sks dog in a bucket" \
    --image_save_path "sks_dog.png" \
    --torch_dtype "fp16" \
