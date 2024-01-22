PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python infer_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path "AI-ModelScope/stable-diffusion-xl-base-1.0" \
    --lora_model_path "train_text_to_image_lora_sdxl/unet" \
    --prompt "A pokemon with green eyes and red legs." \
    --image_save_path "sdxl_lora_pokemon.png" \
    --torch_dtype "fp16" \
