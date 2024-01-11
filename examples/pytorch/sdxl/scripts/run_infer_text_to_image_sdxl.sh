PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python infer_text_to_image_sdxl.py \
    --pretrained_model_name_or_path "AI-ModelScope/stable-diffusion-xl-base-1.0" \
    --unet_model_path "train_text_to_image_sdxl/checkpoint-10000/unet" \
    --prompt "A pokemon with green eyes and red legs." \
    --image_save_path "sdxl_pokemon.png" \
    --torch_dtype "fp16" \
