PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python infer_text_to_image.py \
    --pretrained_model_name_or_path "AI-ModelScope/stable-diffusion-v1-5" \
    --unet_model_path "train_text_to_image/checkpoint-15000/unet" \
    --prompt "yoda" \
    --image_save_path "yoda-pokemon.png" \
    --torch_dtype "fp16" \
