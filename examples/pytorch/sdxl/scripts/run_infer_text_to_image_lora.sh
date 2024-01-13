PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python infer_text_to_image_lora.py \
    --pretrained_model_name_or_path "AI-ModelScope/stable-diffusion-v1-5" \
    --lora_model_path "train_text_to_image_lora/checkpoint-80000" \
    --prompt "A pokemon with green eyes and red legs." \
    --image_save_path "lora_pokemon.png" \
    --torch_dtype "fp16" \
