PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python infer_dreambooth_lora.py \
    --base_model_path "AI-ModelScope/stable-diffusion-v1-5" \
    --lora_model_path "train_dreambooth_lora" \
    --prompt "A picture of a sks dog in a bucket" \
    --image_save_path "dog-bucket.png" \
    --torch_dtype "fp16" \
