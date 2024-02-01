PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python infer_controlnet_sdxl.py \
    --base_model_path "AI-ModelScope/stable-diffusion-xl-base-1.0" \
    --controlnet_path "train_controlnet_sdxl" \
    --prompt "pale golden rod circle with old lace background" \
    --control_image_path "conditioning_image_1.png" \
    --image_save_path "output.png" \
    --torch_dtype "fp16" \
    --seed 0 \
