PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python infer_controlnet.py \
    --base_model_path "AI-ModelScope/stable-diffusion-v1-5" \
    --controlnet_path "train_controlnet" \
    --prompt "pale golden rod circle with old lace background" \
    --control_image_path "conditioning_image_1.png" \
    --image_save_path "output.png" \
    --torch_dtype "fp16" \
    --seed 0 \
