PYTHONPATH=../../.. \
accelerate launch train_controlnet.py \
    --pretrained_model_name_or_path="AI-ModelScope/stable-diffusion-v1-5" \
    --output_dir="train_controlnet" \
    --dataset_name="AI-ModelScope/controlnet_dataset_condition_fill50k" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
    --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
    --train_batch_size=4 \
