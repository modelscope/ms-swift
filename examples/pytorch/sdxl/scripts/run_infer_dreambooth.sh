PYTHONPATH=../../.. \
CUDA_VISIBLE_DEVICES=0 \
python infer_dreambooth.py \
    --model_path "train_dreambooth" \
    --prompt "A photo of sks dog in a bucket" \
    --image_save_path "dog-bucket.png" \
    --torch_dtype "fp16" \
