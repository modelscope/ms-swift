CUDA_VISIBLE_DEVICES=0,1 \
swift infer \
    --model OpenGVLab/InternVL2_5-1B \
    --infer_backend lmdeploy \
    --val_dataset AI-ModelScope/captcha-images#1000 \
    --lmdeploy_tp 2 \
    --lmdeploy_vision_batch_size 8 \
    --max_new_tokens 2048
