CUDA_VISIBLE_DEVICES=0,1 \
swift infer \
    --model OpenGVLab/InternVL2_5-1B \
    --infer_backend lmdeploy \
    --val_dataset AI-ModelScope/captcha-images#1000 \
    --tp 2 \
    --vision_batch_size 8
