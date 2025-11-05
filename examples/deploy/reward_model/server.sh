CUDA_VISIBLE_DEVICES=0 swift deploy \
    --host 0.0.0.0 \
    --port 8000 \
    --model Shanghai_AI_Laboratory/internlm2-1_8b-reward \
    --infer_backend pt
